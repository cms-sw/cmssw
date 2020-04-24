#ifndef FastSimulation_Tracking_SeedingTree_h
#define FastSimulation_Tracking_SeedingTree_h

#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_set>

template<class DATA>
class SeedingNode
{
    protected:
        const DATA _data;
        const std::vector<SeedingNode<DATA>*>& _allNodes;
        unsigned int _index; //the index in allNodes
        int _parentIndex; //the parent's index in allNodes
        int _childIndex; //the index of this Node in its parent's children vector
        unsigned int _depth; //the depth within the tree (for root: depth=0 && parentIndex=-1
        std::vector<unsigned int> _children;

    public:
        SeedingNode(const DATA& data, std::vector<SeedingNode*>& allNodes, int parentIndex=-1):
            _data(data),
            _allNodes(allNodes),
            _index(allNodes.size()),
            _parentIndex(parentIndex)
        {
            allNodes.push_back(this);
        
            if (_parentIndex>=0)
            {
                SeedingNode* parent = allNodes[_parentIndex];
                _depth=parent->_depth+1;
                _childIndex=parent->_children.size();
                parent->_children.push_back(_index);
                
            }
            else
            {
                _depth=0;
                _childIndex=-1;
            }
        }
        
        void sort(std::vector<SeedingNode<DATA>*>& allNodes,unsigned int parentIndex)
        {
            _parentIndex=parentIndex;
            _index=allNodes.size();
            if (_parentIndex>=0)
            {
                allNodes[_parentIndex]->_children[_childIndex]=_index;
            }
            allNodes.push_back(this);
            for (unsigned int ichild=0; ichild<_children.size();++ichild)
            {
                _allNodes[_children[ichild]]->sort(allNodes,_index);
            }
        }
        
        bool insert(const std::vector<DATA>& dataList, std::vector<SeedingNode<DATA>*>& allNodes)
        {
            if (_depth+1>=dataList.size())
            {
                return false;
            }
            for (unsigned int ichild=0; ichild<_children.size();++ichild)
            {
                if (allNodes[_children[ichild]]->getData()==dataList[_depth+1])
                {
                    return allNodes[_children[ichild]]->insert(dataList,allNodes);
                }
            }
            SeedingNode<DATA>* node = new SeedingNode<DATA>(dataList[_depth+1],allNodes,_index);
            if (node->getDepth()+1>=dataList.size())
            {
                return true;
            }
            return node->insert(dataList,allNodes);
        }
        
        inline unsigned int getDepth() const
        {
            return _depth;
        }
        
        inline const SeedingNode<DATA>* next() const
        {
            if (_index+1<_allNodes.size())
            {
                return _allNodes[_index+1];
            }
            return nullptr;
        }
       
        inline const SeedingNode<DATA>* firstChild() const
        {
            if (!_children.empty())
            {
                return _allNodes[_children[0]];
            }
            return nullptr;
        }
        
        inline unsigned int getIndex() const
        {
            return _index;
        }
        
        inline const SeedingNode* getParent() const
        {
            if (_parentIndex>=0)
            {
                return _allNodes[_parentIndex];
            }
            return nullptr;
        }
        
        inline unsigned int getChildrenSize() const
        {
            return _children.size();
        }
        
        inline const SeedingNode<DATA>* getChild(unsigned int ichild) const
        {
            return _allNodes[_children[ichild]];
        }
        
        inline unsigned int getChildIndex() const
        {
            return _childIndex;
        }
        
        inline const DATA& getData() const
        {
            return _data;
        }
        
        void print() const
        {
            
            printf("index=%3i, depth=%2i, childIndex=%2i:  ",_index,_depth,_childIndex);
            for (unsigned int i=0;i<_depth;++i)
            {
                std::cout<<" -- ";
            }
            printf("[%s, %s] \r\n",_data.toString().c_str(),_data.toIdString().c_str());
            
        }
        void printRecursive() const
        {
            print();
            for (unsigned int ichild=0; ichild<_children.size(); ++ichild)
            {
                _allNodes[_children[ichild]]->printRecursive();
            }
        }
};

template<class DATA>
class SeedingTree
{
    public:
        typedef std::unordered_set<DATA,typename DATA:: hashfct, typename DATA:: eqfct> SingleSet;
    protected:
        std::vector<SeedingNode<DATA>*> _roots;
        std::vector<SeedingNode<DATA>*> _allNodes;
        
        SingleSet _singleSet;
    public:
        
        //returns true if successfully inserted into tree
        bool insert(const std::vector<DATA>& dataList)
        {
            for (unsigned int i = 0; i< dataList.size(); ++i)
            {
                _singleSet.insert(dataList[i]);
            }

            if (dataList.empty())
            {
                return false;
            }
            for (unsigned int iroot=0; iroot<_roots.size();++iroot)
            {
                if (_roots[iroot]->getData()==dataList[0])
                {
                    return _roots[iroot]->insert(dataList,_allNodes);
                }
            }
            SeedingNode<DATA>* node = new SeedingNode<DATA>(dataList[0],_allNodes);
            _roots.push_back(node);
            return node->insert(dataList,_allNodes);
        }
        
        inline const SingleSet& getSingleSet() const
        {
            return _singleSet;
        }
        
        void sort()
        {
            //this setups depth first ordered indexes.
            std::vector<SeedingNode<DATA>*> allNodes;
            for (unsigned int iroot=0; iroot<_roots.size();++iroot)
            {
                _roots[iroot]->sort(allNodes,-1);
            }
            _allNodes=allNodes;
        }
        
        
        
        inline unsigned int numberOfRoots() const
        {
            return _roots.size();
        }
        
        inline unsigned int numberOfNodes() const
        {
            return _allNodes.size();
        }
        
        inline const SeedingNode<DATA>* getRoot(unsigned int i) const
        {
            if (i<_roots.size())
            {
                return _roots[i];
            }
            return nullptr;
        }
        
        void printRecursive() const
        {
            std::cout<<"SeedingTree: n="<<_allNodes.size()<<" [recursive]"<<std::endl;
            for (unsigned int iroot=0; iroot<_roots.size();++iroot)
            {
                _roots[iroot]->printRecursive();
            }
        }
        void printOrdered() const
        {
            std::cout<<"SeedingTree: n="<<_allNodes.size()<<" [ordered]"<<std::endl;
            for (unsigned int inode=0; inode<_allNodes.size();++inode)
            {
                _allNodes[inode]->print();
            }
        }
        
        void print() const
        {
            std::cout<<"SeedingTree: n="<<_allNodes.size()<<std::endl;
            for (unsigned int inode=0; inode<_allNodes.size();++inode)
            {
                _allNodes[inode]->print();
            }
        }
        
        ~SeedingTree()
        {
            for (unsigned int iroot=0; iroot<_roots.size();++iroot)
            {
                delete _roots[iroot];
            }
            _roots.clear();
        }
};

#endif


#ifndef FastSimulation_Tracking_SeedingTree_h
#define FastSimulation_Tracking_SeedingTree_h

#include <vector>
#include <iostream>


template<class DATA>
class SeedingNode
{
    protected:
        const DATA _data;
        const std::vector<SeedingNode<DATA>*>& _allNodes;
        const unsigned int _index; //the index in allNodes
        const int _parentIndex; //the parent's index in allNodes
        int _childIndex; //the index of this Node in its parent's children vector
        unsigned int _depth; //the depth within the tree (for root: depth=0 && parentIndex=-1
        std::vector<unsigned int> _children;
        
    public:
        SeedingNode(const DATA& data, std::vector<SeedingNode*>& allNodes, const int parentIndex=-1):
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
                    //std::cout<<"insert: has child"<<std::endl;
                    return allNodes[_children[ichild]]->insert(dataList,allNodes);
                }
            }
            //std::cout<<"insert: create child"<<std::endl;
            SeedingNode<DATA>* node = new SeedingNode<DATA>(dataList[_depth+1],allNodes,_index);
            return node->insert(dataList,allNodes);
        }
        
        inline unsigned int getDepth() const
        {
            return _depth;
        }
        
        inline const SeedingNode<DATA>* next() const
        {
            if (_index+1>=_allNodes.size())
            {
                return nullptr;
            }
            else
            {
                return _allNodes[_index+1];
            }
        }
        
        inline const SeedingNode<DATA>* nextSibling() const
        {
            if (_childIndex+1>=getParent()->_children.size())
            {
                return nullptr;
            }
            else
            {
                return _allNodes[getParent()->_children[_childIndex+1]];
            }
        }
        
        inline unsigned int getIndex() const
        {
            return _index;
        }
        
        inline const SeedingNode* getParent() const
        {
            return _allNodes[_parentIndex];
        }
        
        inline unsigned int getNChildren() const
        {
            return _children.size();
        }
        
        inline unsigned int getChildIndex() const
        {
            return _childIndex;
        }
        
        inline const DATA& getData() const
        {
            return _data;
        }
        
        void print()
        {
            
            printf("index=%3i, depth=%2i, childIndex=%2i:  ",_index,_depth,_childIndex);
            for (unsigned int i=0;i<_depth;++i)
            {
                std::cout<<" -- ";
            }
            printf("[%s] \r\n",_data.print().c_str());
            
        }
};

template<class DATA>
class SeedingTree
{
    protected:
        std::vector<SeedingNode<DATA>*> _roots;
        std::vector<SeedingNode<DATA>*> _allNodes;
    public:
        SeedingTree()
        {
        }
        
        //returns true if successfully inserted into tree
        bool insert(const std::vector<DATA>& dataList)
        {
            if (dataList.size()==0)
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
        
        inline unsigned int size() const
        {
            return _allNodes.size();
        }
        
        inline const SeedingNode<DATA>* getFirst() const
        {
            if (size()>0)
            {
                return _allNodes[0];
            }
            else
            {
                return nullptr;
            }
        }
        
        void print()
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


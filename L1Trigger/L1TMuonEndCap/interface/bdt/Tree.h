// Tree.h

#ifndef L1Trigger_L1TMuonEndCap_emtf_Tree
#define L1Trigger_L1TMuonEndCap_emtf_Tree

#include <list>
#include "Node.h"
#include "TXMLEngine.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"

namespace emtf {

//class Node;

class Tree
{
    public:
        Tree();
        Tree(std::vector< std::vector<Event*> >& cEvents);
        ~Tree();

        Tree(const Tree &tree);
        Tree& operator=(const Tree &tree);
        Tree(Tree && tree);

        void setRootNode(Node *sRootNode);
        Node* getRootNode();

        void setTerminalNodes(std::list<Node*>& sTNodes);
        std::list<Node*>& getTerminalNodes();

        Int_t getNumTerminalNodes();

        void buildTree(Int_t nodeLimit);
        void calcError();
        void filterEvents(std::vector<Event*>& tEvents);
        void filterEventsRecursive(Node* node);
        Node* filterEvent(Event* e);
        Node* filterEventRecursive(Node* node, Event* e);

        void saveToXML(const char* filename);
        void saveToXMLRecursive(TXMLEngine* xml, Node* node, XMLNodePointer_t np);
        void addXMLAttributes(TXMLEngine* xml, Node* node, XMLNodePointer_t np);

        void loadFromXML(const char* filename);
        void loadFromXMLRecursive(TXMLEngine* xml, XMLNodePointer_t node, Node* tnode);
        void loadFromCondPayload(const L1TMuonEndCapForest::DTree& tree);
        void loadFromCondPayloadRecursive(const L1TMuonEndCapForest::DTree& tree, const L1TMuonEndCapForest::DTreeNode& node, Node* tnode);

        void rankVariables(std::vector<Double_t>& v);
        void rankVariablesRecursive(Node* node, std::vector<Double_t>& v);

        void getSplitValues(std::vector<std::vector<Double_t>>& v);
        void getSplitValuesRecursive(Node* node, std::vector<std::vector<Double_t>>& v);

        Double_t getBoostWeight(void) const   { return boostWeight; }
        void     setBoostWeight(Double_t wgt) { boostWeight =  wgt; }

    private:
        Node *rootNode;
        std::list<Node*> terminalNodes;
        Int_t numTerminalNodes;
        Double_t rmsError;
        Double_t boostWeight;
        unsigned xmlVersion; // affects only XML loading part, save uses an old format and looses the boostWeight

        // this is the main recursive workhorse function that compensates for Nodes being non-copyable
        Node* copyFrom(const Node *local_root); // no garantees if throws in the process
        // a dumb DFS tree traversal
        void findLeafs(Node *local_root, std::list<Node*> &tn);
};

} // end of emtf namespace

#endif

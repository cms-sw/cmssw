// Tree.h

#ifndef L1Trigger_L1TMuonEndCap_emtf_Tree
#define L1Trigger_L1TMuonEndCap_emtf_Tree

#include <list>
#include <memory>
#include "Node.h"
#include "TXMLEngine.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"

namespace emtf {

  //class Node;

  class Tree {
  public:
    Tree();
    Tree(std::vector<std::vector<Event*>>& cEvents);
    ~Tree() = default;

    Tree(const Tree& tree);
    Tree& operator=(const Tree& tree);
    Tree(Tree&& tree) noexcept = default;

    void setRootNode(Node* sRootNode);
    Node* getRootNode();
    const Node* getRootNode() const;

    void setTerminalNodes(std::list<Node*>& sTNodes);
    std::list<Node*>& getTerminalNodes();

    int getNumTerminalNodes();

    void buildTree(int nodeLimit);
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
    void loadFromCondPayloadRecursive(const L1TMuonEndCapForest::DTree& tree,
                                      const L1TMuonEndCapForest::DTreeNode& node,
                                      Node* tnode);

    void rankVariables(std::vector<double>& v) const;
    void rankVariablesRecursive(Node* node, std::vector<double>& v) const;

    void getSplitValues(std::vector<std::vector<double>>& v) const;
    void getSplitValuesRecursive(Node* node, std::vector<std::vector<double>>& v) const;

    double getBoostWeight(void) const { return boostWeight; }
    void setBoostWeight(double wgt) { boostWeight = wgt; }

  private:
    std::unique_ptr<Node> rootNode;
    std::list<Node*> terminalNodes;
    int numTerminalNodes;
    double rmsError;
    double boostWeight;
    unsigned xmlVersion;  // affects only XML loading part, save uses an old format and looses the boostWeight

    // this is the main recursive workhorse function that compensates for Nodes being non-copyable
    std::unique_ptr<Node> copyFrom(const Node* local_root);  // no garantees if throws in the process
    // a dumb DFS tree traversal
    void findLeafs(Node* local_root, std::list<Node*>& tn);
  };

}  // namespace emtf

#endif

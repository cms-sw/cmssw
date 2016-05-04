// Tree.h

#ifndef ADD_TREE
#define ADD_TREE

#include <list>
#include "L1Trigger/L1TMuonEndCap/interface/Node.h"
#include "TXMLEngine.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


//class Node;

class Tree
{
    public:
        Tree();
        Tree(std::vector< std::vector<Event*> >& cEvents);
        ~Tree();

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

        void rankVariables(std::vector<Double_t>& v);
        void rankVariablesRecursive(Node* node, std::vector<Double_t>& v);

        void getSplitValues(std::vector<std::vector<Double_t>>& v);
        void getSplitValuesRecursive(Node* node, std::vector<std::vector<Double_t>>& v);

    private:
        Node *rootNode;
        std::list<Node*> terminalNodes;
        Int_t numTerminalNodes;
        Double_t rmsError;
};

#endif

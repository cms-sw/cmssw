//////////////////////////////////////////////////////////////////////////
//                            Tree.cxx                                  //
// =====================================================================//
// This is the object implementation of a decision tree.                //
// References include                                                   //
//    *Elements of Statistical Learning by Hastie,                      //
//     Tibshirani, and Friedman.                                        //
//    *Greedy Function Approximation: A Gradient Boosting Machine.      //
//     Friedman. The Annals of Statistics, Vol. 29, No. 5. Oct 2001.    //
//    *Inductive Learning of Tree-based Regression Models. Luis Torgo.  //    
//                                                                      //
//////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// _______________________Includes_______________________________________//
///////////////////////////////////////////////////////////////////////////

#include "L1Trigger/L1TMuonEndCap/interface/Tree.h"
#include <iostream>
#include <sstream>

//////////////////////////////////////////////////////////////////////////
// _______________________Constructor(s)________________________________//
//////////////////////////////////////////////////////////////////////////

Tree::Tree()
{
    rootNode = new Node("root");

    terminalNodes.push_back(rootNode);
    numTerminalNodes = 1;
}

Tree::Tree(std::vector< std::vector<Event*> >& cEvents)
{
    rootNode = new Node("root");
    rootNode->setEvents(cEvents);

    terminalNodes.push_back(rootNode);
    numTerminalNodes = 1;
}
//////////////////////////////////////////////////////////////////////////
// _______________________Destructor____________________________________//
//////////////////////////////////////////////////////////////////////////


Tree::~Tree()
{
// When the tree is destroyed it will delete all of the nodes in the tree.
// The deletion begins with the rootnode and continues recursively.
    delete rootNode;
}

//////////////////////////////////////////////////////////////////////////
// ______________________Get/Set________________________________________//
//////////////////////////////////////////////////////////////////////////

void Tree::setRootNode(Node *sRootNode)
{
    rootNode = sRootNode;
}
 
Node * Tree::getRootNode()
{
     return rootNode;
}

// ----------------------------------------------------------------------

void Tree::setTerminalNodes(std::list<Node*>& sTNodes)
{
    terminalNodes = sTNodes;
}

std::list<Node*>& Tree::getTerminalNodes()
{
    return terminalNodes;
}

// ----------------------------------------------------------------------

Int_t Tree::getNumTerminalNodes()
{
    return numTerminalNodes;
}

//////////////////////////////////////////////////////////////////////////
// ______________________Performace_____________________________________//
//////////////////////////////////////////////////////////////////////////

void Tree::calcError() 
{ 
// Loop through the separate predictive regions (terminal nodes) and 
// add up the errors to get the error of the entire space.  
 
    Double_t totalSquaredError = 0; 
 
    for(std::list<Node*>::iterator it=terminalNodes.begin(); it!=terminalNodes.end(); it++) 
    { 
        totalSquaredError += (*it)->getTotalError();  
    } 
    rmsError = sqrt( totalSquaredError/rootNode->getNumEvents() ); 
} 

// ----------------------------------------------------------------------

void Tree::buildTree(Int_t nodeLimit)
{
    // We greedily pick the best terminal node to split.
    Double_t bestNodeErrorReduction = -1;
    Node* nodeToSplit = 0;

    if(numTerminalNodes == 1)
    {   
        rootNode->calcOptimumSplit();
        calcError();
//        std::cout << std::endl << "  " << numTerminalNodes << " Nodes : " << rmsError << std::endl;
    }

    for(std::list<Node*>::iterator it=terminalNodes.begin(); it!=terminalNodes.end(); it++)
    {   
       if( (*it)->getErrorReduction() > bestNodeErrorReduction ) 
       {   
           bestNodeErrorReduction = (*it)->getErrorReduction();
           nodeToSplit = (*it);
       }    
    }   

    //std::cout << "nodeToSplit size = " << nodeToSplit->getNumEvents() << std::endl;

    // If all of the nodes have one event we can't add any more nodes and reduce the error.
    if(nodeToSplit == 0) return;

    // Create daughter nodes, and link the nodes together appropriately.
    nodeToSplit->theMiracleOfChildBirth();

    // Get left and right daughters for reference.
    Node* left = nodeToSplit->getLeftDaughter();
    Node* right = nodeToSplit->getRightDaughter();
 
    // Update the list of terminal nodes.
    terminalNodes.remove(nodeToSplit);
    terminalNodes.push_back(left);
    terminalNodes.push_back(right);
    numTerminalNodes++;

    // Filter the events from the parent into the daughters.
    nodeToSplit->filterEventsToDaughters();  

    // Calculate the best splits for the new nodes.
    left->calcOptimumSplit();
    right->calcOptimumSplit();

    // See if the error reduces as we add more nodes.
    calcError();
 
    if(numTerminalNodes % 1 == 0)
    {
//        std::cout << "  " << numTerminalNodes << " Nodes : " << rmsError << std::endl;
    }

    // Repeat until done.
    if(numTerminalNodes <  nodeLimit) buildTree(nodeLimit);
}

// ----------------------------------------------------------------------

void Tree::filterEvents(std::vector<Event*>& tEvents)
{
// Use trees which have already been built to fit a bunch of events
// given by the tEvents vector.

    // Set the events to be filtered.
    rootNode->getEvents() = std::vector< std::vector<Event*> >(1);
    rootNode->getEvents()[0] = tEvents;

    // The tree now knows about the events it needs to fit.
    // Filter them into a predictive region (terminal node).
    filterEventsRecursive(rootNode);
}

// ----------------------------------------------------------------------

void Tree::filterEventsRecursive(Node* node)
{
// Filter the events repeatedly into the daughter nodes until they
// fall into a terminal node.

    Node* left = node->getLeftDaughter();
    Node* right = node->getRightDaughter();

    if(left == 0 || right == 0) return;

    node->filterEventsToDaughters();

    filterEventsRecursive(left);
    filterEventsRecursive(right);
}

// ----------------------------------------------------------------------

Node* Tree::filterEvent(Event* e)
{
// Use trees which have already been built to fit a bunch of events
// given by the tEvents vector.

    // Filter the event into a predictive region (terminal node).
    Node* node = filterEventRecursive(rootNode, e);
    return node;
}

// ----------------------------------------------------------------------

Node* Tree::filterEventRecursive(Node* node, Event* e)
{
// Filter the event repeatedly into the daughter nodes until it
// falls into a terminal node.


    Node* nextNode = node->filterEventToDaughter(e);
    if(nextNode == 0) return node;

    return filterEventRecursive(nextNode, e);
}

// ----------------------------------------------------------------------


void Tree::rankVariablesRecursive(Node* node, std::vector<Double_t>& v)
{
// We recursively go through all of the nodes in the tree and find the
// total error reduction for each variable. The one with the most
// error reduction should be the most important.

    Node* left = node->getLeftDaughter();
    Node* right = node->getRightDaughter();

    // Terminal nodes don't contribute to error reduction.
    if(left==0 || right==0) return;

    Int_t sv =  node->getSplitVariable();
    Double_t er = node->getErrorReduction();

    //if(sv == -1)
    //{
      //std::cout << "ERROR: negative split variable for nonterminal node." << std::endl;
      //std::cout << "rankVarRecursive Split Variable = " << sv << std::endl;
      //std::cout << "rankVarRecursive Error Reduction = " << er << std::endl;
    //}

    // Add error reduction to the current total for the appropriate
    // variable.
    v[sv] += er;

    rankVariablesRecursive(left, v);
    rankVariablesRecursive(right, v); 

}

// ----------------------------------------------------------------------

void Tree::rankVariables(std::vector<Double_t>& v)
{
    rankVariablesRecursive(rootNode, v);
}

// ----------------------------------------------------------------------


void Tree::getSplitValuesRecursive(Node* node, std::vector<std::vector<Double_t>>& v)
{
// We recursively go through all of the nodes in the tree and find the
// split points used for each split variable.

    Node* left = node->getLeftDaughter();
    Node* right = node->getRightDaughter();

    // Terminal nodes don't contribute.
    if(left==0 || right==0) return;

    Int_t sv =  node->getSplitVariable();
    Double_t sp = node->getSplitValue();

    if(sv == -1)
    {
        std::cout << "ERROR: negative split variable for nonterminal node." << std::endl;
        std::cout << "rankVarRecursive Split Variable = " << sv << std::endl;
    }

    // Add the split point to the list for the correct split variable.
    v[sv].push_back(sp);

    getSplitValuesRecursive(left, v);
    getSplitValuesRecursive(right, v); 

}

// ----------------------------------------------------------------------

void Tree::getSplitValues(std::vector<std::vector<Double_t>>& v)
{
    getSplitValuesRecursive(rootNode, v);
}

//////////////////////////////////////////////////////////////////////////
// ______________________Storage/Retrieval______________________________//
//////////////////////////////////////////////////////////////////////////

template <typename T>
std::string numToStr( T num )
{
// Convert a number to a string.
    std::stringstream ss;
    ss << num;
    std::string s = ss.str();
    return  s;
}

// ----------------------------------------------------------------------

void Tree::addXMLAttributes(TXMLEngine* xml, Node* node, XMLNodePointer_t np)
{
    // Convert Node members into XML attributes    
    // and add them to the XMLEngine.
    xml->NewAttr(np, 0, "splitVar", numToStr(node->getSplitVariable()).c_str());
    xml->NewAttr(np, 0, "splitVal", numToStr(node->getSplitValue()).c_str());
    xml->NewAttr(np, 0, "fitVal", numToStr(node->getFitValue()).c_str());
}

// ----------------------------------------------------------------------

void Tree::saveToXML(const char* c)
{

    TXMLEngine* xml = new TXMLEngine();

    // Add the root node.
    XMLNodePointer_t root = xml->NewChild(0, 0, rootNode->getName().c_str());
    addXMLAttributes(xml, rootNode, root);

    // Recursively write the tree to XML.
    saveToXMLRecursive(xml, rootNode, root);

    // Make the XML Document.
    XMLDocPointer_t xmldoc = xml->NewDoc();
    xml->DocSetRootElement(xmldoc, root);

    // Save to file.
    xml->SaveDoc(xmldoc, c);

    // Clean up.
    xml->FreeDoc(xmldoc);
    delete xml;
}

// ----------------------------------------------------------------------

void Tree::saveToXMLRecursive(TXMLEngine* xml, Node* node, XMLNodePointer_t np)
{
    Node* l = node->getLeftDaughter();
    Node* r = node->getRightDaughter();

    if(l==0 || r==0) return;

    // Add children to the XMLEngine. 
    XMLNodePointer_t left = xml->NewChild(np, 0, "left");
    XMLNodePointer_t right = xml->NewChild(np, 0, "right");

    // Add attributes to the children.
    addXMLAttributes(xml, l, left);
    addXMLAttributes(xml, r, right);

    // Recurse.
    saveToXMLRecursive(xml, l, left);
    saveToXMLRecursive(xml, r, right);
}

// ----------------------------------------------------------------------

void Tree::loadFromXML(const char* filename)
{   
    // First create the engine.
    TXMLEngine* xml = new TXMLEngine;

    // Now try to parse xml file.
    XMLDocPointer_t xmldoc = xml->ParseFile(filename);
    if (xmldoc==0)
    {
        delete xml;
        return;  
    }

    // Get access to main node of the xml file.
    XMLNodePointer_t mainnode = xml->DocGetRootElement(xmldoc);
   
    // Recursively connect nodes together.
    loadFromXMLRecursive(xml, mainnode, rootNode);
   
    // Release memory before exit
    xml->FreeDoc(xmldoc);
    delete xml;
}

// ----------------------------------------------------------------------

void Tree::loadFromXMLRecursive(TXMLEngine* xml, XMLNodePointer_t xnode, Node* tnode) 
{

    // Get the split information from xml.
    XMLAttrPointer_t attr = xml->GetFirstAttr(xnode);
    std::vector<std::string> splitInfo(3);
    for(unsigned int i=0; i<3; i++)
    {
        splitInfo[i] = xml->GetAttrValue(attr); 
        attr = xml->GetNextAttr(attr);  
    }

    // Convert strings into numbers.
    std::stringstream converter;
    Int_t splitVar;
    Double_t splitVal;
    Double_t fitVal;  

    converter << splitInfo[0];
    converter >> splitVar;
    converter.str("");
    converter.clear();

    converter << splitInfo[1];
    converter >> splitVal;
    converter.str("");
    converter.clear();

    converter << splitInfo[2];
    converter >> fitVal;
    converter.str("");
    converter.clear();

    // Store gathered splitInfo into the node object.
    tnode->setSplitVariable(splitVar);
    tnode->setSplitValue(splitVal);
    tnode->setFitValue(fitVal);

    // Get the xml daughters of the current xml node. 
    XMLNodePointer_t xleft = xml->GetChild(xnode);
    XMLNodePointer_t xright = xml->GetNext(xleft);

    // If there are no daughters we are done.
    if(xleft == 0 || xright == 0) return;

    // If there are daughters link the node objects appropriately.
    tnode->theMiracleOfChildBirth();
    Node* tleft = tnode->getLeftDaughter();
    Node* tright = tnode->getRightDaughter();

    // Update the list of terminal nodes.
    terminalNodes.remove(tnode);
    terminalNodes.push_back(tleft);
    terminalNodes.push_back(tright);
    numTerminalNodes++;

    loadFromXMLRecursive(xml, xleft, tleft);
    loadFromXMLRecursive(xml, xright, tright);
}

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

#include "L1Trigger/L1TMuonEndCap/interface/bdt/Tree.h"

#include <iostream>
#include <sstream>
#include <cmath>

//////////////////////////////////////////////////////////////////////////
// _______________________Constructor(s)________________________________//
//////////////////////////////////////////////////////////////////////////

using namespace emtf;

Tree::Tree() {
  rootNode = new Node("root");

  terminalNodes.push_back(rootNode);
  numTerminalNodes = 1;
  boostWeight = 0;
  xmlVersion = 2017;
}

Tree::Tree(std::vector<std::vector<Event*>>& cEvents) {
  rootNode = new Node("root");
  rootNode->setEvents(cEvents);

  terminalNodes.push_back(rootNode);
  numTerminalNodes = 1;
  boostWeight = 0;
  xmlVersion = 2017;
}
//////////////////////////////////////////////////////////////////////////
// _______________________Destructor____________________________________//
//////////////////////////////////////////////////////////////////////////

Tree::~Tree() {
  // When the tree is destroyed it will delete all of the nodes in the tree.
  // The deletion begins with the rootnode and continues recursively.
  if (rootNode)
    delete rootNode;
}

Tree::Tree(const Tree& tree) {
  // unfortunately, authors of these classes didn't use const qualifiers
  rootNode = copyFrom(const_cast<Tree&>(tree).getRootNode());
  numTerminalNodes = tree.numTerminalNodes;
  rmsError = tree.rmsError;
  boostWeight = tree.boostWeight;
  xmlVersion = tree.xmlVersion;

  terminalNodes.resize(0);
  // find new leafs
  findLeafs(rootNode, terminalNodes);

  ///    if( numTerminalNodes != terminalNodes.size() ) throw std::runtime_error();
}

Tree& Tree::operator=(const Tree& tree) {
  if (rootNode)
    delete rootNode;
  // unfortunately, authors of these classes didn't use const qualifiers
  rootNode = copyFrom(const_cast<Tree&>(tree).getRootNode());
  numTerminalNodes = tree.numTerminalNodes;
  rmsError = tree.rmsError;
  boostWeight = tree.boostWeight;
  xmlVersion = tree.xmlVersion;

  terminalNodes.resize(0);
  // find new leafs
  findLeafs(rootNode, terminalNodes);

  ///    if( numTerminalNodes != terminalNodes.size() ) throw std::runtime_error();

  return *this;
}

Node* Tree::copyFrom(const Node* local_root) {
  // end-case
  if (!local_root)
    return nullptr;

  Node* lr = const_cast<Node*>(local_root);

  // recursion
  Node* left_new_child = copyFrom(lr->getLeftDaughter());
  Node* right_new_child = copyFrom(lr->getRightDaughter());

  // performing main work at this level
  Node* new_local_root = new Node(lr->getName());
  if (left_new_child)
    left_new_child->setParent(new_local_root);
  if (right_new_child)
    right_new_child->setParent(new_local_root);
  new_local_root->setLeftDaughter(left_new_child);
  new_local_root->setRightDaughter(right_new_child);
  new_local_root->setErrorReduction(lr->getErrorReduction());
  new_local_root->setSplitValue(lr->getSplitValue());
  new_local_root->setSplitVariable(lr->getSplitVariable());
  new_local_root->setFitValue(lr->getFitValue());
  new_local_root->setTotalError(lr->getTotalError());
  new_local_root->setAvgError(lr->getAvgError());
  new_local_root->setNumEvents(lr->getNumEvents());
  //    new_local_root->setEvents( lr->getEvents() ); // no ownership assumed for the events anyways

  return new_local_root;
}

void Tree::findLeafs(Node* local_root, std::list<Node*>& tn) {
  if (!local_root->getLeftDaughter() && !local_root->getRightDaughter()) {
    // leaf or ternimal node found
    tn.push_back(local_root);
    return;
  }

  if (local_root->getLeftDaughter())
    findLeafs(local_root->getLeftDaughter(), tn);

  if (local_root->getRightDaughter())
    findLeafs(local_root->getRightDaughter(), tn);
}

Tree::Tree(Tree&& tree) {
  if (rootNode)
    delete rootNode;  // this line is the only reason not to use default move constructor
  rootNode = tree.rootNode;
  terminalNodes = std::move(tree.terminalNodes);
  numTerminalNodes = tree.numTerminalNodes;
  rmsError = tree.rmsError;
  boostWeight = tree.boostWeight;
  xmlVersion = tree.xmlVersion;
}

//////////////////////////////////////////////////////////////////////////
// ______________________Get/Set________________________________________//
//////////////////////////////////////////////////////////////////////////

void Tree::setRootNode(Node* sRootNode) { rootNode = sRootNode; }

Node* Tree::getRootNode() { return rootNode; }

// ----------------------------------------------------------------------

void Tree::setTerminalNodes(std::list<Node*>& sTNodes) { terminalNodes = sTNodes; }

std::list<Node*>& Tree::getTerminalNodes() { return terminalNodes; }

// ----------------------------------------------------------------------

int Tree::getNumTerminalNodes() { return numTerminalNodes; }

//////////////////////////////////////////////////////////////////////////
// ______________________Performace_____________________________________//
//////////////////////////////////////////////////////////////////////////

void Tree::calcError() {
  // Loop through the separate predictive regions (terminal nodes) and
  // add up the errors to get the error of the entire space.

  double totalSquaredError = 0;

  for (std::list<Node*>::iterator it = terminalNodes.begin(); it != terminalNodes.end(); it++) {
    totalSquaredError += (*it)->getTotalError();
  }
  rmsError = sqrt(totalSquaredError / rootNode->getNumEvents());
}

// ----------------------------------------------------------------------

void Tree::buildTree(int nodeLimit) {
  // We greedily pick the best terminal node to split.
  double bestNodeErrorReduction = -1;
  Node* nodeToSplit = nullptr;

  if (numTerminalNodes == 1) {
    rootNode->calcOptimumSplit();
    calcError();
    //        std::cout << std::endl << "  " << numTerminalNodes << " Nodes : " << rmsError << std::endl;
  }

  for (std::list<Node*>::iterator it = terminalNodes.begin(); it != terminalNodes.end(); it++) {
    if ((*it)->getErrorReduction() > bestNodeErrorReduction) {
      bestNodeErrorReduction = (*it)->getErrorReduction();
      nodeToSplit = (*it);
    }
  }

  //std::cout << "nodeToSplit size = " << nodeToSplit->getNumEvents() << std::endl;

  // If all of the nodes have one event we can't add any more nodes and reduce the error.
  if (nodeToSplit == nullptr)
    return;

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

  if (numTerminalNodes % 1 == 0) {
    //        std::cout << "  " << numTerminalNodes << " Nodes : " << rmsError << std::endl;
  }

  // Repeat until done.
  if (numTerminalNodes < nodeLimit)
    buildTree(nodeLimit);
}

// ----------------------------------------------------------------------

void Tree::filterEvents(std::vector<Event*>& tEvents) {
  // Use trees which have already been built to fit a bunch of events
  // given by the tEvents vector.

  // Set the events to be filtered.
  rootNode->getEvents() = std::vector<std::vector<Event*>>(1);
  rootNode->getEvents()[0] = tEvents;

  // The tree now knows about the events it needs to fit.
  // Filter them into a predictive region (terminal node).
  filterEventsRecursive(rootNode);
}

// ----------------------------------------------------------------------

void Tree::filterEventsRecursive(Node* node) {
  // Filter the events repeatedly into the daughter nodes until they
  // fall into a terminal node.

  Node* left = node->getLeftDaughter();
  Node* right = node->getRightDaughter();

  if (left == nullptr || right == nullptr)
    return;

  node->filterEventsToDaughters();

  filterEventsRecursive(left);
  filterEventsRecursive(right);
}

// ----------------------------------------------------------------------

Node* Tree::filterEvent(Event* e) {
  // Use trees which have already been built to fit a bunch of events
  // given by the tEvents vector.

  // Filter the event into a predictive region (terminal node).
  Node* node = filterEventRecursive(rootNode, e);
  return node;
}

// ----------------------------------------------------------------------

Node* Tree::filterEventRecursive(Node* node, Event* e) {
  // Filter the event repeatedly into the daughter nodes until it
  // falls into a terminal node.

  Node* nextNode = node->filterEventToDaughter(e);
  if (nextNode == nullptr)
    return node;

  return filterEventRecursive(nextNode, e);
}

// ----------------------------------------------------------------------

void Tree::rankVariablesRecursive(Node* node, std::vector<double>& v) {
  // We recursively go through all of the nodes in the tree and find the
  // total error reduction for each variable. The one with the most
  // error reduction should be the most important.

  Node* left = node->getLeftDaughter();
  Node* right = node->getRightDaughter();

  // Terminal nodes don't contribute to error reduction.
  if (left == nullptr || right == nullptr)
    return;

  int sv = node->getSplitVariable();
  double er = node->getErrorReduction();

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

void Tree::rankVariables(std::vector<double>& v) { rankVariablesRecursive(rootNode, v); }

// ----------------------------------------------------------------------

void Tree::getSplitValuesRecursive(Node* node, std::vector<std::vector<double>>& v) {
  // We recursively go through all of the nodes in the tree and find the
  // split points used for each split variable.

  Node* left = node->getLeftDaughter();
  Node* right = node->getRightDaughter();

  // Terminal nodes don't contribute.
  if (left == nullptr || right == nullptr)
    return;

  int sv = node->getSplitVariable();
  double sp = node->getSplitValue();

  if (sv == -1) {
    std::cout << "ERROR: negative split variable for nonterminal node." << std::endl;
    std::cout << "rankVarRecursive Split Variable = " << sv << std::endl;
  }

  // Add the split point to the list for the correct split variable.
  v[sv].push_back(sp);

  getSplitValuesRecursive(left, v);
  getSplitValuesRecursive(right, v);
}

// ----------------------------------------------------------------------

void Tree::getSplitValues(std::vector<std::vector<double>>& v) { getSplitValuesRecursive(rootNode, v); }

//////////////////////////////////////////////////////////////////////////
// ______________________Storage/Retrieval______________________________//
//////////////////////////////////////////////////////////////////////////

template <typename T>
std::string numToStr(T num) {
  // Convert a number to a string.
  std::stringstream ss;
  ss << num;
  std::string s = ss.str();
  return s;
}

// ----------------------------------------------------------------------

void Tree::addXMLAttributes(TXMLEngine* xml, Node* node, XMLNodePointer_t np) {
  // Convert Node members into XML attributes
  // and add them to the XMLEngine.
  xml->NewAttr(np, nullptr, "splitVar", numToStr(node->getSplitVariable()).c_str());
  xml->NewAttr(np, nullptr, "splitVal", numToStr(node->getSplitValue()).c_str());
  xml->NewAttr(np, nullptr, "fitVal", numToStr(node->getFitValue()).c_str());
}

// ----------------------------------------------------------------------

void Tree::saveToXML(const char* c) {
  TXMLEngine* xml = new TXMLEngine();

  // Add the root node.
  XMLNodePointer_t root = xml->NewChild(nullptr, nullptr, rootNode->getName().c_str());
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

void Tree::saveToXMLRecursive(TXMLEngine* xml, Node* node, XMLNodePointer_t np) {
  Node* l = node->getLeftDaughter();
  Node* r = node->getRightDaughter();

  if (l == nullptr || r == nullptr)
    return;

  // Add children to the XMLEngine.
  XMLNodePointer_t left = xml->NewChild(np, nullptr, "left");
  XMLNodePointer_t right = xml->NewChild(np, nullptr, "right");

  // Add attributes to the children.
  addXMLAttributes(xml, l, left);
  addXMLAttributes(xml, r, right);

  // Recurse.
  saveToXMLRecursive(xml, l, left);
  saveToXMLRecursive(xml, r, right);
}

// ----------------------------------------------------------------------

void Tree::loadFromXML(const char* filename) {
  // First create the engine.
  TXMLEngine* xml = new TXMLEngine;

  // Now try to parse xml file.
  XMLDocPointer_t xmldoc = xml->ParseFile(filename);
  if (xmldoc == nullptr) {
    delete xml;
    return;
  }

  // Get access to main node of the xml file.
  XMLNodePointer_t mainnode = xml->DocGetRootElement(xmldoc);

  // the original 2016 pT xmls define the source tree node to be the top-level xml node
  // while in 2017 TMVA's xmls every decision tree is wrapped in an extra block specifying boostWeight parameter
  // I choose to identify the format by checking the top xml node name that is a "BinaryTree" in 2017
  if (std::string("BinaryTree") == xml->GetNodeName(mainnode)) {
    XMLAttrPointer_t attr = xml->GetFirstAttr(mainnode);
    while (std::string("boostWeight") != xml->GetAttrName(attr)) {
      attr = xml->GetNextAttr(attr);
    }
    boostWeight = (attr ? strtod(xml->GetAttrValue(attr), nullptr) : 0);
    // step inside the top-level xml node
    mainnode = xml->GetChild(mainnode);
    xmlVersion = 2017;
  } else {
    boostWeight = 0;
    xmlVersion = 2016;
  }
  // Recursively connect nodes together.
  loadFromXMLRecursive(xml, mainnode, rootNode);

  // Release memory before exit
  xml->FreeDoc(xmldoc);
  delete xml;
}

// ----------------------------------------------------------------------

void Tree::loadFromXMLRecursive(TXMLEngine* xml, XMLNodePointer_t xnode, Node* tnode) {
  // Get the split information from xml.
  XMLAttrPointer_t attr = xml->GetFirstAttr(xnode);
  std::vector<std::string> splitInfo(3);
  if (xmlVersion >= 2017) {
    for (unsigned int i = 0; i < 10; i++) {
      if (std::string("IVar") == xml->GetAttrName(attr)) {
        splitInfo[0] = xml->GetAttrValue(attr);
      }
      if (std::string("Cut") == xml->GetAttrName(attr)) {
        splitInfo[1] = xml->GetAttrValue(attr);
      }
      if (std::string("res") == xml->GetAttrName(attr)) {
        splitInfo[2] = xml->GetAttrValue(attr);
      }
      attr = xml->GetNextAttr(attr);
    }
  } else {
    for (unsigned int i = 0; i < 3; i++) {
      splitInfo[i] = xml->GetAttrValue(attr);
      attr = xml->GetNextAttr(attr);
    }
  }

  // Convert strings into numbers.
  std::stringstream converter;
  int splitVar;
  double splitVal;
  double fitVal;

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
  if (xleft == nullptr || xright == nullptr)
    return;

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

void Tree::loadFromCondPayload(const L1TMuonEndCapForest::DTree& tree) {
  // start fresh in case this is not the only call to construct a tree
  if (rootNode)
    delete rootNode;
  rootNode = new Node("root");

  const L1TMuonEndCapForest::DTreeNode& mainnode = tree[0];
  loadFromCondPayloadRecursive(tree, mainnode, rootNode);
}

void Tree::loadFromCondPayloadRecursive(const L1TMuonEndCapForest::DTree& tree,
                                        const L1TMuonEndCapForest::DTreeNode& node,
                                        Node* tnode) {
  // Store gathered splitInfo into the node object.
  tnode->setSplitVariable(node.splitVar);
  tnode->setSplitValue(node.splitVal);
  tnode->setFitValue(node.fitVal);

  // If there are no daughters we are done.
  if (node.ileft == 0 || node.iright == 0)
    return;  // root cannot be anyone's child
  if (node.ileft >= tree.size() || node.iright >= tree.size())
    return;  // out of range addressing on purpose

  // If there are daughters link the node objects appropriately.
  tnode->theMiracleOfChildBirth();
  Node* tleft = tnode->getLeftDaughter();
  Node* tright = tnode->getRightDaughter();

  // Update the list of terminal nodes.
  terminalNodes.remove(tnode);
  terminalNodes.push_back(tleft);
  terminalNodes.push_back(tright);
  numTerminalNodes++;

  loadFromCondPayloadRecursive(tree, tree[node.ileft], tleft);
  loadFromCondPayloadRecursive(tree, tree[node.iright], tright);
}

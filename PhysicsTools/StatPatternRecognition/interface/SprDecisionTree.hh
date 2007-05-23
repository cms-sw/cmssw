// File and Version Information:
//      $Id: SprDecisionTree.hh,v 1.4 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprDecisionTree :
//          Implements a decision tree with a user-supplied optimization
//          criterion. Nmin is the minimal number of events per node of the
//          decision tree. If Nmin is specified too large, the tree won't
//          be flexible enough to model the data structure. If Nmin is too
//          small, the tree will overtrain and produce too many nodes
//          (rectangular regions). 
//
//          Decision tree are described in many statistical textbooks.
//          Note that, unlike a conventional decision tree, the tree
//          implemented here is asymmetric. This implementation cares
//          only about the signal component of the sample; it does not 
//          optimize the tree with respect to background. As a benefit,
//          the user can supply inherently asymmetric figures of merit
//          such as S/sqrt(S+B), sensible only for optimization of signal.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprDecisionTree_HH
#define _SprDecisionTree_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedDecisionTree.hh"

#include <string>
#include <iostream>
#include <vector>
#include <utility>

class SprAbsTwoClassCriterion;
class SprTreeNode;
class SprIntegerBootstrap;
class SprClass;


class SprDecisionTree : public SprAbsClassifier
{
public:
  virtual ~SprDecisionTree();

  SprDecisionTree(SprAbsFilter* data, 
		  const SprAbsTwoClassCriterion* crit,
		  int nmin, bool doMerge, bool discrete,
		  SprIntegerBootstrap* bootstrap=0);

  /*
    Classifier name.
  */
  std::string name() const { return "DecisionTree"; }

  /*
    Trains classifier on data. Returns true on success, false otherwise.
  */
  bool train(int verbose=0);

  /*
    Reset this classifier to untrained state.
  */
  bool reset();

  /*
    Replace training data.
  */
  bool setData(SprAbsFilter* data);
 
  /*
    Prints results of training.
  */
  void print(std::ostream& os) const;

  /*
    Make a trained classifier.
  */
  SprTrainedDecisionTree* makeTrained() const;

  /*
    Choose two classes.
  */
  bool setClasses(const SprClass& cls0, const SprClass& cls1);

  //
  // Local methods
  //

  // Force tree to have events of both types in each node.
  void forceMixedNodes() { canHavePureNodes_ = false; }
  bool forcedMixedNodes() const { return !canHavePureNodes_; }

  // Force the tree to count the number of splits on input variables.
  // This method has to be called before train().
  void startSplitCounter();

  // Print out split counts;
  void printSplitCounter(std::ostream& os) const;

  // Force the tree to print-out background nodes as well.
  // This flag must be set before call to train().
  // This flag is only in effect if the tree nodes are not merged;
  // if merging is requested (doMerge=true at construction),
  // all nodes are treated as signal candidates.
  void setShowBackgroundNodes(bool show) { showBackgroundNodes_ = show; }

  // returns next leaf node in the tree
  const SprTreeNode* next(const SprTreeNode* node) const;

  // returns the leftmost leaf node in the tree
  const SprTreeNode* first() const;

  // returns overal FOM of the tree
  double fom() const { return fom_; }

protected:
  bool merge(int category, bool doMerge,
	     std::vector<const SprTreeNode*>& nodes,
	     double& fomtot, double& w0tot, double& w1tot, 
	     unsigned& n0tot, unsigned& n1tot, int verbose) const;

  const SprAbsTwoClassCriterion* crit_;
  int nmin_;
  bool doMerge_;
  bool discrete_;
  bool canHavePureNodes_;
  bool showBackgroundNodes_;
  SprIntegerBootstrap* bootstrap_;
  SprTreeNode* root_;
  std::vector<const SprTreeNode*> nodes1_;
  std::vector<const SprTreeNode*> nodes0_;
  std::vector<SprTreeNode*> fullNodeList_;
  double fom_;
  double w0_;
  double w1_;
  unsigned n0_;
  unsigned n1_;
  // number of splits and relative change in FOM for input variables
  std::vector<std::pair<int,double> > splits_;
};

#endif

// File and Version Information:
//      $Id: SprTopdownTree.hh,v 1.3 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprTopdownTree :
//        Reproduces the functionality of SprDecisionTree but makes
//        a trained classifier in a different implementation. 
//        See SprTrainedTopdownTree.hh for details.
//
//        Discrete=true option assigns an integer label to every 
//        terminal node of the tree - 0 for background-dominated and
//        1 for signal-dominated nodes. Discrete=false computes
//        continuous output for each terminal node as w1/(w0+w1),
//        purity of events from category 1.
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
 
#ifndef _SprTopdownTree_HH
#define _SprTopdownTree_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedTopdownTree.hh"

#include <string>
#include <vector>

class SprAbsFilter;
class SprAbsTwoClassCriterion;
class SprTrainedNode;
class SprIntegerBootstrap;


class SprTopdownTree : public SprDecisionTree
{
public:
  virtual ~SprTopdownTree() {}

  SprTopdownTree(SprAbsFilter* data, 
		 const SprAbsTwoClassCriterion* crit,
		 int nmin, bool doMerge, bool discrete,
		 SprIntegerBootstrap* bootstrap=0);

  /*
    Classifier name.
  */
  std::string name() const { return "TopdownTree"; }

  /*
    Make a trained classifier.
  */
  SprTrainedTopdownTree* makeTrained() const;

private:
  bool makeTrainedNodes(std::vector<const SprTrainedNode*>& nodes) const;
};

#endif

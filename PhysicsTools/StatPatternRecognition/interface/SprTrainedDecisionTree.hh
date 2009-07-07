// File and Version Information:
//      $Id: SprTrainedDecisionTree.hh,v 1.2 2007/09/21 22:32:04 narsky Exp $
//
// Description:
//      Class SprTrainedDecisionTree :
//         Implements a trained decision tree. The constructor input
//         is a vector of rectangular regions, where each region is in turn
//         represented by a vector of pairs, lower and upper bounds.
//         The tree can use as many rectangular regions as you'd like.
//         The length of each vector representing one rectangular region
//         must be equal to the dimensionality of the observable space.
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
 
#ifndef _SprTrainedDecisionTree_HH
#define _SprTrainedDecisionTree_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <iostream>
#include <string>
#include <vector>
#include <utility>


class SprTrainedDecisionTree : public SprAbsTrainedClassifier
{
public:
  virtual ~SprTrainedDecisionTree() {}

  /*
    The constructor takes a vector of terminal signal nodes. Each node
    is represented by a map, where the map key is the input dimension, 
    and the map element is a pair that gives the lower and upper
    bounds on the node in this dimension.
  */
  SprTrainedDecisionTree(const std::vector<SprBox>& nodes1)
    :
    SprAbsTrainedClassifier(),
    nodes1_(nodes1)
  {
    this->setCut(SprUtils::lowerBound(0.5));
  }

  SprTrainedDecisionTree(const SprTrainedDecisionTree& other)
    :
    SprAbsTrainedClassifier(other),
    nodes1_(other.nodes1_)
  {}

  /*
    Returns classifier name.
  */
  std::string name() const {
    return "DecisionTree";
  }

  /*
    Make a clone.
  */
  SprTrainedDecisionTree* clone() const {
    return new SprTrainedDecisionTree(*this);
  }

  /*
    Classifier response for a data point. 
  */
  double response(const std::vector<double>& v) const;

  /*
    Generate code.
  */
  bool generateCode(std::ostream& os) const {
    return false;
  }

  /*
    Print out.
  */
  void print(std::ostream& os) const;

  // Local methods.

  // return number of nodes
  unsigned nNodes() const { 
    return nodes1_.size(); 
  }

  // return box for the i-th node
  void box(int i, SprBox& limits) const {
    if( i<0 || i>=(int)nodes1_.size() )
      limits.clear();
    else
      limits = nodes1_[i];
  }

  // return all nodes
  void nodes(std::vector<SprBox>& nodes) const {
    nodes = nodes1_;
  }

  // return the box number to which this vector belongs; -1 if none
  int nBox(const std::vector<double>& v) const;

protected:
  std::vector<SprBox> nodes1_;
};

#endif

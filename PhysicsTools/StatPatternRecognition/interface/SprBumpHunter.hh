// File and Version Information:
//      $Id: SprBumpHunter.hh,v 1.2 2007/09/21 22:32:01 narsky Exp $
//
// Description:
//      Class SprBumpHunter :
//         Implements PRIM algorithm for bump hunting.
//         Input parameters:
//           nbump - number of bumps to look for
//           nmin  - minimal number of events per bump
//           apeel - fraction of (weighted) events that can be peeled off
//                   the signal box in one iteration of the algorithm ([0,1])
//      The "peel" parameter is of crucial importance and should be chosen
//      carefully for each problem. You can easily miss bumps in data
//      if the "peel" parameter is far from optimal. 
//      
//      The recommended optimization criterion to use with this class is
//      purity, S/(S+B).
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
 
#ifndef _SprBumpHunter_HH
#define _SprBumpHunter_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBoxFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <string>
#include <iostream>
#include <vector>
#include <utility>

class SprAbsTwoClassCriterion;


class SprBumpHunter : public SprAbsClassifier
{
public:
  virtual ~SprBumpHunter() { delete box_; }

  SprBumpHunter(SprAbsFilter* data, 
		const SprAbsTwoClassCriterion* crit,
		int nbump,
		int nmin,
		double apeel);

  /*
    Classifier name.
  */
  std::string name() const { return "BumpHunter"; }

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
  bool setClasses(const SprClass& cls0, const SprClass& cls1) {
    cls0_ = cls0; cls1_ = cls1;
    std::cout << "Classes for bump hunter reset to " 
	      << cls0_ << " " << cls1_ << std::endl;
    return true;
  }

private:
  void setClasses();
  bool sort(int dsort, std::vector<std::vector<int> >& sorted,
	    std::vector<std::vector<double> >& division) const;

  // shrink the signal box
  int shrink(SprBox& limits, 
	     unsigned& n0, unsigned& n1,
	     double& w0, double& w1, double& fom0, int verbose);

  // expand the signal box
  int expand(SprBox& limits, 
	     unsigned& n0, unsigned& n1,
	     double& w0, double& w1, double& fom0, int verbose);

  const SprAbsTwoClassCriterion* crit_;
  unsigned int nbump_;
  unsigned int nmin_;
  double apeel_;
  SprBoxFilter* box_;
  std::vector<SprBox> boxes_;
  std::vector<double> fom_;
  std::vector<unsigned> n0_;
  std::vector<unsigned> n1_;
  std::vector<double> w0_;
  std::vector<double> w1_;
  int nsplit_;
  SprClass cls0_;
  SprClass cls1_;
};

#endif

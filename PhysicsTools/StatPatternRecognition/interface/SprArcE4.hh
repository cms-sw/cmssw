//$Id: SprArcE4.hh,v 1.4 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprArcE4 :
//         Implements a version of Breiman's arc-x4 algorithm.
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
 
#ifndef _SprArcE4_HH
#define _SprArcE4_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprBagger.hh"

#include <string>
#include <vector>
#include <utility>

class SprAbsFilter;
class SprAbsTrainedClassifier;


class SprArcE4 : public SprBagger
{
public:
  virtual ~SprArcE4() {}

  SprArcE4(SprAbsFilter* data, unsigned cycles, bool discrete=false);

  /*
    Classifier name.
  */
  std::string name() const { return "ArcE4"; }

 /*
    Trains classifier on data. Returns true on success, false otherwise.
  */
  bool train(int verbose=0);

  /*
    Replace training data.
  */
  bool setData(SprAbsFilter* data);

private:
  bool prepareExit(bool status=true);
  void reweight(const SprAbsTrainedClassifier* t);

  std::vector<double> initialDataWeights_;
  std::vector<std::pair<double,double> > response_;// response and weight
};

#endif

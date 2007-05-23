// File and Version Information:
//      $Id: SprMultiClassReader.hh,v 1.1 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprMultiClassReader :
//          Reads trained multi class learner from a file.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005, 2006        California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprMultiClassReader_HH
#define _SprMultiClassReader_HH

#include "PhysicsTools/StatPatternRecognition/src/SprMatrix.hh"

#include <vector>
#include <utility>

class SprMultiClassLearner;
class SprTrainedMultiClassLearner;
class SprAbsTrainedClassifier;


class SprMultiClassReader
{
public:
  virtual ~SprMultiClassReader();

  SprMultiClassReader() 
  : indicator_(), mapper_(), classifiers_() {}

  bool read(const char* filename);

  void setTrainable(SprMultiClassLearner* multi);
  SprTrainedMultiClassLearner* makeTrained();

private:
  SprMatrix indicator_;
  std::vector<int> mapper_;
  std::vector<std::pair<const SprAbsTrainedClassifier*,bool> > classifiers_;
};

#endif

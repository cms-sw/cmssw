// File and Version Information:
//      $Id: SprMultiClassReader.hh,v 1.3 2007/07/11 19:52:09 narsky Exp $
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
#include <string>
#include <utility>
#include <istream>

class SprMultiClassLearner;
class SprTrainedMultiClassLearner;
class SprAbsTrainedClassifier;


class SprMultiClassReader
{
public:
  virtual ~SprMultiClassReader();

  SprMultiClassReader() 
    : indicator_(), mapper_(), classifiers_(), vars_() {}

  bool read(const char* filename);
  bool read(std::istream& input);

  void setTrainable(SprMultiClassLearner* multi);
  SprTrainedMultiClassLearner* makeTrained();

  static bool readIndicatorMatrix(const char* filename, SprMatrix& indicator);

private:
  SprMatrix indicator_;
  std::vector<int> mapper_;
  std::vector<std::pair<const SprAbsTrainedClassifier*,bool> > classifiers_;
  std::vector<std::string> vars_;
};

#endif

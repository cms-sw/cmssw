// File and Version Information:
//      $Id: SprDataFeeder.hh,v 1.5 2007/05/23 21:18:58 narsky Exp $
//
// Description:
//      Class SprDataFeeder :
//          Feeds data to a set of trained classifiers and
//          writes out data with classifier outputs.
//
//          The optionally supplied mapper can be used to select a subset
//          of variables. For example, the classifiers were trained 
//          on a subset variables but the user would like to save all of
//          them into an output file. Then the user should read all input
//          data, without filtering on variables, and supply an input
//          mapper to the data feeder. For example, input data are 
//          5-dimensional but the classifiers were trained on dimensions
//          0, 2, and 3. Then you should supply
//              mapper[0]=0
//              mapper[1]=2
//              mapper[2]=3 
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
 
#ifndef _SprDataFeeder_HH
#define _SprDataFeeder_HH

#include <vector>

class SprAbsFilter;
class SprAbsTrainedClassifier;
class SprTrainedMultiClassLearner;
class SprAbsWriter;
class SprCoordinateMapper;


class SprDataFeeder
{
public:
  virtual ~SprDataFeeder();

  // Takes the default mapper used for all classifiers.
  SprDataFeeder(const SprAbsFilter* data,
		SprAbsWriter* writer,
		const std::vector<unsigned>& mapper=std::vector<unsigned>());

  // Add a new classifier.
  // If a mapper is provided, it overrides the default mapper
  // provided in the constructor.
  bool addClassifier(const SprAbsTrainedClassifier* c, 
		     const char* name,
		     const std::vector<unsigned>& mapper);
  bool addClassifier(const SprAbsTrainedClassifier* c, 
		     const char* name,
		     SprCoordinateMapper* mapper=0);

  // add a multi class learner
  bool addMultiClassLearner(const SprTrainedMultiClassLearner* c, 
			    const char* name,
			    const std::vector<unsigned>& mapper);
  bool addMultiClassLearner(const SprTrainedMultiClassLearner* c, 
			    const char* name,
			    SprCoordinateMapper* mapper=0);

  // Feed 1D classifier response and print out message every nout points.
  bool feed(int nout=0) const;

private:
  const SprAbsFilter* data_;
  SprAbsWriter* writer_;
  int mode_;// 1 for classifiers, 2 for multi class learners
  std::vector<const SprAbsTrainedClassifier*> classifiers_;
  std::vector<const SprTrainedMultiClassLearner*> multiclass_;
  SprCoordinateMapper* mapper_;
  std::vector<SprCoordinateMapper*> specificMappers_;
  std::vector<SprCoordinateMapper*> multiSpecificMappers_;
};

#endif


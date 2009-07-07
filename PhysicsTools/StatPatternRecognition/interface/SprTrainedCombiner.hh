// File and Version Information:
//      $Id: SprTrainedCombiner.hh,v 1.1 2007/09/21 22:32:04 narsky Exp $
//
// Description:
//      Class SprTrainedCombiner :
//          Implements response of the trained Combiner.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005,2007         California Institute of Technology
//
//------------------------------------------------------------------------

#ifndef _SprTrainedCombiner_HH
#define _SprTrainedCombiner_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

class SprCoordinateMapper;

#include <map>
#include <vector>
#include <utility>
#include <string>


class SprTrainedCombiner : public SprAbsTrainedClassifier
{
public:
  typedef std::map<unsigned,SprCut> SprAllowedIndexMap;

  virtual ~SprTrainedCombiner();

  SprTrainedCombiner(const SprAbsTrainedClassifier* overall,
		     const std::vector<
		     std::pair<const SprAbsTrainedClassifier*,bool> >&
		     trained,
		     const std::vector<std::string>& labels,
		     const std::vector<SprAllowedIndexMap>& constraints,
		     const std::vector<SprCoordinateMapper*>& inputDataMappers,
		     const std::vector<double>& defaultValues,
		     bool ownOverall);

  SprTrainedCombiner(const SprTrainedCombiner& other);

  SprTrainedCombiner* clone() const {
    return new SprTrainedCombiner(*this);
  }

  std::string name() const { return "Combiner"; }

  double response(const std::vector<double>& v) const;

  bool generateCode(std::ostream& os) const {
    return false;
  }

  void print(std::ostream& o) const;

  /*
    Local accessors.
  */
  const SprAbsTrainedClassifier* classifier(int i) const {
    if( i>=0 && i<(int)trained_.size() ) return trained_[i].first;
    return 0;
  }

  std::string label(int i) const {
    if( i>=0 && i<(int)labels_.size() ) return "";
    return labels_[i];
  }

  void classifierList(std::vector<const SprAbsTrainedClassifier*>& classifiers)
    const {
    classifiers.clear();
    classifiers.resize(trained_.size());
    for( int i=0;i<(int)trained_.size();i++ )
      classifiers[i] = trained_[i].first;
  }

  void labels(std::vector<std::string>& ls) const {
    ls = labels_;
  }

  const SprAbsTrainedClassifier* overall() const {
    return overall_;
  }

private:
  bool init();

  const SprAbsTrainedClassifier* overall_;
  std::vector<std::pair<const SprAbsTrainedClassifier*,bool> > trained_;
  std::vector<std::string> labels_;
  std::vector<SprAllowedIndexMap> constraints_;
  std::vector<SprCoordinateMapper*> inputDataMappers_;
  std::vector<double> defaultValues_;
  bool ownOverall_;
};

#endif

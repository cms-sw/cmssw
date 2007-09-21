//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprPlotter.hh,v 1.3 2007/08/30 17:54:38 narsky Exp $
//
// Description:
//      Class SprPlotter :
//         tools for plotting classification results
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2007              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprPlotter_HH
#define _SprPlotter_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <cassert>

class SprAbsTwoClassCriterion;


class SprPlotter
{
public:

  struct Response {
    int cls;// class; only 0 and 1 are allowed
    double weight;
    std::map<std::string,double> response;// responses for classifiers
    
    ~Response() {}

    Response() : cls(0), weight(0), response() {}
    
    Response(int c, double w)
      : cls(c), weight(w), response() 
    {
      assert( cls==0 || cls==1 );
    }
    
    Response(const Response& other)
      : cls(other.cls), weight(other.weight), response(other.response) {}
    
    void set(const char* classifier, double r) {
      std::map<std::string,double>::iterator found = response.find(classifier);
      if( found == response.end() )
	response.insert(std::pair<const std::string,double>(classifier,r));
      else
	found->second = r;
    }
  };

  struct FigureOfMerit {
    double lowerBound;// cut on the classifier output
    double bgrWeight;
    unsigned bgrNevts;
    double fom;
    
    ~FigureOfMerit() {}
    
    FigureOfMerit()
      : lowerBound(SprUtils::min()), bgrWeight(0), bgrNevts(0), fom(0) {}
    
    FigureOfMerit(double c, double w, unsigned n)
      : lowerBound(c), bgrWeight(w), bgrNevts(n), fom(0) {}
    
    FigureOfMerit(double c, double w, unsigned n, double f)
      : lowerBound(c), bgrWeight(w), bgrNevts(n), fom(f) {}
    
    FigureOfMerit(const FigureOfMerit& other)
      : lowerBound(other.lowerBound), 
	bgrWeight(other.bgrWeight), 
	bgrNevts(other.bgrNevts), 
	fom(other.fom) 
    {}
  };
  
  virtual ~SprPlotter() {}

  SprPlotter(const std::vector<Response>& responses);

  SprPlotter(const SprPlotter& other);

  // Return signal and background weights.
  double signalWeight() const {
    return scaleS_*sigW_;
  }
  double bgrndWeight() const {
    return scaleB_*bgrW_;
  }
  unsigned signalNevts() const {
    return sigN_;
  }
  unsigned bgrndNevts() const {
    return bgrN_;
  }

  // Set signal and background scaling factors
  bool setScaleFactors(double scaleS, double scaleB);

  // Set FOM to be used for computation.
  void setCrit(const SprAbsTwoClassCriterion* crit) {
    crit_ = crit;
  }

  // Use absolute weights for signal and background
  // or signal and background efficiencies wrt totals in the
  // provided sample.
  void useAbsolute() {
    useAbsolute_ = true;
  }
  void useRelative() {
    useAbsolute_ = false;
  }
  bool absolute() const {
    return useAbsolute_;
  }

  // Compute background efficiency and FOM for given values of 
  // signal efficiency for a given classifier.
  bool backgroundCurve(const std::vector<double>& signalEff,
		       const char* classifier,
		       std::vector<FigureOfMerit>& bgrndEff) const;

  // Histograms response of a certain classifier for signal and background.
  // Each pair represents value and error in a certain bin.
  // Returns the number of bins (less or equal to 0 upon failure).
  int histogram(const char* classifier, 
		double xlo, double xhi, double dx,
		std::vector<std::pair<double,double> >& sigHist,
		std::vector<std::pair<double,double> >& bgrHist) const;

private:
  bool init();
  bool fillSandB(const std::string& sclassifier,
		 std::vector<std::pair<double,double> >& signal,
		 std::vector<std::pair<double,double> >& bgrnd) const;

  std::vector<Response> responses_;
  const SprAbsTwoClassCriterion* crit_;
  bool useAbsolute_;

  // scaling factors
  double scaleS_;
  double scaleB_;

  // total weights and events in the 2 categories
  double sigW_;
  double bgrW_;
  unsigned sigN_;
  unsigned bgrN_;
};

#endif

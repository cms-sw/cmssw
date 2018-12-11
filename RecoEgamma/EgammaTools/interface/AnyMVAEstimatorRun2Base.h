#ifndef RecoEgamma_EgammaTools_AnyMVAEstimatorRun2Base_H
#define RecoEgamma_EgammaTools_AnyMVAEstimatorRun2Base_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Candidate/interface/Candidate.h"

class AnyMVAEstimatorRun2Base {

 public:
  // Constructor, destructor
  AnyMVAEstimatorRun2Base(const edm::ParameterSet& conf)
    : tag_         (conf.getParameter<std::string>("mvaTag"))
    , nCategories_ (conf.getParameter<int>("nCategories"))
    , debug_       (conf.getUntrackedParameter<bool>("debug", false))
  {}
 
 AnyMVAEstimatorRun2Base(const::std::string& mvaName,
                         const::std::string& mvaTag, 
                         int nCategories, 
                         bool debug)
    : name_        (mvaName)
    , tag_         (mvaTag)
    , nCategories_ (nCategories)
    , debug_       (debug)
  {}
  virtual ~AnyMVAEstimatorRun2Base(){};

  // Functions that must be provided in derived classes
  // These function should work on electrons or photons
  // of the reco or pat type

  virtual float mvaValue( const reco::Candidate* candidate, std::vector<float> const& auxVariables, int &iCategory) const = 0;
  float mvaValue( const reco::Candidate* candidate, std::vector<float> const& auxVariables) const {
      int iCategory;
      return mvaValue(candidate, auxVariables, iCategory);
  };

  // A specific implementation of MVA is expected to have one or more categories
  // defined with respect to eta, pt, etc.
  // This function determines the category for a given candidate.
  virtual int findCategory( const reco::Candidate* candidate) const = 0;
  int getNCategories() const { return nCategories_; }
  const std::string& getName() const { return name_; }
  // An extra variable string set during construction that can be used
  // to distinguish different instances of the estimator configured with
  // different weight files. The tag can be used to construct names of ValueMaps, etc.
  const std::string& getTag() const { return tag_; }

  bool isDebug() const { return debug_; }
  //
  // Extra event content - if needed.
  //
  // Some MVA implementation may require direct access to event content.
  // Implement these methods only if needed in the derived classes (use "override"
  // for certainty).

 private:

  //
  // Data members
  //
  const std::string name_;

  // MVA tag. This is an additional string variable to distinguish
  // instances of the estimator of this class configured with different
  // weight files.
  const std::string tag_;

  // The number of categories and number of variables per category
  const int nCategories_;

  const bool debug_;
};

// define the factory for this base class
#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< AnyMVAEstimatorRun2Base* (const edm::ParameterSet&) >
        AnyMVAEstimatorRun2Factory;

#endif

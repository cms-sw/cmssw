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
    : conf_        (conf)
    , name_        (conf.getParameter<std::string>("mvaName"))
    , tag_         (conf.getParameter<std::string>("mvaTag"))
    , nCategories_ (conf.getParameter<int>("nCategories"))
    , methodName_  ("BDTG method")
    , debug_       (conf.getUntrackedParameter<bool>("debug", false))
  {}
  virtual ~AnyMVAEstimatorRun2Base(){};

  // Functions that must be provided in derived classes
  // These function should work on electrons or photons
  // of the reco or pat type

  virtual float mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::EventBase&, int &iCategory) const = 0;
  float mvaValue( const edm::Ptr<reco::Candidate>& candPtr, const edm::EventBase& iEvent) const {
      int iCategory;
      return mvaValue(candPtr, iEvent, iCategory);
  };

  // A specific implementation of MVA is expected to have one or more categories
  // defined with respect to eta, pt, etc.
  // This function determines the category for a given particle.
  virtual int findCategory( const edm::Ptr<reco::Candidate>& candPtr) const = 0;
  int getNCategories() const { return nCategories_; }
  // The name is a unique name associated with a particular MVA implementation,
  // it is found as a const data member in a derived class.
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

  // This method needs to be used only once after this MVA estimator is constructed
  virtual void setConsumes(edm::ConsumesCollector &&cc) {};

 private:

  //
  // Data members
  //

  // Configuration
  const edm::ParameterSet conf_;

  // MVA name. This is a unique name for this MVA implementation.
  // It will be used as part of ValueMap names.
  // For simplicity, keep it set to the class name.
  const std::string name_;

  // MVA tag. This is an additional string variable to distinguish
  // instances of the estimator of this class configured with different
  // weight files.
  const std::string tag_;

  // The number of categories and number of variables per category
  const int nCategories_;

  const std::string methodName_;

  const bool debug_;
};

// define the factory for this base class
#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< AnyMVAEstimatorRun2Base* (const edm::ParameterSet&) >
        AnyMVAEstimatorRun2Factory;

#endif

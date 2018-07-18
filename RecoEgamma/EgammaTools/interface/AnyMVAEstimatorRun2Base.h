#ifndef RecoEgamma_EgammaTools_AnyMVAEstimatorRun2Base_H
#define RecoEgamma_EgammaTools_AnyMVAEstimatorRun2Base_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Candidate/interface/Candidate.h"

class AnyMVAEstimatorRun2Base {

 public:
  // Constructor, destructor
 AnyMVAEstimatorRun2Base(const edm::ParameterSet& conf) : conf_(conf) {}
  virtual ~AnyMVAEstimatorRun2Base(){};

  // Functions that must be provided in derived classes
  // These function should work on electrons or photons
  // of the reco or pat type

  virtual float mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::EventBase&) const = 0;

  // A specific implementation of MVA is expected to have one or more categories
  // defined with respect to eta, pt, etc.
  // This function determines the category for a given particle.
  virtual int findCategory( const edm::Ptr<reco::Candidate>& particle) const = 0;
  virtual int getNCategories() const = 0;
  // The name is a unique name associated with a particular MVA implementation,
  // it is found as a const data member in a derived class.
  virtual const std::string& getName() const = 0;
  // An extra variable string set during construction that can be used
  // to distinguish different instances of the estimator configured with
  // different weight files. The tag can be used to construct names of ValueMaps, etc.
  virtual const std::string& getTag() const = 0;

  //
  // Extra event content - if needed.
  //
  // Some MVA implementation may require direct access to event content.
  // Implement these methods only if needed in the derived classes (use "override"
  // for certainty).

  // This method needs to be used only once after this MVA estimator is constructed
  virtual void setConsumes(edm::ConsumesCollector &&cc) const {};

 private:

  //
  // Data members
  //
  // Configuration
  const edm::ParameterSet conf_;

};

// define the factory for this base class
#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< AnyMVAEstimatorRun2Base* (const edm::ParameterSet&) >
        AnyMVAEstimatorRun2Factory;

#endif

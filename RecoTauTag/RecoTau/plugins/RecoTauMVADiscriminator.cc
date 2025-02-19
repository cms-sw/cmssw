/*
 * RecoTauMVADiscriminator
 *
 * Apply an MVA discriminator to a collection of PFTaus.  Output is a
 * PFTauDiscriminator.  The module takes the following options:
 *  > dbLabel - should match "appendToDataLabel" option of PoolDBSource
 *              if it exists.
 *  > mvas    - a vector of PSets, each of which contains nCharged, nPiZeros
 *              and a string giving the name of the correct MVA in the
 *              MVA ComputerContainer provided PoolDBSource.  This maps decay
 *              modes to MVA implementations.
 *  > defaultMVA - MVA to use if the decay mode does not match one specified in
 *              mvas.
 *  > remapOutput - TMVA gives its output from (-1, 1).  If this enabled remap
 *              it to (0, 1).
 *
 *  The interface to the MVA framework is handled by the RecoTauMVAHelper class.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_map.hpp>

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/RecoTauMVAHelper.h"
#include "RecoTauTag/RecoTau/interface/PFTauDecayModeTools.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class RecoTauMVADiscriminator : public PFTauDiscriminationProducerBase {
  public:
    explicit RecoTauMVADiscriminator(const edm::ParameterSet& pset);
    ~RecoTauMVADiscriminator() {}

    void beginEvent(const edm::Event&, const edm::EventSetup&);
    double discriminate(const reco::PFTauRef&);

  private:
    // Map a decay mode to an MVA getter
    typedef boost::ptr_map<reco::PFTau::hadronicDecayMode,
            reco::tau::RecoTauMVAHelper> MVAMap;

    std::auto_ptr<reco::tau::RecoTauMVAHelper> defaultMVA_;

    MVAMap mvas_;
    std::string dbLabel_;
    double unsupportedDMValue_;
    bool remapOutput_;
};

RecoTauMVADiscriminator::RecoTauMVADiscriminator(const edm::ParameterSet& pset)
  :PFTauDiscriminationProducerBase(pset) {
  std::string dbLabel;
  if (pset.exists("dbLabel"))
    dbLabel = pset.getParameter<std::string>("dbLabel");

  unsupportedDMValue_ = (pset.exists("unsupportedDecayModeValue")) ?
      pset.getParameter<double>("unsupportedDecayModeValue")
      : prediscriminantFailValue_;

  remapOutput_ = pset.getParameter<bool>("remapOutput");

  edm::ParameterSet discriminantOptions = pset.getParameter<edm::ParameterSet>(
      "discriminantOptions");

  typedef std::vector<edm::ParameterSet> VPSet;
  const VPSet& mvas = pset.getParameter<VPSet>("mvas");

  for (VPSet::const_iterator mva = mvas.begin(); mva != mvas.end(); ++mva) {
    unsigned int nCharged = mva->getParameter<unsigned int>("nCharged");
    unsigned int nPiZeros = mva->getParameter<unsigned int>("nPiZeros");
    reco::PFTau::hadronicDecayMode decayMode = reco::tau::translateDecayMode(
        nCharged, nPiZeros);
    // Check to ensure this decay mode is not already added
    if (!mvas_.count(decayMode)) {
      std::string computerName = mva->getParameter<std::string>("mvaLabel");
      // Add it
      mvas_.insert(
          decayMode, new reco::tau::RecoTauMVAHelper(
            computerName, dbLabel, discriminantOptions));
    } else {
      edm::LogError("DecayModeNotUnique") << "The tau decay mode with "
        "nCharged/nPiZero = " << nCharged << "/" << nPiZeros << " dm: "
        << decayMode <<
        " is associated to multiple MVA implmentations, "
        "the second instantiation is being ignored!!!";
    }
  }

  // Check if we a catch-all MVA is desired.
  if (pset.exists("defaultMVA")) {
    defaultMVA_.reset(new reco::tau::RecoTauMVAHelper(
            pset.getParameter<std::string>("defaultMVA"),
            dbLabel, discriminantOptions));
  }

}

void RecoTauMVADiscriminator::beginEvent(const edm::Event& evt,
                                         const edm::EventSetup& es) {
  // Pass the event setup so the MVAHelpers can get the MVAs from the DB
  BOOST_FOREACH(MVAMap::value_type mva, mvas_) {
      mva.second->setEvent(evt, es);
  }
  if (defaultMVA_.get())
    defaultMVA_->setEvent(evt, es);
}

// Get the MVA output for a given PFTau
double RecoTauMVADiscriminator::discriminate(const reco::PFTauRef& tau) {
  // Find the right MVA for this tau's decay mode
  MVAMap::iterator mva = mvas_.find(tau->decayMode());
  // If this DM has an associated decay mode, get and return the result.
  double output = unsupportedDMValue_;
  if (mva != mvas_.end() || defaultMVA_.get()) {
    if (mva != mvas_.end())
      output = mva->second->operator()(tau);
    else
      output = defaultMVA_->operator()(tau);
    // TMVA produces output from -1 to 1
    if (remapOutput_) {
      output += 1.;
      output /= 2.;
    }
  }
  return output;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauMVADiscriminator);

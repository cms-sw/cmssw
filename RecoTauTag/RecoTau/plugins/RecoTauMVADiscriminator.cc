/*
 * RecoTauMVADiscriminator
 *
 * Apply an MVA discriminator depending on the reconstructed
 * decay mode of the tau.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_map.hpp>

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/RecoTauMVAHelper.h"

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

  typedef std::vector<edm::ParameterSet> VPSet;
  const VPSet& mvas = pset.getParameter<VPSet>("mvas");

  for (VPSet::const_iterator mva = mvas.begin(); mva != mvas.end(); ++mva) {
    unsigned int nCharged = mva->getParameter<unsigned int>("nCharged");
    unsigned int nPiZeros = mva->getParameter<unsigned int>("nPiZeros");
    reco::PFTau::hadronicDecayMode decayMode = reco::PFTau::translateDecayMode(
        nCharged, nPiZeros);
    // Check to ensure this decay mode is not already added
    if (!mvas_.count(decayMode)) {
      std::string computerName = mva->getParameter<std::string>("mvaLabel");
      // Add it
      mvas_.insert(
          decayMode, new reco::tau::RecoTauMVAHelper(computerName, dbLabel));
    } else {
      edm::LogError("DecayModeNotUnique") << "The tau decay mode with "
        "nCharged/nPiZero = " << nCharged << "/" << nPiZeros << " dm: "
        << decayMode <<
        " is associated to multiple MVA implmentations, "
        "the second instantiation is being ignored!!!";
    }
  }
}

void RecoTauMVADiscriminator::beginEvent(const edm::Event& evt,
                                         const edm::EventSetup& es) {
  // Pass the event setup so the MVAHelpers can get the MVAs from the DB
  BOOST_FOREACH(MVAMap::value_type mva, mvas_) {
      mva.second->setEvent(evt, es);
  }
}

// Get the MVA output for a given PFTau
double RecoTauMVADiscriminator::discriminate(const PFTauRef& tau) {
  // Find the right MVA for this tau's decay mode
  MVAMap::iterator mva = mvas_.find(tau->decayMode());
  // If this DM has an associated decay mode, get and return the result.
  double output = unsupportedDMValue_;
  if (mva != mvas_.end()) {
      output= mva->second->operator()(tau);
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

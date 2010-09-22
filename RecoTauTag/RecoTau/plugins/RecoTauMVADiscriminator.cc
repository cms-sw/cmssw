/*
 * RecoTauMVADiscriminator
 *
 * Apply an MVA discriminator depending on the reconstructed
 * decay mode of the tau.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantPlugins.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

#include "CondFormats/DataRecord/interface/TauTagMVAComputerRcd.h"
#include "PhysicsTools/MVAComputer/interface/MVAModuleHelper.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/foreach.hpp>

class RecoTauMVADiscriminator : public PFTauDiscriminationProducerBase
{
  public: 
    explicit RecoTauMVADiscriminator(const edm::ParameterSet& pset);
    ~RecoTauMVADiscriminator() {};

    void beginEvent(const edm::Event&, const edm::EventSetup&);
    double discriminate(const reco::PFTauRef&);

  private:
    class DiscriminantPluginWrapper {
      public:
        explicit DiscriminantPluginWrapper(const std::string& name) {
          // Build a fake PSet containing only the plugin name
          edm::ParameterSet fakePSet;
          fakePSet.addParameter("name", "MVA_" + name);
          // Build the plugin
          plugin_ = std::auto_ptr<reco::tau::RecoTauDiscriminantPlugin>(
              RecoTauDiscriminantPluginFactory::get()->create(
                reco::tau::discPluginName(name), fakePSet));
        }

        // Get the discriminant value(s) for this tau
        std::vector<double> operator()(const reco::PFTauRef& tau) const {
          return (*plugin_)(tau);
        }

      private:
        // Ptr must be copy constructible
        boost::shared_ptr<reco::tau::RecoTauDiscriminantPlugin> plugin_;
    };
              
    // Define the MVA output extractor
    typedef PhysicsTools::MVAModuleHelper<TauTagMVAComputerRcd,
            reco::PFTauRef, DiscriminantPluginWrapper> MVAHelper;

    // Map a decay mode to an MVA getter
    typedef std::map<reco::PFTau::hadronicDecayMode, MVAHelper> MVAMap;

    MVAMap mvas_;
    std::string dbLabel_;
    double unsupportedDMValue_;
};

RecoTauMVADiscriminator::RecoTauMVADiscriminator(const edm::ParameterSet& pset):
  PFTauDiscriminationProducerBase(pset) {

  if (pset.exists("dbLabel"))
    dbLabel_ = pset.getParameter<std::string>("dbLabel");

  unsupportedDMValue_ = (pset.exists("unsupportedDecayModeValue")) ?
      pset.getParameter<double>("unsupportedDecayModeValue") : -5.0;

  typedef std::vector<edm::ParameterSet> VPSet;
  const VPSet& mvas = pset.getParameter<VPSet>("mvas");

  for (VPSet::const_iterator mva = mvas.begin(); mva != mvas.end(); ++mva) {
    unsigned int nCharged = mva->getParameter<unsigned int>("nCharged");
    unsigned int nPiZeros = mva->getParameter<unsigned int>("nPiZeros");
    reco::PFTau::hadronicDecayMode decayMode = reco::PFTau::translateDecayMode(
        nCharged, nPiZeros);
    // Check to ensure this decay mode is not already added
    if (!mvas_.count(decayMode)) {
      // Add it
      mvas_.insert(std::make_pair(decayMode,
            MVAHelper(mva->getParameter<std::string>("mvaLabel"))));
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
  using boost::bind;
  // Pass the event setup so the MVAHelpers can get the MVAs from the DB
  if (!dbLabel_.empty()) {
    BOOST_FOREACH(MVAMap::value_type &mva, mvas_) {
      mva.second.setEventSetup(es);
    }
  } else {
    BOOST_FOREACH(MVAMap::value_type &mva, mvas_) {
      mva.second.setEventSetup(es, dbLabel_.c_str());
    }
  }
}

// Get the MVA output for a given PFTau
double RecoTauMVADiscriminator::discriminate(const PFTauRef& tau) {
  // Find the right MVA for this tau's decay mode
  MVAMap::iterator mva = mvas_.find(tau->decayMode());
  // If this DM has an associated decay mode, get and return the result.
  double output =  (mva != mvas_.end()) ? mva->second(tau) : unsupportedDMValue_;
  //std::cout << " tau: " << *tau << " Decay mode: " << tau->decayMode() <<
  //  " mva: " << (mva != mvas_.end()) << " tanc: " << output << std::endl;
  return output;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauMVADiscriminator);

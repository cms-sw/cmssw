#ifndef PhysicsTools_PatAlgos_TauJetCorrFactorsProducer_h
#define PhysicsTools_PatAlgos_TauJetCorrFactorsProducer_h

/**
  \class    pat::TauJetCorrFactorsProducer TauJetCorrFactorsProducer.h "PhysicsTools/PatAlgos/interface/TauJetCorrFactorsProducer.h"
  \brief    Produces a ValueMap between TauJetCorrFactors and the originating reco taus

   The TauJetCorrFactorsProducer produces a set of tau-jet energy correction factors, defined in the class pat::TauJetCorrFactors.
   This vector is linked to the originating reco taus through an edm::ValueMap. The initializing parameters of the module can be found
   in the recoLayer1/tauJetCorrFactors_cfi.py of the PatAlgos package. In the standard PAT workflow the module has to be run
   before the creation of the pat::Tau. The edm::ValueMap will then be embedded into the pat::Tau.

   Jets corrected up to a given correction level can then be accessed via the pat::Tau member function correctedJet. For
   more details have a look into the class description of the pat::Tau.
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/PatCandidates/interface/TauJetCorrFactors.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <map>
#include <string>

namespace pat {

  class TauJetCorrFactorsProducer : public edm::stream::EDProducer<> {
  public:
    /// value map for JetCorrFactors (to be written into the event)
    typedef edm::ValueMap<pat::TauJetCorrFactors> JetCorrFactorsMap;

  public:
    /// default constructor
    explicit TauJetCorrFactorsProducer(const edm::ParameterSet&);
    /// default destructor
    ~TauJetCorrFactorsProducer() override{};

    /// everything that needs to be done per event
    void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    /// return the jec parameters as input to the FactorizedJetCorrector for different flavors
    std::vector<JetCorrectorParameters> params(const JetCorrectorParametersCollection&,
                                               const std::vector<std::string>&) const;

    /// evaluate jet correction factor up to a given level
    float evaluate(edm::View<reco::BaseTau>::const_iterator&, std::shared_ptr<FactorizedJetCorrector>&, int);

  private:
    /// python label of this TauJetCorrFactorsProducer module
    std::string moduleLabel_;

    /// input tau-jet collection
    edm::EDGetTokenT<edm::View<reco::BaseTau> > srcToken_;

    /// mapping of reconstructed tau decay modes to payloads
    typedef std::vector<int> vint;
    struct payloadMappingType {
      /// reconstructed tau decay modes associated to this payload,
      /// as defined in DataFormats/TauReco/interface/PFTau.h
      vint decayModes_;

      /// payload label
      std::string payload_;
    };
    std::vector<payloadMappingType> payloadMappings_;

    /// payload to be used for decay modes not explicitely specified
    ///
    /// NOTE: no decay mode reconstruction implemented for CaloTaus so far
    ///      --> this payload is used for all CaloTaus
    ///
    std::string defaultPayload_;

    /// jec levels
    typedef std::vector<std::string> vstring;
    vstring levels_;
  };
}  // namespace pat

#endif

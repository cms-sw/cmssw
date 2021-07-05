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

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TauReco/interface/PFTau.h"
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
#include <memory>
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

/// value map for JetCorrFactors (to be written into the event)
typedef edm::ValueMap<pat::TauJetCorrFactors> TauJetCorrFactorsMap;

using namespace pat;

TauJetCorrFactorsProducer::TauJetCorrFactorsProducer(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      srcToken_(consumes<edm::View<reco::BaseTau> >(cfg.getParameter<edm::InputTag>("src"))),
      levels_(cfg.getParameter<std::vector<std::string> >("levels")) {
  typedef std::vector<edm::ParameterSet> vParameterSet;
  vParameterSet parameters = cfg.getParameter<vParameterSet>("parameters");
  for (vParameterSet::const_iterator param = parameters.begin(); param != parameters.end(); ++param) {
    payloadMappingType payloadMapping;

    payloadMapping.payload_ = param->getParameter<std::string>("payload");

    vstring decayModes_string = param->getParameter<vstring>("decayModes");
    for (vstring::const_iterator decayMode = decayModes_string.begin(); decayMode != decayModes_string.end();
         ++decayMode) {
      if ((*decayMode) == "*") {
        defaultPayload_ = payloadMapping.payload_;
      } else {
        payloadMapping.decayModes_.push_back(atoi(decayMode->data()));
      }
    }

    if (!payloadMapping.decayModes_.empty())
      payloadMappings_.push_back(payloadMapping);
  }

  produces<TauJetCorrFactorsMap>();
}

std::vector<JetCorrectorParameters> TauJetCorrFactorsProducer::params(
    const JetCorrectorParametersCollection& jecParameters, const std::vector<std::string>& levels) const {
  std::vector<JetCorrectorParameters> retVal;
  for (std::vector<std::string>::const_iterator corrLevel = levels.begin(); corrLevel != levels.end(); ++corrLevel) {
    const JetCorrectorParameters& jecParameter_level = jecParameters[*corrLevel];
    retVal.push_back(jecParameter_level);
  }
  return retVal;
}

float TauJetCorrFactorsProducer::evaluate(edm::View<reco::BaseTau>::const_iterator& tauJet,
                                          std::shared_ptr<FactorizedJetCorrector>& corrector,
                                          int corrLevel) {
  corrector->setJetEta(tauJet->eta());
  corrector->setJetPt(tauJet->pt());
  corrector->setJetE(tauJet->energy());
  return corrector->getSubCorrections()[corrLevel];
}

void TauJetCorrFactorsProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  // get tau-jet collection from the event
  edm::Handle<edm::View<reco::BaseTau> > tauJets;
  evt.getByToken(srcToken_, tauJets);

  typedef std::shared_ptr<FactorizedJetCorrector> FactorizedJetCorrectorPtr;
  std::map<std::string, FactorizedJetCorrectorPtr> correctorMapping;

  // fill the tauJetCorrFactors
  std::vector<TauJetCorrFactors> tauJetCorrections;
  for (edm::View<reco::BaseTau>::const_iterator tauJet = tauJets->begin(); tauJet != tauJets->end(); ++tauJet) {
    // the TauJetCorrFactors::CorrectionFactor is a std::pair<std::string, float>
    // the string corresponds to the label of the correction level, the float to the tau-jet energy correction factor.
    // The first correction level is predefined with label 'Uncorrected'. The correction factor is 1.
    std::vector<TauJetCorrFactors::CorrectionFactor> jec;
    jec.push_back(std::make_pair(std::string("Uncorrected"), 1.0));

    if (levels_.empty())
      throw cms::Exception("No JECFactors")
          << "You request to create a jetCorrFactors object with no JEC Levels indicated. \n"
          << "This makes no sense, either you should correct this or drop the module from \n"
          << "the sequence.";

    std::string payload = defaultPayload_;
    if (dynamic_cast<const reco::PFTau*>(&(*tauJet))) {
      const reco::PFTau* pfTauJet = dynamic_cast<const reco::PFTau*>(&(*tauJet));
      for (std::vector<payloadMappingType>::const_iterator payloadMapping = payloadMappings_.begin();
           payloadMapping != payloadMappings_.end();
           ++payloadMapping) {
        for (vint::const_iterator decayMode = payloadMapping->decayModes_.begin();
             decayMode != payloadMapping->decayModes_.end();
             ++decayMode) {
          if (pfTauJet->decayMode() == (*decayMode))
            payload = payloadMapping->payload_;
        }
      }
    }

    // retrieve JEC parameters from the DB and build a new corrector,
    // in case it does not exist already for current payload
    if (correctorMapping.find(payload) == correctorMapping.end()) {
      edm::ESHandle<JetCorrectorParametersCollection> jecParameters;
      es.get<JetCorrectionsRecord>().get(payload, jecParameters);

      correctorMapping[payload] = std::make_shared<FactorizedJetCorrector>(params(*jecParameters, levels_));
    }
    FactorizedJetCorrectorPtr& corrector = correctorMapping[payload];

    // evaluate tau-jet energy corrections
    size_t numLevels = levels_.size();
    for (size_t idx = 0; idx < numLevels; ++idx) {
      const std::string& corrLevel = levels_[idx];

      float jecFactor = evaluate(tauJet, corrector, idx);

      // push back the set of JetCorrFactors: the first entry corresponds to the label
      // of the correction level. The second parameter corresponds to the jec factor.
      // In the default configuration the CorrectionFactor will look like this:
      //   'Uncorrected' : 1 ;
      //   'L2Relative'  : x ;
      //   'L3Absolute'  : x ;
      jec.push_back(std::make_pair(corrLevel.substr(0, corrLevel.find('_')), jecFactor));
    }

    // create the actual object with the scale factors we want the valuemap to refer to
    // moduleLabel_ corresponds to the python label of the TauJetCorrFactorsProducer module instance
    TauJetCorrFactors tauJetCorrection(moduleLabel_, jec);
    tauJetCorrections.push_back(tauJetCorrection);
  }

  // build the valuemap
  auto jecMapping = std::make_unique<TauJetCorrFactorsMap>();
  TauJetCorrFactorsMap::Filler filler(*jecMapping);
  // tauJets and tauJetCorrections vectors have their indices aligned by construction
  filler.insert(tauJets, tauJetCorrections.begin(), tauJetCorrections.end());
  filler.fill();  // do the actual filling

  // add valuemap to the event
  evt.put(std::move(jecMapping));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TauJetCorrFactorsProducer);

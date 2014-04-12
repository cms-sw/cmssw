#include "PhysicsTools/PatAlgos/plugins/TauJetCorrFactorsProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TauReco/interface/PFTau.h"

/// value map for JetCorrFactors (to be written into the event)
typedef edm::ValueMap<pat::TauJetCorrFactors> TauJetCorrFactorsMap;

using namespace pat;

TauJetCorrFactorsProducer::TauJetCorrFactorsProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    srcToken_(consumes<edm::View<reco::BaseTau> >(cfg.getParameter<edm::InputTag>("src"))),
    levels_(cfg.getParameter<std::vector<std::string> >("levels"))
{
  typedef std::vector<edm::ParameterSet> vParameterSet;
  vParameterSet parameters = cfg.getParameter<vParameterSet>("parameters");
  for ( vParameterSet::const_iterator param = parameters.begin();
	param != parameters.end(); ++param ) {
    payloadMappingType payloadMapping;

    payloadMapping.payload_ = param->getParameter<std::string>("payload");

    vstring decayModes_string = param->getParameter<vstring>("decayModes");
    for ( vstring::const_iterator decayMode = decayModes_string.begin();
	  decayMode != decayModes_string.end(); ++decayMode ) {
      if ( (*decayMode) == "*" ) {
	defaultPayload_ = payloadMapping.payload_;
      } else {
	payloadMapping.decayModes_.push_back(atoi(decayMode->data()));
      }
    }

    if ( payloadMapping.decayModes_.size() > 0 ) payloadMappings_.push_back(payloadMapping);
  }

  produces<TauJetCorrFactorsMap>();
}

std::vector<JetCorrectorParameters>
TauJetCorrFactorsProducer::params(const JetCorrectorParametersCollection& jecParameters, const std::vector<std::string>& levels) const
{
  std::vector<JetCorrectorParameters> retVal;
  for ( std::vector<std::string>::const_iterator corrLevel = levels.begin();
	corrLevel != levels.end(); ++corrLevel ) {
    const JetCorrectorParameters& jecParameter_level = jecParameters[*corrLevel];
    retVal.push_back(jecParameter_level);
  }
  return retVal;
}

float
TauJetCorrFactorsProducer::evaluate(edm::View<reco::BaseTau>::const_iterator& tauJet,
				    boost::shared_ptr<FactorizedJetCorrector>& corrector, int corrLevel)
{
  corrector->setJetEta(tauJet->eta());
  corrector->setJetPt(tauJet->pt());
  corrector->setJetE(tauJet->energy());
  return corrector->getSubCorrections()[corrLevel];
}

void
TauJetCorrFactorsProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // get tau-jet collection from the event
  edm::Handle<edm::View<reco::BaseTau> > tauJets;
  evt.getByToken(srcToken_, tauJets);

  typedef boost::shared_ptr<FactorizedJetCorrector> FactorizedJetCorrectorPtr;
  std::map<std::string, FactorizedJetCorrectorPtr> correctorMapping;

  // fill the tauJetCorrFactors
  std::vector<TauJetCorrFactors> tauJetCorrections;
  for ( edm::View<reco::BaseTau>::const_iterator tauJet = tauJets->begin();
	tauJet != tauJets->end(); ++tauJet ) {

    // the TauJetCorrFactors::CorrectionFactor is a std::pair<std::string, float>
    // the string corresponds to the label of the correction level, the float to the tau-jet energy correction factor.
    // The first correction level is predefined with label 'Uncorrected'. The correction factor is 1.
    std::vector<TauJetCorrFactors::CorrectionFactor> jec;
    jec.push_back(std::make_pair(std::string("Uncorrected"), 1.0));

    if ( levels_.size() == 0 )
      throw cms::Exception("No JECFactors")
	<< "You request to create a jetCorrFactors object with no JEC Levels indicated. \n"
	<< "This makes no sense, either you should correct this or drop the module from \n"
	<< "the sequence.";

    std::string payload = defaultPayload_;
    if ( dynamic_cast<const reco::PFTau*>(&(*tauJet)) ) {
      const reco::PFTau* pfTauJet = dynamic_cast<const reco::PFTau*>(&(*tauJet));
      for ( std::vector<payloadMappingType>::const_iterator payloadMapping = payloadMappings_.begin();
	    payloadMapping != payloadMappings_.end(); ++payloadMapping ) {
	for( vint::const_iterator decayMode = payloadMapping->decayModes_.begin();
	     decayMode != payloadMapping->decayModes_.end(); ++decayMode ) {
	  if ( pfTauJet->decayMode() == (*decayMode) ) payload = payloadMapping->payload_;
	}
      }
    }

    // retrieve JEC parameters from the DB and build a new corrector,
    // in case it does not exist already for current payload
    if ( correctorMapping.find(payload) == correctorMapping.end() ) {
      edm::ESHandle<JetCorrectorParametersCollection> jecParameters;
      es.get<JetCorrectionsRecord>().get(payload, jecParameters);

      correctorMapping[payload] = FactorizedJetCorrectorPtr(new FactorizedJetCorrector(params(*jecParameters, levels_)));
    }
    FactorizedJetCorrectorPtr& corrector = correctorMapping[payload];

    // evaluate tau-jet energy corrections
    size_t numLevels = levels_.size();
    for ( size_t idx = 0; idx < numLevels; ++idx ) {
      const std::string& corrLevel = levels_[idx];

      float jecFactor = evaluate(tauJet, corrector, idx);

      // push back the set of JetCorrFactors: the first entry corresponds to the label
      // of the correction level. The second parameter corresponds to the jec factor.
      // In the default configuration the CorrectionFactor will look like this:
      //   'Uncorrected' : 1 ;
      //   'L2Relative'  : x ;
      //   'L3Absolute'  : x ;
      jec.push_back(std::make_pair(corrLevel.substr(0, corrLevel.find("_")), jecFactor));
    }

    // create the actual object with the scale factors we want the valuemap to refer to
    // moduleLabel_ corresponds to the python label of the TauJetCorrFactorsProducer module instance
    TauJetCorrFactors tauJetCorrection(moduleLabel_, jec);
    tauJetCorrections.push_back(tauJetCorrection);
  }

  // build the valuemap
  std::auto_ptr<TauJetCorrFactorsMap> jecMapping(new TauJetCorrFactorsMap());
  TauJetCorrFactorsMap::Filler filler(*jecMapping);
  // tauJets and tauJetCorrections vectors have their indices aligned by construction
  filler.insert(tauJets, tauJetCorrections.begin(), tauJetCorrections.end());
  filler.fill(); // do the actual filling

  // add valuemap to the event
  evt.put(jecMapping);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TauJetCorrFactorsProducer);

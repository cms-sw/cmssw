#include "JetMETCorrections/Type1MET/plugins/PFJetMETcorrInputProducer.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

#include "JetMETCorrections/Type1MET/interface/metCorrAuxFunctions.h"

using namespace metCorr_namespace;

PFJetMETcorrInputProducer::PFJetMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    skipMuonSelection_(0)
{
  src_ = cfg.getParameter<edm::InputTag>("src");

  offsetCorrLabel_ = ( cfg.exists("offsetCorrLabel") ) ?
    cfg.getParameter<std::string>("offsetCorrLabel") : "";
  jetCorrLabel_ = cfg.getParameter<std::string>("jetCorrLabel");

  jetCorrEtaMax_ = cfg.getParameter<double>("jetCorrEtaMax");

  type1JetPtThreshold_ = cfg.getParameter<double>("type1JetPtThreshold");

  skipEM_ = cfg.getParameter<bool>("skipEM");
  if ( skipEM_ ) {
    skipEMfractionThreshold_ = cfg.getParameter<double>("skipEMfractionThreshold");
  }

  skipMuons_ = cfg.getParameter<bool>("skipMuons");
  if ( skipMuons_ ) {
    std::string skipMuonSelection_string = cfg.getParameter<std::string>("skipMuonSelection");
    skipMuonSelection_ = new StringCutObjectSelector<reco::Muon>(skipMuonSelection_string);
  }

  produces<CorrMETData>("type1");
  produces<CorrMETData>("type2");
  produces<CorrMETData>("offset");
}

PFJetMETcorrInputProducer::~PFJetMETcorrInputProducer()
{
  delete skipMuonSelection_;
}

void PFJetMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
 std::auto_ptr<CorrMETData> type1Correction(new CorrMETData());
 std::auto_ptr<CorrMETData> unclEnergySum(new CorrMETData());
 std::auto_ptr<CorrMETData> offsetEnergySum(new CorrMETData());

  edm::Handle<reco::PFJetCollection> jets;
  evt.getByLabel(src_, jets);

  int numJets = jets->size();
  for ( int jetIndex = 0; jetIndex < numJets; ++jetIndex ) {
    const reco::PFJet& rawJet = jets->at(jetIndex);

    edm::RefToBase<reco::Jet> rawJetRef(reco::PFJetRef(jets, jetIndex));
    
    if ( skipEM_ && rawJet.photonEnergyFraction() > skipEMfractionThreshold_ ) continue;

    reco::Candidate::LorentzVector rawJetP4 = rawJet.p4();
    if ( skipMuons_ ) {
      std::vector<reco::PFCandidatePtr> cands = rawJet.getPFConstituents();
      for ( std::vector<reco::PFCandidatePtr>::const_iterator cand = cands.begin();
	    cand != cands.end(); ++cand ) {
	if ( (*cand)->muonRef().isNonnull() && (*skipMuonSelection_)(*(*cand)->muonRef()) ) {
	  reco::Candidate::LorentzVector muonP4 = (*cand)->p4();
	  rawJetP4 -= muonP4;
	}
      }
    }

    const JetCorrector* jetCorrector = JetCorrector::getJetCorrector(jetCorrLabel_, es);
    if ( !jetCorrector )  
      throw cms::Exception("PFJetMETcorrInputProducer::produce")
	<< "Failed to access Jet corrections for = " << jetCorrLabel_ << " !!\n";
    double jetScaleFactor = getJetCorrFactor(jetCorrector, rawJet, rawJetRef, rawJetP4, evt, es, jetCorrEtaMax_);

    double corrJetPt = jetScaleFactor*rawJetP4.pt();
    if ( corrJetPt > type1JetPtThreshold_ ) {

      double offsetScaleFactor = 0.;
      if ( offsetCorrLabel_ != "" ) {
	const JetCorrector* offsetCorrector = JetCorrector::getJetCorrector(offsetCorrLabel_, es);
	if ( !offsetCorrector )  
	  throw cms::Exception("PFJetMETcorrInputProducer::produce")
	    << "Failed to access Jet corrections for = " << offsetCorrLabel_ << " !!\n";
	offsetScaleFactor = getJetCorrFactor(offsetCorrector, rawJet, rawJetRef, rawJetP4, evt, es, jetCorrEtaMax_);
      }

//--- MET balances momentum of reconstructed particles,
//    hence correction to jets and corresponding Type 1 MET correction are of opposite sign
      type1Correction->mex   -= (jetScaleFactor - offsetScaleFactor)*rawJetP4.px();
      type1Correction->mey   -= (jetScaleFactor - offsetScaleFactor)*rawJetP4.py();
      type1Correction->sumet += (jetScaleFactor - offsetScaleFactor)*rawJetP4.Et();
      
      offsetEnergySum->mex   += (1. - offsetScaleFactor)*rawJetP4.px();
      offsetEnergySum->mey   += (1. - offsetScaleFactor)*rawJetP4.py();
      offsetEnergySum->sumet += (1. - offsetScaleFactor)*rawJetP4.Et();
    } else {
      unclEnergySum->mex     += rawJetP4.px();
      unclEnergySum->mey     += rawJetP4.py();
      unclEnergySum->sumet   += rawJetP4.Et();
    }
  }

//--- add 
//     o Type 1 MET correction                (difference corrected-uncorrected jet energy for jets of (corrected) Pt > 10 GeV)
//     o momentum sum of "unclustered energy" (jets of (corrected) Pt < 10 GeV)
//     o momentum sum of "offset energy"      (sum of energy attributed to pile-up/underlying event)
//    to the event
  evt.put(type1Correction, "type1");
  evt.put(unclEnergySum,   "type2");
  evt.put(offsetEnergySum, "offset");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFJetMETcorrInputProducer);

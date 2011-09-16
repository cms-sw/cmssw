#ifndef JetMETCorrections_Type1MET_PFJetMETcorrInputProducerT_h
#define JetMETCorrections_Type1MET_PFJetMETcorrInputProducerT_h

/** \class PFJetMETcorrInputProducerT
 *
 * Produce Type 1 + 2 MET corrections corresponding to differences
 * between raw PFJets and PFJets with jet energy corrections (JECs) applied
 *
 * NOTE: class is templated to that it works with reco::PFJets as well as with pat::Jets of PF-type as input
 *
 * \authors Michael Schmitt, Richard Cavanaugh, The University of Florida
 *          Florent Lacroix, University of Illinois at Chicago
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: PFJetMETcorrInputProducerT.h,v 1.1 2011/09/13 14:35:35 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

#include <string>

namespace PFJetMETcorrInputProducer_namespace
{
  template <typename T>
  class InputTypeCheckerT
  {
    public:

     void operator()(const T&) const {} // no type-checking needed for reco::PFJet input
  };
}

template <typename T>
class PFJetMETcorrInputProducerT : public edm::EDProducer  
{
 public:

  explicit PFJetMETcorrInputProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      skipMuonSelection_(0)
  {
    src_ = cfg.getParameter<edm::InputTag>("src");
    
    offsetCorrLabel_ = ( cfg.exists("offsetCorrLabel") ) ?
      cfg.getParameter<std::string>("offsetCorrLabel") : "";
    jetCorrLabel_ = cfg.getParameter<std::string>("jetCorrLabel");
    
    jetCorrEtaMax_ = ( cfg.exists("jetCorrEtaMax") ) ?
      cfg.getParameter<double>("jetCorrEtaMax") : 9.9;
    
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
  ~PFJetMETcorrInputProducerT()
  {
    delete skipMuonSelection_;
  }
    
 private:

  void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    std::auto_ptr<CorrMETData> type1Correction(new CorrMETData());
    std::auto_ptr<CorrMETData> unclEnergySum(new CorrMETData());
    std::auto_ptr<CorrMETData> offsetEnergySum(new CorrMETData());

    typedef std::vector<T> JetCollection;
    edm::Handle<JetCollection> jets;
    evt.getByLabel(src_, jets);

    int numJets = jets->size();
    for ( int jetIndex = 0; jetIndex < numJets; ++jetIndex ) {
      const T& rawJet = jets->at(jetIndex);
      
      static PFJetMETcorrInputProducer_namespace::InputTypeCheckerT<T> checkInputType;
      checkInputType(rawJet);
      
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

      edm::RefToBase<reco::Jet> rawJetRef(edm::Ref<JetCollection>(jets, jetIndex));

      reco::Candidate::LorentzVector corrJetP4 = jetCorrExtractor_(rawJet, jetCorrLabel_, 
								   &evt, &es, &rawJetRef, jetCorrEtaMax_, &rawJetP4);
      if ( corrJetP4.pt() > type1JetPtThreshold_ ) {
	
	reco::Candidate::LorentzVector rawJetP4offsetCorr = rawJetP4;
	if ( offsetCorrLabel_ != "" ) {
	  rawJetP4offsetCorr = jetCorrExtractor_(rawJet, offsetCorrLabel_, 
						 &evt, &es, &rawJetRef, jetCorrEtaMax_, &rawJetP4);
	  
	  offsetEnergySum->mex   += (rawJetP4.px() - rawJetP4offsetCorr.px());
	  offsetEnergySum->mey   += (rawJetP4.py() - rawJetP4offsetCorr.py());
	  offsetEnergySum->sumet += (rawJetP4.Et() - rawJetP4offsetCorr.Et());
	}

//--- MET balances momentum of reconstructed particles,
//    hence correction to jets and corresponding Type 1 MET correction are of opposite sign
	type1Correction->mex   -= (corrJetP4.px() - rawJetP4offsetCorr.px());
	type1Correction->mey   -= (corrJetP4.py() - rawJetP4offsetCorr.py());
	type1Correction->sumet += (corrJetP4.Et() - rawJetP4offsetCorr.Et());
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

  std::string moduleLabel_;

  edm::InputTag src_; // PFJet input collection

  std::string offsetCorrLabel_; // e.g. 'ak5PFJetL1Fastjet'
  std::string jetCorrLabel_;    // e.g. 'ak5PFJetL1FastL2L3' (MC) / 'ak5PFJetL1FastL2L3Residual' (Data)
  JetCorrExtractorT<T> jetCorrExtractor_;

  double jetCorrEtaMax_; // do not use JEC factors for |eta| above this threshold (recommended default = 4.7),
                         // in order to work around problem with CMSSW_4_2_x JEC factors at high eta,
                         // reported in
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/jes/270.html
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1259/1.html

  double type1JetPtThreshold_; // threshold to distinguish between jets entering Type 1 MET correction
                               // and jets entering "unclustered energy" sum
                               // NOTE: threshold is applied on **corrected** jet energy (recommended default = 10 GeV)

  bool skipEM_; // flag to exclude jets with large fraction of electromagnetic energy (electrons/photons) 
                // from Type 1 + 2 MET corrections
  double skipEMfractionThreshold_;

  bool skipMuons_; // flag to subtract momentum of muons (provided muons pass selection cuts) which are within jets
                   // from jet energy before compute JECs/propagating JECs to Type 1 + 2 MET corrections
  StringCutObjectSelector<reco::Muon>* skipMuonSelection_;
};

#endif



 

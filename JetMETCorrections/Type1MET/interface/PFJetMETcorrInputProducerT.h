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
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
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
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

#include <string>
#include <type_traits>

namespace PFJetMETcorrInputProducer_namespace
{
  template <typename T, typename Textractor>
  class InputTypeCheckerT
  {
    public:

     void operator()(const T&) const {} // no type-checking needed for reco::PFJet input
     bool isPatJet(const T&) const {
       return std::is_base_of<class pat::Jet, T>::value;
     }
  };

  template <typename T>
  class RawJetExtractorT // this template is neccessary to support pat::Jets
                         // (because pat::Jet->p4() returns the JES corrected, not the raw, jet momentum)
    // But it does not handle the muon removal!!!!! MM
  {
    public:

     RawJetExtractorT(){}
     reco::Candidate::LorentzVector operator()(const T& jet) const
     {
       return jet.p4();
     }
  };
}

template <typename T, typename Textractor>
class PFJetMETcorrInputProducerT : public edm::stream::EDProducer<>
{
 public:

  explicit PFJetMETcorrInputProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      offsetCorrLabel_(""),
      skipMuonSelection_(0)
  {
    token_ = consumes<std::vector<T> >(cfg.getParameter<edm::InputTag>("src"));

    if ( cfg.exists("offsetCorrLabel") ) {
      offsetCorrLabel_ = cfg.getParameter<edm::InputTag>("offsetCorrLabel");
      offsetCorrToken_ = consumes<reco::JetCorrector>(offsetCorrLabel_);
    }
    jetCorrLabel_ = cfg.getParameter<edm::InputTag>("jetCorrLabel"); //for MC
    jetCorrLabelRes_ = cfg.getParameter<edm::InputTag>("jetCorrLabelRes"); //for data
    jetCorrToken_ = mayConsume<reco::JetCorrector>(jetCorrLabel_);
    jetCorrResToken_ = mayConsume<reco::JetCorrector>(jetCorrLabelRes_);

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
      skipMuonSelection_ = new StringCutObjectSelector<reco::Candidate>(skipMuonSelection_string,true);
    }

    if ( cfg.exists("type2Binning") ) {
      typedef std::vector<edm::ParameterSet> vParameterSet;
      vParameterSet cfgType2Binning = cfg.getParameter<vParameterSet>("type2Binning");
      for ( vParameterSet::const_iterator cfgType2BinningEntry = cfgType2Binning.begin();
	    cfgType2BinningEntry != cfgType2Binning.end(); ++cfgType2BinningEntry ) {
	type2Binning_.push_back(new type2BinningEntryType(*cfgType2BinningEntry));
      }
    } else {
      type2Binning_.push_back(new type2BinningEntryType());
    }

    produces<CorrMETData>("type1");
    for ( typename std::vector<type2BinningEntryType*>::const_iterator type2BinningEntry = type2Binning_.begin();
	  type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {
      produces<CorrMETData>((*type2BinningEntry)->getInstanceLabel_full("type2"));
      produces<CorrMETData>((*type2BinningEntry)->getInstanceLabel_full("offset"));
    }
  }
  ~PFJetMETcorrInputProducerT()
  {
    delete skipMuonSelection_;

    for ( typename std::vector<type2BinningEntryType*>::const_iterator it = type2Binning_.begin();
	  it != type2Binning_.end(); ++it ) {
      delete (*it);
    }
  }

 private:

  void produce(edm::Event& evt, const edm::EventSetup& es)
  {

    std::auto_ptr<CorrMETData> type1Correction(new CorrMETData());
    for ( typename std::vector<type2BinningEntryType*>::iterator type2BinningEntry = type2Binning_.begin();
	  type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {
      (*type2BinningEntry)->binUnclEnergySum_   = CorrMETData();
      (*type2BinningEntry)->binOffsetEnergySum_ = CorrMETData();
    }

    edm::Handle<reco::JetCorrector> jetCorr;
    //automatic switch for residual corrections
    if(evt.isRealData() ) {
      jetCorrLabel_ = jetCorrLabelRes_;
      evt.getByToken(jetCorrResToken_, jetCorr);
    } else {
      evt.getByToken(jetCorrToken_, jetCorr);
    }

    typedef std::vector<T> JetCollection;
    edm::Handle<JetCollection> jets;
    evt.getByToken(token_, jets);

    int numJets = jets->size();
    for ( int jetIndex = 0; jetIndex < numJets; ++jetIndex ) {
      const T& jet = jets->at(jetIndex);

      const static PFJetMETcorrInputProducer_namespace::InputTypeCheckerT<T, Textractor> checkInputType {};
      checkInputType(jet);

      double emEnergyFraction = jet.chargedEmEnergyFraction() + jet.neutralEmEnergyFraction();
      if ( skipEM_ && emEnergyFraction > skipEMfractionThreshold_ ) continue;

      const static PFJetMETcorrInputProducer_namespace::RawJetExtractorT<T> rawJetExtractor {};
      reco::Candidate::LorentzVector rawJetP4 = rawJetExtractor(jet);

      if ( skipMuons_ ) {
	const std::vector<reco::CandidatePtr> & cands = jet.daughterPtrVector();
	for ( std::vector<reco::CandidatePtr>::const_iterator cand = cands.begin();
	      cand != cands.end(); ++cand ) {
          const reco::PFCandidate *pfcand = dynamic_cast<const reco::PFCandidate *>(cand->get());
          const reco::Candidate *mu = (pfcand != 0 ? ( pfcand->muonRef().isNonnull() ? pfcand->muonRef().get() : 0) : cand->get());
	  if ( mu != 0 && (*skipMuonSelection_)(*mu) ) {
	    reco::Candidate::LorentzVector muonP4 = (*cand)->p4();
	    rawJetP4 -= muonP4;
	  }
	}
      }

      reco::Candidate::LorentzVector corrJetP4;
      if ( checkInputType.isPatJet(jet) )
        corrJetP4 = jetCorrExtractor_(jet, jetCorrLabel_.label(), jetCorrEtaMax_, &rawJetP4);
      else
        corrJetP4 = jetCorrExtractor_(jet, jetCorr.product(), jetCorrEtaMax_, &rawJetP4);
      
      if ( corrJetP4.pt() > type1JetPtThreshold_ ) {

	reco::Candidate::LorentzVector rawJetP4offsetCorr = rawJetP4;
	if ( !offsetCorrLabel_.label().empty() ) {
          edm::Handle<reco::JetCorrector> offsetCorr;
          evt.getByToken(offsetCorrToken_, offsetCorr);
          if ( checkInputType.isPatJet(jet) )
            rawJetP4offsetCorr = jetCorrExtractor_(jet, offsetCorrLabel_.label(), jetCorrEtaMax_, &rawJetP4);
          else
	    rawJetP4offsetCorr = jetCorrExtractor_(jet, offsetCorr.product(), jetCorrEtaMax_, &rawJetP4);

	  for ( typename std::vector<type2BinningEntryType*>::iterator type2BinningEntry = type2Binning_.begin();
		type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {
	    if ( !(*type2BinningEntry)->binSelection_ || (*(*type2BinningEntry)->binSelection_)(corrJetP4) ) {
	      (*type2BinningEntry)->binOffsetEnergySum_.mex   += (rawJetP4.px() - rawJetP4offsetCorr.px());
	      (*type2BinningEntry)->binOffsetEnergySum_.mey   += (rawJetP4.py() - rawJetP4offsetCorr.py());
	      (*type2BinningEntry)->binOffsetEnergySum_.sumet += (rawJetP4.Et() - rawJetP4offsetCorr.Et());
	    }
	  }
	}

//--- MET balances momentum of reconstructed particles,
//    hence correction to jets and corresponding Type 1 MET correction are of opposite sign
	type1Correction->mex   -= (corrJetP4.px() - rawJetP4offsetCorr.px());
	type1Correction->mey   -= (corrJetP4.py() - rawJetP4offsetCorr.py());
	type1Correction->sumet += (corrJetP4.Et() - rawJetP4offsetCorr.Et());
      } else {
	for ( typename std::vector<type2BinningEntryType*>::iterator type2BinningEntry = type2Binning_.begin();
	      type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {
	  if ( !(*type2BinningEntry)->binSelection_ || (*(*type2BinningEntry)->binSelection_)(corrJetP4) ) {
	    (*type2BinningEntry)->binUnclEnergySum_.mex   += rawJetP4.px();
	    (*type2BinningEntry)->binUnclEnergySum_.mey   += rawJetP4.py();
	    (*type2BinningEntry)->binUnclEnergySum_.sumet += rawJetP4.Et();
	  }
	}
      }
    }

//--- add
//     o Type 1 MET correction                (difference corrected-uncorrected jet energy for jets of (corrected) Pt > 10 GeV)
//     o momentum sum of "unclustered energy" (jets of (corrected) Pt < 10 GeV)
//     o momentum sum of "offset energy"      (sum of energy attributed to pile-up/underlying event)
//    to the event
    evt.put(type1Correction, "type1");
    for ( typename std::vector<type2BinningEntryType*>::const_iterator type2BinningEntry = type2Binning_.begin();
	  type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {
      evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*type2BinningEntry)->binUnclEnergySum_)), (*type2BinningEntry)->getInstanceLabel_full("type2"));
      evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*type2BinningEntry)->binOffsetEnergySum_)), (*type2BinningEntry)->getInstanceLabel_full("offset"));
    }
  }

  std::string moduleLabel_;

  edm::EDGetTokenT<std::vector<T> > token_;

  edm::InputTag offsetCorrLabel_;
  edm::EDGetTokenT<reco::JetCorrector> offsetCorrToken_; // e.g. 'ak5CaloJetL1Fastjet'
  edm::InputTag jetCorrLabel_;
  edm::InputTag jetCorrLabelRes_;
  edm::EDGetTokenT<reco::JetCorrector> jetCorrToken_;    // e.g. 'ak5CaloJetL1FastL2L3' (MC) / 'ak5CaloJetL1FastL2L3Residual' (Data)
  edm::EDGetTokenT<reco::JetCorrector> jetCorrResToken_;    // e.g. 'ak5CaloJetL1FastL2L3' (MC) / 'ak5CaloJetL1FastL2L3Residual' (Data)
  Textractor jetCorrExtractor_;

  double jetCorrEtaMax_; // do not use JEC factors for |eta| above this threshold

  double type1JetPtThreshold_; // threshold to distinguish between jets entering Type 1 MET correction
                               // and jets entering "unclustered energy" sum
                               // NOTE: threshold is applied on **corrected** jet energy (recommended default = 10 GeV)

  bool skipEM_; // flag to exclude jets with large fraction of electromagnetic energy (electrons/photons)
                // from Type 1 + 2 MET corrections
  double skipEMfractionThreshold_;

  bool skipMuons_; // flag to subtract momentum of muons (provided muons pass selection cuts) which are within jets
                   // from jet energy before compute JECs/propagating JECs to Type 1 + 2 MET corrections
  StringCutObjectSelector<reco::Candidate>* skipMuonSelection_;

  struct type2BinningEntryType
  {
    type2BinningEntryType()
      : binLabel_(""),
        binSelection_(0)
    {}
    type2BinningEntryType(const edm::ParameterSet& cfg)
      : binLabel_(cfg.getParameter<std::string>("binLabel")),
        binSelection_(new StringCutObjectSelector<reco::Candidate::LorentzVector>(cfg.getParameter<std::string>("binSelection")))
    {}
    ~type2BinningEntryType()
    {
      delete binSelection_;
    }
    std::string getInstanceLabel_full(const std::string& instanceLabel)
    {
      std::string retVal = instanceLabel;
      if ( instanceLabel != "" && binLabel_ != "" ) retVal.append("#");
      retVal.append(binLabel_);
      return retVal;
    }
    std::string binLabel_;
    StringCutObjectSelector<reco::Candidate::LorentzVector>* binSelection_;
    CorrMETData binUnclEnergySum_;
    CorrMETData binOffsetEnergySum_;
  };
  std::vector<type2BinningEntryType*> type2Binning_;
};

#endif





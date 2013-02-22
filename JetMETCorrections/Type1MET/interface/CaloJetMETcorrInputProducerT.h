#ifndef JetMETCorrections_Type1MET_CaloJetMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_CaloJetMETcorrInputProducer_h

/** \class CaloJetMETcorrInputProducer
 *
 * Produce Type 1 + 2 MET corrections corresponding to differences
 * between raw CaloJets and CaloJets with jet energy corrections (JECs) applied
 *
 * \authors Michael Schmitt, Richard Cavanaugh, The University of Florida
 *          Florent Lacroix, University of Illinois at Chicago
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.5 $
 *
 * $Id: CaloJetMETcorrInputProducerT.h,v 1.5 2012/02/13 14:18:39 veelken Exp $
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
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/MET.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

#include <string>

namespace CaloJetMETcorrInputProducer_namespace
{
  template <typename T>
  class InputTypeCheckerT
  {
    public:

     void operator()(const T&) const {} // no type-checking needed for reco::CaloJet input
  };

  template <typename T>
  class RawJetExtractorT // this template is neccessary to support pat::Jets
                         // (because pat::Jet->p4() returns the JES corrected, not the raw, jet momentum)
  {
    public:

     reco::Candidate::LorentzVector  operator()(const T& jet) const 
     { 
       return jet.p4();
     } 
  };
}

template <typename T, typename Textractor>
class CaloJetMETcorrInputProducerT : public edm::EDProducer  
{
 public:

  explicit CaloJetMETcorrInputProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
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

    if ( cfg.exists("srcMET") ) {
      srcMET_ = cfg.getParameter<edm::InputTag>("srcMET");
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

    type2ResidualCorrLabel_ = cfg.getParameter<std::string>("type2ResidualCorrLabel");
    type2ResidualCorrEtaMax_ = cfg.getParameter<double>("type2ResidualCorrEtaMax");

    produces<CorrMETData>("type1");
    if ( srcMET_.label() != "" ) {
      produces<CorrMETData>("type2fromMEt");
    }
    for ( typename std::vector<type2BinningEntryType*>::const_iterator type2BinningEntry = type2Binning_.begin();
	  type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {   
      produces<CorrMETData>((*type2BinningEntry)->getInstanceLabel_full("type2"));
      produces<CorrMETData>((*type2BinningEntry)->getInstanceLabel_full("offset"));
    }
  }
  ~CaloJetMETcorrInputProducerT() {}
    
 private:

  void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    std::auto_ptr<CorrMETData> type1Correction(new CorrMETData());
    std::auto_ptr<CorrMETData> type2Correction_from_MEt(new CorrMETData());
    for ( typename std::vector<type2BinningEntryType*>::iterator type2BinningEntry = type2Binning_.begin();
	  type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {
      (*type2BinningEntry)->binUnclEnergySum_   = CorrMETData();
      (*type2BinningEntry)->binOffsetEnergySum_ = CorrMETData();
    }

    const JetCorrector* type2ResidualCorrector = 0;
    if ( type2ResidualCorrLabel_ != "" ) {
      type2ResidualCorrector = JetCorrector::getJetCorrector(type2ResidualCorrLabel_, es);
      if ( !type2ResidualCorrector )  
	throw cms::Exception("CaloJetMETcorrInputProducer")
	  << "Failed to access Residual corrections = " << type2ResidualCorrLabel_ << " !!\n";
    }

    typedef std::vector<T> JetCollection;
    edm::Handle<JetCollection> jets;
    evt.getByLabel(src_, jets);

    typedef edm::View<reco::MET> METView;
    edm::Handle<METView> met;
    if ( srcMET_.label() != "" ) {
      // CV: allow to compute energy sum for Type 2 CaloMET correction 
      //     using uncorrected CaloMEt as input, as in order to allow CaloMET corrections
      //     to be computed without requiring all CaloTowers as input (aim is to save disk space)
      evt.getByLabel(srcMET_, met);
      if ( met->size() != 1 )
	throw cms::Exception("CaloJetMETcorrInputProducer::produce") 
	  << "Failed to find unique MET in the event, src = " << srcMET_.label() << " !!\n";

//--- compute "unclustered energy" by sutracting from the reconstructed MET 
//   (i.e. from the negative vectorial sum of all particles reconstructed in the event)
//    the momenta of (high Pt) jets which enter Type 1 MET corrections 
//
//    NOTE: MET = -(jets + muons + "unclustered energy"),
//          so "unclustered energy" = -(MET + jets + muons),
//          i.e. (high Pt) jets enter the sum of "unclustered energy" with negative sign.
//
      type2Correction_from_MEt->mex   = -met->front().px();
      type2Correction_from_MEt->mey   = -met->front().py();
      type2Correction_from_MEt->sumet =  met->front().sumEt();
    }

    int numJets = jets->size();
    for ( int jetIndex = 0; jetIndex < numJets; ++jetIndex ) {
      const T& rawJet = jets->at(jetIndex);

      static CaloJetMETcorrInputProducer_namespace::InputTypeCheckerT<T> checkInputType;
      checkInputType(rawJet);

      double emEnergyFraction = rawJet.emEnergyFraction();
      if ( skipEM_ && emEnergyFraction > skipEMfractionThreshold_ ) continue;

      static CaloJetMETcorrInputProducer_namespace::RawJetExtractorT<T> rawJetExtractor;
      reco::Candidate::LorentzVector rawJetP4 = rawJetExtractor(rawJet);
      
      reco::Candidate::LorentzVector corrJetP4 = jetCorrExtractor_(rawJet, jetCorrLabel_, &evt, &es, jetCorrEtaMax_);

      if ( corrJetP4.pt() > type1JetPtThreshold_ ) {
	
	reco::Candidate::LorentzVector rawJetP4offsetCorr = rawJetP4;
	if ( offsetCorrLabel_ != "" ) {
	  rawJetP4offsetCorr = jetCorrExtractor_(rawJet, offsetCorrLabel_, &evt, &es, jetCorrEtaMax_);

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

	type2Correction_from_MEt->mex   -= rawJetP4.px();
	type2Correction_from_MEt->mey   -= rawJetP4.py();
	type2Correction_from_MEt->sumet -= rawJetP4.Et();	
      } else {
	//std::cout << "jet #" << jetIndex << " (raw): Pt = " << rawJetP4.pt() << "," 
	//	    << " eta = " << rawJetP4.eta() << ", phi = " << rawJetP4.phi() << std::endl;
	
	double residualCorrFactor = 1.;
	if ( type2ResidualCorrector && fabs(rawJetP4.eta()) < type2ResidualCorrEtaMax_ ) {
	  residualCorrFactor = type2ResidualCorrector->correction(rawJetP4);
	  //std::cout << " residualCorrFactor = " << residualCorrFactor << std::endl;
	}
	
	type2Correction_from_MEt->mex   += ((residualCorrFactor - 1.)*rawJetP4.px());
	type2Correction_from_MEt->mey   += ((residualCorrFactor - 1.)*rawJetP4.py());
        type2Correction_from_MEt->sumet += ((residualCorrFactor - 1.)*rawJetP4.Et());

	for ( typename std::vector<type2BinningEntryType*>::iterator type2BinningEntry = type2Binning_.begin();
	      type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {
	  if ( !(*type2BinningEntry)->binSelection_ || (*(*type2BinningEntry)->binSelection_)(corrJetP4) ) {
	    (*type2BinningEntry)->binUnclEnergySum_.mex   += (residualCorrFactor*rawJetP4.px());
	    (*type2BinningEntry)->binUnclEnergySum_.mey   += (residualCorrFactor*rawJetP4.py());
	    (*type2BinningEntry)->binUnclEnergySum_.sumet += (residualCorrFactor*rawJetP4.Et());
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
    if ( srcMET_.label() != "" ) {
      evt.put(type2Correction_from_MEt, "type2fromMEt");
    }
    for ( typename std::vector<type2BinningEntryType*>::const_iterator type2BinningEntry = type2Binning_.begin();
	  type2BinningEntry != type2Binning_.end(); ++type2BinningEntry ) {
      evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*type2BinningEntry)->binUnclEnergySum_)), (*type2BinningEntry)->getInstanceLabel_full("type2"));
      evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*type2BinningEntry)->binOffsetEnergySum_)), (*type2BinningEntry)->getInstanceLabel_full("offset"));
    }
  }

  std::string moduleLabel_;

  edm::InputTag src_; // CaloJet input collection

  std::string offsetCorrLabel_; // e.g. 'ak5CaloJetL1Fastjet'
  std::string jetCorrLabel_;    // e.g. 'ak5CaloJetL1FastL2L3' (MC) / 'ak5CaloJetL1FastL2L3Residual' (Data)
  Textractor jetCorrExtractor_;

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

  edm::InputTag srcMET_; // MET input, needed to compute "unclustered energy" sum for Type 2 MET correction
  
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

  std::string type2ResidualCorrLabel_;
  double type2ResidualCorrEtaMax_;
};

#endif



 


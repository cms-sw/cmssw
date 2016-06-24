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
 *
 *
 */

#include "FWCore/Framework/interface/global/EDProducer.h"
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
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

#include <string>
#include <type_traits>

namespace CaloJetMETcorrInputProducer_namespace
{
  template <typename T>
  class InputTypeCheckerT
  {
    public:
     void operator()(const T&) const {} // no type-checking needed for reco::CaloJet input
     bool isPatJet(const T&) const {
       return std::is_base_of<class pat::Jet, T>::value;
     }
  };

  template <typename T>
  class RawJetExtractorT // this template is neccessary to support pat::Jets
                         // (because pat::Jet->p4() returns the JES corrected, not the raw, jet momentum)
  {
    public:
     RawJetExtractorT() {}

     reco::Candidate::LorentzVector  operator()(const T& jet) const
     {
       return jet.p4();
     }
  };
}

template <typename T, typename Textractor>
class CaloJetMETcorrInputProducerT final : public edm::global::EDProducer<>
{
 public:

  explicit CaloJetMETcorrInputProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      offsetCorrLabel_("")
  {
    token_ = consumes<std::vector<T> >(cfg.getParameter<edm::InputTag>("src"));

    if ( cfg.exists("offsetCorrLabel") ) {
      offsetCorrLabel_ = cfg.getParameter<edm::InputTag>("offsetCorrLabel");
      offsetCorrToken_ = consumes<reco::JetCorrector>(offsetCorrLabel_);
    }
    jetCorrLabel_ = cfg.getParameter<edm::InputTag>("jetCorrLabel");
    jetCorrToken_ = consumes<reco::JetCorrector>(jetCorrLabel_);

    jetCorrEtaMax_ = ( cfg.exists("jetCorrEtaMax") ) ?
      cfg.getParameter<double>("jetCorrEtaMax") : 9.9;

    type1JetPtThreshold_ = cfg.getParameter<double>("type1JetPtThreshold");

    skipEM_ = cfg.getParameter<bool>("skipEM");
    if ( skipEM_ ) {
      skipEMfractionThreshold_ = cfg.getParameter<double>("skipEMfractionThreshold");
    }

    if(cfg.exists("srcMET"))
      {
	srcMET_ = cfg.getParameter<edm::InputTag>("srcMET");
	metToken_ = consumes<edm::View<reco::MET> >(cfg.getParameter<edm::InputTag>("srcMET"));
      }

    produces<CorrMETData>("type1");
    produces<CorrMETData>("type2");
    produces<CorrMETData>("offset");
  }
  ~CaloJetMETcorrInputProducerT() {}

 private:

  void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const override
  {
    std::unique_ptr<CorrMETData> type1Correction(new CorrMETData());
    std::unique_ptr<CorrMETData> unclEnergySum(new CorrMETData());
    std::unique_ptr<CorrMETData> offsetEnergySum(new CorrMETData());

    edm::Handle<reco::JetCorrector> jetCorr;
    evt.getByToken(jetCorrToken_, jetCorr);

    typedef std::vector<T> JetCollection;
    edm::Handle<JetCollection> jets;
    evt.getByToken(token_, jets);

    typedef edm::View<reco::MET> METView;
    edm::Handle<METView> met;
    if ( srcMET_.label() != "" ) {
      evt.getByToken(metToken_, met);
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
      unclEnergySum->mex   = -met->front().px();
      unclEnergySum->mey   = -met->front().py();
      unclEnergySum->sumet =  met->front().sumEt();
    }

    int numJets = jets->size();
    for ( int jetIndex = 0; jetIndex < numJets; ++jetIndex ) {
      const T& rawJet = jets->at(jetIndex);

      const CaloJetMETcorrInputProducer_namespace::InputTypeCheckerT<T> checkInputType{};
      checkInputType(rawJet);

      const CaloJetMETcorrInputProducer_namespace::RawJetExtractorT<T> rawJetExtractor;
      reco::Candidate::LorentzVector rawJetP4 = rawJetExtractor(rawJet);

      reco::Candidate::LorentzVector corrJetP4;
      if ( checkInputType.isPatJet(rawJet) )
        corrJetP4 = jetCorrExtractor_(rawJet, jetCorrLabel_.label(), jetCorrEtaMax_);
      else
        corrJetP4 = jetCorrExtractor_(rawJet, jetCorr.product(), jetCorrEtaMax_);

      if ( corrJetP4.pt() > type1JetPtThreshold_ ) {

	unclEnergySum->mex   -= rawJetP4.px();
	unclEnergySum->mey   -= rawJetP4.py();
	unclEnergySum->sumet -= rawJetP4.Et();

	if ( skipEM_ && rawJet.emEnergyFraction() > skipEMfractionThreshold_ ) continue;

	reco::Candidate::LorentzVector rawJetP4offsetCorr = rawJetP4;
	if ( !offsetCorrLabel_.label().empty() ) {
          edm::Handle<reco::JetCorrector> offsetCorr;
          evt.getByToken(offsetCorrToken_, offsetCorr);
          if (  checkInputType.isPatJet(rawJet) )
            rawJetP4offsetCorr = jetCorrExtractor_(rawJet, offsetCorrLabel_.label(), jetCorrEtaMax_);
          else
	    rawJetP4offsetCorr = jetCorrExtractor_(rawJet, offsetCorr.product(), jetCorrEtaMax_);

	  offsetEnergySum->mex   += (rawJetP4.px() - rawJetP4offsetCorr.px());
	  offsetEnergySum->mey   += (rawJetP4.py() - rawJetP4offsetCorr.py());
	  offsetEnergySum->sumet += (rawJetP4.Et() - rawJetP4offsetCorr.Et());
	}

//--- MET balances momentum of reconstructed particles,
//    hence correction to jets and corresponding Type 1 MET correction are of opposite sign
	type1Correction->mex   -= (corrJetP4.px() - rawJetP4offsetCorr.px());
	type1Correction->mey   -= (corrJetP4.py() - rawJetP4offsetCorr.py());
	type1Correction->sumet += (corrJetP4.Et() - rawJetP4offsetCorr.Et());
      }
    }

//--- add
//     o Type 1 MET correction                (difference corrected-uncorrected jet energy for jets of (corrected) Pt > 20 GeV)
//     o momentum sum of "unclustered energy" (jets of (corrected) Pt < 20 GeV)
//     o momentum sum of "offset energy"      (sum of energy attributed to pile-up/underlying event)
//    to the event
    evt.put(std::move(type1Correction), "type1");
    evt.put(std::move(unclEnergySum),   "type2");
    evt.put(std::move(offsetEnergySum), "offset");
  }

  std::string moduleLabel_;

  edm::EDGetTokenT<std::vector<T> > token_;

  edm::InputTag offsetCorrLabel_;
  edm::EDGetTokenT<reco::JetCorrector> offsetCorrToken_; // e.g. 'ak5CaloJetL1Fastjet'
  edm::InputTag jetCorrLabel_;
  edm::EDGetTokenT<reco::JetCorrector> jetCorrToken_;    // e.g. 'ak5CaloJetL1FastL2L3' (MC) / 'ak5CaloJetL1FastL2L3Residual' (Data)
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
  edm::EDGetTokenT<edm::View<reco::MET> > metToken_;

};

#endif






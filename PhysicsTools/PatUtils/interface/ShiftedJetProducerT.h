#ifndef PhysicsTools_PatUtils_ShiftedJetProducerT_h
#define PhysicsTools_PatUtils_ShiftedJetProducerT_h

/** \class ShiftedJetProducerT
 *
 * Vary energy of jets by +/- 1 standard deviation,
 * in order to estimate resulting uncertainty on MET
 *
 * NOTE: energy scale uncertainties are taken from the Database
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "PhysicsTools/PatUtils/interface/SmearedJetProducerT.h"

#include <TMath.h>

#include <string>

template <typename T, typename Textractor>
class ShiftedJetProducerT : public edm::EDProducer
{
  typedef std::vector<T> JetCollection;

 public:

  explicit ShiftedJetProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      src_(cfg.getParameter<edm::InputTag>("src")),
      srcToken_(consumes<JetCollection>(src_)),
      jetCorrPayloadName_(""),
      jetCorrParameters_(0),
      jecUncertainty_(0),
      jecUncertaintyValue_(-1.)
  {
    if ( cfg.exists("jecUncertaintyValue") ) {
      jecUncertaintyValue_ = cfg.getParameter<double>("jecUncertaintyValue");
    } else {
      jetCorrUncertaintyTag_ = cfg.getParameter<std::string>("jetCorrUncertaintyTag");
      if ( cfg.exists("jetCorrInputFileName") ) {
	jetCorrInputFileName_ = cfg.getParameter<edm::FileInPath>("jetCorrInputFileName");
	if ( jetCorrInputFileName_.location() == edm::FileInPath::Unknown) throw cms::Exception("ShiftedJetProducerT")
	  << " Failed to find JEC parameter file = " << jetCorrInputFileName_ << " !!\n";
	std::cout << "Reading JEC parameters = " << jetCorrUncertaintyTag_
		  << " from file = " << jetCorrInputFileName_.fullPath() << "." << std::endl;
	jetCorrParameters_ = new JetCorrectorParameters(jetCorrInputFileName_.fullPath().data(), jetCorrUncertaintyTag_);
	jecUncertainty_ = new JetCorrectionUncertainty(*jetCorrParameters_);
      } else {
	std::cout << "Reading JEC parameters = " << jetCorrUncertaintyTag_
		  << " from DB/SQLlite file." << std::endl;
	jetCorrPayloadName_ = cfg.getParameter<std::string>("jetCorrPayloadName");
      }
    }

    addResidualJES_ = cfg.getParameter<bool>("addResidualJES");
    jetCorrLabelUpToL3_ = ( cfg.exists("jetCorrLabelUpToL3") ) ?
      cfg.getParameter<std::string>("jetCorrLabelUpToL3") : "";
    jetCorrLabelUpToL3Res_ = ( cfg.exists("jetCorrLabelUpToL3Res") ) ?
      cfg.getParameter<std::string>("jetCorrLabelUpToL3Res") : "";
    jetCorrEtaMax_ = ( cfg.exists("jetCorrEtaMax") ) ?
      cfg.getParameter<double>("jetCorrEtaMax") : 9.9;

    shiftBy_ = cfg.getParameter<double>("shiftBy");

    verbosity_ = ( cfg.exists("verbosity") ) ?
      cfg.getParameter<int>("verbosity") : 0;

    produces<JetCollection>();
  }
  ~ShiftedJetProducerT()
  {
    delete jetCorrParameters_;
    delete jecUncertainty_;
  }

 private:

  void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    if ( verbosity_ ) {
      std::cout << "<ShiftedJetProducerT::produce>:" << std::endl;
      std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
      std::cout << " src = " << src_.label() << std::endl;
    }

    edm::Handle<JetCollection> originalJets;
    evt.getByToken(srcToken_, originalJets);

    std::auto_ptr<JetCollection> shiftedJets(new JetCollection);

    if ( jetCorrPayloadName_ != "" ) {
      edm::ESHandle<JetCorrectorParametersCollection> jetCorrParameterSet;
      es.get<JetCorrectionsRecord>().get(jetCorrPayloadName_, jetCorrParameterSet);
      const JetCorrectorParameters& jetCorrParameters = (*jetCorrParameterSet)[jetCorrUncertaintyTag_];
      delete jecUncertainty_;
      jecUncertainty_ = new JetCorrectionUncertainty(jetCorrParameters);
    }

    for ( typename JetCollection::const_iterator originalJet = originalJets->begin();
	  originalJet != originalJets->end(); ++originalJet ) {
      reco::Candidate::LorentzVector originalJetP4 = originalJet->p4();
      if ( verbosity_ ) {
	std::cout << "originalJet: Pt = " << originalJetP4.pt() << ", eta = " << originalJetP4.eta() << ", phi = " << originalJetP4.phi() << std::endl;
      }

      double shift = 0.;
      if ( jecUncertaintyValue_ != -1. ) {
	shift = jecUncertaintyValue_;
      } else {
	jecUncertainty_->setJetEta(originalJetP4.eta());
	jecUncertainty_->setJetPt(originalJetP4.pt());

	shift = jecUncertainty_->getUncertainty(true);
      }
      if ( verbosity_ ) {
	std::cout << "shift = " << shift << std::endl;
      }

      if ( addResidualJES_ ) {
	const static SmearedJetProducer_namespace::RawJetExtractorT<T> rawJetExtractor;
	reco::Candidate::LorentzVector rawJetP4 = rawJetExtractor(*originalJet);
	if ( rawJetP4.E() > 1.e-1 ) {
	  reco::Candidate::LorentzVector corrJetP4upToL3 =
	    jetCorrExtractor_(*originalJet, jetCorrLabelUpToL3_, &evt, &es, jetCorrEtaMax_, &rawJetP4);
	  reco::Candidate::LorentzVector corrJetP4upToL3Res =
	    jetCorrExtractor_(*originalJet, jetCorrLabelUpToL3Res_, &evt, &es, jetCorrEtaMax_, &rawJetP4);
	  if ( corrJetP4upToL3.E() > 1.e-1 && corrJetP4upToL3Res.E() > 1.e-1 ) {
	    double residualJES = (corrJetP4upToL3Res.E()/corrJetP4upToL3.E()) - 1.;
	    shift = TMath::Sqrt(shift*shift + residualJES*residualJES);
	  }
	}
      }

      shift *= shiftBy_;
      if ( verbosity_ ) {
	std::cout << "shift*shiftBy = " << shift << std::endl;
      }

      T shiftedJet(*originalJet);
      shiftedJet.setP4((1. + shift)*originalJetP4);
      if ( verbosity_ ) {
	std::cout << "shiftedJet: Pt = " << shiftedJet.pt() << ", eta = " << shiftedJet.eta() << ", phi = " << shiftedJet.phi() << std::endl;
      }

      shiftedJets->push_back(shiftedJet);
    }

    evt.put(shiftedJets);
  }

  std::string moduleLabel_;

  edm::InputTag src_;
  edm::EDGetTokenT<JetCollection> srcToken_;

  edm::FileInPath jetCorrInputFileName_;
  std::string jetCorrPayloadName_;
  std::string jetCorrUncertaintyTag_;
  JetCorrectorParameters* jetCorrParameters_;
  JetCorrectionUncertainty* jecUncertainty_;

  bool addResidualJES_;
  std::string jetCorrLabelUpToL3_;    // L1+L2+L3 correction
  std::string jetCorrLabelUpToL3Res_; // L1+L2+L3+Residual correction
  double jetCorrEtaMax_; // do not use JEC factors for |eta| above this threshold (recommended default = 4.7),
                         // in order to work around problem with CMSSW_4_2_x JEC factors at high eta,
                         // reported in
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/jes/270.html
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1259/1.html
  Textractor jetCorrExtractor_;

  double jecUncertaintyValue_;

  double shiftBy_; // set to +1.0/-1.0 for up/down variation of energy scale

  int verbosity_; // flag to enabled/disable debug output
};

#endif




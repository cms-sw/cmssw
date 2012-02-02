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
 * \version $Revision: 1.1 $
 *
 * $Id: ShiftedJetProducerT.h,v 1.1 2011/10/14 11:18:24 veelken Exp $
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
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <string>

template <typename T>
class ShiftedJetProducerT : public edm::EDProducer  
{
  typedef std::vector<T> JetCollection;

 public:

  explicit ShiftedJetProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      src_(cfg.getParameter<edm::InputTag>("src")),
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
	if ( !jetCorrInputFileName_.isLocal()) throw cms::Exception("ShiftedJetProducerT") 
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

    shiftBy_ = cfg.getParameter<double>("shiftBy");

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
    edm::Handle<JetCollection> originalJets;
    evt.getByLabel(src_, originalJets);

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
      T shiftedJet(*originalJet);

      double shift = 0.;
      if ( jecUncertaintyValue_ != -1. ) {
	shift = jecUncertaintyValue_;
      } else {
	jecUncertainty_->setJetEta(originalJet->eta());
	jecUncertainty_->setJetPt(originalJet->pt());
	
	shift = jecUncertainty_->getUncertainty(true);
      }
      shift *= shiftBy_;

      reco::Candidate::LorentzVector originalJetP4 = originalJet->p4();
      shiftedJet.setP4((1. + shift)*originalJetP4);
    
      shiftedJets->push_back(shiftedJet);
    }
  
    evt.put(shiftedJets);
  }

  std::string moduleLabel_;

  edm::InputTag src_; 

  edm::FileInPath jetCorrInputFileName_;
  std::string jetCorrPayloadName_;
  std::string jetCorrUncertaintyTag_;
  JetCorrectorParameters* jetCorrParameters_;
  JetCorrectionUncertainty* jecUncertainty_;

  double jecUncertaintyValue_;

  double shiftBy_; // set to +1.0/-1.0 for up/down variation of energy scale
};

#endif

 


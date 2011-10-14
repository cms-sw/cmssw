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
 * $Id: ShiftedJetProducerT.h,v 1.1 2011/09/13 14:35:34 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
      jecUncertainty_(0),
      jecUncertaintyValue_(-1.)
  {
    if ( cfg.exists("jecUncertaintyValue") ) {
      jecUncertaintyValue_ = cfg.getParameter<double>("jecUncertaintyValue");
    } else {
      jetCorrPayloadName_ = cfg.getParameter<std::string>("jetCorrPayloadName");
      jetCorrUncertaintyTag_ = cfg.getParameter<std::string>("jetCorrUncertaintyTag");
    }

    shiftBy_ = cfg.getParameter<double>("shiftBy");

    produces<JetCollection>();
  }
  ~ShiftedJetProducerT()
  {
    delete jecUncertainty_;
  }
    
 private:

  void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    edm::Handle<JetCollection> originalJets;
    evt.getByLabel(src_, originalJets);

    std::auto_ptr<JetCollection> shiftedJets(new JetCollection);
    
    if ( jecUncertaintyValue_ == -1. ) {
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

  std::string jetCorrPayloadName_;
  std::string jetCorrUncertaintyTag_;
  JetCorrectionUncertainty* jecUncertainty_;

  double jecUncertaintyValue_;

  double shiftBy_; // set to +1.0/-1.0 for up/down variation of energy scale
};

#endif

 


#ifndef RecoTauTag_RecoTau_PFRecoTauDiscriminationAgainstElectron_H_
#define RecoTauTag_RecoTau_PFRecoTauDiscriminationAgainstElectron_H_

/* class PFRecoTauDiscriminationAgainstElectron
 * created : May 02 2008,
 * revised : ,
 * Authorss : Chi Nhan Nguyen (Texas A&M)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

using namespace std; 
using namespace edm;
using namespace edm::eventsetup; 
using namespace reco;

class PFRecoTauDiscriminationAgainstElectron : public EDProducer {
 public:
  explicit PFRecoTauDiscriminationAgainstElectron(const ParameterSet& iConfig){   
    PFTauProducer_        = iConfig.getParameter<InputTag>("PFTauProducer");
    emFraction_maxValue_  = iConfig.getParameter<double>("EmFraction_maxValue");  
    applyCut_emFraction_  = iConfig.getParameter<bool>("ApplyCut_EmFraction");
    hcalTotOverPLead_minValue_  = iConfig.getParameter<double>("HcalTotOverPLead_minValue");  
    applyCut_hcalTotOverPLead_  = iConfig.getParameter<bool>("ApplyCut_HcalTotOverPLead");
    hcalMaxOverPLead_minValue_  = iConfig.getParameter<double>("HcalMaxOverPLead_minValue");  
    applyCut_hcalMaxOverPLead_  = iConfig.getParameter<bool>("ApplyCut_HcalMaxOverPLead");
    hcal3x3OverPLead_minValue_  = iConfig.getParameter<double>("Hcal3x3OverPLead_minValue");  

    applyCut_hcal3x3OverPLead_  = iConfig.getParameter<bool>("ApplyCut_Hcal3x3OverPLead");
    EOverPLead_minValue_  = iConfig.getParameter<double>("EOverPLead_minValue");  
    EOverPLead_maxValue_  = iConfig.getParameter<double>("EOverPLead_maxValue");  
    applyCut_EOverPLead_  = iConfig.getParameter<bool>("ApplyCut_EOverPLead");
    bremsRecoveryEOverPLead_minValue_  = iConfig.getParameter<double>("BremsRecoveryEOverPLead_minValue");  
    bremsRecoveryEOverPLead_maxValue_  = iConfig.getParameter<double>("BremsRecoveryEOverPLead_maxValue");  

    applyCut_bremsRecoveryEOverPLead_  = iConfig.getParameter<bool>("ApplyCut_BremsRecoveryEOverPLead");

    applyCut_electronPreID_  = iConfig.getParameter<bool>("ApplyCut_ElectronPreID");

    applyCut_electronPreID_2D_  = iConfig.getParameter<bool>("ApplyCut_ElectronPreID_2D");

    elecPreID0_EOverPLead_maxValue_  = iConfig.getParameter<double>("ElecPreID0_EOverPLead_maxValue");
    elecPreID0_HOverPLead_minValue_  = iConfig.getParameter<double>("ElecPreID0_HOverPLead_minValue");
    elecPreID1_EOverPLead_maxValue_  = iConfig.getParameter<double>("ElecPreID1_EOverPLead_maxValue");
    elecPreID1_HOverPLead_minValue_  = iConfig.getParameter<double>("ElecPreID1_HOverPLead_minValue");


    applyCut_PFElectronMVA_  = iConfig.getParameter<bool>("ApplyCut_PFElectronMVA");
    pfelectronMVA_maxValue_  = iConfig.getParameter<double>("PFElectronMVA_maxValue"); 

    applyCut_ecalCrack_  = iConfig.getParameter<bool>("ApplyCut_EcalCrackCut");
    
    produces<PFTauDiscriminator>();
  }
  ~PFRecoTauDiscriminationAgainstElectron(){} 
  virtual void produce(Event&, const EventSetup&);
 private:
  bool isInEcalCrack(double) const; 
  InputTag PFTauProducer_;
  bool applyCut_emFraction_;
  double emFraction_maxValue_;   
  bool applyCut_hcalTotOverPLead_;
  double hcalTotOverPLead_minValue_;   
  bool applyCut_hcalMaxOverPLead_;
  double hcalMaxOverPLead_minValue_;   
  bool applyCut_hcal3x3OverPLead_;
  double hcal3x3OverPLead_minValue_;   

  bool applyCut_EOverPLead_;
  double EOverPLead_minValue_;   
  double EOverPLead_maxValue_;   
  bool applyCut_bremsRecoveryEOverPLead_;
  double bremsRecoveryEOverPLead_minValue_;   
  double bremsRecoveryEOverPLead_maxValue_;   

  bool applyCut_electronPreID_;

  bool applyCut_electronPreID_2D_;
  double elecPreID0_EOverPLead_maxValue_;
  double elecPreID0_HOverPLead_minValue_;
  double elecPreID1_EOverPLead_maxValue_;
  double elecPreID1_HOverPLead_minValue_;

  bool applyCut_PFElectronMVA_;
  double pfelectronMVA_maxValue_; 

  bool applyCut_ecalCrack_;

};
#endif

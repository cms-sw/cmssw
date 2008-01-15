#include "HLTriggerOffline/Egamma/interface/EmDQM.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"



#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TFile.h"
#include "TDirectory.h"
#include "TH1F.h"
#include <iostream>
#include <string>


/// Constructor
EmDQM::EmDQM(const edm::ParameterSet& pset)  
{
  theHLTCollectionLabels = pset.getParameter<std::vector<edm::InputTag> >("HLTCollectionLabels");
  theHLTOutputTypes = pset.getParameter<std::vector<int> >("theHLTOutputTypes");
  reqNum = pset.getParameter<unsigned int>("reqNum");
  pdgGen =  pset.getParameter<int>("pdgGen");
  genEtaAcc = pset.getParameter<double>("genEtaAcc");
  genEtAcc = pset.getParameter<double>("genEtAcc");
  thePtMin = pset.getUntrackedParameter<double>("PtMin",0.);
  thePtMax = pset.getUntrackedParameter<double>("PtMax",1000.);
  theNbins = pset.getUntrackedParameter<unsigned int>("Nbins",40);
  
  assert (theHLTCollectionLabels.size()==theHLTOutputTypes.size());
}


void EmDQM::beginJob(const edm::EventSetup&){
  edm::Service<TFileService> fs;
  std::string histoname="total eff";

  total = fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theHLTCollectionLabels.size()+2,0,theHLTCollectionLabels.size()+2);
  total->GetXaxis()->SetBinLabel(theHLTCollectionLabels.size()+1,"Total");
  total->GetXaxis()->SetBinLabel(theHLTCollectionLabels.size()+2,"Gen");
  for (unsigned int u=0; u<theHLTCollectionLabels.size(); u++){total->GetXaxis()->SetBinLabel(u+1,theHLTCollectionLabels[u].label().c_str());}

  TH1F* tmphisto;
  
  histoname = "gen et";
  etgen =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
  histoname = "gen eta";
  etagen = fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
  
  for(unsigned int i = 0; i< theHLTCollectionLabels.size() ; i++){
    histoname = theHLTCollectionLabels[i].label()+"et";
    tmphisto =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
    ethist.push_back(tmphisto);
    
    histoname = theHLTCollectionLabels[i].label()+"eta";
    tmphisto =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
    etahist.push_back(tmphisto);          
  } 

}


/// Destructor
EmDQM::~EmDQM(){
}

void EmDQM::analyze(const edm::Event & event , const edm::EventSetup& setup){

  // total event number
  total->Fill(theHLTCollectionLabels.size()+0.5);

  // fill generator info
  edm::Handle<edm::HepMCProduct> genEvt;
  event.getByLabel("source", genEvt);
  
  const HepMC::GenEvent * myGenEvent = genEvt->GetEvent();
  unsigned int ncand = 0;
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p ) {
    if (  !( abs((*p)->pdg_id())==pdgGen  && (*p)->status()==1 )   )  continue;
    float eta   =(*p)->momentum().eta();
    float e     =(*p)->momentum().e();
    float theta =2*atan(exp(-eta));
    float Et    =e*sin(theta);
    if(fabs(eta)<genEtaAcc  &&  Et > genEtAcc) {
      ncand++;
      etgen->Fill(Et);
      etagen->Fill(eta);
    }
  }//end of loop over MC particles
  if (ncand >= reqNum) total->Fill(theHLTCollectionLabels.size()+1.5);
	  

  // fill L1 and HLT info
  // get objects possed by each filter
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  event.getByLabel("triggerSummaryRAW",triggerObj); 
  if(!triggerObj.isValid()) throw(cms::Exception("Release Validation Error")<< "RAW-type HLT results not found" );

  for(unsigned int n=0; n < theHLTCollectionLabels.size() ; n++) { //loop over filter modules
    switch(theHLTOutputTypes[n]){
    case 82: // non-iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,theHLTOutputTypes,n);break;
    case 83: // iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,theHLTOutputTypes,n);break;
    case 92: //electron 
      fillHistos<reco::ElectronCollection>(triggerObj,theHLTOutputTypes,n);break;
    case 100: // TriggerCluster
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,theHLTOutputTypes,n);break;
    default: throw(cms::Exception("Release Validation Error")<< "HLT output type not implemented: theHLTOutputTypes[n]" );
    }
  }
}


template <class T> void EmDQM::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj, std::vector<int>&  theHLTOutputTypes,int n){
  
  std::vector<edm::Ref<T> > recoecalcands;
  if (!( triggerObj->filterIndex(theHLTCollectionLabels[n].label())>=triggerObj->size() )){ // only process if availabel
  
    // retrieve saved filter objects
    triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n].label()),theHLTOutputTypes[n],recoecalcands);
    //Danger: special case, L1 non-isolated
    // needs to be merged with L1 iso
    if(theHLTOutputTypes[n]==82){
      std::vector<edm::Ref<T> > isocands;
      triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n].label()),83,isocands);
      if(isocands.size()>0)
	for(unsigned int i=0; i < isocands.size(); i++)
	  recoecalcands.push_back(isocands[i]);
    }

    //fill filter objects into histos
    if (recoecalcands.size()!=0){
      if(recoecalcands.size() >= reqNum ) 
	total->Fill(n+0.5);
      for (unsigned int i=0; i<recoecalcands.size(); i++) {
	ethist[n]->Fill(recoecalcands[i]->et() );
	etahist[n]->Fill(recoecalcands[i]->eta() );
      }
    }
  }
}


void EmDQM::endJob(){
  //  total->Scale(1./total->GetBinContent(1));
  //for(unsigned int n= theHLTCollectionLabels.size()-1 ; n>0;n--){
  //  ethist[n]->Divide(ethist[n-1]);
  //  etahist[n]->Divide(etahist[n-1]);
  //}
}

DEFINE_FWK_MODULE(EmDQM);

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HLTEvF/interface/HLTMonElectron.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Common/interface/RefToBase.h"

using namespace edm;

HLTMonElectron::HLTMonElectron(const edm::ParameterSet& iConfig)
{

   // verbosity switch
   verbose_ = iConfig.getUntrackedParameter < bool > ("verbose", false);
 
   if (verbose_)
     std::cout << "HLTMonElectron: constructor...." << std::endl;
 
   logFile_.open("HLTMonElectron.log");
 
  theHLTCollectionLabels = iConfig.getParameter<std::vector<edm::InputTag> >("HLTCollectionLabels");
  theHLTOutputTypes = iConfig.getParameter<std::vector<int> >("theHLTOutputTypes");
  assert (theHLTCollectionLabels.size()==theHLTOutputTypes.size());

   dbe = NULL;
   if (iConfig.getUntrackedParameter < bool > ("DaqMonitorBEInterface", false)) {
     dbe = Service < DaqMonitorBEInterface > ().operator->();
     dbe->setVerbose(0);
   }
 
   monitorDaemon_ = false;
   if (iConfig.getUntrackedParameter < bool > ("MonitorDaemon", false)) {
     Service < MonitorDaemon > daemon;
     daemon.operator->();
     monitorDaemon_ = true;
   }
 
   outputFile_ =
       iConfig.getUntrackedParameter < std::string > ("outputFile", "");
   if (outputFile_.size() != 0) {
     std::cout << "L1T Monitoring histograms will be saved to " 
	       << outputFile_ << std::endl;
   }
   else {
     outputFile_ = "L1TDQM.root";
   }
 
   bool disable =
       iConfig.getUntrackedParameter < bool > ("disableROOToutput", false);
   if (disable) {
     outputFile_ = "";
   }
 
 
   if (dbe != NULL) {
     dbe->setCurrentFolder("HLT/HLTMonElectron");
   }
 

}


HLTMonElectron::~HLTMonElectron()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HLTMonElectron::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace trigger;
   nev_++;
   if (verbose_) {
     std::cout << "HLTMonElectron: analyze...." << std::endl;
   }
  
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  iEvent.getByLabel("triggerSummaryRAW",triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogWarning("HLTMonElectron") << "RAW-type HLT results not found, skipping event";
    return;
  }
       
  for(unsigned int n=0; n < theHLTCollectionLabels.size() ; n++) { //loop over filter modules
    switch(theHLTOutputTypes[n]){
    case 82: // non-iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,theHLTOutputTypes,n);break;
    case 83: // iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,theHLTOutputTypes,n);break;
    case 91: //photon 
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,theHLTOutputTypes,n);break;
    case 92: //electron 
      fillHistos<reco::ElectronCollection>(triggerObj,theHLTOutputTypes,n);break;
    case 100: // TriggerCluster
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,theHLTOutputTypes,n);break;
    default: throw(cms::Exception("Release Validation Error")<< "HLT output type not implemented: theHLTOutputTypes[n]" );
    }
  }
}

template <class T> void HLTMonElectron::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj, std::vector<int>&  theHLTOutputTypes,int n){
  
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
      for (unsigned int i=0; i<recoecalcands.size(); i++) {
	//unmatched
	ethist[n]->Fill(recoecalcands[i]->et() );
	etahist[n]->Fill(recoecalcands[i]->eta() );
	}
      }
    }
}

// ------------ method called once each job just before starting event loop  ------------
void 
HLTMonElectron::beginJob(const edm::EventSetup&)
{
   nev_ = 0;
   DaqMonitorBEInterface *dbe = 0;
   dbe = Service < DaqMonitorBEInterface > ().operator->();
 
   if (dbe) {
     dbe->setCurrentFolder("HLT/HLTMonElectron");
     dbe->rmdir("HLT/HLTMonElectron");
   }
 
 
   if (dbe) {
     dbe->setCurrentFolder("HLT/HLTMonElectron");
     //h1_ = 
       //dbe->book1D("ElectronCandEt", "ElectronCandEt", 100, 0,100);
  std::string histoname;
  MonitorElement* tmphisto;
  for(unsigned int i = 0; i< theHLTCollectionLabels.size() ; i++){
    histoname = theHLTCollectionLabels[i].label()+"et";
    tmphisto =  dbe->book1D(histoname.c_str(),histoname.c_str(),100, 0,100);
    ethist.push_back(tmphisto);
    
    histoname = theHLTCollectionLabels[i].label()+"eta";
    tmphisto =  dbe->book1D(histoname.c_str(),histoname.c_str(),50,-2.7,2.7);
    etahist.push_back(tmphisto);          

  } 


   }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTMonElectron::endJob() {
   if (verbose_)
//     std::cout << "HLTMonElectron: end job...." << std::endl;
   LogInfo("HLTMonElectron") << "analyzed " << nev_ << " events";
 
   if (outputFile_.size() != 0 && dbe)
     dbe->save(outputFile_);
 
   return;
}

//DEFINE_FWK_MODULE(HLTMonElectron);

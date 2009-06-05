#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HLTEvF/interface/HLTMonPhotonSource.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"


#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;


HLTMonPhotonSource::HLTMonPhotonSource(const edm::ParameterSet& iConfig)
{
  
  LogDebug("HLTMonPhotonSource") << "constructor...." ;
  
  logFile_.open("HLTMonPhotonSource.log");
  
  dbe = NULL;
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }

  //std::cout<<"supposed to be message about output file \n\n\n\n";
  
  outputFile_ =
    iConfig.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    edm::LogInfo("HLTMonPhotonSource") << "Photon Trigger Monitoring histograms will be saved to " << 
outputFile_ ;
    //    std::cout<< "Photon Trigger Monitoring histograms will be saved to " << outputFile_ ;
  }
  else {

    //std::cout<<"output filename not read in, setting as: PhotonDQM.root\n\n";
    outputFile_ = "PhotonDQM.root";
  }
  
  bool disable =
    iConfig.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
std::cout<<"output disabled?\n\n\n";
    outputFile_ = "";
  }
  
  dirname_="HLT/HLTMonPhoton/"+iConfig.getParameter<std::string>("@module_label");
  
  if (dbe != NULL) {
    dbe->setCurrentFolder(dirname_);
  }
  
//std::cout<<"finished setting up directories\n\n";
  
  //plotting paramters
  thePtMin = iConfig.getUntrackedParameter<double>("PtMin",0.);
  thePtMax = iConfig.getUntrackedParameter<double>("PtMax",1000.);
  theNbins = iConfig.getUntrackedParameter<unsigned int>("Nbins",40);
  
  //info for each filter-step
  reqNum = iConfig.getParameter<unsigned int>("reqNum");
  std::vector<edm::ParameterSet> filters = iConfig.getParameter<std::vector<edm::ParameterSet> >("filters");
  
  for(std::vector<edm::ParameterSet>::iterator filterconf = filters.begin() ; filterconf != filters.end() ; filterconf++){
    theHLTCollectionLabels.push_back(filterconf->getParameter<edm::InputTag>("HLTCollectionLabels"));
    theHLTOutputTypes.push_back(filterconf->getParameter<unsigned int>("theHLTOutputTypes"));
    std::vector<double> bounds = filterconf->getParameter<std::vector<double> >("PlotBounds");
    assert(bounds.size() == 2);
    plotBounds.push_back(std::pair<double,double>(bounds[0],bounds[1]));
    isoNames.push_back(filterconf->getParameter<std::vector<edm::InputTag> >("IsoCollections"));
    assert(isoNames.back().size()>0);
    if (isoNames.back().at(0).label()=="none")
      plotiso.push_back(false);
    else{
      plotiso.push_back(true);
    }
  }

//std::cout<<"finished setup\n\n";
  
}


HLTMonPhotonSource::~HLTMonPhotonSource()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HLTMonPhotonSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace trigger;
   nev_++;
//   LogDebug("HLTMonPhotonSource")<< "HLTMonPhotonSource: analyze...." ;
   edm::LogInfo("HLTMonPhotonSource")<< "HLTMonPhotonSource: analyze...." ;
   

  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  iEvent.getByLabel("hltTriggerSummaryRAW",triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogWarning("HLTMonPhotonSource") << "RAW-type HLT results not found, skipping event";
    return;
  } 

  maxEt = 0.;
  eta = 999.;
  phi = 999.;

  //testing:
  sigmaetaeta=0.;

  for(unsigned int n=0; n < theHLTCollectionLabels.size() ; n++) { //loop over filter modules
//std::cout<<"\n\n\n HLT codes:\n\n";
//std::cout<<theHLTOutputTypes[n]<<"\t";

    switch(theHLTOutputTypes[n]){
    case 82: // non-iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,iEvent,n);break;
    case 83: // iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,iEvent,n);break;
    case 91: //photon 
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,iEvent,n);break;
    case 92: //electron 
      fillHistos<reco::ElectronCollection>(triggerObj,iEvent,n);break;
    case 100: // TriggerCluster
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,iEvent,n);break;
    default: throw(cms::Exception("Release Validation Error")<< "HLT output type not implemented: theHLTOutputTypes[n]" );
    }
  }
}

template <class T> void HLTMonPhotonSource::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj,const edm::Event& iEvent  ,unsigned int n){
  
  std::vector<edm::Ref<T> > recoecalcands;

//std::cout<<"next step should be analyze\n\n";

//std::cout<<"filterIndex result: "<<triggerObj->filterIndex(theHLTCollectionLabels[n])<<std::endl;

  if (!( triggerObj->filterIndex(theHLTCollectionLabels[n])>=triggerObj->size() )){ // only process if available

    //  std::cout<<theHLTCollectionLabels[n]<<" "<<n<<std::endl;

    //std::cout<<"analyzing recoecalcands\n\n";
  
    // retrieve saved filter objects
    triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),theHLTOutputTypes[n],recoecalcands);
    //Danger: special case, L1 non-isolated
    // needs to be merged with L1 iso
    if(theHLTOutputTypes[n]==82){
      std::vector<edm::Ref<T> > isocands;
      triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),83,isocands);
      if(isocands.size()>0)
	for(unsigned int i=0; i < isocands.size(); i++)
	  recoecalcands.push_back(isocands[i]);
    }



    //get the parameters of the candidate with highest et
    if(n == 0 && recoecalcands.size()!=0){
      for (unsigned int i=0; i<recoecalcands.size(); i++) {
	if(recoecalcands[i]->et() > maxEt){
	  maxEt = recoecalcands[i]->et();
	  eta = recoecalcands[i]->eta();
	  phi = recoecalcands[i]->phi();

	  if ( theHLTOutputTypes[n]==100){
	    //	  std::cout<<recoecalcands[i]->superCluster()->seed()->sigmaEtaEta()<<std::endl;
	  }
	}
      }
    }


    //fill filter objects into histos
    if (recoecalcands.size()!=0){
      if(recoecalcands.size() >= reqNum){ 
	eventCounter->Fill(n+0.5);
	ethist[n]->Fill(maxEt);
	phihist[n]->Fill(phi);
	etahist[n]->Fill(eta);

	//std::cout<<"Et, eta, phi: "<<maxEt<<" "<<eta<<" "<<phi<<"\n\n\n";

	for (unsigned int i=0; i<recoecalcands.size(); i++) {
	  //plot isolation variables before the isolation filter has been applied
	  if(n+1 < theHLTCollectionLabels.size()){ // can't plot beyond last
	    if(plotiso[n+1]){
	      for(unsigned int j =  0 ; j < isoNames[n+1].size() ;j++  ){
		edm::Handle<edm::AssociationMap<edm::OneToValue< T , float > > > depMap; 
		iEvent.getByLabel(isoNames[n+1].at(j).label(),depMap);
		typename edm::AssociationMap<edm::OneToValue< T , float > >::const_iterator mapi = depMap->find(recoecalcands[i]);
		if(mapi!=depMap->end()){  // found candidate in isolation map! 
		  histiso[n+1]->Fill(mapi->val);
		  etahistiso[n+1]->Fill(recoecalcands[i]->eta(),mapi->val);
		  phihistiso[n+1]->Fill(recoecalcands[i]->phi(),mapi->val);
		  ethistiso[n+1]->Fill(recoecalcands[i]->et(),mapi->val);
		  break; // to avoid multiple filling we only look until we found the candidate once.
		}
	      }
	    }	  	  
	  }
	}
      }
    }
  }
}
// ------------ method called once each job just before starting event loop  ------------
void 
HLTMonPhotonSource::beginJob(const edm::EventSetup&)
{
  nev_ = 0;
  DQMStore *dbe = 0;
  dbe = Service < DQMStore > ().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    dbe->rmdir(dirname_);
  }
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);

    std::string histoname;
    MonitorElement* tmphisto;

    eventCounter = dbe->book1D("Evts Passing Filters","Evts Passing Filters",5,0,5);
    eventCounter->setBinLabel(1,"SinglePhotonEtFilter");
    eventCounter->setBinLabel(2,"SinglePhotonEcalIsolFilter");
    eventCounter->setBinLabel(3,"SinglePhotonHcalIsolFilter");
    eventCounter->setBinLabel(4,"SinglePhotonTrackIsolFilter");
    /*
    eventCounter->setBinLabel(5,"HLTIsoSinglePhotonEt10EtFilter");
    eventCounter->setBinLabel(6,"HLTIsoSinglePhotonEt10EcalIsolFilter");
    eventCounter->setBinLabel(7,"HLTIsoSinglePhotonEt10HcalIsolFilter");
    eventCounter->setBinLabel(8,"HLTIsoSinglePhotonEt10TrackIsolFilter");
    eventCounter->setBinLabel(9,"HLTIsoSinglePhotonEt15EtFilter");
    eventCounter->setBinLabel(10,"HLTIsoSinglePhotonEt15EcalIsolFilter");
    eventCounter->setBinLabel(11,"HLTIsoSinglePhotonEt15HcalIsolFilter");
    eventCounter->setBinLabel(12,"HLTIsoSinglePhotonEt15TrackIsolFilter");
    eventCounter->setBinLabel(13,"HLTIsoSinglePhotonEt20EtFilter");
    eventCounter->setBinLabel(14,"HLTIsoSinglePhotonEt20EcalIsolFilter");
    eventCounter->setBinLabel(15,"HLTIsoSinglePhotonEt20HcalIsolFilter");
    eventCounter->setBinLabel(16,"HLTIsoSinglePhotonEt20TrackIsolFilter");
    eventCounter->setBinLabel(17,"HLTIsoSinglePhotonEt25EtFilter");
    eventCounter->setBinLabel(18,"HLTIsoSinglePhotonEt25EcalIsolFilter");
    eventCounter->setBinLabel(19,"HLTIsoSinglePhotonEt25HcalIsolFilter");
    eventCounter->setBinLabel(20,"HLTIsoSinglePhotonEt25TrackIsolFilter");
    eventCounter->setBinLabel(21,"HLTNonIsoSinglePhotonEtFilter");
    eventCounter->setBinLabel(22,"HLTNonIsoSinglePhotonEcalIsolFilter");
    eventCounter->setBinLabel(23,"HLTNonIsoSinglePhotonHcalIsolFilter");
    eventCounter->setBinLabel(24,"HLTNonIsoSinglePhotonTrackIsolFilter");
    eventCounter->setBinLabel(25,"HLTNonIsoSinglePhotonEt15EtFilter");
    eventCounter->setBinLabel(26,"HLTNonIsoSinglePhotonEt15EcalIsolFilter");
    eventCounter->setBinLabel(27,"HLTNonIsoSinglePhotonEt15HcalIsolFilter");
    eventCounter->setBinLabel(28,"HLTNonIsoSinglePhotonEt15TrackIsolFilter");
    eventCounter->setBinLabel(29,"HLTNonIsoSinglePhotonEt25EtFilter");
    eventCounter->setBinLabel(30,"HLTNonIsoSinglePhotonEt25EcalIsolFilter");
    eventCounter->setBinLabel(31,"HLTNonIsoSinglePhotonEt25HcalIsolFilter");
    eventCounter->setBinLabel(32,"HLTNonIsoSinglePhotonEt25TrackIsolFilter");
    */


    for(unsigned int i = 0; i< theHLTCollectionLabels.size() ; i++){
      histoname = theHLTCollectionLabels[i].label()+" Et Dist";
      tmphisto =  dbe->book1D(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
      ethist.push_back(tmphisto);
      
      histoname = theHLTCollectionLabels[i].label()+" Eta Dist";
      tmphisto =  dbe->book1D(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
      etahist.push_back(tmphisto);          
 
      histoname = theHLTCollectionLabels[i].label()+" Phi Dist";
      tmphisto =  dbe->book1D(histoname.c_str(),histoname.c_str(),theNbins,-3.2,3.2);
      phihist.push_back(tmphisto);  

      if(plotiso[i]){
	histoname = theHLTCollectionLabels[i].label()+" Isolation Variable";
	tmphisto = dbe->book1D(histoname.c_str(),histoname.c_str(),theNbins,plotBounds[i].first,plotBounds[i].second);
      }
      else{
	tmphisto = NULL;
      }
      histiso.push_back(tmphisto);
 
      if(plotiso[i]){
	histoname = theHLTCollectionLabels[i].label()+" Isolation vs Eta";
	tmphisto = dbe->book2D(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7,theNbins,plotBounds[i].first,plotBounds[i].second);
      }
      else{
	tmphisto = NULL;
      }
      etahistiso.push_back(tmphisto);
      
      if(plotiso[i]){
	histoname = theHLTCollectionLabels[i].label()+" Isolation vs Phi";
	tmphisto = dbe->book2D(histoname.c_str(),histoname.c_str(),theNbins,-3.2,3.2,theNbins,plotBounds[i].first,plotBounds[i].second);
      }
      else{
	tmphisto = NULL;
      }
      phihistiso.push_back(tmphisto);
      
      if(plotiso[i]){
	histoname = theHLTCollectionLabels[i].label()+" Isolation vs Et";
	tmphisto = dbe->book2D(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax,theNbins,plotBounds[i].first,plotBounds[i].second);
      }
      else{
	tmphisto = NULL;
      }
      ethistiso.push_back(tmphisto);
    } 
  } // end "if(dbe)"
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTMonPhotonSource::endJob() {

  
//     std::cout << "HLTMonPhotonSource: end job...." << std::endl;
   LogInfo("HLTMonPhotonSource") << "analyzed " << nev_ << " events";
 
   if (outputFile_.size() != 0 && dbe)
     dbe->save(outputFile_);

   //std::cout<<"outputfile size: " <<outputFile_.size();

//don't do this, you'll break root

//std::cout<<"saving file\n\n";
//dbe->save(outputFile_);

   return;
}

DEFINE_FWK_MODULE(HLTMonPhotonSource);

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HLTEvF/interface/HLTMon.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//copied mostly from HLTMuonDQMSource.cc
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"


using namespace edm;

HLTMon::HLTMon(const edm::ParameterSet& iConfig)
{
  
  LogDebug("HLTMon") << "constructor...." ;
  
  logFile_.open("HLTMon.log");
  
  dbe = NULL;
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }
  
  outputFile_ =
    iConfig.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    LogInfo("HLTMon") << "L1T Monitoring histograms will be saved to " 
			      << outputFile_ ;
  }
  else {
    outputFile_ = "L1TDQM.root";
  }
  
  bool disable =
    iConfig.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }
  
  dirname_="HLT/HLTMon"+iConfig.getParameter<std::string>("@module_label");
  
  if (dbe != NULL) {
    dbe->setCurrentFolder(dirname_);
  }
  
  
  //plotting paramters
  thePtMin = iConfig.getUntrackedParameter<double>("PtMin",0.);
  thePtMax = iConfig.getUntrackedParameter<double>("PtMax",300.);
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
  
}


HLTMon::~HLTMon()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HLTMon::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace trigger;
   nev_++;
   LogDebug("HLTMon")<< "HLTMon: analyze...." ;

  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  iEvent.getByLabel("hltTriggerSummaryRAW",triggerObj);   //Gets the data product which holds
  if(!triggerObj.isValid()) {                             //all of the information. 
    edm::LogWarning("HLTMon") << "RAW-type HLT results not found, skipping event";
    return;
  }

  // total event number
  total->Fill(theHLTCollectionLabels.size()+0.5);

  //Each individual "Collection Label" has a numbered category that can be found on doxygen
  for(unsigned int n=0; n < theHLTCollectionLabels.size() ; n++) { //loop over filter modules
    switch(theHLTOutputTypes[n]){
    case 81: //L1 Muons
      fillHistos<l1extra::L1MuonParticleCollection>(triggerObj,iEvent,n);break;
    case 82: // non-iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,iEvent,n);break;
    case 83: // iso L1http://slashdot.org/
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,iEvent,n);break;
    case 91: //photon 
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,iEvent,n);break;
    case 92: //electron 
      fillHistos<reco::ElectronCollection>(triggerObj,iEvent,n);break;
    case 93: //Muon 
      fillHistos<reco::RecoChargedCandidateCollection>(triggerObj,iEvent,n);break;
    case 100: // TriggerCluster
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,iEvent,n);break;
    default: throw(cms::Exception("Release Validation Error")<< "HLT output type not implemented: theHLTOutputTypes[n]" );
    }
  }
}

template <class T> void HLTMon::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj,const edm::Event& iEvent  ,unsigned int n){
  

  std::vector<edm::Ref<T> > particlecands;
  // To keep track of what particlecands have passed a filter in TriggerEventWithRefs
  // it adds the name to a list of filter names. To check whether a filter got passed, 
  // one just looks at its index  in the list...if it is not there it (for some reason) 
  // returns the size of the list.
  if (!( triggerObj->filterIndex(theHLTCollectionLabels[n])>=triggerObj->size() )){ // only process if availabel  
    // retrieve saved filter objects
    triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),theHLTOutputTypes[n],particlecands);
    //Danger: special case, L1 non-isolated
    // needs to be merged with L1 iso
    if(theHLTOutputTypes[n]==82){
      std::vector<edm::Ref<T> > isocands;
      triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),83,isocands);
      if(isocands.size()>0)
	for(unsigned int i=0; i < isocands.size(); i++)
	  particlecands.push_back(isocands[i]);
    }

    //fill filter objects into histos
    if (particlecands.size()!=0){
      if(particlecands.size() >= reqNum ) 
	total->Fill(n+0.5);
      for (unsigned int i=0; i<particlecands.size() && particlecands[i].isAvailable(); i++) {
	//unmatched
	ethist[n]->Fill(particlecands[i]->et() );
	etahist[n]->Fill(particlecands[i]->eta() );
	phihist[n]->Fill(particlecands[i]->phi() );
	eta_phihist[n]->Fill(particlecands[i]->eta(), particlecands[i]->phi() );

	//plot isolation variables (show not yet cut  iso, i.e. associated to next filter)
	if(n+1 < theHLTCollectionLabels.size()){ // can't plot beyond last
	  if(plotiso[n+1]){
	    for(unsigned int j =  0 ; j < isoNames[n+1].size() ;j++  ){
	      edm::Handle<edm::AssociationMap<edm::OneToValue< T , float > > > depMap; 
              try{
	           iEvent.getByLabel(isoNames[n+1].at(j).label(),depMap);
                   typename edm::AssociationMap<edm::OneToValue< T , float > >::const_iterator mapi = depMap->find(particlecands[i]);
	           if(mapi!=depMap->end()){  // found candidate in isolation map! 
		     etahistiso[n+1]->Fill(particlecands[i]->eta(),mapi->val);
		     ethistiso[n+1]->Fill(particlecands[i]->et(),mapi->val);
		     phihistiso[n+1]->Fill(particlecands[i]->phi(),mapi->val);
		     break; // to avoid multiple filling we only look until we found the candidate once.
	           }
                 }
               catch(...){
               edm::LogWarning("HLTMon") << "IsoName collection not found";  
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
HLTMon::beginJob(const edm::EventSetup&)
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

    std::string histoname="total eff";
    MonitorElement* tmphisto;

     
    total = dbe->book1D(histoname.c_str(),histoname.c_str(),theHLTCollectionLabels.size()+1,0,theHLTCollectionLabels.size()+1);
    total->setBinLabel(theHLTCollectionLabels.size()+1,"Total",1);
    for (unsigned int u=0; u<theHLTCollectionLabels.size(); u++){total->setBinLabel(u+1,theHLTCollectionLabels[u].label().c_str());}
  
    for(unsigned int i = 0; i< theHLTCollectionLabels.size() ; i++){
      histoname = theHLTCollectionLabels[i].label()+"et";
	//ability to customize histograms
/*
	if(theHLTCollectionLabels[i].label() == "hltL1seedRelaxedSingleEt8")
	{
		thePtMinTemp = thePtMin;
		thePtMaxTemp = thePtMax;
	}
	else if(theHLTCollectionLabels[i].label() == "hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional")
	{
		thePtMinTemp = thePtMin;
		thePtMaxTemp = thePtMax;
	}
	else if(theHLTCollectionLabels[i].label() == "hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter")
	{
		thePtMinTemp = thePtMin;
		thePtMaxTemp = thePtMax;
	}
	else if(theHLTCollectionLabels[i].label() == "hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter")
	{
		thePtMinTemp = thePtMin;
		thePtMaxTemp = thePtMax;
	}
	else if(theHLTCollectionLabels[i].label() == "hltZMML2Filtered")
	{
		thePtMinTemp = thePtMin;
		thePtMaxTemp = thePtMax;
	}
	else if(theHLTCollectionLabels[i].label() == "hltEMuL1MuonFilter")
	{
		thePtMinTemp = thePtMin;
		thePtMaxTemp = thePtMax;
	}
      else
	{ */
	thePtMaxTemp = thePtMax;
	thePtMinTemp = thePtMin;
	//}
        // Formatting of various plots.
	histoTitle = theHLTCollectionLabels[i].label() + " Et";
      tmphisto =  dbe->book1D(histoname.c_str(),histoTitle.c_str(),theNbins,thePtMinTemp,thePtMaxTemp);
      tmphisto->setAxisTitle("Number of Events", 2);
      tmphisto->setAxisTitle("p_{T}", 1);
      ethist.push_back(tmphisto);
      
      histoname = theHLTCollectionLabels[i].label()+"eta";
      histoTitle = theHLTCollectionLabels[i].label() + " #eta";
      tmphisto =  dbe->book1D(histoname.c_str(),histoTitle.c_str(),theNbins,-2.7,2.7);
      tmphisto->setAxisTitle("Number of Events", 2);
      tmphisto->setAxisTitle("#eta", 1);
      etahist.push_back(tmphisto);          
 
      histoname = theHLTCollectionLabels[i].label()+"phi";
      histoTitle = theHLTCollectionLabels[i].label() + " #phi";
      tmphisto =  dbe->book1D(histoname.c_str(),histoTitle.c_str(),theNbins,-3.14,3.14);
      tmphisto->setAxisTitle("Number of Events", 2);
      tmphisto->setAxisTitle("#phi", 1);
      phihist.push_back(tmphisto);  
       
      histoname = theHLTCollectionLabels[i].label()+"eta_phi";
      histoTitle = theHLTCollectionLabels[i].label() + " #eta vs. #phi";
      tmphisto =  dbe->book2D(histoname.c_str(),histoTitle.c_str(),theNbins,-2.7,2.7, theNbins,
      -3.14, 3.14);
      tmphisto->setAxisTitle("#phi", 2);
      tmphisto->setAxisTitle("#eta", 1);
      eta_phihist.push_back(tmphisto);	

      if(plotiso[i]){
	histoname = theHLTCollectionLabels[i].label()+"eta isolation";
	tmphisto = dbe->book2D(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7,theNbins,plotBounds[i].first,plotBounds[i].second);
      }
      else{
	tmphisto = NULL;
      }
      etahistiso.push_back(tmphisto);
      
      if(plotiso[i]){
	histoname = theHLTCollectionLabels[i].label()+"et isolation";
	tmphisto = dbe->book2D(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax,theNbins,plotBounds[i].first,plotBounds[i].second);
      }
      else{
	tmphisto = NULL;
      }
      ethistiso.push_back(tmphisto);
      

      if(plotiso[i]){
	histoname = theHLTCollectionLabels[i].label()+"phi isolation";
	tmphisto = dbe->book2D(histoname.c_str(),histoname.c_str(),theNbins,-3.14,3.14,theNbins,plotBounds[i].first,plotBounds[i].second);
      }
      else{
	tmphisto = NULL;
      }
      phihistiso.push_back(tmphisto);
	

    } 
  } // end "if(dbe)"
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTMon::endJob() {

//     std::cout << "HLTMonElectron: end job...." << std::endl;
   LogInfo("HLTMon") << "analyzed " << nev_ << " events";
 
   if (outputFile_.size() != 0 && dbe)
     dbe->save(outputFile_);
 
   return;
}

//DEFINE_FWK_MODULE(HLTMonElectron);

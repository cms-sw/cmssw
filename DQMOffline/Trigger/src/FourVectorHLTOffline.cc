// $Id: FourVectorHLTOffline.cc,v 1.9 2008/08/15 20:17:34 berryhil Exp $
// See header file for information. 
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/Trigger/interface/FourVectorHLTOffline.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

#include "PhysicsTools/Utilities/interface/deltaR.h"

#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;

FourVectorHLTOffline::FourVectorHLTOffline(const edm::ParameterSet& iConfig):
  resetMe_(true),  currentRun_(-99)
{
  LogDebug("FourVectorHLTOffline") << "constructor...." ;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("FourVectorHLTOffline") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  
  dirname_="HLTOffline/FourVectorHLTOffline" + 
    iConfig.getParameter<std::string>("@module_label");
  
  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirname_);
  }
  
  
  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",40);
  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);

  // this is the list of paths to look at.
  std::vector<edm::ParameterSet> paths = 
    iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");
  for(std::vector<edm::ParameterSet>::iterator 
	pathconf = paths.begin() ; pathconf != paths.end(); 
      pathconf++) {
    std::string pathname = pathconf->getParameter<std::string>("pathname");  
    edm::InputTag filtername = pathconf->getParameter<edm::InputTag>("filtername");
    int objectType = pathconf->getParameter<unsigned int>("type");
    float ptMin = pathconf->getUntrackedParameter<double>("ptMin");
    float ptMax = pathconf->getUntrackedParameter<double>("ptMax");
    hltPaths_.push_back(PathInfo(pathname, filtername, objectType, ptMin, ptMax));
  }
  if ( hltPaths_.size() && plotAll_) {
    // these two ought to be mutually exclusive....
    LogInfo("FourVectorHLTOffline") << "Using both plotAll and a list. "
      "list will be ignored." ;
    hltPaths_.clear();
  }

  if (hltPaths_.size() > 0)
    {
      // book a histogram of scalers
     scalersSelect = dbe_->book1D("selectedScalers","Selected Scalers", hltPaths_.size(), 0.0, (double)hltPaths_.size());
    }
  triggerSummaryLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
 
  
}


FourVectorHLTOffline::~FourVectorHLTOffline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
FourVectorHLTOffline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace trigger;
  using namespace l1extra;
  ++nev_;
  LogDebug("FourVectorHLTOffline")<< "FourVectorHLTOffline: analyze...." ;
  
  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(triggerResultsLabel_,triggerResults);
  if(!triggerResults.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "TriggerResults not found, "
      "skipping event"; 
    return;
  }
    


  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "Summary HLT objects not found, "
      "skipping event"; 
    return;
  }
  
  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());



    for(PathInfoCollection::iterator v = hltPaths_.begin();
	v!= hltPaths_.end(); ++v ) 
{ 
  // fill scaler histograms
           


      const int index = triggerObj->filterIndex(v->getTag());
      if ( index >= triggerObj->sizeFilters() ) {
	//        cout << "WTF no index "<< index << " of that name "
	//	     << v->getTag().label() << "\t" << v->getTag().instance() << "\t" << v->getTag().process() << endl;
	continue; // not in this event
      }
      LogDebug("FourVectorHLTOffline") << "filling ... " ;
      const trigger::Keys & k = triggerObj->filterKeys(index);
      const trigger::Vids & idtype = triggerObj->filterIds(index);
      // assume for now the first object type is the same as all objects in the collection
      // cout << idtype.size() << "\t" << k.size() << endl;
      int triggertype = 0;     
      if (idtype.size() > 0) triggertype = *idtype.begin();
      //      cout << "path " << v->getPath() << "trigger type "<<triggertype << endl;
      if (k.size() > 0) v->getNOnHisto()->Fill(k.size());
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	v->getEtOnHisto()->Fill(toc[*ki].pt());
	v->getEtaOnHisto()->Fill(toc[*ki].eta());
	v->getPhiOnHisto()->Fill(toc[*ki].phi());
	v->getEtaVsPhiOnHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
	//  cout << "pdgId "<<toc[*ki].id() << endl;
      // for muon triggers, loop over and fill offline 4-vectors
      if (triggertype == trigger::TriggerMuon || triggertype == trigger::TriggerL1Mu)
	{
         edm::Handle<reco::MuonCollection> muonHandle;
         iEvent.getByLabel("muons",muonHandle);

         if(!muonHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "muonHandle not found, "
            "skipping event"; 
            return;
          }
         const reco::MuonCollection muonCollection = *(muonHandle.product());

         for (reco::MuonCollection::const_iterator muonIter=muonCollection.begin(); muonIter!=muonCollection.end(); muonIter++)
         {
	   if (reco::deltaR((*muonIter).eta(),(*muonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getEtOffHisto()->Fill((*muonIter).pt());
	  v->getEtaOffHisto()->Fill((*muonIter).eta());
	  v->getPhiOffHisto()->Fill((*muonIter).phi());
	  v->getEtaVsPhiOffHisto()->Fill((*muonIter).eta(),(*muonIter).phi());
	   }
         }


         edm::Handle<l1extra::L1MuonParticleCollection> l1MuonHandle;
         iEvent.getByType(l1MuonHandle);

         if(!l1MuonHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "l1MuonHandle not found, "
            "skipping event"; 
            return;
         }
         const l1extra::L1MuonParticleCollection l1MuonCollection = *(l1MuonHandle.product());

         for (l1extra::L1MuonParticleCollection::const_iterator l1MuonIter=l1MuonCollection.begin(); l1MuonIter!=l1MuonCollection.end(); l1MuonIter++)
         {
	   if (reco::deltaR((*l1MuonIter).eta(),(*l1MuonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getEtL1Histo()->Fill((*l1MuonIter).pt());
	  v->getEtaL1Histo()->Fill((*l1MuonIter).eta());
	  v->getPhiL1Histo()->Fill((*l1MuonIter).phi());
	  v->getEtaVsPhiL1Histo()->Fill((*l1MuonIter).eta(),(*l1MuonIter).phi());
	   }
         }
	}

      // for electron triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerElectron)
	{
	  //	  std::cout << "Electron trigger" << std::endl;
         edm::Handle<reco::PixelMatchGsfElectronCollection> gsfElectrons;
         iEvent.getByLabel("pixelMatchGsfElectrons",gsfElectrons); 
         if(!gsfElectrons.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "gsfElectrons not found, "
            "skipping event"; 
            return;
         }
         for (reco::PixelMatchGsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin(); gsfIter!=gsfElectrons->end(); gsfIter++)
         {
	   if (reco::deltaR((*gsfIter).eta(),(*gsfIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getEtOffHisto()->Fill(gsfIter->pt());
	  v->getEtaOffHisto()->Fill(gsfIter->eta());
	  v->getPhiOffHisto()->Fill(gsfIter->phi());
	  v->getEtaVsPhiOffHisto()->Fill(gsfIter->eta(), gsfIter->phi());
	   }
         }

         std::vector<edm::Handle<l1extra::L1EmParticleCollection> > l1ElectronHandleList;
         iEvent.getManyByType(l1ElectronHandleList);        
         std::vector<edm::Handle<l1extra::L1EmParticleCollection> >::iterator l1ElectronHandle;

         for (l1ElectronHandle=l1ElectronHandleList.begin(); l1ElectronHandle!=l1ElectronHandleList.end(); l1ElectronHandle++) {

         const L1EmParticleCollection l1ElectronCollection = *(l1ElectronHandle->product());
	   for (L1EmParticleCollection::const_iterator l1ElectronIter=l1ElectronCollection.begin(); l1ElectronIter!=l1ElectronCollection.end(); l1ElectronIter++){
	   if (reco::deltaR((*l1ElectronIter).eta(),(*l1ElectronIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getEtL1Histo()->Fill((*l1ElectronIter).pt());
     	  v->getEtaL1Histo()->Fill((*l1ElectronIter).eta());
          v->getPhiL1Histo()->Fill((*l1ElectronIter).phi());
     	  v->getEtaVsPhiL1Histo()->Fill((*l1ElectronIter).eta(),(*l1ElectronIter).phi());
	   }
	   }
         }
	}


      // for tau triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerTau)
	{
         edm::Handle<reco::CaloTauCollection> tauHandle;
         iEvent.getByLabel("caloRecoTauProducer",tauHandle);

         if(!tauHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "tauHandle not found, "
            "skipping event"; 
            return;
         }
         const reco::CaloTauCollection tauCollection = *(tauHandle.product());

         for (reco::CaloTauCollection::const_iterator tauIter=tauCollection.begin(); tauIter!=tauCollection.end(); tauIter++)
         {
	   if (reco::deltaR((*tauIter).eta(),(*tauIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getEtOffHisto()->Fill((*tauIter).pt());
	  v->getEtaOffHisto()->Fill((*tauIter).eta());
	  v->getPhiOffHisto()->Fill((*tauIter).phi());
	  v->getEtaVsPhiOffHisto()->Fill((*tauIter).eta(),(*tauIter).phi());
	   }
         }


         std::vector<edm::Handle<l1extra::L1JetParticleCollection> > l1TauHandleList;
         iEvent.getManyByType(l1TauHandleList);        
         std::vector<edm::Handle<l1extra::L1JetParticleCollection> >::iterator l1TauHandle;

         for (l1TauHandle=l1TauHandleList.begin(); l1TauHandle!=l1TauHandleList.end(); l1TauHandle++) {
	   if (!l1TauHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOffline") << "photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1TauCollection = *(l1TauHandle->product());
	   for (L1JetParticleCollection::const_iterator l1TauIter=l1TauCollection.begin(); l1TauIter!=l1TauCollection.end(); l1TauIter++){
	   if (reco::deltaR((*l1TauIter).eta(),(*l1TauIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getEtL1Histo()->Fill((*l1TauIter).pt());
     	  v->getEtaL1Histo()->Fill((*l1TauIter).eta());
          v->getPhiL1Histo()->Fill((*l1TauIter).phi());
     	  v->getEtaVsPhiL1Histo()->Fill((*l1TauIter).eta(),(*l1TauIter).phi());
	   }
	   }
         }
	}


      // for jet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerJet)
	{
         edm::Handle<reco::CaloJetCollection> jetHandle;
         iEvent.getByLabel("iterativeCone5CaloJets",jetHandle);

         if(!jetHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "jetHandle not found, "
            "skipping event"; 
            return;
         }
         const reco::CaloJetCollection jetCollection = *(jetHandle.product());

         for (reco::CaloJetCollection::const_iterator jetIter=jetCollection.begin(); jetIter!=jetCollection.end(); jetIter++)
         {
	   if (reco::deltaR((*jetIter).eta(),(*jetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getEtOffHisto()->Fill((*jetIter).pt());
	  v->getEtaOffHisto()->Fill((*jetIter).eta());
	  v->getPhiOffHisto()->Fill((*jetIter).phi());
	  v->getEtaVsPhiOffHisto()->Fill((*jetIter).eta(),(*jetIter).phi());
	   }
         }



         std::vector<edm::Handle<l1extra::L1JetParticleCollection> > l1JetHandleList;
         iEvent.getManyByType(l1JetHandleList);        
         std::vector<edm::Handle<l1extra::L1JetParticleCollection> >::iterator l1JetHandle;

         for (l1JetHandle=l1JetHandleList.begin(); l1JetHandle!=l1JetHandleList.end(); l1JetHandle++) {
	   if (!l1JetHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOffline") << "photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1JetCollection = *(l1JetHandle->product());
	   for (L1JetParticleCollection::const_iterator l1JetIter=l1JetCollection.begin(); l1JetIter!=l1JetCollection.end(); l1JetIter++){
	   if (reco::deltaR((*l1JetIter).eta(),(*l1JetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getEtL1Histo()->Fill((*l1JetIter).pt());
     	  v->getEtaL1Histo()->Fill((*l1JetIter).eta());
          v->getPhiL1Histo()->Fill((*l1JetIter).phi());
     	  v->getEtaVsPhiL1Histo()->Fill((*l1JetIter).eta(),(*l1JetIter).phi());
	   }
	   }
         }
	}

      // for bjet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerBJet)
	{
	}
      // for met triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerMET)
	{
         edm::Handle<reco::CaloMETCollection> metHandle;
         iEvent.getByLabel("met",metHandle);

         if(!metHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "metHandle not found, "
            "skipping event"; 
            return;
         }
         const reco::CaloMETCollection metCollection = *(metHandle.product());

         for (reco::CaloMETCollection::const_iterator metIter=metCollection.begin(); metIter!=metCollection.end(); metIter++)
         {
	   if (reco::deltaR((*metIter).eta(),(*metIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getEtOffHisto()->Fill((*metIter).pt());
	  v->getEtaOffHisto()->Fill((*metIter).eta());
	  v->getPhiOffHisto()->Fill((*metIter).phi());
	  v->getEtaVsPhiOffHisto()->Fill((*metIter).eta(),(*metIter).phi());
	   }
         }

         Handle< L1EtMissParticleCollection > l1MetHandle ;
         iEvent.getByType(l1MetHandle) ;

         if(!l1MetHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "l1MetHandle not found, "
            "skipping event"; 
            return;
         }
         const l1extra::L1EtMissParticleCollection l1MetCollection = *(l1MetHandle.product());

         for (l1extra::L1EtMissParticleCollection::const_iterator l1MetIter=l1MetCollection.begin(); l1MetIter!=l1MetCollection.end(); l1MetIter++)
         {
	   if (reco::deltaR((*l1MetIter).eta(),(*l1MetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getEtL1Histo()->Fill((*l1MetIter).pt());
	  v->getEtaL1Histo()->Fill((*l1MetIter).eta());
	  v->getPhiL1Histo()->Fill((*l1MetIter).phi());
	  v->getEtaVsPhiL1Histo()->Fill((*l1MetIter).eta(),(*l1MetIter).phi());
	   }
         }

	}


      // for photon triggers, loop over and fill offline and L1 4-vectors
      else if (triggertype == trigger::TriggerPhoton)
	{

         edm::Handle<reco::PhotonCollection> photonHandle;
         iEvent.getByLabel("photons",photonHandle);

         if(!photonHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "photonHandle not found, "
            "skipping event"; 
            return;
         }
         const reco::PhotonCollection photonCollection = *(photonHandle.product());

         for (reco::PhotonCollection::const_iterator photonIter=photonCollection.begin(); photonIter!=photonCollection.end(); photonIter++)
         {
	   if (reco::deltaR((*photonIter).eta(),(*photonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getEtOffHisto()->Fill((*photonIter).pt());
	  v->getEtaOffHisto()->Fill((*photonIter).eta());
	  v->getPhiOffHisto()->Fill((*photonIter).phi());
	  v->getEtaVsPhiOffHisto()->Fill((*photonIter).eta(),(*photonIter).phi());
	   }
         }


         std::vector<edm::Handle<l1extra::L1EmParticleCollection> > l1PhotonHandleList;
         iEvent.getManyByType(l1PhotonHandleList);        
         std::vector<edm::Handle<l1extra::L1EmParticleCollection> >::iterator l1PhotonHandle;

         for (l1PhotonHandle=l1PhotonHandleList.begin(); l1PhotonHandle!=l1PhotonHandleList.end(); l1PhotonHandle++) {
	   if (!l1PhotonHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOffline") << "photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1EmParticleCollection l1PhotonCollection = *(l1PhotonHandle->product());
	   for (L1EmParticleCollection::const_iterator l1PhotonIter=l1PhotonCollection.begin(); l1PhotonIter!=l1PhotonCollection.end(); l1PhotonIter++){
	   if (reco::deltaR((*l1PhotonIter).eta(),(*l1PhotonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getEtL1Histo()->Fill((*l1PhotonIter).pt());
     	  v->getEtaL1Histo()->Fill((*l1PhotonIter).eta());
          v->getPhiL1Histo()->Fill((*l1PhotonIter).phi());
     	  v->getEtaVsPhiL1Histo()->Fill((*l1PhotonIter).eta(),(*l1PhotonIter).phi());
	   }
	   

	 }
       }
     }

    }
  }
}



// -- method called once each job just before starting event loop  --------
void 
FourVectorHLTOffline::beginJob(const edm::EventSetup&)
{
  nev_ = 0;
  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    dbe->rmdir(dirname_);
  }
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);

    if ( ! plotAll_ ) {
      for(PathInfoCollection::iterator v = hltPaths_.begin();
	  v!= hltPaths_.end(); ++v ) {
	MonitorElement *NOn, *etOn, *etaOn, *phiOn, *etavsphiOn=0;
	MonitorElement *etOff, *etaOff, *phiOff, *etavsphiOff=0;
	MonitorElement *etL1, *etaL1, *phiL1, *etavsphiL1=0;
	std::string labelname("dummy");
        labelname = v->getPath();
	std::string histoname(labelname+"_NOn");
	std::string title(labelname+" N online");
	NOn =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);
      
	histoname = labelname+"_etOn";
	title = labelname+" E_t offline";
	etOn =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_etOff";
	title = labelname+" E_t offline";
	etOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_etL1";
	title = labelname+" E_t L1";
	etL1 =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_etaOn";
	title = labelname+" #eta online";
	etaOn =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_etaOff";
	title = labelname+" #eta offline";
	etaOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_etaL1";
	title = labelname+" #eta L1";
	etaL1 =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_phiOn";
	title = labelname+" #phi online";
	phiOn =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);

	histoname = labelname+"_phiOff";
	title = labelname+" #phi offline";
	phiOff =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);

	histoname = labelname+"_phiL1";
	title = labelname+" #phi L1";
	phiL1 =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);
 

	histoname = labelname+"_etaphiOn";
	title = labelname+" #eta vs #phi online";
	etavsphiOn =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	histoname = labelname+"_etaphiOff";
	title = labelname+" #eta vs #phi offline";
	etavsphiOff =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	histoname = labelname+"_etaphiL1";
	title = labelname+" #eta vs #phi L1";
	etavsphiL1 =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	v->setHistos( NOn, etOn, etaOn, phiOn, etavsphiOn, etOff, etaOff, phiOff, etavsphiOff, etL1, etaL1, phiL1, etavsphiL1);
      }
    }
  }
}

// - method called once each job just after ending the event loop  ------------
void 
FourVectorHLTOffline::endJob() 
{
   LogInfo("FourVectorHLTOffline") << "analyzed " << nev_ << " events";
   return;
}


// BeginRun
void FourVectorHLTOffline::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTOffline") << "beginRun, run " << run.id();
}

/// EndRun
void FourVectorHLTOffline::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTOffline") << "endRun, run " << run.id();
}

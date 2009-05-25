// $Id: FourVectorHLTOffline.cc,v 1.17 2008/12/04 18:37:39 berryhil Exp $
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
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

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
  
  processname_ = iConfig.getParameter<std::string>("processname");

  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",40);
  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);

  if (!plotAll_)
 {
  // this is the list of paths to look at.
  std::vector<edm::ParameterSet> paths = 
    iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");
  for(std::vector<edm::ParameterSet>::iterator 
	pathconf = paths.begin() ; pathconf != paths.end(); 
      pathconf++) {
    std::string denompathname = pathconf->getParameter<std::string>("denompathname");  
    std::string pathname = pathconf->getParameter<std::string>("pathname");  
    std::string l1pathname = pathconf->getParameter<std::string>("l1pathname");  
    std::string filtername = pathconf->getParameter<std::string>("filtername");
    int objectType = pathconf->getParameter<unsigned int>("type");
    float ptMin = pathconf->getUntrackedParameter<double>("ptMin");
    float ptMax = pathconf->getUntrackedParameter<double>("ptMax");
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));
  }

  if (hltPaths_.size() > 0)
    {
      // book a histogram of scalers
     scalersSelect = dbe_->book1D("selectedScalers","Selected Scalers", hltPaths_.size(), 0.0, (double)hltPaths_.size());
    }

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
  TriggerNames triggerNames(*triggerResults);  
  int npath = triggerResults->size();

  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "Summary HLT objects not found, "
      "skipping event"; 
    return;
  }
  
  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());


  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByLabel("muons",muonHandle);
  if(!muonHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "muonHandle not found, ";
    //  "skipping event"; 
    //  return;
   }

  edm::Handle<l1extra::L1MuonParticleCollection> l1MuonHandle;
  iEvent.getByType(l1MuonHandle);
  if(!l1MuonHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "l1MuonHandle not found, "
    "skipping event"; 
    return;
   }
  const l1extra::L1MuonParticleCollection l1MuonCollection = *(l1MuonHandle.product());

  edm::Handle<reco::PixelMatchGsfElectronCollection> gsfElectrons;
  iEvent.getByLabel("pixelMatchGsfElectrons",gsfElectrons); 
  if(!gsfElectrons.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "gsfElectrons not found, ";
      //"skipping event"; 
      //return;
  }

  std::vector<edm::Handle<l1extra::L1EmParticleCollection> > l1ElectronHandleList;
  iEvent.getManyByType(l1ElectronHandleList);        
  std::vector<edm::Handle<l1extra::L1EmParticleCollection> >::iterator l1ElectronHandle;

  
  edm::Handle<reco::CaloTauCollection> tauHandle;
  iEvent.getByLabel("caloRecoTauProducer",tauHandle);
  if(!tauHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "tauHandle not found, ";
      //"skipping event"; 
      //return;
  }



  std::vector<edm::Handle<l1extra::L1JetParticleCollection> > l1TauHandleList;
  iEvent.getManyByType(l1TauHandleList);        
  std::vector<edm::Handle<l1extra::L1JetParticleCollection> >::iterator l1TauHandle;

  edm::Handle<reco::CaloJetCollection> jetHandle;
  iEvent.getByLabel("iterativeCone5CaloJets",jetHandle);
  if(!jetHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "jetHandle not found, ";
      //"skipping event"; 
      //return;
  }


  std::vector<edm::Handle<l1extra::L1JetParticleCollection> > l1JetHandleList;
  iEvent.getManyByType(l1JetHandleList);        
  std::vector<edm::Handle<l1extra::L1JetParticleCollection> >::iterator l1JetHandle;

  edm::Handle<reco::CaloMETCollection> metHandle;
  iEvent.getByLabel("met",metHandle);
  if(!metHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "metHandle not found, ";
      //"skipping event"; 
      //return;
  }


  Handle< L1EtMissParticleCollection > l1MetHandle ;
  iEvent.getByType(l1MetHandle) ;
  if(!l1MetHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "l1MetHandle not found, "
    "skipping event"; 
    return;
  }
  const l1extra::L1EtMissParticleCollection l1MetCollection = *(l1MetHandle.product());

  edm::Handle<reco::PhotonCollection> photonHandle;
  iEvent.getByLabel("photons",photonHandle);
  if(!photonHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "photonHandle not found, ";
      //"skipping event"; 
      //return;
  }


  std::vector<edm::Handle<l1extra::L1EmParticleCollection> > l1PhotonHandleList;
  iEvent.getManyByType(l1PhotonHandleList);        
  std::vector<edm::Handle<l1extra::L1EmParticleCollection> >::iterator l1PhotonHandle;
 
    for(PathInfoCollection::iterator v = hltPaths_.begin();
	v!= hltPaths_.end(); ++v ) 
{ 

  // did we pass the denomPath?
  bool denompassed = false;
  for(int i = 0; i < npath; ++i) {
     if (triggerNames.triggerName(i) == v->getDenomPath() && triggerResults->accept(i)) denompassed = true;
  }

  if (denompassed)
    {  



      // ok plot denominator L1, and denominator offline, and numerator L1Off objects 
      int triggertype = 0;     
      //if (idtype.size() > 0) triggertype = *idtype.begin();
      triggertype = v->getObjectType();


      // for muon triggers, loop over and fill offline 4-vectors
      if (triggertype == trigger::TriggerMuon || triggertype == trigger::TriggerL1Mu){
	if (muonHandle.isValid()){
         const reco::MuonCollection muonCollection = *(muonHandle.product());
         for (reco::MuonCollection::const_iterator muonIter=muonCollection.begin(); muonIter!=muonCollection.end(); muonIter++)
         {
	  v->getOffEtOffHisto()->Fill((*muonIter).pt());
	  v->getOffEtaOffHisto()->Fill((*muonIter).eta());
	  v->getOffPhiOffHisto()->Fill((*muonIter).phi());
	  v->getOffEtaVsOffPhiOffHisto()->Fill((*muonIter).eta(),(*muonIter).phi());
	 }
	}
         for (l1extra::L1MuonParticleCollection::const_iterator l1MuonIter=l1MuonCollection.begin(); l1MuonIter!=l1MuonCollection.end(); l1MuonIter++)
         {
	  v->getL1EtL1Histo()->Fill((*l1MuonIter).pt());
	  v->getL1EtaL1Histo()->Fill((*l1MuonIter).eta());
	  v->getL1PhiL1Histo()->Fill((*l1MuonIter).phi());
	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1MuonIter).eta(),(*l1MuonIter).phi());
	  if (muonHandle.isValid()){
         const reco::MuonCollection muonCollection = *(muonHandle.product());
         for (reco::MuonCollection::const_iterator muonIter=muonCollection.begin(); muonIter!=muonCollection.end(); muonIter++)
         {
	   if (reco::deltaR((*muonIter).eta(),(*muonIter).phi(),(*l1MuonIter).eta(),(*l1MuonIter).phi()) < 0.3){
	  v->getOffEtL1OffHisto()->Fill((*muonIter).pt());
	  v->getOffEtaL1OffHisto()->Fill((*muonIter).eta());
	  v->getOffPhiL1OffHisto()->Fill((*muonIter).phi());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill((*muonIter).eta(),(*muonIter).phi());
	   }}
	 }
	 }
      }
      // for electron triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerElectron)
	{
	  //	  std::cout << "Electron trigger" << std::endl;
	  if (gsfElectrons.isValid()){
         for (reco::PixelMatchGsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin(); gsfIter!=gsfElectrons->end(); gsfIter++)
         {
	  v->getOffEtOffHisto()->Fill(gsfIter->pt());
	  v->getOffEtaOffHisto()->Fill(gsfIter->eta());
	  v->getOffPhiOffHisto()->Fill(gsfIter->phi());
	  v->getOffEtaVsOffPhiOffHisto()->Fill(gsfIter->eta(), gsfIter->phi());
         }}
         for (l1ElectronHandle=l1ElectronHandleList.begin(); l1ElectronHandle!=l1ElectronHandleList.end(); l1ElectronHandle++) {

         const L1EmParticleCollection l1ElectronCollection = *(l1ElectronHandle->product());
	   for (L1EmParticleCollection::const_iterator l1ElectronIter=l1ElectronCollection.begin(); l1ElectronIter!=l1ElectronCollection.end(); l1ElectronIter++){
     	  v->getL1EtL1Histo()->Fill((*l1ElectronIter).pt());
     	  v->getL1EtaL1Histo()->Fill((*l1ElectronIter).eta());
          v->getL1PhiL1Histo()->Fill((*l1ElectronIter).phi());
     	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1ElectronIter).eta(),(*l1ElectronIter).phi());
	  if (gsfElectrons.isValid()){
         for (reco::PixelMatchGsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin(); gsfIter!=gsfElectrons->end(); gsfIter++)
         {
	   if (reco::deltaR(gsfIter->eta(),gsfIter->phi(),(*l1ElectronIter).eta(),(*l1ElectronIter).phi()) < 0.3){
	  v->getOffEtL1OffHisto()->Fill(gsfIter->pt());
	  v->getOffEtaL1OffHisto()->Fill(gsfIter->eta());
	  v->getOffPhiL1OffHisto()->Fill(gsfIter->phi());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill(gsfIter->eta(), gsfIter->phi());}
	 }}
	   }
	   }
	}
    

      // for tau triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerTau)
	{
	  if (tauHandle.isValid()){
	    const reco::CaloTauCollection tauCollection = *(tauHandle.product());
         for (reco::CaloTauCollection::const_iterator tauIter=tauCollection.begin(); tauIter!=tauCollection.end(); tauIter++)
         {
	  v->getOffEtOffHisto()->Fill((*tauIter).pt());
	  v->getOffEtaOffHisto()->Fill((*tauIter).eta());
	  v->getOffPhiOffHisto()->Fill((*tauIter).phi());
	  v->getOffEtaVsOffPhiOffHisto()->Fill((*tauIter).eta(),(*tauIter).phi());
         }}
         for (l1TauHandle=l1TauHandleList.begin(); l1TauHandle!=l1TauHandleList.end(); l1TauHandle++) {
	   if (!l1TauHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOffline") << "l1TauHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1TauCollection = *(l1TauHandle->product());
	   for (L1JetParticleCollection::const_iterator l1TauIter=l1TauCollection.begin(); l1TauIter!=l1TauCollection.end(); l1TauIter++){
     	  v->getL1EtL1Histo()->Fill((*l1TauIter).pt());
     	  v->getL1EtaL1Histo()->Fill((*l1TauIter).eta());
          v->getL1PhiL1Histo()->Fill((*l1TauIter).phi());
     	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1TauIter).eta(),(*l1TauIter).phi());
         if (tauHandle.isValid()){
	   const reco::CaloTauCollection tauCollection = *(tauHandle.product());
         for (reco::CaloTauCollection::const_iterator tauIter=tauCollection.begin(); tauIter!=tauCollection.end(); tauIter++)
         {
	   if (reco::deltaR((*tauIter).eta(),(*tauIter).phi(),(*l1TauIter).eta(),(*l1TauIter).phi()) < 0.3){
	  v->getOffEtL1OffHisto()->Fill((*tauIter).pt());
	  v->getOffEtaL1OffHisto()->Fill((*tauIter).eta());
	  v->getOffPhiL1OffHisto()->Fill((*tauIter).phi());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill((*tauIter).eta(),(*tauIter).phi());}
         }}
	   }
         }
	}


      // for jet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerJet)
	{
	  if (jetHandle.isValid()){
         const reco::CaloJetCollection jetCollection = *(jetHandle.product());
         for (reco::CaloJetCollection::const_iterator jetIter=jetCollection.begin(); jetIter!=jetCollection.end(); jetIter++)
         {
	  v->getOffEtOffHisto()->Fill((*jetIter).pt());
	  v->getOffEtaOffHisto()->Fill((*jetIter).eta());
	  v->getOffPhiOffHisto()->Fill((*jetIter).phi());
	  v->getOffEtaVsOffPhiOffHisto()->Fill((*jetIter).eta(),(*jetIter).phi());
         }}
         for (l1JetHandle=l1JetHandleList.begin(); l1JetHandle!=l1JetHandleList.end(); l1JetHandle++) {
	   if (!l1JetHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOffline") << "l1JetHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1JetCollection = *(l1JetHandle->product());
	   for (L1JetParticleCollection::const_iterator l1JetIter=l1JetCollection.begin(); l1JetIter!=l1JetCollection.end(); l1JetIter++){
     	  v->getL1EtL1Histo()->Fill((*l1JetIter).pt());
     	  v->getL1EtaL1Histo()->Fill((*l1JetIter).eta());
          v->getL1PhiL1Histo()->Fill((*l1JetIter).phi());
     	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1JetIter).eta(),(*l1JetIter).phi());
	  if (jetHandle.isValid()){
         const reco::CaloJetCollection jetCollection = *(jetHandle.product());
         for (reco::CaloJetCollection::const_iterator jetIter=jetCollection.begin(); jetIter!=jetCollection.end(); jetIter++)
         {
	   if (reco::deltaR((*jetIter).eta(),(*jetIter).phi(),(*l1JetIter).eta(),(*l1JetIter).phi()) < 0.3){
	  v->getOffEtL1OffHisto()->Fill((*jetIter).pt());
	  v->getOffEtaL1OffHisto()->Fill((*jetIter).eta());
	  v->getOffPhiL1OffHisto()->Fill((*jetIter).phi());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill((*jetIter).eta(),(*jetIter).phi());}
         }}
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
	  if (metHandle.isValid()){
         const reco::CaloMETCollection metCollection = *(metHandle.product());
         for (reco::CaloMETCollection::const_iterator metIter=metCollection.begin(); metIter!=metCollection.end(); metIter++)
         {
	  v->getOffEtOffHisto()->Fill((*metIter).pt());
	  v->getOffEtaOffHisto()->Fill((*metIter).eta());
	  v->getOffPhiOffHisto()->Fill((*metIter).phi());
	  v->getOffEtaVsOffPhiOffHisto()->Fill((*metIter).eta(),(*metIter).phi());
         }}

         for (l1extra::L1EtMissParticleCollection::const_iterator l1MetIter=l1MetCollection.begin(); l1MetIter!=l1MetCollection.end(); l1MetIter++)
         {
	  v->getL1EtL1Histo()->Fill((*l1MetIter).pt());
	  v->getL1EtaL1Histo()->Fill((*l1MetIter).eta());
	  v->getL1PhiL1Histo()->Fill((*l1MetIter).phi());
	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1MetIter).eta(),(*l1MetIter).phi());
	  if (metHandle.isValid()){
         const reco::CaloMETCollection metCollection = *(metHandle.product());
         for (reco::CaloMETCollection::const_iterator metIter=metCollection.begin(); metIter!=metCollection.end(); metIter++)
         {
	   if (reco::deltaR((*metIter).eta(),(*metIter).phi(),(*l1MetIter).eta(),(*l1MetIter).phi()) < 0.3){
	  v->getOffEtL1OffHisto()->Fill((*metIter).pt());
	  v->getOffEtaL1OffHisto()->Fill((*metIter).eta());
	  v->getOffPhiL1OffHisto()->Fill((*metIter).phi());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill((*metIter).eta(),(*metIter).phi());}
         }}
	  }

	}


      // for photon triggers, loop over and fill offline and L1 4-vectors
      else if (triggertype == trigger::TriggerPhoton)
	{
	  if (photonHandle.isValid()){
          const reco::PhotonCollection photonCollection = *(photonHandle.product());
         for (reco::PhotonCollection::const_iterator photonIter=photonCollection.begin(); photonIter!=photonCollection.end(); photonIter++)
         {
	  v->getOffEtOffHisto()->Fill((*photonIter).pt());
	  v->getOffEtaOffHisto()->Fill((*photonIter).eta());
	  v->getOffPhiOffHisto()->Fill((*photonIter).phi());
	  v->getOffEtaVsOffPhiOffHisto()->Fill((*photonIter).eta(),(*photonIter).phi());
         }
	  }

         for (l1PhotonHandle=l1PhotonHandleList.begin(); l1PhotonHandle!=l1PhotonHandleList.end(); l1PhotonHandle++) {
	   if (!l1PhotonHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOffline") << "photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1EmParticleCollection l1PhotonCollection = *(l1PhotonHandle->product());
	   for (L1EmParticleCollection::const_iterator l1PhotonIter=l1PhotonCollection.begin(); l1PhotonIter!=l1PhotonCollection.end(); l1PhotonIter++){
     	  v->getL1EtL1Histo()->Fill((*l1PhotonIter).pt());
     	  v->getL1EtaL1Histo()->Fill((*l1PhotonIter).eta());
          v->getL1PhiL1Histo()->Fill((*l1PhotonIter).phi());
     	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1PhotonIter).eta(),(*l1PhotonIter).phi());
	  if (photonHandle.isValid()){
          const reco::PhotonCollection photonCollection = *(photonHandle.product());
         for (reco::PhotonCollection::const_iterator photonIter=photonCollection.begin(); photonIter!=photonCollection.end(); photonIter++)
         {
	   if (reco::deltaR((*photonIter).eta(),(*photonIter).phi(),(*l1PhotonIter).eta(),(*l1PhotonIter).phi()) < 0.3){
	  v->getOffEtL1OffHisto()->Fill((*photonIter).pt());
	  v->getOffEtaL1OffHisto()->Fill((*photonIter).eta());
	  v->getOffPhiL1OffHisto()->Fill((*photonIter).phi());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill((*photonIter).eta(),(*photonIter).phi());}
         }}
           }
	 }
       }

    // did we pass the numerator path?
  bool numpassed = false;
  for(int i = 0; i < npath; ++i) {
     if (triggerNames.triggerName(i) == v->getPath() && triggerResults->accept(i)) numpassed = true;
  }

  if (numpassed)
    {  
    // ok plot On, L1On, and OnOff objects

    // fill scaler histograms
      edm::InputTag filterTag = v->getTag();
      if (plotAll_)
	{
	// loop through indices and see if the filter is on the list of filters used by this path
      
    if (v->getLabel() == "dummy"){
        const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
	//loop over labels
        for (std::vector<std::string>::const_iterator labelIter= filterLabels.begin(); labelIter!=filterLabels.end(); labelIter++)          
	 {
	   //cout << v->getPath() << "\t" << *labelIter << endl;
           // last match wins...
	   edm::InputTag testTag(*labelIter,"",processname_);
	   //           cout << v->getPath() << "\t" << testTag.label() << "\t" << testTag.process() << endl;
           int testindex = triggerObj->filterIndex(testTag);
           if ( !(testindex >= triggerObj->sizeFilters()) ) {
	     //cout << "found one! " << v->getPath() << "\t" << testTag.label() << endl; 
            filterTag = testTag; v->setLabel(*labelIter);}
	 }
         }
	}

      const int index = triggerObj->filterIndex(filterTag);
      if ( index >= triggerObj->sizeFilters() ) {
	//        cout << "WTF no index "<< index << " of that name "
	//	     << filterTag << endl;
	continue; // not in this event
      }
      LogDebug("FourVectorHLTOffline") << "filling ... " ;
      const trigger::Keys & k = triggerObj->filterKeys(index);
      //      const trigger::Vids & idtype = triggerObj->filterIds(index);
      // assume for now the first object type is the same as all objects in the collection
      //    cout << filterTag << "\t" << idtype.size() << "\t" << k.size() << endl;
      //     cout << "path " << v->getPath() << " trigger type "<<triggertype << endl;
      if (k.size() > 0) v->getNOnHisto()->Fill(k.size());
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	v->getOnEtOnHisto()->Fill(toc[*ki].pt());
	v->getOnEtaOnHisto()->Fill(toc[*ki].eta());
	v->getOnPhiOnHisto()->Fill(toc[*ki].phi());
	v->getOnEtaVsOnPhiOnHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
	//	  cout << "pdgId "<<toc[*ki].id() << endl;
      // for muon triggers, loop over and fill offline 4-vectors
      if (triggertype == trigger::TriggerMuon || triggertype == trigger::TriggerL1Mu)
	{
	  if (muonHandle.isValid()){
         const reco::MuonCollection muonCollection = *(muonHandle.product());
         for (reco::MuonCollection::const_iterator muonIter=muonCollection.begin(); muonIter!=muonCollection.end(); muonIter++)
         {
	   if (reco::deltaR((*muonIter).eta(),(*muonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getOffEtOnOffHisto()->Fill((*muonIter).pt());
	  v->getOffEtaOnOffHisto()->Fill((*muonIter).eta());
	  v->getOffPhiOnOffHisto()->Fill((*muonIter).phi());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill((*muonIter).eta(),(*muonIter).phi());
	   }
         }}
         for (l1extra::L1MuonParticleCollection::const_iterator l1MuonIter=l1MuonCollection.begin(); l1MuonIter!=l1MuonCollection.end(); l1MuonIter++)
         {
	   if (reco::deltaR((*l1MuonIter).eta(),(*l1MuonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getL1EtL1OnHisto()->Fill((*l1MuonIter).pt());
	  v->getL1EtaL1OnHisto()->Fill((*l1MuonIter).eta());
	  v->getL1PhiL1OnHisto()->Fill((*l1MuonIter).phi());
	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1MuonIter).eta(),(*l1MuonIter).phi());
	   }
         }
	}

      // for electron triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerElectron)
	{
	  //	  std::cout << "Electron trigger" << std::endl;
	  if (gsfElectrons.isValid()){
         for (reco::PixelMatchGsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin(); gsfIter!=gsfElectrons->end(); gsfIter++)
         {
	   if (reco::deltaR((*gsfIter).eta(),(*gsfIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getOffEtOnOffHisto()->Fill(gsfIter->pt());
	  v->getOffEtaOnOffHisto()->Fill(gsfIter->eta());
	  v->getOffPhiOnOffHisto()->Fill(gsfIter->phi());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill(gsfIter->eta(), gsfIter->phi());
	   }
         }}
         for (l1ElectronHandle=l1ElectronHandleList.begin(); l1ElectronHandle!=l1ElectronHandleList.end(); l1ElectronHandle++) {

         const L1EmParticleCollection l1ElectronCollection = *(l1ElectronHandle->product());
	   for (L1EmParticleCollection::const_iterator l1ElectronIter=l1ElectronCollection.begin(); l1ElectronIter!=l1ElectronCollection.end(); l1ElectronIter++){
	   if (reco::deltaR((*l1ElectronIter).eta(),(*l1ElectronIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getL1EtL1OnHisto()->Fill((*l1ElectronIter).pt());
     	  v->getL1EtaL1OnHisto()->Fill((*l1ElectronIter).eta());
          v->getL1PhiL1OnHisto()->Fill((*l1ElectronIter).phi());
     	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1ElectronIter).eta(),(*l1ElectronIter).phi());
	   }
	   }
         }
	}


      // for tau triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerTau)
	{
	  if (tauHandle.isValid()){
	    const reco::CaloTauCollection tauCollection = *(tauHandle.product());
         for (reco::CaloTauCollection::const_iterator tauIter=tauCollection.begin(); tauIter!=tauCollection.end(); tauIter++)
         {
	   if (reco::deltaR((*tauIter).eta(),(*tauIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getOffEtOnOffHisto()->Fill((*tauIter).pt());
	  v->getOffEtaOnOffHisto()->Fill((*tauIter).eta());
	  v->getOffPhiOnOffHisto()->Fill((*tauIter).phi());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill((*tauIter).eta(),(*tauIter).phi());
	   }
         }}


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
     	  v->getL1EtL1OnHisto()->Fill((*l1TauIter).pt());
     	  v->getL1EtaL1OnHisto()->Fill((*l1TauIter).eta());
          v->getL1PhiL1OnHisto()->Fill((*l1TauIter).phi());
     	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1TauIter).eta(),(*l1TauIter).phi());
	   }
	   }
         }
	}


      // for jet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerJet)
	{
	  if (jetHandle.isValid()){
         const reco::CaloJetCollection jetCollection = *(jetHandle.product());
         for (reco::CaloJetCollection::const_iterator jetIter=jetCollection.begin(); jetIter!=jetCollection.end(); jetIter++)
         {
	   if (reco::deltaR((*jetIter).eta(),(*jetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getOffEtOnOffHisto()->Fill((*jetIter).pt());
	  v->getOffEtaOnOffHisto()->Fill((*jetIter).eta());
	  v->getOffPhiOnOffHisto()->Fill((*jetIter).phi());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill((*jetIter).eta(),(*jetIter).phi());
	   }
         }}
         for (l1JetHandle=l1JetHandleList.begin(); l1JetHandle!=l1JetHandleList.end(); l1JetHandle++) {
	   if (!l1JetHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOffline") << "l1JetHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1JetCollection = *(l1JetHandle->product());
	   for (L1JetParticleCollection::const_iterator l1JetIter=l1JetCollection.begin(); l1JetIter!=l1JetCollection.end(); l1JetIter++){
	   if (reco::deltaR((*l1JetIter).eta(),(*l1JetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getL1EtL1OnHisto()->Fill((*l1JetIter).pt());
     	  v->getL1EtaL1OnHisto()->Fill((*l1JetIter).eta());
          v->getL1PhiL1OnHisto()->Fill((*l1JetIter).phi());
     	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1JetIter).eta(),(*l1JetIter).phi());
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
	  if (metHandle.isValid()){
         const reco::CaloMETCollection metCollection = *(metHandle.product());
         for (reco::CaloMETCollection::const_iterator metIter=metCollection.begin(); metIter!=metCollection.end(); metIter++)
         {
	   if (reco::deltaR((*metIter).eta(),(*metIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getOffEtOnOffHisto()->Fill((*metIter).pt());
	  v->getOffEtaOnOffHisto()->Fill((*metIter).eta());
	  v->getOffPhiOnOffHisto()->Fill((*metIter).phi());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill((*metIter).eta(),(*metIter).phi());
	   }
         }}

         for (l1extra::L1EtMissParticleCollection::const_iterator l1MetIter=l1MetCollection.begin(); l1MetIter!=l1MetCollection.end(); l1MetIter++)
         {
	   if (reco::deltaR((*l1MetIter).eta(),(*l1MetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getL1EtL1OnHisto()->Fill((*l1MetIter).pt());
	  v->getL1EtaL1OnHisto()->Fill((*l1MetIter).eta());
	  v->getL1PhiL1OnHisto()->Fill((*l1MetIter).phi());
	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1MetIter).eta(),(*l1MetIter).phi());
	   }
         }

	}


      // for photon triggers, loop over and fill offline and L1 4-vectors
      else if (triggertype == trigger::TriggerPhoton)
	{
	  if (photonHandle.isValid()){
          const reco::PhotonCollection photonCollection = *(photonHandle.product());
         for (reco::PhotonCollection::const_iterator photonIter=photonCollection.begin(); photonIter!=photonCollection.end(); photonIter++)
         {
	   if (reco::deltaR((*photonIter).eta(),(*photonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getOffEtOnOffHisto()->Fill((*photonIter).pt());
	  v->getOffEtaOnOffHisto()->Fill((*photonIter).eta());
	  v->getOffPhiOnOffHisto()->Fill((*photonIter).phi());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill((*photonIter).eta(),(*photonIter).phi());
	   }
         }}


         for (l1PhotonHandle=l1PhotonHandleList.begin(); l1PhotonHandle!=l1PhotonHandleList.end(); l1PhotonHandle++) {
	   if (!l1PhotonHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOffline") << "l1photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1EmParticleCollection l1PhotonCollection = *(l1PhotonHandle->product());
	   for (L1EmParticleCollection::const_iterator l1PhotonIter=l1PhotonCollection.begin(); l1PhotonIter!=l1PhotonCollection.end(); l1PhotonIter++){
	   if (reco::deltaR((*l1PhotonIter).eta(),(*l1PhotonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getL1EtL1OnHisto()->Fill((*l1PhotonIter).pt());
     	  v->getL1EtaL1OnHisto()->Fill((*l1PhotonIter).eta());
          v->getL1PhiL1OnHisto()->Fill((*l1PhotonIter).phi());
     	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1PhotonIter).eta(),(*l1PhotonIter).phi());
	   }
	   

	 }
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
// HLT config does not change within runs!
 
  if (!hltConfig_.init(processname_)) {
  LogDebug("FourVectorHLTOffline") << "HLTConfigProvider failed to initialize.";
    // check if trigger name in (new) config
    //	cout << "Available TriggerNames are: " << endl;
	//	hltConfig_.dump("Triggers");
      }


    // get provenance, HLT PSet 
  //     std::vector<edm::ParameterSet> ps;
  //   if (run.getProcessParameterSet(processname_,ps))
  //   {
  //    cout << ps << endl;
  //   }






  if (1)
 {
  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
  }


    const unsigned int n(hltConfig_.size());
    for (unsigned int j=0; j!=n; ++j) {
    for (unsigned int i=0; i!=n; ++i) {
      // cout << hltConfig_.triggerName(i) << endl;
    
    std::string denompathname = hltConfig_.triggerName(j);  
    std::string pathname = hltConfig_.triggerName(i);  
    std::string l1pathname = "dummy";
    int objectType = 0;
    int denomobjectType = 0;
    //parse pathname to guess object type
    if (pathname.find("Jet") != std::string::npos) 
      objectType = trigger::TriggerJet;    
    if (pathname.find("BJet") != std::string::npos) 
      objectType = trigger::TriggerBJet;    
    if (pathname.find("MET") != std::string::npos) 
      objectType = trigger::TriggerMET;    
    if (pathname.find("Mu") != std::string::npos) 
      objectType = trigger::TriggerMuon;    
    if (pathname.find("Ele") != std::string::npos) 
      objectType = trigger::TriggerElectron;    
    if (pathname.find("Photon") != std::string::npos) 
      objectType = trigger::TriggerPhoton;    
    if (pathname.find("Tau") != std::string::npos) 
      objectType = trigger::TriggerTau;    


    //parse denompathname to guess denomobject type
    if (denompathname.find("Jet") != std::string::npos) 
      denomobjectType = trigger::TriggerJet;    
    if (denompathname.find("BJet") != std::string::npos) 
      denomobjectType = trigger::TriggerBJet;    
    if (denompathname.find("MET") != std::string::npos) 
      denomobjectType = trigger::TriggerMET;    
    if (denompathname.find("Mu") != std::string::npos) 
      denomobjectType = trigger::TriggerMuon;    
    if (denompathname.find("Ele") != std::string::npos) 
      denomobjectType = trigger::TriggerElectron;    
    if (denompathname.find("Photon") != std::string::npos) 
      denomobjectType = trigger::TriggerPhoton;    
    if (denompathname.find("Tau") != std::string::npos) 
      denomobjectType = trigger::TriggerTau;    

    // find L1 condition for numpath with numpath objecttype 
    // find PSet for L1 global seed for numpath, 
    // get L1 path name which has numpath objecttype (pick just one for now)
    



    std::string filtername("dummy");
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (pathname.find("HLT_") != std::string::npos && plotAll_ && denomobjectType == objectType && objectType != 0)
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));
    }
    }
    // now set up all of the histos for each path
    for(PathInfoCollection::iterator v = hltPaths_.begin();
	  v!= hltPaths_.end(); ++v ) {
    	MonitorElement *NOn, *onEtOn, *onEtaOn, *onPhiOn, *onEtavsonPhiOn=0;
	MonitorElement *offEtOff, *offEtaOff, *offPhiOff, *offEtavsoffPhiOff=0;
	MonitorElement *l1EtL1, *l1EtaL1, *l1PhiL1, *l1Etavsl1PhiL1=0;
    	MonitorElement *l1EtL1On, *l1EtaL1On, *l1PhiL1On, *l1Etavsl1PhiL1On=0;
	MonitorElement *offEtL1Off, *offEtaL1Off, *offPhiL1Off, *offEtavsoffPhiL1Off=0;
	MonitorElement *offEtOnOff, *offEtaOnOff, *offPhiOnOff, *offEtavsoffPhiOnOff=0;
	std::string labelname("dummy");
        labelname = v->getPath() + " " + v->getDenomPath();
	std::string histoname(labelname+"_NOn");
	std::string title(labelname+" N online");
	NOn =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);
      
	histoname = labelname+"_onEtOn";
	title = labelname+" onE_t online";
	onEtOn =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_offEtOff";
	title = labelname+" offE_t offline";
	offEtOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_l1EtL1";
	title = labelname+" l1E_t L1";
	l1EtL1 =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_onEtaOn";
	title = labelname+" on#eta online";
	onEtaOn =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_offEtaOff";
	title = labelname+" off#eta offline";
	offEtaOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_l1EtaL1";
	title = labelname+" l1#eta L1";
	l1EtaL1 =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_onPhiOn";
	title = labelname+" on#phi online";
	onPhiOn =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);

	histoname = labelname+"_offPhiOff";
	title = labelname+" off#phi offline";
	offPhiOff =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);

	histoname = labelname+"_l1PhiL1";
	title = labelname+" l1#phi L1";
	l1PhiL1 =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);
 

	histoname = labelname+"_onEtaonPhiOn";
	title = labelname+" on#eta vs on#phi online";
	onEtavsonPhiOn =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	histoname = labelname+"_offEtaoffPhiOff";
	title = labelname+" off#eta vs off#phi offline";
	offEtavsoffPhiOff =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	histoname = labelname+"_l1Etal1PhiL1";
	title = labelname+" l1#eta vs l1#phi L1";
	l1Etavsl1PhiL1 =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	histoname = labelname+"_l1EtL1On";
	title = labelname+" l1E_t L1+online";
	l1EtL1On =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_offEtL1Off";
	title = labelname+" offE_t L1+offline";
	offEtL1Off =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_offEtOnOff";
	title = labelname+" offE_t online+offline";
	offEtOnOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_l1EtaL1On";
	title = labelname+" l1#eta L1+online";
	l1EtaL1On =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_offEtaL1Off";
	title = labelname+" off#eta L1+offline";
	offEtaL1Off =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_offEtaOnOff";
	title = labelname+" off#eta online+offline";
	offEtaOnOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_l1PhiL1On";
	title = labelname+" l1#phi L1+online";
	l1PhiL1On =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);

	histoname = labelname+"_offPhiL1Off";
	title = labelname+" off#phi L1+offline";
	offPhiL1Off =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);

	histoname = labelname+"_offPhiOnOff";
	title = labelname+" off#phi online+offline";
	offPhiOnOff =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);
 

	histoname = labelname+"_l1Etal1PhiL1On";
	title = labelname+" l1#eta vs l1#phi L1+online";
	l1Etavsl1PhiL1On =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	histoname = labelname+"_offEtaoffPhiL1Off";
	title = labelname+" off#eta vs off#phi L1+offline";
	offEtavsoffPhiL1Off =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	histoname = labelname+"_offEtaoffPhiOnOff";
	title = labelname+" off#eta vs off#phi online+offline";
	offEtavsoffPhiOnOff =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	v->setHistos( NOn, onEtOn, onEtaOn, onPhiOn, onEtavsonPhiOn, offEtOff, offEtaOff, offPhiOff, offEtavsoffPhiOff, l1EtL1, l1EtaL1, l1PhiL1, l1Etavsl1PhiL1, l1EtL1On, l1EtaL1On, l1PhiL1On, l1Etavsl1PhiL1On, offEtL1Off, offEtaL1Off, offPhiL1Off, offEtavsoffPhiL1Off, offEtOnOff, offEtaOnOff, offPhiOnOff, offEtavsoffPhiOnOff);


    }
 }
 return;



}

/// EndRun
void FourVectorHLTOffline::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTOffline") << "endRun, run " << run.id();
}

// $Id: FourVectorHLTriggerOffline.cc,v 1.6 2009/02/12 15:43:54 berryhil Exp $
// See header file for information. 
#include "TMath.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTriggerOffline/Common/interface/FourVectorHLTriggerOffline.h"

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
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "PhysicsTools/Utilities/interface/deltaR.h"

#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;

FourVectorHLTriggerOffline::FourVectorHLTriggerOffline(const edm::ParameterSet& iConfig):
  resetMe_(true),  currentRun_(-99)
{
  LogDebug("FourVectorHLTriggerOffline") << "constructor...." ;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("FourVectorHLTriggerOffline") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  dirname_ = iConfig.getUntrackedParameter("dirname",
					   std::string("HLT/FourVector/"));
  //dirname_ +=  iConfig.getParameter<std::string>("@module_label");
  
  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirname_);
  }
  
  processname_ = iConfig.getParameter<std::string>("processname");

  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",20);
  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);
     // this is the list of paths to look at.
     std::vector<edm::ParameterSet> paths = 
     iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");
     for(std::vector<edm::ParameterSet>::iterator 
	pathconf = paths.begin() ; pathconf != paths.end(); 
      pathconf++) {
       std::pair<std::string, std::string> custompathnamepair;
       custompathnamepair.first =pathconf->getParameter<std::string>("pathname"); 
       custompathnamepair.second = pathconf->getParameter<std::string>("denompathname");   
       custompathnamepairs_.push_back(custompathnamepair);
       //    customdenompathnames_.push_back(pathconf->getParameter<std::string>("denompathname"));  
       // custompathnames_.push_back(pathconf->getParameter<std::string>("pathname"));  
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
  gtObjectMapRecordLabel_ = 
    iConfig.getParameter<edm::InputTag>("gtObjectMapRecordLabel");
  l1GTRRLabel_ = 
    iConfig.getParameter<edm::InputTag>("l1GTRRLabel");
  l1GtMenuCacheIDtemp_ = 0ULL;
 

  electronEtaMax_ = iConfig.getUntrackedParameter<double>("electronEtaMax",2.5);
  electronEtMin_ = iConfig.getUntrackedParameter<double>("electronEtMin",3.0);
  muonEtaMax_ = iConfig.getUntrackedParameter<double>("muonEtaMax",2.5);
  muonEtMin_ = iConfig.getUntrackedParameter<double>("muonEtMin",3.0);
  tauEtaMax_ = iConfig.getUntrackedParameter<double>("tauEtaMax",2.5);
  tauEtMin_ = iConfig.getUntrackedParameter<double>("tauEtMin",3.0);
  jetEtaMax_ = iConfig.getUntrackedParameter<double>("jetEtaMax",5.0);
  jetEtMin_ = iConfig.getUntrackedParameter<double>("jetEtMin",10.0);
  bjetEtaMax_ = iConfig.getUntrackedParameter<double>("bjetEtaMax",2.5);
  bjetEtMin_ = iConfig.getUntrackedParameter<double>("bjetEtMin",10.0);
  metEtMin_ = iConfig.getUntrackedParameter<double>("metEtMin",10.0);
  photonEtaMax_ = iConfig.getUntrackedParameter<double>("photonEtaMax",2.5);
  photonEtMin_ = iConfig.getUntrackedParameter<double>("photonEtMin",3.0);

  
}


FourVectorHLTriggerOffline::~FourVectorHLTriggerOffline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
FourVectorHLTriggerOffline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace trigger;
  using namespace l1extra;
  ++nev_;
  LogDebug("FourVectorHLTriggerOffline")<< "FourVectorHLTriggerOffline: analyze...." ;
  
  Handle<GenParticleCollection> genParticles;
  iEvent.getByLabel("genParticles", genParticles);
  if(!genParticles.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "genParticles not found, "
      "skipping event"; 
    return;
  }


  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(triggerResultsLabel_,triggerResults);
  if(!triggerResults.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "TriggerResults not found, "
      "skipping event"; 
    return;
  }
  TriggerNames triggerNames(*triggerResults);  
  int npath = triggerResults->size();

  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "Summary HLT objects not found, "
      "skipping event"; 
    return;
  }
  
  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());

  // get handle to object maps (one object map per algorithm)
  edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
  iEvent.getByLabel(gtObjectMapRecordLabel_, gtObjectMapRecord);
  if(!gtObjectMapRecord.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "L1GlobalTriggerObjectMapRecord not found, ";
    //  "skipping event"; 
    // return;
  }
    unsigned long long l1GtMenuCacheID = iSetup.get<L1GtTriggerMenuRcd>().cacheIdentifier();
    
     if (l1GtMenuCacheIDtemp_ != l1GtMenuCacheID) {
 
         edm::ESHandle< L1GtTriggerMenu> l1GtMenuHandle;
         iSetup.get< L1GtTriggerMenuRcd>().get(l1GtMenuHandle) ;
         l1GtMenu = l1GtMenuHandle.product();
         (const_cast<L1GtTriggerMenu*>(l1GtMenu))->buildGtConditionMap(); 
         l1GtMenuCacheIDtemp_ = l1GtMenuCacheID;
 
         // update also the tokenNumber members (holding the bit numbers) from m_l1AlgoLogicParser
	 //         updateAlgoLogicParser(m_l1GtMenu);
     }
     //  const AlgorithmMap& algorithmMap = l1GtMenu->gtAlgorithmMap();

  edm::Handle<L1GlobalTriggerReadoutRecord> l1GTRR;
  iEvent.getByLabel(l1GTRRLabel_,l1GTRR);
  if(!l1GTRR.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "L1GlobalTriggerReadoutRecord "<< l1GTRRLabel_ << " not found, ";
      //  "skipping event"; 
      //return;
  }
  const DecisionWord gtDecisionWord = l1GTRR->decisionWord();

  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByLabel("muons",muonHandle);
  if(!muonHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "muonHandle not found, ";
    //  "skipping event"; 
    //  return;
   }

  edm::Handle<l1extra::L1MuonParticleCollection> l1MuonHandle;
  iEvent.getByType(l1MuonHandle);
  if(!l1MuonHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "l1MuonHandle not found, ";
      //"skipping event"; 
      //return;
   }
  const l1extra::L1MuonParticleCollection l1MuonCollection = *(l1MuonHandle.product());

  edm::Handle<reco::PixelMatchGsfElectronCollection> gsfElectrons;
  iEvent.getByLabel("pixelMatchGsfElectrons",gsfElectrons); 
  if(!gsfElectrons.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "gsfElectrons not found, ";
      //"skipping event"; 
      //return;
  }

  std::vector<edm::Handle<l1extra::L1EmParticleCollection> > l1ElectronHandleList;
  iEvent.getManyByType(l1ElectronHandleList);        
  std::vector<edm::Handle<l1extra::L1EmParticleCollection> >::iterator l1ElectronHandle;

  
  edm::Handle<reco::CaloTauCollection> tauHandle;
  iEvent.getByLabel("caloRecoTauProducer",tauHandle);
  if(!tauHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "tauHandle not found, ";
      //"skipping event"; 
      //return;
  }



  std::vector<edm::Handle<l1extra::L1JetParticleCollection> > l1TauHandleList;
  iEvent.getManyByType(l1TauHandleList);        
  std::vector<edm::Handle<l1extra::L1JetParticleCollection> >::iterator l1TauHandle;

  edm::Handle<reco::CaloJetCollection> jetHandle;
  iEvent.getByLabel("iterativeCone5CaloJets",jetHandle);
  if(!jetHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "jetHandle not found, ";
      //"skipping event"; 
      //return;
  }

 
   // Get b tag information
 edm::Handle<reco::JetTagCollection> bTagIPHandle;
 iEvent.getByLabel("jetProbabilityBJetTags", bTagIPHandle);
 if (!bTagIPHandle.isValid()) {
    edm::LogInfo("FourVectorHLTriggerOffline") << "bTagIPHandle trackCountingHighEffJetTags not found, ";
      //"skipping event"; 
      //return;
  }


   // Get b tag information
 edm::Handle<reco::JetTagCollection> bTagMuHandle;
 iEvent.getByLabel("softMuonBJetTags", bTagMuHandle);
 if (!bTagMuHandle.isValid()) {
    edm::LogInfo("FourVectorHLTriggerOffline") << "bTagMuHandle  not found, ";
      //"skipping event"; 
      //return;
  }


  std::vector<edm::Handle<l1extra::L1JetParticleCollection> > l1JetHandleList;
  iEvent.getManyByType(l1JetHandleList);        
  std::vector<edm::Handle<l1extra::L1JetParticleCollection> >::iterator l1JetHandle;

  edm::Handle<reco::CaloMETCollection> metHandle;
  iEvent.getByLabel("met",metHandle);
  if(!metHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "metHandle not found, ";
      //"skipping event"; 
      //return;
  }


  Handle< L1EtMissParticleCollection > l1MetHandle ;
  iEvent.getByType(l1MetHandle) ;
  if(!l1MetHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "l1MetHandle not found, ";
    //"skipping event"; 
    // return;
  }
  const l1extra::L1EtMissParticleCollection l1MetCollection = *(l1MetHandle.product());

  edm::Handle<reco::PhotonCollection> photonHandle;
  iEvent.getByLabel("photons",photonHandle);
  if(!photonHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "photonHandle not found, ";
      //"skipping event"; 
      //return;
  }


  std::vector<edm::Handle<l1extra::L1EmParticleCollection> > l1PhotonHandleList;
  iEvent.getManyByType(l1PhotonHandleList);        
  std::vector<edm::Handle<l1extra::L1EmParticleCollection> >::iterator l1PhotonHandle;
 
    for(PathInfoCollection::iterator v = hltPaths_.begin();
	v!= hltPaths_.end(); ++v ) 
{ 

  int NMc = 0;
  int NOn = 0;
  int NOff = 0;
  int NL1 = 0;
  int NL1Mc = 0;
  int NOnMc = 0;
  int NOnOff = 0;
  int NL1On = 0;
  int NL1Off = 0;

  // did we pass the denomPath?
  bool denompassed = false;
  for(int i = 0; i < npath; ++i) {
     if (triggerNames.triggerName(i) == v->getDenomPath() && triggerResults->accept(i)) denompassed = true;
  }

  if (denompassed)
    {  


      int triggertype = 0;     
      triggertype = v->getObjectType();


      // plot denominator MC, denominator L1, and denominator offline, and numerator L1Off objects
 
      // test whether the L1 seed path for the numerator path passed

      // get the list of L1seed algortihms.  
      //Let's assume they are always OR'ed for now
        L1GtLogicParser l1AlgoLogicParser = L1GtLogicParser(v->getl1Path());
	std::vector<L1GtLogicParser::TokenRPN> l1RpnVector = l1AlgoLogicParser.rpnVector();
        l1AlgoLogicParser.buildOperandTokenVector();
	std::vector<L1GtLogicParser::OperandToken> l1AlgoSeeds = l1AlgoLogicParser.operandTokenVector();

        std::vector< const std::vector<L1GtLogicParser::TokenRPN>* > l1AlgoSeedsRpn;
        std::vector< std::vector< const std::vector<L1GtObject>* > > l1AlgoSeedsObjType;
        
	//	cout << v->getl1Path() << "\t" << l1AlgoLogicParser.logicalExpression() << "\t" << l1RpnVector.size() << "\t" << l1AlgoSeeds.size() << endl;
        //l1AlgoSeeds = l1AlgoLogicParser->expressionSeedsOperandList();

	// loop over the algorithms
         int iAlgo = -1;
         bool l1accept = false;
	 for (std::vector<L1GtLogicParser::OperandToken>::const_iterator
	   itSeed = l1AlgoSeeds.begin(); itSeed != l1AlgoSeeds.end(); 
	  ++itSeed) 
	  {
	    // determine whether this algo passed, go to the next one if not
	    iAlgo++;
	    //  cout << (*itSeed).tokenName << endl;
            //int algBit = (*itSeed).tokenNumber;
            std::string algName = (*itSeed).tokenName;
            const bool algResult = l1GtMenu->gtAlgorithmResult(algName,
             gtDecisionWord);

            //bool algResult = (*itSeed).tokenResult;
            if ( algResult) {
	      //cout << "found one" << "\t" << v->getl1Path() << "\t" << algName << endl;
              //   continue;
              l1accept = true;
            }
	  }


      // for muon triggers, loop over and fill offline 4-vectors
      if (triggertype == trigger::TriggerMuon || triggertype == trigger::TriggerL1Mu){
	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 13 && p.status() == 3 && fabs(p.eta()) <= muonEtaMax_ && p.pt() >= muonEtMin_){
            NMc++;
	    v->getMcEtMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiMcHisto()->Fill(p.eta(),p.phi());
	  }
	 }
	}

	if (muonHandle.isValid()){
         const reco::MuonCollection muonCollection = *(muonHandle.product());
         for (reco::MuonCollection::const_iterator muonIter=muonCollection.begin(); muonIter!=muonCollection.end(); muonIter++)
         {
	   if (fabs((*muonIter).eta()) <= muonEtaMax_ && (*muonIter).pt() >= muonEtMin_){
	  NOff++;
	  v->getOffEtOffHisto()->Fill((*muonIter).pt());
	  v->getOffEtaVsOffPhiOffHisto()->Fill((*muonIter).eta(),(*muonIter).phi());
	   }
         }
	}

        if (l1accept){
         for (l1extra::L1MuonParticleCollection::const_iterator l1MuonIter=l1MuonCollection.begin(); l1MuonIter!=l1MuonCollection.end(); l1MuonIter++)
         {
	   if (fabs((*l1MuonIter).eta()) <= muonEtaMax_ && (*l1MuonIter).pt() >= muonEtMin_){
	  NL1++;
	  v->getL1EtL1Histo()->Fill((*l1MuonIter).pt());
	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1MuonIter).eta(),(*l1MuonIter).phi());
	   }

	  if (muonHandle.isValid()){
         const reco::MuonCollection muonCollection = *(muonHandle.product());
         for (reco::MuonCollection::const_iterator muonIter=muonCollection.begin(); muonIter!=muonCollection.end(); muonIter++)
         {
	   if (reco::deltaR((*muonIter).eta(),(*muonIter).phi(),(*l1MuonIter).eta(),(*l1MuonIter).phi()) < 0.3 && fabs((*muonIter).eta()) <= muonEtaMax_ && (*muonIter).pt() >= muonEtMin_ ){
	  NL1Off++;
	  v->getOffEtL1OffHisto()->Fill((*muonIter).pt());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill((*muonIter).eta(),(*muonIter).phi());
	   }
	  }
	 }

	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 13 && p.status() == 3 && fabs(p.eta()) <= muonEtaMax_ && p.pt() >= muonEtMin_){ 
	   if (reco::deltaR(p.eta(),p.phi(),(*l1MuonIter).eta(),(*l1MuonIter).phi()) < 0.3){
	    NL1Mc++;
	    v->getMcEtL1McHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiL1McHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
	} 
	 
      }
	}
     }
      // for electron triggers, loop over and fill offline 4-vectors
     else if (triggertype == trigger::TriggerElectron || triggertype == trigger::TriggerL1NoIsoEG || triggertype == trigger::TriggerL1IsoEG)
	{
	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 11 && p.status() == 3 && fabs(p.eta()) <= electronEtaMax_ && p.pt() >= electronEtMin_ ){
            NMc++;
	    v->getMcEtMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiMcHisto()->Fill(p.eta(),p.phi());
	  }
	 }
	}


	  //	  std::cout << "Electron trigger" << std::endl;
	  if (gsfElectrons.isValid()){
         for (reco::PixelMatchGsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin(); gsfIter!=gsfElectrons->end(); gsfIter++)
         {
	   if (fabs(gsfIter->eta()) <= electronEtaMax_ && gsfIter->pt() >= electronEtMin_ ){
	  NOff++;
	  v->getOffEtOffHisto()->Fill(gsfIter->pt());
	  v->getOffEtaVsOffPhiOffHisto()->Fill(gsfIter->eta(), gsfIter->phi());
	   }
         }
         }

	  if (l1accept){
         for (l1ElectronHandle=l1ElectronHandleList.begin(); l1ElectronHandle!=l1ElectronHandleList.end(); l1ElectronHandle++) {

         const L1EmParticleCollection l1ElectronCollection = *(l1ElectronHandle->product());
	   for (L1EmParticleCollection::const_iterator l1ElectronIter=l1ElectronCollection.begin(); l1ElectronIter!=l1ElectronCollection.end(); l1ElectronIter++){
	     if (fabs((*l1ElectronIter).eta()) <= electronEtaMax_ && (*l1ElectronIter).pt() >= electronEtMin_ ){
	  NL1++;
     	  v->getL1EtL1Histo()->Fill((*l1ElectronIter).pt());
     	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1ElectronIter).eta(),(*l1ElectronIter).phi());
	     }    
	  if (gsfElectrons.isValid()){
         for (reco::PixelMatchGsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin(); gsfIter!=gsfElectrons->end(); gsfIter++)
         {
	   if (reco::deltaR(gsfIter->eta(),gsfIter->phi(),(*l1ElectronIter).eta(),(*l1ElectronIter).phi()) < 0.3 && fabs(gsfIter->eta()) <= electronEtaMax_ && gsfIter->pt() >= electronEtMin_ ){
	  NL1Off++;
	  v->getOffEtL1OffHisto()->Fill(gsfIter->pt());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill(gsfIter->eta(), gsfIter->phi());}
	 }
         }

	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 11 && p.status() == 3 && fabs(p.eta()) <= electronEtaMax_ && p.pt() >= electronEtMin_ ){ 
	   if (reco::deltaR(p.eta(),p.phi(),(*l1ElectronIter).eta(),(*l1ElectronIter).phi()) < 0.3){
	    NL1Mc++;
	    v->getMcEtL1McHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiL1McHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
        }

       }
	 }
	  }
    }
    

      // for tau triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerTau || triggertype == trigger::TriggerL1TauJet)
	{

	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 15 && p.status() == 3 && fabs(p.eta()) <= tauEtaMax_ && p.pt() >= tauEtMin_){
            NMc++; 
	    v->getMcEtMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiMcHisto()->Fill(p.eta(),p.phi());
	  }
	 }
	}

	  if (tauHandle.isValid()){
	    const reco::CaloTauCollection tauCollection = *(tauHandle.product());
         for (reco::CaloTauCollection::const_iterator tauIter=tauCollection.begin(); tauIter!=tauCollection.end(); tauIter++)
         {
	   if (fabs((*tauIter).eta()) <= tauEtaMax_ && (*tauIter).pt() >= tauEtMin_ ){
	  NOff++;
	  v->getOffEtOffHisto()->Fill((*tauIter).pt());
	  v->getOffEtaVsOffPhiOffHisto()->Fill((*tauIter).eta(),(*tauIter).phi());
	   }
         }
         }

	  if (l1accept){

         for (l1TauHandle=l1TauHandleList.begin(); l1TauHandle!=l1TauHandleList.end(); l1TauHandle++) {
	   if (!l1TauHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTriggerOffline") << "l1TauHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1TauCollection = *(l1TauHandle->product());
	   for (L1JetParticleCollection::const_iterator l1TauIter=l1TauCollection.begin(); l1TauIter!=l1TauCollection.end(); l1TauIter++){
	     if (fabs((*l1TauIter).eta()) <= tauEtaMax_ && (*l1TauIter).pt() >= tauEtMin_ ){
	  NL1++;
     	  v->getL1EtL1Histo()->Fill((*l1TauIter).pt());
     	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1TauIter).eta(),(*l1TauIter).phi());
	     }

         if (tauHandle.isValid()){
	   const reco::CaloTauCollection tauCollection = *(tauHandle.product());
         for (reco::CaloTauCollection::const_iterator tauIter=tauCollection.begin(); tauIter!=tauCollection.end(); tauIter++)
         {
	   if (reco::deltaR((*tauIter).eta(),(*tauIter).phi(),(*l1TauIter).eta(),(*l1TauIter).phi()) < 0.3 && fabs((*tauIter).eta()) <= tauEtaMax_ && (*tauIter).pt() >= tauEtMin_ ){
	  NL1Off++;
	  v->getOffEtL1OffHisto()->Fill((*tauIter).pt());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill((*tauIter).eta(),(*tauIter).phi());}
         }}

	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 15 && p.status() == 3 && fabs(p.eta()) <= tauEtaMax_ && p.pt() >= tauEtMin_ ){ 
	   if (reco::deltaR(p.eta(),p.phi(),(*l1TauIter).eta(),(*l1TauIter).phi()) < 0.3){
	    NL1Mc++;
	    v->getMcEtL1McHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiL1McHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
        }

       }
	 }
	  }
    }



      // for jet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerJet || triggertype == trigger::TriggerL1CenJet || triggertype == trigger::TriggerL1ForJet)
	{
	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if ((abs(p.pdgId()) == 21 || (abs(p.pdgId()) <= 5 && abs(p.pdgId()) >=1)) && p.status() == 3 && fabs(p.eta()) <= jetEtaMax_ && p.pt() >= jetEtMin_ ){
            NMc++; 
	    v->getMcEtMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiMcHisto()->Fill(p.eta(),p.phi());
	  }
	 }
	}

	  if (jetHandle.isValid()){
         const reco::CaloJetCollection jetCollection = *(jetHandle.product());
         for (reco::CaloJetCollection::const_iterator jetIter=jetCollection.begin(); jetIter!=jetCollection.end(); jetIter++)
         {
	   if (fabs((*jetIter).eta()) <= jetEtaMax_ && (*jetIter).pt() >= jetEtMin_ ){
	  NOff++;
	  v->getOffEtOffHisto()->Fill((*jetIter).pt());
	  v->getOffEtaVsOffPhiOffHisto()->Fill((*jetIter).eta(),(*jetIter).phi());
	   }
         }
         }

	  if (l1accept){

         for (l1JetHandle=l1JetHandleList.begin(); l1JetHandle!=l1JetHandleList.end(); l1JetHandle++) {
	   if (!l1JetHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTriggerOffline") << "l1JetHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1JetCollection = *(l1JetHandle->product());
	   for (L1JetParticleCollection::const_iterator l1JetIter=l1JetCollection.begin(); l1JetIter!=l1JetCollection.end(); l1JetIter++){
	     if (fabs((*l1JetIter).eta()) <= jetEtaMax_ && (*l1JetIter).pt() >= jetEtMin_ ){
	  NL1++;
     	  v->getL1EtL1Histo()->Fill((*l1JetIter).pt());
     	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1JetIter).eta(),(*l1JetIter).phi());
	     }

	  if (jetHandle.isValid()){
         const reco::CaloJetCollection jetCollection = *(jetHandle.product());
         for (reco::CaloJetCollection::const_iterator jetIter=jetCollection.begin(); jetIter!=jetCollection.end(); jetIter++)
         {
	   if (reco::deltaR((*jetIter).eta(),(*jetIter).phi(),(*l1JetIter).eta(),(*l1JetIter).phi()) < 0.3 && fabs((*jetIter).eta()) <= jetEtaMax_ && (*jetIter).pt() >= jetEtMin_ ){
	  NL1Off++;
	  v->getOffEtL1OffHisto()->Fill((*jetIter).pt());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill((*jetIter).eta(),(*jetIter).phi());}
         }}

	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if ((abs(p.pdgId()) == 21 || (abs(p.pdgId()) <= 5 && abs(p.pdgId())>=1))&& p.status() == 3 && fabs(p.eta()) <= jetEtaMax_ && p.pt() >= jetEtMin_ ){ 
	   if (reco::deltaR(p.eta(),p.phi(),(*l1JetIter).eta(),(*l1JetIter).phi()) < 0.3){
	    NL1Mc++;
	    v->getMcEtL1McHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiL1McHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
        }


	  }
         }
	  }
	}

      // for bjet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerBJet)
	{ 
	  if (v->getPath().find("BTagIP") != std::string::npos && bTagIPHandle.isValid()){
          const reco::JetTagCollection & bTags = *(bTagIPHandle.product());
          for (size_t i = 0; i != bTags.size(); ++i) {
           edm::RefToBase<reco::Jet>  BRefJet=bTags[i].first;
 	   if (fabs(BRefJet->eta()) <= bjetEtaMax_ && BRefJet->pt() >= bjetEtMin_ ){
	      NOff++;
	      v->getOffEtOffHisto()->Fill(BRefJet->pt());
	      v->getOffEtaVsOffPhiOffHisto()->Fill(BRefJet->eta(),BRefJet->phi());
	   }
	  }
	 }

	  if (v->getPath().find("BTagMu") != std::string::npos && bTagMuHandle.isValid()){
          const reco::JetTagCollection & bTags = *(bTagMuHandle.product());
          for (size_t i = 0; i != bTags.size(); ++i) {
           edm::RefToBase<reco::Jet>  BRefJet=bTags[i].first;
 	   if (fabs(BRefJet->eta()) <= bjetEtaMax_ && BRefJet->pt() >= bjetEtMin_ ){
	      NOff++;
	      v->getOffEtOffHisto()->Fill(BRefJet->pt());
	      v->getOffEtaVsOffPhiOffHisto()->Fill(BRefJet->eta(),BRefJet->phi());
	   }
	  }
	 }

	

	 if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 5 && p.status() == 3 && fabs(p.eta()) <= bjetEtaMax_ && p.pt() >= bjetEtMin_){
            NMc++; 
	    v->getMcEtMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiMcHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }

	 if (l1accept)
	   {


         for (l1JetHandle=l1JetHandleList.begin(); l1JetHandle!=l1JetHandleList.end(); l1JetHandle++) {
	   if (!l1JetHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTriggerOffline") << "l1JetHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1JetCollection = *(l1JetHandle->product());
	   for (L1JetParticleCollection::const_iterator l1JetIter=l1JetCollection.begin(); l1JetIter!=l1JetCollection.end(); l1JetIter++){
	     if (fabs((*l1JetIter).eta()) <= bjetEtaMax_ && (*l1JetIter).pt() >= bjetEtMin_ ){
	  NL1++;
     	  v->getL1EtL1Histo()->Fill((*l1JetIter).pt());
     	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1JetIter).eta(),(*l1JetIter).phi());
	    }


	  if (v->getPath().find("BTagIP") != std::string::npos && bTagIPHandle.isValid()){
          const reco::JetTagCollection & bTags = *(bTagIPHandle.product());
          for (size_t i = 0; i != bTags.size(); ++i) {
           edm::RefToBase<reco::Jet>  BRefJet=bTags[i].first;
           
 	   if (fabs(BRefJet->eta()) <= bjetEtaMax_ && BRefJet->pt() >= bjetEtMin_ ){
	   if (reco::deltaR(BRefJet->eta(),BRefJet->phi(),(*l1JetIter).eta(),(*l1JetIter).phi()) < 0.3){
	      NL1Off++;
	      v->getOffEtL1OffHisto()->Fill(BRefJet->pt());
	      v->getOffEtaVsOffPhiL1OffHisto()->Fill(BRefJet->eta(),BRefJet->phi());
	    }
	   }
	  }
	 }


	  if (v->getPath().find("BTagMu") != std::string::npos && bTagMuHandle.isValid()){
          const reco::JetTagCollection & bTags = *(bTagMuHandle.product());
          for (size_t i = 0; i != bTags.size(); ++i) {
           edm::RefToBase<reco::Jet>  BRefJet=bTags[i].first;
           
 	   if (fabs(BRefJet->eta()) <= bjetEtaMax_ && BRefJet->pt() >= bjetEtMin_ ){
	   if (reco::deltaR(BRefJet->eta(),BRefJet->phi(),(*l1JetIter).eta(),(*l1JetIter).phi()) < 0.3){
	      NL1Off++;
	      v->getOffEtL1OffHisto()->Fill(BRefJet->pt());
	      v->getOffEtaVsOffPhiL1OffHisto()->Fill(BRefJet->eta(),BRefJet->phi());
	    }
	   }
	  }
	 }


	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 5 && p.status() == 3 && fabs(p.eta()) <= bjetEtaMax_ && p.pt() >= bjetEtMin_ ){ 
	   if (reco::deltaR(p.eta(),p.phi(),(*l1JetIter).eta(),(*l1JetIter).phi()) < 0.3){
	    NL1Mc++;
	    v->getMcEtL1McHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiL1McHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
	}       

	   }
	 }


	   
	   }

	}
      // for met triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerMET || triggertype == trigger::TriggerL1ETM)
	{

	if (genParticles.isValid()){
          double metx = 0.0; double mety = 0.0; 
          double met = 0.0; //double metphi = 0.0;
          for(size_t i = 0; i < genParticles->size(); ++ i) {
           const GenParticle & p = (*genParticles)[i];
          if ((abs(p.pdgId()) == 12 || abs(p.pdgId()) == 14 || abs(p.pdgId()) == 16 || abs(p.pdgId()) == 18 || abs(p.pdgId()) == 1000022 || abs(p.pdgId()) == 1000039) && p.status() == 3){ 
	    metx += p.pt()*cos(p.phi());
	    mety += p.pt()*sin(p.phi());
	  }
	 }
	    met = sqrt(metx*metx+mety*mety);
            if (met >= metEtMin_){
	    NMc++;
            v->getMcEtMcHisto()->Fill(met);
	    v->getMcEtaVsMcPhiMcHisto()->Fill(0.0,atan2(mety,metx));
	    }
	}

	  if (metHandle.isValid()){
         const reco::CaloMETCollection metCollection = *(metHandle.product());
         for (reco::CaloMETCollection::const_iterator metIter=metCollection.begin(); metIter!=metCollection.end(); metIter++)
         {
	   if ((*metIter).pt() >= metEtMin_){
	  NOff++;
	  v->getOffEtOffHisto()->Fill((*metIter).pt());
	  v->getOffEtaVsOffPhiOffHisto()->Fill((*metIter).eta(),(*metIter).phi());
	   }
         }
         }

	 if (l1accept){

         for (l1extra::L1EtMissParticleCollection::const_iterator l1MetIter=l1MetCollection.begin(); l1MetIter!=l1MetCollection.end(); l1MetIter++)
         {
	   if ((*l1MetIter).pt() >= metEtMin_){
	  NL1++;
	  v->getL1EtL1Histo()->Fill((*l1MetIter).pt());
	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1MetIter).eta(),(*l1MetIter).phi());
	   }

	  if (metHandle.isValid()){
         const reco::CaloMETCollection metCollection = *(metHandle.product());
         for (reco::CaloMETCollection::const_iterator metIter=metCollection.begin(); metIter!=metCollection.end(); metIter++)
         {
	   if (reco::deltaR((*metIter).eta(),(*metIter).phi(),(*l1MetIter).eta(),(*l1MetIter).phi()) < 0.3 && (*metIter).pt() >= metEtMin_){
	  NL1Off++;
	  v->getOffEtL1OffHisto()->Fill((*metIter).pt());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill((*metIter).eta(),(*metIter).phi());}
         }}

	if (genParticles.isValid()){
          double metx = 0.0; double mety = 0.0; 
          double met = 0.0; //double metphi = 0.0;
          for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if ((abs(p.pdgId()) == 12 || abs(p.pdgId()) == 14 || abs(p.pdgId()) == 16 || abs(p.pdgId()) == 18 || abs(p.pdgId()) == 1000022 || abs(p.pdgId()) == 1000039) && p.status() == 3){ 
	    metx += p.pt()*cos(p.phi());
	    mety += p.pt()*sin(p.phi());
	  }
	 }
	 met = sqrt(metx*metx+mety*mety);
	 if (met >= metEtMin_){
	 NL1Mc++;
	 v->getMcEtL1McHisto()->Fill(met);
	 v->getMcEtaVsMcPhiL1McHisto()->Fill(0.0,atan2(mety,metx));
	 }
	}


	 }
	 }
	}


      // for photon triggers, loop over and fill offline and L1 4-vectors
      else if (triggertype == trigger::TriggerPhoton)
	{

	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 22 && p.status() == 3 && fabs(p.eta()) <= photonEtaMax_ && p.pt() >= photonEtMin_){
            NMc++; 
	    v->getMcEtMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiMcHisto()->Fill(p.eta(),p.phi());
	  }
	 }
	}

	  if (photonHandle.isValid()){
          const reco::PhotonCollection photonCollection = *(photonHandle.product());
         for (reco::PhotonCollection::const_iterator photonIter=photonCollection.begin(); photonIter!=photonCollection.end(); photonIter++)
         {
	   if (fabs((*photonIter).eta()) <= photonEtaMax_ && (*photonIter).pt() >= photonEtMin_ ){
	  NOff++;
	  v->getOffEtOffHisto()->Fill((*photonIter).pt());
	  v->getOffEtaVsOffPhiOffHisto()->Fill((*photonIter).eta(),(*photonIter).phi());
	   }
         }
	  }

	  if (l1accept){

         for (l1PhotonHandle=l1PhotonHandleList.begin(); l1PhotonHandle!=l1PhotonHandleList.end(); l1PhotonHandle++) {
	   if (!l1PhotonHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTriggerOffline") << "photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1EmParticleCollection l1PhotonCollection = *(l1PhotonHandle->product());
	   for (L1EmParticleCollection::const_iterator l1PhotonIter=l1PhotonCollection.begin(); l1PhotonIter!=l1PhotonCollection.end(); l1PhotonIter++){
	     if ((*l1PhotonIter).eta() <= photonEtaMax_ && (*l1PhotonIter).pt() >= photonEtMin_ ){
	  NL1++;
     	  v->getL1EtL1Histo()->Fill((*l1PhotonIter).pt());
     	  v->getL1EtaVsL1PhiL1Histo()->Fill((*l1PhotonIter).eta(),(*l1PhotonIter).phi());
	     }

	  if (photonHandle.isValid()){
          const reco::PhotonCollection photonCollection = *(photonHandle.product());
         for (reco::PhotonCollection::const_iterator photonIter=photonCollection.begin(); photonIter!=photonCollection.end(); photonIter++)
         {
	   if (reco::deltaR((*photonIter).eta(),(*photonIter).phi(),(*l1PhotonIter).eta(),(*l1PhotonIter).phi()) < 0.3 && fabs((*photonIter).eta()) <= photonEtaMax_ && (*photonIter).pt() >= photonEtMin_ ){
	  NL1Off++;
	  v->getOffEtL1OffHisto()->Fill((*photonIter).pt());
	  v->getOffEtaVsOffPhiL1OffHisto()->Fill((*photonIter).eta(),(*photonIter).phi());}
         }}

	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 22 && p.status() == 3 && fabs(p.eta()) <= photonEtaMax_ && p.pt() >= photonEtMin_ ){ 
	   if (reco::deltaR(p.eta(),p.phi(),(*l1PhotonIter).eta(),(*l1PhotonIter).phi()) < 0.3){
	    NL1Mc++;
	    v->getMcEtL1McHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiL1McHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
        }

	   }
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
 
      if (!l1accept) {
            edm::LogInfo("FourVectorHLTriggerOffline") << "l1 seed path not accepted for hlt path "<< v->getPath() << "\t" << v->getl1Path();
      }
    // ok plot On, L1On, OnOff, and OnMc objects

    // fill scaler histograms
      edm::InputTag filterTag = v->getTag();

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
	

      const int index = triggerObj->filterIndex(filterTag);
      if ( index >= triggerObj->sizeFilters() ) {
	//        cout << "WTF no index "<< index << " of that name "
	//	     << filterTag << endl;
	continue; // not in this event
      }
      LogDebug("FourVectorHLTriggerOffline") << "filling ... " ;
      const trigger::Keys & k = triggerObj->filterKeys(index);
      //      const trigger::Vids & idtype = triggerObj->filterIds(index);
      // assume for now the first object type is the same as all objects in the collection
      //    cout << filterTag << "\t" << idtype.size() << "\t" << k.size() << endl;
      //     cout << "path " << v->getPath() << " trigger type "<<triggertype << endl;
      //if (k.size() > 0) v->getNOnHisto()->Fill(k.size());
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {

        double tocEtaMax = 2.5;
        double tocEtMin = 3.0;
        if (triggertype == trigger::TriggerMuon || triggertype == trigger::TriggerL1Mu) 
	  {
	    tocEtaMax = muonEtaMax_; tocEtMin = muonEtMin_;
	  }
        else if (triggertype == trigger::TriggerElectron || triggertype == trigger::TriggerL1NoIsoEG || triggertype == trigger::TriggerL1IsoEG )
	  {
	    tocEtaMax = electronEtaMax_; tocEtMin = electronEtMin_;
	  }
        else if (triggertype == trigger::TriggerTau || triggertype == trigger::TriggerL1TauJet )
	  {
	    tocEtaMax = tauEtaMax_; tocEtMin = tauEtMin_;
	  }
        else if (triggertype == trigger::TriggerJet || triggertype == trigger::TriggerL1CenJet || triggertype == trigger::TriggerL1ForJet )
	  {
	    tocEtaMax = jetEtaMax_; tocEtMin = jetEtMin_;
	  }
        else if (triggertype == trigger::TriggerBJet)
	  {
	    tocEtaMax = bjetEtaMax_; tocEtMin = bjetEtMin_;
	  }
        else if (triggertype == trigger::TriggerMET || triggertype == trigger::TriggerL1ETM )
	  {
	    tocEtaMax = 999.0; tocEtMin = metEtMin_;
	  }
        else if (triggertype == trigger::TriggerPhoton)
	  {
	    tocEtaMax = photonEtaMax_; tocEtMin = photonEtMin_;
	  }

        if (fabs(toc[*ki].eta()) <= tocEtaMax && toc[*ki].pt() >= tocEtMin)
	  {
	NOn++;    
        v->getOnEtOnHisto()->Fill(toc[*ki].pt());
	v->getOnEtaVsOnPhiOnHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
	  }
	//	  cout << "pdgId "<<toc[*ki].id() << endl;
      // for muon triggers, loop over and fill offline 4-vectors
      if (triggertype == trigger::TriggerMuon || triggertype == trigger::TriggerL1Mu)
	{

	  if (muonHandle.isValid()){
         const reco::MuonCollection muonCollection = *(muonHandle.product());
         for (reco::MuonCollection::const_iterator muonIter=muonCollection.begin(); muonIter!=muonCollection.end(); muonIter++)
         {
	   if (reco::deltaR((*muonIter).eta(),(*muonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*muonIter).eta())<= muonEtaMax_ && (*muonIter).pt() >= muonEtMin_){
	  NOnOff++;
	  v->getOffEtOnOffHisto()->Fill((*muonIter).pt());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill((*muonIter).eta(),(*muonIter).phi());
	   }
         }

}

         for (l1extra::L1MuonParticleCollection::const_iterator l1MuonIter=l1MuonCollection.begin(); l1MuonIter!=l1MuonCollection.end(); l1MuonIter++)
         {
	   if (reco::deltaR((*l1MuonIter).eta(),(*l1MuonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*l1MuonIter).eta()) <= muonEtaMax_ && (*l1MuonIter).pt() >= muonEtMin_ ){
	  NL1On++;
	  v->getL1EtL1OnHisto()->Fill((*l1MuonIter).pt());
	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1MuonIter).eta(),(*l1MuonIter).phi());
	   }
         }

	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 13 && p.status() == 3 && fabs(p.eta()) <= muonEtaMax_ && p.pt() >= muonEtMin_){ 
	   if (reco::deltaR(p.eta(),p.phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	    NOnMc++;
	    v->getMcEtOnMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiOnMcHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
	}


      }

      // for electron triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerElectron || triggertype == trigger::TriggerL1NoIsoEG || triggertype == trigger::TriggerL1IsoEG )
	{
	  //	  std::cout << "Electron trigger" << std::endl;

	  if (gsfElectrons.isValid()){
         for (reco::PixelMatchGsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin(); gsfIter!=gsfElectrons->end(); gsfIter++)
         {
	   if (reco::deltaR((*gsfIter).eta(),(*gsfIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*gsfIter).eta()) <= electronEtaMax_ && (*gsfIter).pt() >= electronEtMin_ ){
	  NOnOff++;
	  v->getOffEtOnOffHisto()->Fill(gsfIter->pt());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill(gsfIter->eta(), gsfIter->phi());
	   }
         }}

         for (l1ElectronHandle=l1ElectronHandleList.begin(); l1ElectronHandle!=l1ElectronHandleList.end(); l1ElectronHandle++) {

         const L1EmParticleCollection l1ElectronCollection = *(l1ElectronHandle->product());
	   for (L1EmParticleCollection::const_iterator l1ElectronIter=l1ElectronCollection.begin(); l1ElectronIter!=l1ElectronCollection.end(); l1ElectronIter++){
	   if (reco::deltaR((*l1ElectronIter).eta(),(*l1ElectronIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*l1ElectronIter).eta()) <= electronEtaMax_ && (*l1ElectronIter).pt() >= electronEtMin_ ){
	  NL1On++;
     	  v->getL1EtL1OnHisto()->Fill((*l1ElectronIter).pt());
     	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1ElectronIter).eta(),(*l1ElectronIter).phi());
	   }
	   }
         }


	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 11 && p.status() == 3 && fabs(p.eta()) <= electronEtaMax_ && p.pt() >= electronEtMin_ ){ 
	   if (reco::deltaR(p.eta(),p.phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	    NOnMc++;
	    v->getMcEtOnMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiOnMcHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
	}

      }


      // for tau triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerTau || triggertype == trigger::TriggerL1TauJet)
	{

	  if (tauHandle.isValid()){
	    const reco::CaloTauCollection tauCollection = *(tauHandle.product());
         for (reco::CaloTauCollection::const_iterator tauIter=tauCollection.begin(); tauIter!=tauCollection.end(); tauIter++)
         {
	   if (reco::deltaR((*tauIter).eta(),(*tauIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*tauIter).eta()) <= tauEtaMax_ && (*tauIter).pt() >= tauEtMin_ ){
	  NOnOff++;
	  v->getOffEtOnOffHisto()->Fill((*tauIter).pt());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill((*tauIter).eta(),(*tauIter).phi());
	   }
         }}


         for (l1TauHandle=l1TauHandleList.begin(); l1TauHandle!=l1TauHandleList.end(); l1TauHandle++) {
	   if (!l1TauHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTriggerOffline") << "photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1TauCollection = *(l1TauHandle->product());
	   for (L1JetParticleCollection::const_iterator l1TauIter=l1TauCollection.begin(); l1TauIter!=l1TauCollection.end(); l1TauIter++){
	   if (reco::deltaR((*l1TauIter).eta(),(*l1TauIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*l1TauIter).eta()) <= tauEtaMax_ && (*l1TauIter).pt() >= tauEtMin_ ){
	  NL1On++;
     	  v->getL1EtL1OnHisto()->Fill((*l1TauIter).pt());
     	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1TauIter).eta(),(*l1TauIter).phi());
	   }
	  }
         }


	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 15 && p.status() == 3 && fabs(p.eta()) <= tauEtaMax_ && p.pt() >= tauEtMin_ ){ 
	   if (reco::deltaR(p.eta(),p.phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 ){
	    NOnMc++;
	    v->getMcEtOnMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiOnMcHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
	}

      }


      // for jet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerJet || triggertype == trigger::TriggerL1CenJet || triggertype == trigger::TriggerL1ForJet )
	{

	  if (jetHandle.isValid()){
         const reco::CaloJetCollection jetCollection = *(jetHandle.product());
         for (reco::CaloJetCollection::const_iterator jetIter=jetCollection.begin(); jetIter!=jetCollection.end(); jetIter++)
         {
	   if (reco::deltaR((*jetIter).eta(),(*jetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*jetIter).eta()) <= jetEtaMax_ && (*jetIter).pt() >= jetEtMin_ ){
	  NOnOff++;
	  v->getOffEtOnOffHisto()->Fill((*jetIter).pt());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill((*jetIter).eta(),(*jetIter).phi());
	   }
         }}

         for (l1JetHandle=l1JetHandleList.begin(); l1JetHandle!=l1JetHandleList.end(); l1JetHandle++) {
	   if (!l1JetHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTriggerOffline") << "l1JetHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1JetCollection = *(l1JetHandle->product());
	   for (L1JetParticleCollection::const_iterator l1JetIter=l1JetCollection.begin(); l1JetIter!=l1JetCollection.end(); l1JetIter++){
	   if (reco::deltaR((*l1JetIter).eta(),(*l1JetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*l1JetIter).eta()) <= jetEtaMax_ && (*l1JetIter).pt() >= jetEtMin_ ){
	  NL1On++;
     	  v->getL1EtL1OnHisto()->Fill((*l1JetIter).pt());
     	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1JetIter).eta(),(*l1JetIter).phi());
	   }
	   }
         }


	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if ((abs(p.pdgId()) == 21 ||(abs(p.pdgId()) <=5 && abs(p.pdgId()) >=1)) && p.status() == 3 && fabs(p.eta()) <= jetEtaMax_ && p.pt() >= jetEtMin_ ){ 
	   if (reco::deltaR(p.eta(),p.phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	    NOnMc++;
	    v->getMcEtOnMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiOnMcHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
	}

      }

      // for bjet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerBJet)
	{

	  if (v->getPath().find("BTagIP") != std::string::npos && bTagIPHandle.isValid()){
          const reco::JetTagCollection & bTags = *(bTagIPHandle.product());
          for (size_t i = 0; i != bTags.size(); ++i) {
           edm::RefToBase<reco::Jet>  BRefJet=bTags[i].first;
           
 	   if (fabs(BRefJet->eta()) <= bjetEtaMax_ && BRefJet->pt() >= bjetEtMin_ ){
	   if (reco::deltaR(BRefJet->eta(),BRefJet->phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	      NOnOff++;
	      v->getOffEtOnOffHisto()->Fill(BRefJet->pt());
	      v->getOffEtaVsOffPhiOnOffHisto()->Fill(BRefJet->eta(),BRefJet->phi());
	    }
	   }
	  }
	}


	  if (v->getPath().find("BTagMu") != std::string::npos && bTagMuHandle.isValid()){
          const reco::JetTagCollection & bTags = *(bTagMuHandle.product());
          for (size_t i = 0; i != bTags.size(); ++i) {
           edm::RefToBase<reco::Jet>  BRefJet=bTags[i].first;
           
 	   if (fabs(BRefJet->eta()) <= bjetEtaMax_ && BRefJet->pt() >= bjetEtMin_ ){
	   if (reco::deltaR(BRefJet->eta(),BRefJet->phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	      NOnOff++;
	      v->getOffEtOnOffHisto()->Fill(BRefJet->pt());
	      v->getOffEtaVsOffPhiOnOffHisto()->Fill(BRefJet->eta(),BRefJet->phi());
	    }
	   }
	  }
	}

         for (l1JetHandle=l1JetHandleList.begin(); l1JetHandle!=l1JetHandleList.end(); l1JetHandle++) {
	   if (!l1JetHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTriggerOffline") << "l1JetHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1JetCollection = *(l1JetHandle->product());
	   for (L1JetParticleCollection::const_iterator l1JetIter=l1JetCollection.begin(); l1JetIter!=l1JetCollection.end(); l1JetIter++){
	   if (reco::deltaR((*l1JetIter).eta(),(*l1JetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*l1JetIter).eta()) <= bjetEtaMax_ && (*l1JetIter).pt() >= bjetEtMin_ ){
	  NL1On++;
     	  v->getL1EtL1OnHisto()->Fill((*l1JetIter).pt());
     	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1JetIter).eta(),(*l1JetIter).phi());
	   }
	  }
         }


	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 5 && p.status() == 3 && fabs(p.eta()) <= bjetEtaMax_ && p.pt() >= bjetEtMin_ ){ 
	   if (reco::deltaR(p.eta(),p.phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	    NOnMc++;
	    v->getMcEtOnMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiOnMcHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
	}

	}
      // for met triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerMET || triggertype == trigger::TriggerL1ETM )
	{

	  if (metHandle.isValid()){
         const reco::CaloMETCollection metCollection = *(metHandle.product());
         for (reco::CaloMETCollection::const_iterator metIter=metCollection.begin(); metIter!=metCollection.end(); metIter++)
         {
	   if (reco::deltaR((*metIter).eta(),(*metIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && (*metIter).pt() >= metEtMin_ ){
	  NOnOff++;
	  v->getOffEtOnOffHisto()->Fill((*metIter).pt());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill((*metIter).eta(),(*metIter).phi());
	   }
         }}

         for (l1extra::L1EtMissParticleCollection::const_iterator l1MetIter=l1MetCollection.begin(); l1MetIter!=l1MetCollection.end(); l1MetIter++)
         {
	   if (reco::deltaR((*l1MetIter).eta(),(*l1MetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && (*l1MetIter).pt() >= metEtMin_ ){
	  NL1On++;
	  v->getL1EtL1OnHisto()->Fill((*l1MetIter).pt());
	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1MetIter).eta(),(*l1MetIter).phi());
	   }
         }

	if (genParticles.isValid()){
          double metx = 0.0; double mety = 0.0; 
          double met = 0.0; //double metphi = 0.0;
          for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if ((abs(p.pdgId()) == 12 || abs(p.pdgId()) == 14 || abs(p.pdgId()) == 16 || abs(p.pdgId()) == 18 || abs(p.pdgId()) == 1000022 || abs(p.pdgId()) == 1000039) && p.status() == 3){ 
	    metx += p.pt()*cos(p.phi());
	    mety += p.pt()*sin(p.phi());
	  }
	 }
	 met = sqrt(metx*metx+mety*mety);
         if (met > metEtMin_){
	 NOnMc++;  
	 v->getMcEtOnMcHisto()->Fill(met);
	 v->getMcEtaVsMcPhiOnMcHisto()->Fill(0.0,atan2(mety,metx));
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
	   if (reco::deltaR((*photonIter).eta(),(*photonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*photonIter).eta()) <= photonEtaMax_ && (*photonIter).pt() >= photonEtMin_ ){
	  NOnOff++;
	  v->getOffEtOnOffHisto()->Fill((*photonIter).pt());
	  v->getOffEtaVsOffPhiOnOffHisto()->Fill((*photonIter).eta(),(*photonIter).phi());
	   }
         }}


         for (l1PhotonHandle=l1PhotonHandleList.begin(); l1PhotonHandle!=l1PhotonHandleList.end(); l1PhotonHandle++) {
	   if (!l1PhotonHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTriggerOffline") << "l1photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1EmParticleCollection l1PhotonCollection = *(l1PhotonHandle->product());
	   for (L1EmParticleCollection::const_iterator l1PhotonIter=l1PhotonCollection.begin(); l1PhotonIter!=l1PhotonCollection.end(); l1PhotonIter++){
	   if (reco::deltaR((*l1PhotonIter).eta(),(*l1PhotonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && fabs((*l1PhotonIter).eta()) <= photonEtaMax_ && (*l1PhotonIter).pt() >= photonEtMin_ ){
	  NL1On++;
     	  v->getL1EtL1OnHisto()->Fill((*l1PhotonIter).pt());
     	  v->getL1EtaVsL1PhiL1OnHisto()->Fill((*l1PhotonIter).eta(),(*l1PhotonIter).phi());
	   }
	   

	 }
	 }

	if (genParticles.isValid()){
           for(size_t i = 0; i < genParticles->size(); ++ i) {
          const GenParticle & p = (*genParticles)[i];
          if (abs(p.pdgId()) == 22 && p.status() == 3 && fabs(p.eta()) <= photonEtaMax_ && p.pt() >= photonEtMin_ ){ 
	   if (reco::deltaR(p.eta(),p.phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	    NOnMc++;
	    v->getMcEtOnMcHisto()->Fill(p.pt());
	    v->getMcEtaVsMcPhiOnMcHisto()->Fill(p.eta(),p.phi());
	   }
	  }
	 }
	}
       

	}// photon trigger type

      } //online object loop

      v->getNOnHisto()->Fill(NOn);      
      v->getNL1OnHisto()->Fill(NL1On);      
      v->getNOnOffHisto()->Fill(NOnOff);
      v->getNOnMcHisto()->Fill(NOnMc);
  

    } //numpassed
    
      v->getNMcHisto()->Fill(NMc);      
      v->getNOffHisto()->Fill(NOff);      
      v->getNL1Histo()->Fill(NL1);
      v->getNL1OffHisto()->Fill(NL1Off);
      v->getNL1McHisto()->Fill(NL1Mc);

    } //denompassed
  } //pathinfo loop

}



// -- method called once each job just before starting event loop  --------
void 
FourVectorHLTriggerOffline::beginJob(const edm::EventSetup&)
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
FourVectorHLTriggerOffline::endJob() 
{
   LogInfo("FourVectorHLTriggerOffline") << "analyzed " << nev_ << " events";
   return;
}


// BeginRun
void FourVectorHLTriggerOffline::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTriggerOffline") << "beginRun, run " << run.id();
// HLT config does not change within runs!
 
  if (!hltConfig_.init(processname_)) {
  LogDebug("FourVectorHLTriggerOffline") << "HLTConfigProvider failed to initialize.";
    // check if trigger name in (new) config
    //	cout << "Available TriggerNames are: " << endl;
    //	hltConfig_.dump("Triggers");
      }


  if (1)
 {
  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
  }


    const unsigned int n(hltConfig_.size());
    if (plotAll_){
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
    if (pathname.find("BTag") != std::string::npos) 
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
    if (denompathname.find("BTag") != std::string::npos) 
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
    // list module labels for numpath
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);

            for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
    	  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      //  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
	      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
		{
		  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
		  //                  cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
                  l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression"); 
                  break; 
		}
    	} 
   
    



    std::string filtername("dummy");
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (pathname.find("HLT_") != std::string::npos && plotAll_ && denomobjectType == objectType && objectType != 0)
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));

    }
    }

    }
    else
    {
     // plot all diagonal combinations plus any other specified pairs
    for (unsigned int i=0; i!=n; ++i) {
      std::string denompathname = hltConfig_.triggerName(i);  
      std::string pathname = hltConfig_.triggerName(i);  
      std::string l1pathname = "dummy";
      int objectType = 0;
      int denomobjectType = 0;
    //parse pathname to guess object type
    if (pathname.find("Jet") != std::string::npos) 
      objectType = trigger::TriggerJet;    
    if (pathname.find("BTag") != std::string::npos) 
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
    if (denompathname.find("BTag") != std::string::npos) 
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
    // list module labels for numpath
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);

    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
    	  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      //  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
	      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
		{
		  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
		  //                  cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
                  l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression"); 
                  break; 
		}
    } 
   
    



    std::string filtername("dummy");
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (pathname.find("HLT_") != std::string::npos && objectType != 0)
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));

    
    }
    // now loop over denom/num path pairs specified in cfg, 
    // recording the off-diagonal ones
    for (std::vector<std::pair<std::string, std::string> >::iterator custompathnamepair = custompathnamepairs_.begin(); custompathnamepair != custompathnamepairs_.end(); ++custompathnamepair)
    {
      if (custompathnamepair->first != custompathnamepair->second)
	{

      std::string denompathname = custompathnamepair->second;  
      std::string pathname = custompathnamepair->first;  
     
      // check that these exist
      bool foundfirst = false;
      bool foundsecond = false;
      for (unsigned int i=0; i!=n; ++i) {
	if (hltConfig_.triggerName(i) == denompathname) foundsecond = true;
	if (hltConfig_.triggerName(i) == pathname) foundfirst = true;
      } 
      if (!foundfirst)
	{
	  edm::LogInfo("FourVectorHLTriggerOffline") << "pathname not found, ignoring " << pathname;
          continue;
	}
      if (!foundsecond)
	{
	  edm::LogInfo("FourVectorHLTriggerOffline") << "denompathname not found, ignoring " << pathname;
          continue;
	}

     //cout << pathname << "\t" << denompathname << endl;
      std::string l1pathname = "dummy";
      int objectType = 0;
      int denomobjectType = 0;
    //parse pathname to guess object type
    if (pathname.find("Jet") != std::string::npos) 
      objectType = trigger::TriggerJet;    
    if (pathname.find("BTag") != std::string::npos) 
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

    // find L1 condition for numpath with numpath objecttype 

    // find PSet for L1 global seed for numpath, 
    // list module labels for numpath
  
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
    
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
    	  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      //  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
	      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
		{
		  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
		  //                  cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
                  l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression"); 
                  break; 
		}
    }
    
    



    std::string filtername("dummy");
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (pathname.find("HLT_") != std::string::npos && objectType != 0)
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));
    
	}
    }

    }



    // now set up all of the histos for each path
    for(PathInfoCollection::iterator v = hltPaths_.begin();
	  v!= hltPaths_.end(); ++v ) {
    	MonitorElement *NOn, *onEtOn, *onEtavsonPhiOn=0;
    	MonitorElement *NMc, *mcEtMc, *mcEtavsmcPhiMc=0;
	MonitorElement *NOff, *offEtOff, *offEtavsoffPhiOff=0;
	MonitorElement *NL1, *l1EtL1, *l1Etavsl1PhiL1=0;
    	MonitorElement *NL1On, *l1EtL1On, *l1Etavsl1PhiL1On=0;
	MonitorElement *NL1Off, *offEtL1Off, *offEtavsoffPhiL1Off=0;
	MonitorElement *NOnOff, *offEtOnOff, *offEtavsoffPhiOnOff=0;
	MonitorElement *NL1Mc, *mcEtL1Mc, *mcEtavsmcPhiL1Mc=0;
	MonitorElement *NOnMc, *mcEtOnMc, *mcEtavsmcPhiOnMc=0;
	std::string labelname("dummy");
        labelname = v->getPath() + "_wrt_" + v->getDenomPath();
	std::string histoname(labelname+"_NOn");
	std::string title(labelname+" N online");



        double histEtaMax = 2.5;
        if (v->getObjectType() == trigger::TriggerMuon || v->getObjectType() == trigger::TriggerL1Mu) 
	  {
	    histEtaMax = muonEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerElectron || v->getObjectType() == trigger::TriggerL1NoIsoEG || v->getObjectType() == trigger::TriggerL1IsoEG )
	  {
	    histEtaMax = electronEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerTau || v->getObjectType() == trigger::TriggerL1TauJet )
	  {
	    histEtaMax = tauEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerJet || v->getObjectType() == trigger::TriggerL1CenJet || v->getObjectType() == trigger::TriggerL1ForJet )
	  {
	    histEtaMax = jetEtaMax_; 
	  }
        else if (v->getObjectType() == trigger::TriggerBJet)
	  {
	    histEtaMax = bjetEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerMET || v->getObjectType() == trigger::TriggerL1ETM )
	  {
	    histEtaMax = 5.0; 
	  }
        else if (v->getObjectType() == trigger::TriggerPhoton)
	  {
	    histEtaMax = photonEtaMax_; 
	  }

	NOn =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NMc";
	title = labelname+" N Mc";
	NMc =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NOff";
	title = labelname+" N Off";
	NOff =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);
      
	histoname = labelname+"_NL1";
	title = labelname+" N L1";
	NL1 =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1On";
	title = labelname+" N L1On";
	NL1On =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1Off";
	title = labelname+" N L1Off";
	NL1Off =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NOnOff";
	title = labelname+" N OnOff";
	NOnOff =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1Mc";
	title = labelname+" N L1Mc";
	NL1Mc =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NOnMc";
	title = labelname+" N OnMc";
	NOnMc =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_mcEtMc";
	title = labelname+" mcE_t Mc";
	mcEtMc =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

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

        int nBins2D = 10;

	histoname = labelname+"_mcEtamcPhiMc";
	title = labelname+" mc#eta vs mc#phi Mc";
	mcEtavsmcPhiMc =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_onEtaonPhiOn";
	title = labelname+" on#eta vs on#phi online";
	onEtavsonPhiOn =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_offEtaoffPhiOff";
	title = labelname+" off#eta vs off#phi offline";
	offEtavsoffPhiOff =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_l1Etal1PhiL1";
	title = labelname+" l1#eta vs l1#phi L1";
	l1Etavsl1PhiL1 =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

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

	histoname = labelname+"_mcEtL1Mc";
	title = labelname+" mcE_t L1+MC truth";
	mcEtL1Mc =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_mcEtOnMc";
	title = labelname+" mcE_t online+MC truth";
	mcEtOnMc =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_l1Etal1PhiL1On";
	title = labelname+" l1#eta vs l1#phi L1+online";
	l1Etavsl1PhiL1On =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_offEtaoffPhiL1Off";
	title = labelname+" off#eta vs off#phi L1+offline";
	offEtavsoffPhiL1Off =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_offEtaoffPhiOnOff";
	title = labelname+" off#eta vs off#phi online+offline";
	offEtavsoffPhiOnOff =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_mcEtamcPhiL1Mc";
	title = labelname+" mc#eta vs mc#phi L1+MC truth";
	mcEtavsmcPhiL1Mc =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_mcEtamcPhiOnMc";
	title = labelname+" mc#eta vs mc#phi online+MC truth";
	mcEtavsmcPhiOnMc =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	v->setHistos( NMc, mcEtMc, mcEtavsmcPhiMc, NOn, onEtOn, onEtavsonPhiOn, NOff, offEtOff, offEtavsoffPhiOff, NL1, l1EtL1, l1Etavsl1PhiL1, NL1On, l1EtL1On, l1Etavsl1PhiL1On, NL1Off, offEtL1Off, offEtavsoffPhiL1Off, NOnOff, offEtOnOff, offEtavsoffPhiOnOff, NL1Mc, mcEtL1Mc, mcEtavsmcPhiL1Mc, NOnMc, mcEtOnMc, mcEtavsmcPhiOnMc);


    }
 }
 return;



}

/// EndRun
void FourVectorHLTriggerOffline::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTriggerOffline") << "endRun, run " << run.id();
}

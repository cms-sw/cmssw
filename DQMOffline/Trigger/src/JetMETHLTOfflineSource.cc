/*
  New version of HLT Offline DQM code for JetMET
  responsible: Sunil Bansal, Shabnam jabeen 

*/

#include "TMath.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/Trigger/interface/JetMETHLTOfflineSource.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "math.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TH2F.h"
#include "TPRegexp.h"

using namespace edm;
using namespace reco;
using namespace std;

  
JetMETHLTOfflineSource::JetMETHLTOfflineSource(const edm::ParameterSet& iConfig):
  isSetup_(false)
{
  LogDebug("JetMETHLTOfflineSource") << "constructor....";

  dbe = Service < DQMStore > ().operator->();
  if ( ! dbe ) {
    LogDebug("JetMETHLTOfflineSource") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe->setVerbose(0);
  }
  
  dirname_ = iConfig.getUntrackedParameter("dirname",
					   std::string("HLT/JetMET/"));
  
  
  processname_ = iConfig.getParameter<std::string>("processname");
  verbose_     = iConfig.getUntrackedParameter< bool >("verbose", false);
  plotAll_     = iConfig.getUntrackedParameter< bool >("plotAll", false);
  plotAllwrtMu_     = iConfig.getUntrackedParameter< bool >("plotAllwrtMu", false);
  plotEff_     = iConfig.getUntrackedParameter< bool >("plotEff", false);
  nameForEff_ =  iConfig.getUntrackedParameter< bool >("nameForEff", true);
  jetID = new reco::helper::JetIDHelper(iConfig.getParameter<ParameterSet>("JetIDParams"));
  
  // plotting paramters
  MuonTrigPaths_ = iConfig.getUntrackedParameter<vector<std::string> >("pathnameMuon");
  MBTrigPaths_ = iConfig.getUntrackedParameter<vector<std::string> >("pathnameMB");
  caloJetsTag_ = iConfig.getParameter<edm::InputTag>("CaloJetCollectionLabel");
  caloMETTag_ = iConfig.getParameter<edm::InputTag>("CaloMETCollectionLabel"); 
  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  custompathname = iConfig.getUntrackedParameter<vector<std::string> >("paths");
  _fEMF  = iConfig.getUntrackedParameter< double >("fEMF", 0.01);
  _feta  = iConfig.getUntrackedParameter< double >("feta", 2.60);
  _fHPD  = iConfig.getUntrackedParameter< double >("fHPD", 0.98);
  _n90Hits = iConfig.getUntrackedParameter< double >("n90Hits", 1.0);
  // this is the list of paths to look at.
  std::vector<edm::ParameterSet> paths =  iConfig.getParameter<std::vector<edm::ParameterSet> >("pathPairs");
  for(std::vector<edm::ParameterSet>::iterator pathconf = paths.begin() ; pathconf != paths.end();  pathconf++) {

    std::pair<std::string, std::string> custompathnamepair;
    custompathnamepair.first =pathconf->getParameter<std::string>("pathname"); 
    custompathnamepair.second = pathconf->getParameter<std::string>("denompathname");   
    custompathnamepairs_.push_back(custompathnamepair);
  } 


}


JetMETHLTOfflineSource::~JetMETHLTOfflineSource() {
 
 
  //
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


void
JetMETHLTOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm;
  using namespace trigger;
  using namespace reco;
  //---------- triggerResults ----------
  iEvent.getByLabel(triggerResultsLabel_, triggerResults_);
  if(!triggerResults_.isValid()) {
    edm::InputTag triggerResultsLabelFU(triggerResultsLabel_.label(),triggerResultsLabel_.instance(), "FU");
    iEvent.getByLabel(triggerResultsLabelFU,triggerResults_);
    if(!triggerResults_.isValid()) {
      edm::LogInfo("FourVectorHLTOffline") << "TriggerResults not found, "
	"skipping event";
      return;
    }
  }
  
  int npath;
  if(&triggerResults_) {
  
    // Check how many HLT triggers are in triggerResults
    npath = triggerResults_->size();
    triggerNames_ = iEvent.triggerNames(*triggerResults_);


  } else {
  
    edm::LogInfo("CaloMETHLTOfflineSource") << "TriggerResults::HLT not found, "
      "automatically select events";
    return;
    
  }
  //---------- triggerSummary ----------
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj_);
  if(!triggerObj_.isValid()) {
    edm::InputTag triggerSummaryLabelFU(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(), "FU");
    iEvent.getByLabel(triggerSummaryLabelFU,triggerObj_);
    if(!triggerObj_.isValid()) {
      edm::LogInfo("FourVectorHLTOffline") << "TriggerEvent not found, "
	"skipping event";
      return;
    }
  }
  //------------Offline Objects-------
  bool ValidJetColl_ = iEvent.getByLabel(caloJetsTag_,calojetColl_);
  if(!ValidJetColl_)return;
  calojet = *calojetColl_; 
  std::stable_sort( calojet.begin(), calojet.end(), PtSorter() );

  bool ValidMETColl_ = iEvent.getByLabel(caloMETTag_, calometColl_);
  if(!ValidMETColl_)return; 
 
  if(calometColl_.isValid()){
    const CaloMETCollection *calometcol = calometColl_.product();
    const CaloMET met = calometcol->front();
  }   
  fillMEforMonTriggerSummary();
  if(plotAll_)fillMEforMonAllTrigger(iEvent);
  if(plotAllwrtMu_)fillMEforMonAllTriggerwrtMuonTrigger(iEvent);
  if(plotEff_)
  {
    fillMEforEffAllTrigger(iEvent); 
    fillMEforEffWrtMuTrigger(iEvent);
    fillMEforEffWrtMBTrigger(iEvent);
  }
  fillMEforTriggerNTfired();
}


void JetMETHLTOfflineSource::fillMEforMonTriggerSummary(){
  // Trigger summary for all paths

  bool muTrig = false;
  bool mbTrig = false;
  for(size_t i=0;i<MuonTrigPaths_.size();++i){
    if(isHLTPathAccepted(MuonTrigPaths_[i])){
      muTrig = true;
      break;
    } 
  }
  for(size_t i=0;i<MBTrigPaths_.size();++i){
    if(isHLTPathAccepted(MBTrigPaths_[i])){
      mbTrig = true;
      break;
    }
  }
  for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v )
  {
    bool trigFirst= false;  
    double binV = TriggerPosition(v->getPath());       
    if(isHLTPathAccepted(v->getPath())) trigFirst = true;
    if(!trigFirst)continue;
    if(trigFirst)
    {
      rate_All->Fill(binV);
      correlation_All->Fill(binV,binV);
      if(muTrig){
	rate_AllWrtMu->Fill(binV);
	correlation_AllWrtMu->Fill(binV,binV);
      }
      if(mbTrig){
	rate_AllWrtMB->Fill(binV);
	correlation_AllWrtMB->Fill(binV,binV);
      }
    }
    for(PathInfoCollection::iterator w = v+1; w!= hltPathsAll_.end(); ++w )
    {
      bool trigSec = false; 
      double binW = TriggerPosition(w->getPath()); 
      if(isHLTPathAccepted(w->getPath()))trigSec = true;
      if(trigSec && trigFirst)
      {
	correlation_All->Fill(binV,binW);
	if(muTrig)correlation_AllWrtMu->Fill(binV,binW);
	if(mbTrig)correlation_AllWrtMB->Fill(binV,binW); 
      }
      if(!trigSec && trigFirst)
      {
	correlation_All->Fill(binW,binV); 
	if(muTrig)correlation_AllWrtMu->Fill(binW,binV);
	if(mbTrig)correlation_AllWrtMB->Fill(binW,binV);

      }
    }
  }
}

void JetMETHLTOfflineSource::fillMEforTriggerNTfired(){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v )
  {
    unsigned index = triggerNames_.triggerIndex(v->getPath()); 
    if (index < triggerNames_.size() )
    {
      v->getMEhisto_TriggerSummary()->Fill(0.);
      edm::InputTag l1Tag(v->getl1Path(),"",processname_);
      const int l1Index = triggerObj_->filterIndex(l1Tag);
      bool l1found = false;
      if ( l1Index < triggerObj_->sizeFilters() ) l1found = true;
      if(!l1found)v->getMEhisto_TriggerSummary()->Fill(1.);
      if(!l1found && !(triggerResults_->accept(index)))v->getMEhisto_TriggerSummary()->Fill(2.);
      if(!l1found && (triggerResults_->accept(index)))v->getMEhisto_TriggerSummary()->Fill(3.);
      if(l1found)v->getMEhisto_TriggerSummary()->Fill(4.);
      if(l1found && (triggerResults_->accept(index)))v->getMEhisto_TriggerSummary()->Fill(5.); 
      if(l1found && !(triggerResults_->accept(index)))v->getMEhisto_TriggerSummary()->Fill(6.);
      if(!(triggerResults_->accept(index)) && l1found)
      { 
             
	if((v->getTriggerType().compare("SingleJet_Trigger") == 0) && (calojetColl_.isValid()) && calojet.size())
              
	{
	  CaloJetCollection::const_iterator jet = calojet.begin();
	  v->getMEhisto_JetPt()->Fill(jet->pt());
	  v->getMEhisto_EtavsPt()->Fill(jet->eta(),jet->pt());
	  v->getMEhisto_PhivsPt()->Fill(jet->phi(),jet->pt());
                 
	}// single jet trigger is not fired

	if((v->getTriggerType().compare("DiJet_Trigger") == 0) && calojetColl_.isValid()  && calojet.size())
	{
	  v->getMEhisto_JetSize()->Fill(calojet.size()) ;
	  if (calojet.size()>=2){
	    CaloJetCollection::const_iterator jet = calojet.begin();
	    CaloJetCollection::const_iterator jet2= calojet.begin(); jet2++;
	    double jet3pt = 0.;
	    if(calojet.size()>2)   
	    {
	      CaloJetCollection::const_iterator jet3 = jet2++;
	      jet3pt = jet3->pt();
	    }
	    v->getMEhisto_Pt12()->Fill((jet->pt()+jet2->pt())/2.);
	    v->getMEhisto_Eta12()->Fill((jet->eta()+jet2->eta())/2.);
	    v->getMEhisto_Phi12()->Fill(deltaPhi(jet->phi(),jet2->phi()));
	    v->getMEhisto_Pt3()->Fill(jet3pt);
	    v->getMEhisto_Pt12Pt3()->Fill((jet->pt()+jet2->pt())/2., jet3pt);
	    v->getMEhisto_Pt12Phi12()->Fill((jet->pt()+jet2->pt())/2., deltaPhi(jet->phi(),jet2->phi()));

	  }
	}// di jet trigger is not fired 

	if((v->getTriggerType().compare("MET_Trigger") == 0) && calometColl_.isValid() )
	{
	  const CaloMETCollection *calometcol = calometColl_.product();
	  const CaloMET met = calometcol->front();
	  v->getMEhisto_JetPt()->Fill(met.pt());
	}//MET trigger is not fired   
      } // L1 is fired
    }//
  }// trigger not fired
 


}


void JetMETHLTOfflineSource::fillMEforMonAllTrigger(const Event & iEvent){
  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  //-----------------------------------------------------  
  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects()); 
  for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v )
  {
    if (isHLTPathAccepted(v->getPath()))
    {
      std::vector<double>jetPtVec;
      std::vector<double>jetPhiVec; 
      std::vector<double>jetEtaVec;
      std::vector<double>jetPxVec;
      std::vector<double>jetPyVec;
            
      std::vector<double>hltPtVec;
      std::vector<double>hltPhiVec;
      std::vector<double>hltEtaVec;
      std::vector<double>hltPxVec;
      std::vector<double>hltPyVec;
      bool fillL1HLT = false;  // This will be used to find out punch throgh trigger 
      //---------------------------------------------
      edm::InputTag l1Tag(v->getl1Path(),"",processname_);
      const int l1Index = triggerObj_->filterIndex(l1Tag);
      edm::InputTag hltTag(v->getLabel(),"",processname_);
      const int hltIndex = triggerObj_->filterIndex(hltTag);
      bool l1TrigBool = false;
      bool hltTrigBool = false;
      bool diJetFire = false;
      int jetsize = 0;
      if ( l1Index >= triggerObj_->sizeFilters() ) {
	edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< l1Index << " of that name "<<l1Tag;
      } else {
	l1TrigBool = true;
	const trigger::Keys & kl1 = triggerObj_->filterKeys(l1Index);
	if(v->getObjectType() == trigger::TriggerJet)v->getMEhisto_N_L1()->Fill(kl1.size());
	for( trigger::Keys::const_iterator ki = kl1.begin(); ki != kl1.end(); ++ki)
	{
	  double l1TrigEta = -100;
	  double l1TrigPhi = -100;
	  //-------------------------------------------
	  if(v->getObjectType() == trigger::TriggerJet)
	  { 
	    l1TrigEta = toc[*ki].eta();
	    l1TrigPhi = toc[*ki].phi();
	    v->getMEhisto_Pt_L1()->Fill(toc[*ki].pt());
	    if (isBarrel(toc[*ki].eta()))  v->getMEhisto_PtBarrel_L1()->Fill(toc[*ki].pt());
	    if (isEndCap(toc[*ki].eta()))  v->getMEhisto_PtEndcap_L1()->Fill(toc[*ki].pt());
	    if (isForward(toc[*ki].eta())) v->getMEhisto_PtForward_L1()->Fill(toc[*ki].pt());
	    v->getMEhisto_Eta_L1()->Fill(toc[*ki].eta());
	    v->getMEhisto_Phi_L1()->Fill(toc[*ki].phi());
	    v->getMEhisto_EtaPhi_L1()->Fill(toc[*ki].eta(),toc[*ki].phi());
	  }
	  if(v->getObjectType() == trigger::TriggerMET)
	  {
	    v->getMEhisto_Pt_L1()->Fill(toc[*ki].pt());
	    v->getMEhisto_Phi_L1()->Fill(toc[*ki].phi());
	  }
	  //-----------------------------------------------  
	  if ( hltIndex >= triggerObj_->sizeFilters() ) {
	    edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt"<< hltIndex << " of that name ";
	  } else {
	    const trigger::Keys & khlt = triggerObj_->filterKeys(hltIndex);
	    if((v->getObjectType() == trigger::TriggerJet) && (ki == kl1.begin()))v->getMEhisto_N_HLT()->Fill(khlt.size());

	    for(trigger::Keys::const_iterator kj = khlt.begin();kj != khlt.end(); ++kj)
	    {
	      double hltTrigEta = -100;
	      double hltTrigPhi = -100;
	      if(v->getObjectType() == trigger::TriggerJet)
	      {
		hltTrigEta = toc[*kj].eta();
		hltTrigPhi = toc[*kj].phi();
		if((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi)) < 0.4 && (v->getTriggerType().compare("DiJet_Trigger") == 0))hltTrigBool = true;
	      }    
	    }
	    for(trigger::Keys::const_iterator kj = khlt.begin();kj != khlt.end(); ++kj)
	    {
	      double hltTrigEta = -100;
	      double hltTrigPhi = -100;
	      fillL1HLT = true;
	      //--------------------------------------------------
	      if(v->getObjectType() == trigger::TriggerMET)
	      {
		v->getMEhisto_Pt_HLT()->Fill(toc[*kj].pt());
		v->getMEhisto_Phi_HLT()->Fill(toc[*kj].phi());
		v->getMEhisto_PtCorrelation_L1HLT()->Fill(toc[*ki].pt(),toc[*kj].pt());
		v->getMEhisto_PhiCorrelation_L1HLT()->Fill(toc[*ki].phi(),toc[*kj].phi());
		v->getMEhisto_PtResolution_L1HLT()->Fill((toc[*ki].pt()-toc[*kj].pt())/(toc[*ki].pt()));
		v->getMEhisto_PhiResolution_L1HLT()->Fill((toc[*ki].phi()-toc[*kj].phi())/(toc[*ki].phi()));
	      }
	      //--------------------------------------------------
	      if(v->getObjectType() == trigger::TriggerJet)
	      {
		hltTrigEta = toc[*kj].eta();
		hltTrigPhi = toc[*kj].phi();

		if((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi)) < 0.4)
		{
                    
		  v->getMEhisto_PtCorrelation_L1HLT()->Fill(toc[*ki].pt(),toc[*kj].pt());
		  v->getMEhisto_EtaCorrelation_L1HLT()->Fill(toc[*ki].eta(),toc[*kj].eta());
		  v->getMEhisto_PhiCorrelation_L1HLT()->Fill(toc[*ki].phi(),toc[*kj].phi());

		  v->getMEhisto_PtResolution_L1HLT()->Fill((toc[*ki].pt()-toc[*kj].pt())/(toc[*ki].pt()));
		  v->getMEhisto_EtaResolution_L1HLT()->Fill((toc[*ki].eta()-toc[*kj].eta())/(toc[*ki].eta()));
		  v->getMEhisto_PhiResolution_L1HLT()->Fill((toc[*ki].phi()-toc[*kj].phi())/(toc[*ki].phi()));
		} 

		if(((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi) < 0.4 ) || ((v->getTriggerType().compare("DiJet_Trigger") == 0)  && hltTrigBool)) && !diJetFire)
		{
		  v->getMEhisto_Pt_HLT()->Fill(toc[*kj].pt());
		  if (isBarrel(toc[*kj].eta()))  v->getMEhisto_PtBarrel_HLT()->Fill(toc[*kj].pt());
		  if (isEndCap(toc[*kj].eta()))  v->getMEhisto_PtEndcap_HLT()->Fill(toc[*kj].pt());
		  if (isForward(toc[*kj].eta())) v->getMEhisto_PtForward_HLT()->Fill(toc[*kj].pt());
		  v->getMEhisto_Eta_HLT()->Fill(toc[*kj].eta());
		  v->getMEhisto_Phi_HLT()->Fill(toc[*kj].phi());
		  v->getMEhisto_EtaPhi_HLT()->Fill(toc[*kj].eta(),toc[*kj].phi());
		  //-------------------------------------------------           
 
		  if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
                    
                    for(CaloJetCollection::const_iterator jet = calojet.begin(); jet != calojet.end(); ++jet ) {
                      double jetEta = jet->eta();
                      double jetPhi = jet->phi();
                      if(deltaR(hltTrigEta, hltTrigPhi, jetEta, jetPhi) < 0.4)
		      {
                        jetsize++;
                        v->getMEhisto_Pt()->Fill(jet->pt());
                        if (isBarrel(jet->eta()))  v->getMEhisto_PtBarrel()->Fill(jet->pt());
                        if (isEndCap(jet->eta()))  v->getMEhisto_PtEndcap()->Fill(jet->pt());
                        if (isForward(jet->eta())) v->getMEhisto_PtForward()->Fill(jet->pt());
			v->getMEhisto_Eta()->Fill(jet->eta());
			v->getMEhisto_Phi()->Fill(jet->phi());
			v->getMEhisto_EtaPhi()->Fill(jet->eta(),jet->phi()); 

			v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(),jet->pt());
			v->getMEhisto_EtaCorrelation_HLTRecObj()->Fill(toc[*kj].eta(),jet->eta());
			v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(),jet->phi());

			v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt()-jet->pt())/(toc[*kj].pt()));
			v->getMEhisto_EtaResolution_HLTRecObj()->Fill((toc[*kj].eta()-jet->eta())/(toc[*kj].eta()));
			v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*kj].phi()-jet->phi())/(toc[*kj].phi()));
                         
			//-------------------------------------------------------    
			if((v->getTriggerType().compare("DiJet_Trigger") == 0))
			{
			  jetPhiVec.push_back(jet->phi());
			  jetPtVec.push_back(jet->pt());
			  jetEtaVec.push_back(jet->eta());         
			  jetPxVec.push_back(jet->px());
			  jetPyVec.push_back(jet->py()); 

			  hltPhiVec.push_back(toc[*kj].phi());
			  hltPtVec.push_back(toc[*kj].pt());
			  hltEtaVec.push_back(toc[*kj].eta());
			  hltPxVec.push_back(toc[*kj].px()); 
			  hltPyVec.push_back(toc[*kj].py());

			}
                          

		      }// matching jet
                                    
		    }// Jet Loop
		  }// valid jet collection
		} // hlt matching with l1 
                      
	      }// jet trigger
	      //------------------------------------------------------
	      if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
		const CaloMETCollection *calometcol = calometColl_.product();
		const CaloMET met = calometcol->front();
		v->getMEhisto_Pt()->Fill(met.pt()); 
		v->getMEhisto_Phi()->Fill(met.phi());
     
		v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(),met.pt());
		v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(),met.phi());
		v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt()-met.pt())/(toc[*kj].pt()));
		v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*kj].phi()-met.phi())/(toc[*kj].phi())); 
	      }// valid MET Collection 

	      //--------------------------------------------------------

            }//Loop over HLT trigger candidates
	    if((v->getTriggerType().compare("DiJet_Trigger") == 0))diJetFire = true;
	  }// valid hlt trigger object
	}// Loop over L1 objects
      }// valid L1
      v->getMEhisto_N()->Fill(jetsize);
      //--------------------------------------------------------
      if((v->getTriggerType().compare("DiJet_Trigger") == 0) && jetPtVec.size() >1)
      {
	double AveJetPt = (jetPtVec[0] + jetPtVec[1])/2;
	double AveJetEta = (jetEtaVec[0] + jetEtaVec[1])/2;               
	double JetDelPhi = deltaPhi(jetPhiVec[0],jetPhiVec[1]);
	double AveHLTPt = (hltPtVec[0] + hltPtVec[1])/2;
	double AveHLTEta = (hltEtaVec[0] + hltEtaVec[1])/2;
	double HLTDelPhi = deltaPhi(hltPhiVec[0],hltPhiVec[1]);
	v->getMEhisto_AveragePt_RecObj()->Fill(AveJetPt);
	v->getMEhisto_AverageEta_RecObj()->Fill(AveJetEta);
	v->getMEhisto_DeltaPhi_RecObj()->Fill(JetDelPhi);
 
	v->getMEhisto_AveragePt_HLTObj()->Fill(AveHLTPt);
	v->getMEhisto_AverageEta_HLTObj()->Fill(AveHLTEta);
	v->getMEhisto_DeltaPhi_HLTObj()->Fill(HLTDelPhi);       

            
      }
      //-----------------------------------------------------      
      if(v->getPath().find("L1") != std::string::npos && !fillL1HLT)
      {
	if ( l1Index >= triggerObj_->sizeFilters() ) {
          edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< l1Index << " of that name "<<l1Tag;
	} else {
	  l1TrigBool = true;
	  const trigger::Keys & kl1 = triggerObj_->filterKeys(l1Index);
	  for( trigger::Keys::const_iterator ki = kl1.begin(); ki != kl1.end(); ++ki)
	  {
	    double l1TrigEta = toc[*ki].eta();
	    double l1TrigPhi = toc[*ki].phi();
	    if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType().compare("SingleJet_Trigger") == 0) ){
	      for(CaloJetCollection::const_iterator jet = calojet.begin(); jet != calojet.end(); ++jet ) {
		double jetEta = jet->eta();
		double jetPhi = jet->phi();
		if(deltaR(l1TrigEta, l1TrigPhi, jetEta, jetPhi) < 0.4)
		{
		  jetsize++;
		  v->getMEhisto_Pt()->Fill(jet->pt());
		  if (isBarrel(jet->eta()))  v->getMEhisto_PtBarrel()->Fill(jet->pt());
		  if (isEndCap(jet->eta()))  v->getMEhisto_PtEndcap()->Fill(jet->pt());
		  if (isForward(jet->eta())) v->getMEhisto_PtForward()->Fill(jet->pt());
		  v->getMEhisto_Eta()->Fill(jet->eta());
		  v->getMEhisto_Phi()->Fill(jet->phi());
		  v->getMEhisto_EtaPhi()->Fill(jet->eta(),jet->phi()); 

		  v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*ki].pt(),jet->pt());
		  v->getMEhisto_EtaCorrelation_HLTRecObj()->Fill(toc[*ki].eta(),jet->eta());
		  v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*ki].phi(),jet->phi());

		  v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*ki].pt()-jet->pt())/(toc[*ki].pt()));
		  v->getMEhisto_EtaResolution_HLTRecObj()->Fill((toc[*ki].eta()-jet->eta())/(toc[*ki].eta()));
		  v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*ki].phi()-jet->phi())/(toc[*ki].phi()));

		}// matching jet
                                    
	      }// Jet Loop
	      v->getMEhisto_N()->Fill(jetsize);
	    }// valid Jet collection

	    if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
	      const CaloMETCollection *calometcol = calometColl_.product();
	      const CaloMET met = calometcol->front();
	      v->getMEhisto_Pt()->Fill(met.pt()); 
	      v->getMEhisto_Phi()->Fill(met.phi());
                
	      v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*ki].pt(),met.pt());
	      v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*ki].phi(),met.phi());
	      v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*ki].pt()-met.pt())/(toc[*ki].pt()));
	      v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*ki].phi()-met.phi())/(toc[*ki].phi()));
	    }// valid MET Collection         
             

	  }// Loop over keys
	}// valid object
      }// L1 is fired but not HLT       
      //-----------------------------------    
    }//Trigger is fired
  }//Loop over all trigger paths

}

//-------------plots wrt Muon Trigger------------
void JetMETHLTOfflineSource::fillMEforMonAllTriggerwrtMuonTrigger(const Event & iEvent){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }

  bool muTrig = false;
  for(size_t i=0;i<MuonTrigPaths_.size();++i){
    if(isHLTPathAccepted(MuonTrigPaths_[i])){
      muTrig = true;
      break;
    }
  }
  if(muTrig)
  {
    //-----------------------------------------------------  
    const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects()); 
    for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v )
    {
      if (isHLTPathAccepted(v->getPath()))     
      {
	std::vector<double>jetPtVec;
	std::vector<double>jetPhiVec; 
	std::vector<double>jetEtaVec;
	std::vector<double>jetPxVec;
	std::vector<double>jetPyVec;
            
	std::vector<double>hltPtVec;
	std::vector<double>hltPhiVec;
	std::vector<double>hltEtaVec;
	std::vector<double>hltPxVec;
	std::vector<double>hltPyVec;
	bool fillL1HLT = false;
	//---------------------------------------------
	edm::InputTag l1Tag(v->getl1Path(),"",processname_);
	const int l1Index = triggerObj_->filterIndex(l1Tag);
	edm::InputTag hltTag(v->getLabel(),"",processname_);
	const int hltIndex = triggerObj_->filterIndex(hltTag);
	bool l1TrigBool = false;
	bool hltTrigBool = false;
	bool diJetFire = false;
	int jetsize = 0;
	if ( l1Index >= triggerObj_->sizeFilters() ) {
	  edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< l1Index << " of that name "<<l1Tag;
	} else {
	  l1TrigBool = true;
	  const trigger::Keys & kl1 = triggerObj_->filterKeys(l1Index);
	  if(v->getObjectType() == trigger::TriggerJet)v->getMEhisto_N_L1()->Fill(kl1.size());
	  for( trigger::Keys::const_iterator ki = kl1.begin(); ki != kl1.end(); ++ki)
	  {
	    double l1TrigEta = -100;
	    double l1TrigPhi = -100;
	    //-------------------------------------------
	    if(v->getObjectType() == trigger::TriggerJet)
	    { 
	      l1TrigEta = toc[*ki].eta();
	      l1TrigPhi = toc[*ki].phi();
	      v->getMEhisto_Pt_L1()->Fill(toc[*ki].pt());
	      if (isBarrel(toc[*ki].eta()))  v->getMEhisto_PtBarrel_L1()->Fill(toc[*ki].pt());
	      if (isEndCap(toc[*ki].eta()))  v->getMEhisto_PtEndcap_L1()->Fill(toc[*ki].pt());
	      if (isForward(toc[*ki].eta())) v->getMEhisto_PtForward_L1()->Fill(toc[*ki].pt());
	      v->getMEhisto_Eta_L1()->Fill(toc[*ki].eta());
	      v->getMEhisto_Phi_L1()->Fill(toc[*ki].phi());
	      v->getMEhisto_EtaPhi_L1()->Fill(toc[*ki].eta(),toc[*ki].phi());
	    }
	    if(v->getObjectType() == trigger::TriggerMET)
	    {
	      v->getMEhisto_Pt_L1()->Fill(toc[*ki].pt());
	      v->getMEhisto_Phi_L1()->Fill(toc[*ki].phi());
	    }
	    //-----------------------------------------------  
	    if ( hltIndex >= triggerObj_->sizeFilters() ) {
	      edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt"<< hltIndex << " of that name ";
	    } else {
	      const trigger::Keys & khlt = triggerObj_->filterKeys(hltIndex);
	      if((v->getObjectType() == trigger::TriggerJet) && (ki == kl1.begin()))v->getMEhisto_N_HLT()->Fill(khlt.size());

	      for(trigger::Keys::const_iterator kj = khlt.begin();kj != khlt.end(); ++kj)
	      {
		double hltTrigEta = -100;
		double hltTrigPhi = -100;
		if(v->getObjectType() == trigger::TriggerJet)
		{
		  hltTrigEta = toc[*kj].eta();
		  hltTrigPhi = toc[*kj].phi();
		  if((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi)) < 0.4 && (v->getTriggerType().compare("DiJet_Trigger") == 0))hltTrigBool = true;
		}    
	      }
	      for(trigger::Keys::const_iterator kj = khlt.begin();kj != khlt.end(); ++kj)
	      {
		double hltTrigEta = -100;
		double hltTrigPhi = -100;
		fillL1HLT = true;
		//--------------------------------------------------
		if(v->getObjectType() == trigger::TriggerMET)
		{
		  v->getMEhisto_Pt_HLT()->Fill(toc[*kj].pt());
		  v->getMEhisto_Phi_HLT()->Fill(toc[*kj].phi());
		  v->getMEhisto_PtCorrelation_L1HLT()->Fill(toc[*ki].pt(),toc[*kj].pt());
		  v->getMEhisto_PhiCorrelation_L1HLT()->Fill(toc[*ki].phi(),toc[*kj].phi());
		  v->getMEhisto_PtResolution_L1HLT()->Fill((toc[*ki].pt()-toc[*kj].pt())/(toc[*ki].pt()));
		  v->getMEhisto_PhiResolution_L1HLT()->Fill((toc[*ki].phi()-toc[*kj].phi())/(toc[*ki].phi()));
		}
		//--------------------------------------------------
		if(v->getObjectType() == trigger::TriggerJet)
		{
		  hltTrigEta = toc[*kj].eta();
		  hltTrigPhi = toc[*kj].phi();

		  if((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi)) < 0.4)
		  {
                    
                    v->getMEhisto_PtCorrelation_L1HLT()->Fill(toc[*ki].pt(),toc[*kj].pt());
                    v->getMEhisto_EtaCorrelation_L1HLT()->Fill(toc[*ki].eta(),toc[*kj].eta());
                    v->getMEhisto_PhiCorrelation_L1HLT()->Fill(toc[*ki].phi(),toc[*kj].phi());

                    v->getMEhisto_PtResolution_L1HLT()->Fill((toc[*ki].pt()-toc[*kj].pt())/(toc[*ki].pt()));
                    v->getMEhisto_EtaResolution_L1HLT()->Fill((toc[*ki].eta()-toc[*kj].eta())/(toc[*ki].eta()));
                    v->getMEhisto_PhiResolution_L1HLT()->Fill((toc[*ki].phi()-toc[*kj].phi())/(toc[*ki].phi()));
		  } 

		  if(((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi) < 0.4 ) || ((v->getTriggerType().compare("DiJet_Trigger") == 0)  && hltTrigBool)
		       ) && !diJetFire)
		  {
                    v->getMEhisto_Pt_HLT()->Fill(toc[*kj].pt());
                    if (isBarrel(toc[*kj].eta()))  v->getMEhisto_PtBarrel_HLT()->Fill(toc[*kj].pt());
                    if (isEndCap(toc[*kj].eta()))  v->getMEhisto_PtEndcap_HLT()->Fill(toc[*kj].pt());
                    if (isForward(toc[*kj].eta())) v->getMEhisto_PtForward_HLT()->Fill(toc[*kj].pt());
                    v->getMEhisto_Eta_HLT()->Fill(toc[*kj].eta());
                    v->getMEhisto_Phi_HLT()->Fill(toc[*kj].phi());
                    v->getMEhisto_EtaPhi_HLT()->Fill(toc[*kj].eta(),toc[*kj].phi());
		    //-------------------------------------------------           
 
		    if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
                    
		      for(CaloJetCollection::const_iterator jet = calojet.begin(); jet != calojet.end(); ++jet ) {
			double jetEta = jet->eta();
			double jetPhi = jet->phi();
			if(deltaR(hltTrigEta, hltTrigPhi, jetEta, jetPhi) < 0.4)
                        {
			  jetsize++;
			  v->getMEhisto_Pt()->Fill(jet->pt());
			  if (isBarrel(jet->eta()))  v->getMEhisto_PtBarrel()->Fill(jet->pt());
			  if (isEndCap(jet->eta()))  v->getMEhisto_PtEndcap()->Fill(jet->pt());
			  if (isForward(jet->eta())) v->getMEhisto_PtForward()->Fill(jet->pt());
			  v->getMEhisto_Eta()->Fill(jet->eta());
			  v->getMEhisto_Phi()->Fill(jet->phi());
			  v->getMEhisto_EtaPhi()->Fill(jet->eta(),jet->phi()); 

			  v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(),jet->pt());
			  v->getMEhisto_EtaCorrelation_HLTRecObj()->Fill(toc[*kj].eta(),jet->eta());
			  v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(),jet->phi());

			  v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt()-jet->pt())/(toc[*kj].pt()));
			  v->getMEhisto_EtaResolution_HLTRecObj()->Fill((toc[*kj].eta()-jet->eta())/(toc[*kj].eta()));
			  v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*kj].phi()-jet->phi())/(toc[*kj].phi()));
                         
			  //-------------------------------------------------------    
			  if((v->getTriggerType().compare("DiJet_Trigger") == 0))
			  {
                            jetPhiVec.push_back(jet->phi());
                            jetPtVec.push_back(jet->pt());
                            jetEtaVec.push_back(jet->eta());         
                            jetPxVec.push_back(jet->px());
                            jetPyVec.push_back(jet->py()); 

                            hltPhiVec.push_back(toc[*kj].phi());
                            hltPtVec.push_back(toc[*kj].pt());
                            hltEtaVec.push_back(toc[*kj].eta());
                            hltPxVec.push_back(toc[*kj].px()); 
                            hltPyVec.push_back(toc[*kj].py());

			  }
                          

                        }// matching jet
                                    
		      }// Jet Loop
                    }// valid jet collection
                  } // hlt matching with l1 
                      
                }// jet trigger
		//------------------------------------------------------
		if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
		  const CaloMETCollection *calometcol = calometColl_.product();
		  const CaloMET met = calometcol->front();
		  v->getMEhisto_Pt()->Fill(met.pt()); 
		  v->getMEhisto_Phi()->Fill(met.phi());
     
		  v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(),met.pt());
		  v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(),met.phi());
		  v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt()-met.pt())/(toc[*kj].pt()));
		  v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*kj].phi()-met.phi())/(toc[*kj].phi())); 
                }// valid MET Collection 

		//--------------------------------------------------------

	      }//Loop over HLT trigger candidates
	      if((v->getTriggerType().compare("DiJet_Trigger") == 0))diJetFire = true;
	    }// valid hlt trigger object
	  }// Loop over L1 objects
	}// valid L1
	v->getMEhisto_N()->Fill(jetsize);
	//--------------------------------------------------------
	if((v->getTriggerType().compare("DiJet_Trigger") == 0) && jetPtVec.size() >1)
	{
	  double AveJetPt = (jetPtVec[0] + jetPtVec[1])/2;
	  double AveJetEta = (jetEtaVec[0] + jetEtaVec[1])/2;               
	  double JetDelPhi = deltaPhi(jetPhiVec[0],jetPhiVec[1]);
	  double AveHLTPt = (hltPtVec[0] + hltPtVec[1])/2;
	  double AveHLTEta = (hltEtaVec[0] + hltEtaVec[1])/2;
	  double HLTDelPhi = deltaPhi(hltPhiVec[0],hltPhiVec[1]);
	  v->getMEhisto_AveragePt_RecObj()->Fill(AveJetPt);
	  v->getMEhisto_AverageEta_RecObj()->Fill(AveJetEta);
	  v->getMEhisto_DeltaPhi_RecObj()->Fill(JetDelPhi);
 
	  v->getMEhisto_AveragePt_HLTObj()->Fill(AveHLTPt);
	  v->getMEhisto_AverageEta_HLTObj()->Fill(AveHLTEta);
	  v->getMEhisto_DeltaPhi_HLTObj()->Fill(HLTDelPhi);       

            
	}
	//-----------------------------------------------------      
	if(v->getPath().find("L1") != std::string::npos && !fillL1HLT)
	{
	  if ( l1Index >= triggerObj_->sizeFilters() ) {
	    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< l1Index << " of that name "<<l1Tag;
	  } else {
	    l1TrigBool = true;
	    const trigger::Keys & kl1 = triggerObj_->filterKeys(l1Index);
            for( trigger::Keys::const_iterator ki = kl1.begin(); ki != kl1.end(); ++ki)
	    {
              double l1TrigEta = toc[*ki].eta();
              double l1TrigPhi = toc[*ki].phi();
              if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType().compare("SingleJet_Trigger") == 0) ){
		for(CaloJetCollection::const_iterator jet = calojet.begin(); jet != calojet.end(); ++jet ) {
		  double jetEta = jet->eta();
		  double jetPhi = jet->phi();
		  if(deltaR(l1TrigEta, l1TrigPhi, jetEta, jetPhi) < 0.4)
		  {
		    jetsize++;
		    v->getMEhisto_Pt()->Fill(jet->pt());
		    if (isBarrel(jet->eta()))  v->getMEhisto_PtBarrel()->Fill(jet->pt());
		    if (isEndCap(jet->eta()))  v->getMEhisto_PtEndcap()->Fill(jet->pt());
		    if (isForward(jet->eta())) v->getMEhisto_PtForward()->Fill(jet->pt());
		    v->getMEhisto_Eta()->Fill(jet->eta());
		    v->getMEhisto_Phi()->Fill(jet->phi());
		    v->getMEhisto_EtaPhi()->Fill(jet->eta(),jet->phi()); 

		    v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*ki].pt(),jet->pt());
		    v->getMEhisto_EtaCorrelation_HLTRecObj()->Fill(toc[*ki].eta(),jet->eta());
		    v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*ki].phi(),jet->phi());

		    v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*ki].pt()-jet->pt())/(toc[*ki].pt()));
		    v->getMEhisto_EtaResolution_HLTRecObj()->Fill((toc[*ki].eta()-jet->eta())/(toc[*ki].eta()));
		    v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*ki].phi()-jet->phi())/(toc[*ki].phi()));

		  }// matching jet
                                    
		}// Jet Loop
		v->getMEhisto_N()->Fill(jetsize);
	      }// valid Jet collection

	      if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
		const CaloMETCollection *calometcol = calometColl_.product();
		const CaloMET met = calometcol->front();
		v->getMEhisto_Pt()->Fill(met.pt()); 
		v->getMEhisto_Phi()->Fill(met.phi());
                
		v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*ki].pt(),met.pt());
		v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*ki].phi(),met.phi());
		v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*ki].pt()-met.pt())/(toc[*ki].pt()));
		v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*ki].phi()-met.phi())/(toc[*ki].phi()));
	      }// valid MET Collection         
             

	    }// Loop over keys
	  }// valid object
	}// L1 is fired but not HLT       
	//-----------------------------------    
      }//Trigger is fired
    }//Loop over all trigger paths
  

  }// Muon trigger fired


}

void JetMETHLTOfflineSource::fillMEforEffAllTrigger(const Event & iEvent){
  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  int num = -1;
  int denom = -1;
  for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v )
  {
    num++;
    denom++;
    bool denompassed = false;
    bool numpassed   = false; 

    unsigned indexNum = triggerNames_.triggerIndex(v->getPath());
    unsigned indexDenom = triggerNames_.triggerIndex(v->getDenomPath());

    if(indexNum < triggerNames_.size() && triggerResults_->accept(indexNum))numpassed   = true;
    if(indexDenom < triggerNames_.size() && triggerResults_->accept(indexDenom))denompassed   = true; 

    if(denompassed){
      if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
	bool jetIDbool = false;
	if((v->getTriggerType().compare("SingleJet_Trigger") == 0) && calojet.size())
	{

	  CaloJetCollection::const_iterator jet = calojet.begin();
	  jetID->calculate(iEvent, *jet);
 
	  if(verbose_)cout<<"n90Hits==="<<jetID->n90Hits()<<"==fHPDs=="<<jetID->fHPD()<<endl;
	  if((jet->emEnergyFraction()>_fEMF || fabs(jet->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits)
	  {
	    jetIDbool = true;
	    v->getMEhisto_DenominatorPt()->Fill(jet->pt());
	    if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
	    if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
	    if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
	    v->getMEhisto_DenominatorEta()->Fill(jet->eta());
	    v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
	    v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
	  }
	}
	if((v->getTriggerType().compare("DiJet_Trigger") == 0) && calojet.size()>1)
	{
	  CaloJetCollection::const_iterator jet = calojet.begin();
	  CaloJetCollection::const_iterator jet2 = jet++;
	  jetID->calculate(iEvent, *jet2);
	  if(jetIDbool && ((jet2->emEnergyFraction()>_fEMF || fabs(jet2->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits))
	  {
	    v->getMEhisto_DenominatorPt()->Fill((jet->pt() + jet2->pt())/2.);
	    v->getMEhisto_DenominatorEta()->Fill((jet->eta() + jet2->eta())/2.);
	  }
	}
            
      }// Jet trigger and valid jet collection
      if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
	const CaloMETCollection *calometcol = calometColl_.product();
	const CaloMET met = calometcol->front();
	v->getMEhisto_DenominatorPt()->Fill(met.pt());
	v->getMEhisto_DenominatorPhi()->Fill(met.phi());     

      }// MET trigger and valid MET collection 

      if (numpassed)
      {
	if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
	  bool jetIDbool = false;
	  if((v->getTriggerType().compare("SingleJet_Trigger") == 0) && calojet.size())
	  {
	    CaloJetCollection::const_iterator jet = calojet.begin();
	    jetID->calculate(iEvent, *jet);
	    if((jet->emEnergyFraction()>_fEMF || fabs(jet->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits)
	    {
              jetIDbool = true; 
              v->getMEhisto_NumeratorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
              v->getMEhisto_NumeratorEta()->Fill(jet->eta());
              v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
              v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());
	    }
	  }
	  if((v->getTriggerType().compare("DiJet_Trigger") == 0) && calojet.size() > 1)
	  {
	    CaloJetCollection::const_iterator jet = calojet.begin();
	    CaloJetCollection::const_iterator jet2 = jet++;
	    jetID->calculate(iEvent, *jet2);
	    if(jetIDbool && ((jet2->emEnergyFraction()>_fEMF || fabs(jet2->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits))
	    {
              v->getMEhisto_NumeratorPt()->Fill((jet->pt() + jet2->pt())/2.);
              v->getMEhisto_NumeratorEta()->Fill((jet->eta() + jet2->eta())/2.);
	    }       
	  }
	}// Jet trigger and valid jet collection
	if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
	  const CaloMETCollection *calometcol = calometColl_.product();
	  const CaloMET met = calometcol->front();
	  v->getMEhisto_NumeratorPt()->Fill(met.pt());
	  v->getMEhisto_NumeratorPhi()->Fill(met.phi());

	}// MET trigger and valid MET collection 
      }//Numerator is fired
    }//Denominator is fired
  }// trigger under study

}


void JetMETHLTOfflineSource::fillMEforEffWrtMuTrigger(const Event & iEvent){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  bool muTrig = false;
  bool denompassed = false;
  for(size_t i=0;i<MuonTrigPaths_.size();++i){
    if(isHLTPathAccepted(MuonTrigPaths_[i])){
      muTrig = true;
      break;
    }
  }
  for(PathInfoCollection::iterator v = hltPathsEffWrtMu_.begin(); v!= hltPathsEffWrtMu_.end(); ++v )
  {
    bool numpassed   = false; 
    if(muTrig)denompassed = true;
     
    unsigned indexNum = triggerNames_.triggerIndex(v->getPath());
    if(indexNum < triggerNames_.size() && triggerResults_->accept(indexNum))numpassed   = true;

    if(denompassed){
      if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
	bool jetIDbool = false;
	if((v->getTriggerType().compare("SingleJet_Trigger") == 0) && calojet.size())
	{
	  CaloJetCollection::const_iterator jet = calojet.begin();
	  jetID->calculate(iEvent, *jet);
	  if((jet->emEnergyFraction()>_fEMF || fabs(jet->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits)
	  {
	    jetIDbool = true;
	    v->getMEhisto_DenominatorPt()->Fill(jet->pt());
	    if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
	    if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
	    if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
	    v->getMEhisto_DenominatorEta()->Fill(jet->eta());
	    v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
	    v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
	  }
	}
	if((v->getTriggerType().compare("DiJet_Trigger") == 0) && calojet.size() > 1)
	{
	  CaloJetCollection::const_iterator jet = calojet.begin();
	  CaloJetCollection::const_iterator jet2 = jet++;
	  jetID->calculate(iEvent, *jet2);
	  if(jetIDbool && ((jet2->emEnergyFraction()>_fEMF || fabs(jet2->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits))
	  { 
	    v->getMEhisto_DenominatorPt()->Fill((jet->pt() + jet2->pt())/2.);
	    v->getMEhisto_DenominatorEta()->Fill((jet->eta() + jet2->eta())/2.);
	  }
	}
            
      }// Jet trigger and valid jet collection
      if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
	const CaloMETCollection *calometcol = calometColl_.product();
	const CaloMET met = calometcol->front();
	v->getMEhisto_DenominatorPt()->Fill(met.pt());
	v->getMEhisto_DenominatorPhi()->Fill(met.phi());     

      }// MET trigger and valid MET collection 

      if (numpassed)
      {
	if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
	  bool jetIDbool = false;     
	  if((v->getTriggerType().compare("SingleJet_Trigger") == 0) && calojet.size())
	  {
	    CaloJetCollection::const_iterator jet = calojet.begin();
	    jetID->calculate(iEvent, *jet);
	    if((jet->emEnergyFraction()>_fEMF || fabs(jet->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits)
	    {
              jetIDbool = true;
              v->getMEhisto_NumeratorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
              v->getMEhisto_NumeratorEta()->Fill(jet->eta());
              v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
              v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());
	    }
	  }
	  if((v->getTriggerType().compare("DiJet_Trigger") == 0) && calojet.size() > 1)     
	  {
	    CaloJetCollection::const_iterator jet = calojet.begin();
	    CaloJetCollection::const_iterator jet2 = jet++; 
	    jetID->calculate(iEvent, *jet2);
	    if(jetIDbool && ((jet2->emEnergyFraction()>_fEMF || fabs(jet2->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits))
	    {
              v->getMEhisto_NumeratorPt()->Fill((jet->pt() + jet2->pt())/2.);  
              v->getMEhisto_NumeratorEta()->Fill((jet->eta() + jet2->eta())/2.);
	    }
	  }

	}// Jet trigger and valid jet collection
	if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
	  const CaloMETCollection *calometcol = calometColl_.product();
	  const CaloMET met = calometcol->front();
	  v->getMEhisto_NumeratorPt()->Fill(met.pt());
	  v->getMEhisto_NumeratorPhi()->Fill(met.phi());

	}// MET trigger and valid MET collection 
      }//Numerator is fired
    }//Denominator is fired
  }// trigger under study


}


void JetMETHLTOfflineSource::fillMEforEffWrtMBTrigger(const Event & iEvent){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  bool mbTrig = false;
  bool denompassed = false;
  for(size_t i=0;i<MBTrigPaths_.size();++i){
    if(isHLTPathAccepted(MBTrigPaths_[i])){
      mbTrig = true;
      break;
    }
  }
  for(PathInfoCollection::iterator v = hltPathsEffWrtMB_.begin(); v!= hltPathsEffWrtMB_.end(); ++v )
  {
    bool numpassed   = false; 
    if(mbTrig)denompassed = true;

    unsigned indexNum = triggerNames_.triggerIndex(v->getPath());
    if(indexNum < triggerNames_.size() && triggerResults_->accept(indexNum))numpassed   = true;

    if(denompassed){
      if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
	bool jetIDbool = false;
	if((v->getTriggerType().compare("SingleJet_Trigger") == 0) && calojet.size()) 
	{
	  CaloJetCollection::const_iterator jet = calojet.begin();
	  jetID->calculate(iEvent, *jet);
	  if((jet->emEnergyFraction()>_fEMF || fabs(jet->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits)
	  {
	    jetIDbool = true;
	    v->getMEhisto_DenominatorPt()->Fill(jet->pt());
	    if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
	    if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
	    if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
	    v->getMEhisto_DenominatorEta()->Fill(jet->eta());
	    v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
	    v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
	  }
	}
	if((v->getTriggerType().compare("DiJet_Trigger") == 0) && calojet.size() >1) 
	{
	  CaloJetCollection::const_iterator jet = calojet.begin();
	  CaloJetCollection::const_iterator jet2 = jet++;
	  jetID->calculate(iEvent, *jet2);
	  if(jetIDbool && ((jet2->emEnergyFraction()>_fEMF || fabs(jet2->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits))
	  {
	    v->getMEhisto_DenominatorPt()->Fill((jet->pt() + jet2->pt())/2.);  
	    v->getMEhisto_DenominatorEta()->Fill((jet->eta() + jet2->eta())/2.);
	  } 
	}
      }// Jet trigger and valid jet collection
      if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
	const CaloMETCollection *calometcol = calometColl_.product();
	const CaloMET met = calometcol->front();
	v->getMEhisto_DenominatorPt()->Fill(met.pt());
	v->getMEhisto_DenominatorPhi()->Fill(met.phi());     

      }// MET trigger and valid MET collection 

      if (numpassed)
      {
	if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
	  bool jetIDbool = false;
	  if((v->getTriggerType().compare("SingleJet_Trigger") == 0) && calojet.size())
	  {
	    CaloJetCollection::const_iterator jet = calojet.begin();
	    jetID->calculate(iEvent, *jet);
	    if((jet->emEnergyFraction()>_fEMF || fabs(jet->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits)
	    {
              jetIDbool = true; 
              v->getMEhisto_NumeratorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
              v->getMEhisto_NumeratorEta()->Fill(jet->eta());
              v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
              v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());
	    }
	  }
	  if((v->getTriggerType().compare("DiJet_Trigger") == 0) && calojet.size() > 1)
	  {
	    CaloJetCollection::const_iterator jet = calojet.begin();   
	    CaloJetCollection::const_iterator jet2 = jet++;
	    jetID->calculate(iEvent, *jet2);
	    if(jetIDbool && ((jet2->emEnergyFraction()>_fEMF || fabs(jet2->eta()) > _feta) && (jetID->fHPD()) < _fHPD && (jetID->n90Hits()) > _n90Hits))
	    {
              v->getMEhisto_NumeratorPt()->Fill((jet->pt() + jet2->pt())/2.);
              v->getMEhisto_NumeratorEta()->Fill((jet->eta() + jet2->eta())/2.);
	    }
	  }
	}// Jet trigger and valid jet collection
	if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
	  const CaloMETCollection *calometcol = calometColl_.product();
	  const CaloMET met = calometcol->front();
	  v->getMEhisto_NumeratorPt()->Fill(met.pt());
	  v->getMEhisto_NumeratorPhi()->Fill(met.phi());

	}// MET trigger and valid MET collection 
      }//Numerator is fired
    }//Denominator is fired
  }// trigger under study


}

// -- method called once each job just before starting event loop  --------

void JetMETHLTOfflineSource::beginJob(){
 
}

// BeginRun
void JetMETHLTOfflineSource::beginRun(const edm::Run& run, const edm::EventSetup& c){
  if(!isSetup_)
  { 
    DQMStore *dbe = 0;
    dbe = Service<DQMStore>().operator->();
    if (dbe) {
      dbe->setCurrentFolder(dirname_);
      dbe->rmdir(dirname_);
    }
    if (dbe) {
      dbe->setCurrentFolder(dirname_);
    }

    //--- htlConfig_
    bool changed(true);
    if (!hltConfig_.init(run, c, processname_, changed)) {
      LogDebug("HLTJetMETDQMSource") << "HLTConfigProvider failed to initialize.";
    }

   

    /*
      Here we select the Single Jet, DiJet, MET trigger. SingleJet and DiJet trigger are saved under same object type "TriggerJet". We can easily separate out sing
      le   and di jet trigger later. For the first trigger in the list, denominator trigger is dummy (empty) whereas for other triggers denom is previous trigger o
      f same type. e.g. SingleJet50 has singleJet30 as denominator.
      For defining histos wrt muon trigger, denominator is always set "MuonTrigger". This string later can be checked and condition can be applied on muon triggers
      .
    */

    const unsigned int n(hltConfig_.size());
    int singleJet = 0;
    int diJet     = 0;
    int met       = 0;
    for (unsigned int i=0; i!=n; ++i) {
      bool denomFound = false;
      bool numFound = false; 
      bool mbFound = false;
      bool muFound = false; 
      std::string pathname = hltConfig_.triggerName(i);
      if(verbose_)cout<<"==pathname=="<<pathname<<endl;
      std::string dpathname = MuonTrigPaths_[0];
      std::string l1pathname = "dummy";
      std::string denompathname = "";
      unsigned int usedPrescale = 1;
      unsigned int objectType = 0;
      std::string triggerType = "";
      std::string filtername("dummy");
      std::string Denomfiltername("denomdummy");

      if (pathname.find("Jet") != std::string::npos && !(pathname.find("DoubleJet") != std::string::npos) && !(pathname.find("DiJet") != std::string::npos) && !(pathname.find("BTag") != std::string::npos) && !(pathname.find("Mu") != std::string::npos) && !(pathname.find("Fwd") != std::string::npos)){
	triggerType = "SingleJet_Trigger"; 
	objectType = trigger::TriggerJet;
      }
      if (pathname.find("DiJet") != std::string::npos || pathname.find("DoubleJet") != std::string::npos){
	triggerType = "DiJet_Trigger";
	objectType = trigger::TriggerJet;
      }
      if (pathname.find("MET") != std::string::npos || pathname.find("HT") != std::string::npos){
	triggerType = "MET_Trigger";  
	objectType = trigger::TriggerMET;
      }
    

      if(objectType == trigger::TriggerJet  && !(pathname.find("DiJet") != std::string::npos) && !(pathname.find("DoubleJet") != std::string::npos))
      {
	singleJet++;
	if(singleJet > 1)dpathname = dpathname = hltConfig_.triggerName(i-1);
	if(singleJet == 1)dpathname = MuonTrigPaths_[0];
      }  

      if(objectType == trigger::TriggerJet  && (pathname.find("DiJet") != std::string::npos))
      {
	diJet++;
	if(diJet > 1)dpathname = dpathname = hltConfig_.triggerName(i-1);
	if(diJet == 1)dpathname = MuonTrigPaths_[0];
      } 
      if(objectType == trigger::TriggerMET  )
      {
	met++;
	if(met > 1)dpathname = dpathname = hltConfig_.triggerName(i-1);
	if(met == 1)dpathname = MuonTrigPaths_[0];
      }
      // find L1 condition for numpath with numpath objecttype 
      // find PSet for L1 global seed for numpath, 
      // list module labels for numpath
        // Checking if the trigger exist in HLT table or not
         for (unsigned int i=0; i!=n; ++i) {
          std::string HLTname = hltConfig_.triggerName(i);
          if(HLTname == pathname)numFound = true;
          if(HLTname == dpathname)denomFound = true;
          if(HLTname == MBTrigPaths_[0])mbFound = true;
          if(HLTname == MuonTrigPaths_[0])muFound = true; 
        }
 
      if(numFound)//make trigger exist in the menu
       {
      std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
      for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin(); numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
        edm::InputTag testTag(*numpathmodule,"",processname_);
        if ((hltConfig_.moduleType(*numpathmodule) == "HLT1CaloJet")|| (hltConfig_.moduleType(*numpathmodule) == "HLTDiJetAveFilter") || (hltConfig_.moduleType(*numpathmodule) == "HLT1CaloMET" ) || (hltConfig_.moduleType(*numpathmodule) == "HLTPrescaler") )filtername = *numpathmodule;
        if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")l1pathname = *numpathmodule;
       }
      }

      if(objectType != 0 && denomFound)
      {
        std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(dpathname);
        for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin(); numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	  edm::InputTag testTag(*numpathmodule,"",processname_);
	  if ((hltConfig_.moduleType(*numpathmodule) == "HLT1CaloJet")|| (hltConfig_.moduleType(*numpathmodule) == "HLTDiJetAveFilter") || (hltConfig_.moduleType(*numpathmodule) == "HLT1CaloMET" ) || (hltConfig_.moduleType(*numpathmodule) == "HLTPrescaler") )Denomfiltername = *numpathmodule;
	}
      }

      if(objectType != 0 && numFound)
      {
	if(verbose_)cout<<"==pathname=="<<pathname<<"==denompath=="<<dpathname<<"==filtername=="<<filtername<<"==denomfiltername=="<<Denomfiltername<<"==l1pathname=="<<l1pathname<<"==objectType=="<<objectType<<endl;    
	if(!((pathname.find("HT") != std::string::npos) || (pathname.find("Quad") != std::string::npos)))
	{     
	  hltPathsAll_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
	  if(muFound)hltPathsAllWrtMu_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
	  if(muFound)hltPathsEffWrtMu_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
	  if(mbFound)hltPathsEffWrtMB_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));

	  if(!nameForEff_ && denomFound) hltPathsEff_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
	}

	hltPathsAllTriggerSummary_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));

      }
    } //Loop over paths

    //---------bool to pick trigger names pair from config file-------------
    if(nameForEff_)
    {
      std::string l1pathname = "dummy";
      std::string denompathname = "";
      unsigned int usedPrescale = 1;
      unsigned int objectType = 0;
      std::string triggerType = "";
      std::string filtername("dummy");
      std::string Denomfiltername("denomdummy");
      for (std::vector<std::pair<std::string, std::string> >::iterator custompathnamepair = custompathnamepairs_.begin(); custompathnamepair != custompathnamepairs_.end(); ++custompathnamepair)
      {
	std::string pathname  = custompathnamepair->first;
	std::string dpathname = custompathnamepair->second;
	bool numFound = false;
	bool denomFound = false;
	// Checking if the trigger exist in HLT table or not
	for (unsigned int i=0; i!=n; ++i) {
	  std::string HLTname = hltConfig_.triggerName(i);
	  if(HLTname == pathname)numFound = true;
	  if(HLTname == dpathname)denomFound = true;
	}
	if(numFound && denomFound)
	{
	  if (pathname.find("Jet") != std::string::npos && !(pathname.find("DiJet") != std::string::npos) && !(pathname.find("DoubleJet") != std::string::npos) && !(pathname.find("BTag") != std::string::npos) && !(pathname.find("Mu") != std::string::npos) && !(pathname.find("Fwd") != std::string::npos)){
	    triggerType = "SingleJet_Trigger";
	    objectType = trigger::TriggerJet;
	  }
	  if (pathname.find("DiJet") != std::string::npos || pathname.find("DoubleJet") != std::string::npos ){
	    triggerType = "DiJet_Trigger";
	    objectType = trigger::TriggerJet;
	  }
	  if (pathname.find("MET") != std::string::npos ){
	    triggerType = "MET_Trigger";
	    objectType = trigger::TriggerMET;
	  }

	  std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
	  for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin(); numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	    edm::InputTag testTag(*numpathmodule,"",processname_);
	    if ((hltConfig_.moduleType(*numpathmodule) == "HLT1CaloJet")|| (hltConfig_.moduleType(*numpathmodule) == "HLTDiJetAveFilter") || (hltConfig_.moduleType(*numpathmodule) == "HLT1CaloMET" ) || (hltConfig_.moduleType(*numpathmodule) == "HLTPrescaler") )filtername = *numpathmodule;
	    if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")l1pathname = *numpathmodule;
	  }

	  if(objectType != 0)
	  {
	    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(dpathname);
	    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin(); numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      edm::InputTag testTag(*numpathmodule,"",processname_);
	      if ((hltConfig_.moduleType(*numpathmodule) == "HLT1CaloJet")|| (hltConfig_.moduleType(*numpathmodule) == "HLTDiJetAveFilter") || (hltConfig_.moduleType(*numpathmodule) == "HLT1CaloMET" ) || (hltConfig_.moduleType(*numpathmodule) == "HLTPrescaler") )Denomfiltername = *numpathmodule;
	    }
     
	    if(verbose_)cout<<"==pathname=="<<pathname<<"==denompath=="<<dpathname<<"==filtername=="<<filtername<<"==denomfiltername=="<<Denomfiltername<<"==l1pathname=="<<l1pathname<<"==objectType=="<<objectType<<endl;
	    hltPathsEff_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType)); 

	  }
	}
      }
    }
    //-----------------------------------------------------------------
    //---book trigger summary histos
    if(!isSetup_)
    {
      std::string foldernm = "/TriggerSummary/";
      if (dbe)   {
	dbe->setCurrentFolder(dirname_ + foldernm);
      }
      int     TrigBins_ = hltPathsAllTriggerSummary_.size();
      double  TrigMin_ = -0.5;
      double  TrigMax_ = hltPathsAllTriggerSummary_.size()-0.5;
      std::string histonm="JetMET_TriggerRate";
      std::string histot="JetMET TriggerRate Summary";
     
      rate_All = dbe->book1D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_);


      histonm="JetMET_TriggerRate_Correlation";
      histot="JetMET TriggerRate Correlation Summary;y&&!x;x&&y";

      correlation_All = dbe->book2D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_,TrigBins_,TrigMin_,TrigMax_);


      histonm="JetMET_TriggerRate_WrtMuTrigger";
      histot="JetMET TriggerRate Summary Wrt Muon Trigger ";
    
      rate_AllWrtMu = dbe->book1D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_);


      histonm="JetMET_TriggerRate_Correlation_WrtMuTrigger";
      histot="JetMET TriggerRate Correlation Summary Wrt Muon Trigger;y&&!x;x&&y";

      correlation_AllWrtMu = dbe->book2D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_,TrigBins_,TrigMin_,TrigMax_);

      histonm="JetMET_TriggerRate_WrtMBTrigger";
      histot="JetMET TriggerRate Summary Wrt MB Trigger";

      rate_AllWrtMB = dbe->book1D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_);


      histonm="JetMET_TriggerRate_Correlation_WrtMBTrigger";
      histot="JetMET TriggerRate Correlation Wrt MB Trigger;y&&!x;x&&y";

      correlation_AllWrtMB = dbe->book2D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_,TrigBins_,TrigMin_,TrigMax_);
      isSetup_ = true;

    }
    //---Set bin label
 
    for(PathInfoCollection::iterator v = hltPathsAllTriggerSummary_.begin(); v!= hltPathsAllTriggerSummary_.end(); ++v ){
      std::string labelnm("dummy");
      labelnm = v->getPath(); 
      int nbins = rate_All->getTH1()->GetNbinsX();
      for(int ibin=1; ibin<nbins+1; ibin++)
      {
	const char * binLabel = rate_All->getTH1()->GetXaxis()->GetBinLabel(ibin);
	std::string binLabel_str = string(binLabel);
	if(binLabel_str.compare(labelnm)==0)break;
	if(binLabel[0]=='\0')
        {
	  rate_All->setBinLabel(ibin,labelnm);
	  rate_AllWrtMu->setBinLabel(ibin,labelnm);
	  rate_AllWrtMB->setBinLabel(ibin,labelnm);
	  correlation_All->setBinLabel(ibin,labelnm,1);
	  correlation_AllWrtMu->setBinLabel(ibin,labelnm,1);
	  correlation_AllWrtMB->setBinLabel(ibin,labelnm,1);
	  correlation_All->setBinLabel(ibin,labelnm,2);
	  correlation_AllWrtMu->setBinLabel(ibin,labelnm,2);
	  correlation_AllWrtMB->setBinLabel(ibin,labelnm,2);
	  break; 
        } 
      }     

    }

    // Now define histos for All triggers
    if(plotAll_)
    {
      int Nbins_       = 10;
      int Nmin_        = -0.5;
      int Nmax_        = 9.5;
      int Ptbins_      = 40;
      int Etabins_     = 40;
      int Phibins_     = 35;
      int Resbins_     = 30;
      double PtMin_    = 0.;
      double PtMax_    = 200.;
      double EtaMin_   = -5.;
      double EtaMax_   =  5.;
      double PhiMin_   = -3.14159;
      double PhiMax_   =  3.14159;
      double ResMin_   =  -1.5;
      double ResMax_   =   1.5;
      std::string dirName = dirname_ + "/MonitorAllTriggers/";
      for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){  

	std::string subdirName = dirName + v->getPath();
	std::string trigPath = "("+v->getPath()+")";
	dbe->setCurrentFolder(subdirName);  
 
	MonitorElement *dummy, *N, *Pt,  *PtBarrel, *PtEndcap, *PtForward, *Eta, *Phi, *EtaPhi,
	  *N_L1, *Pt_L1,  *PtBarrel_L1, *PtEndcap_L1, *PtForward_L1, *Eta_L1, *Phi_L1, *EtaPhi_L1,
	  *N_HLT, *Pt_HLT,  *PtBarrel_HLT, *PtEndcap_HLT, *PtForward_HLT, *Eta_HLT, *Phi_HLT, *EtaPhi_HLT,
	  *PtResolution_L1HLT, *EtaResolution_L1HLT,*PhiResolution_L1HLT,
	  *PtResolution_HLTRecObj, *EtaResolution_HLTRecObj,*PhiResolution_HLTRecObj,
	  *PtCorrelation_L1HLT,*EtaCorrelation_L1HLT,*PhiCorrelation_L1HLT,
	  *PtCorrelation_HLTRecObj,*EtaCorrelation_HLTRecObj,*PhiCorrelation_HLTRecObj,
	  *jetAveragePt, *jetAverageEta, *jetPhiDifference, *hltAveragePt, *hltAverageEta, *hltPhiDifference;

	dummy =  dbe->bookFloat("dummy");    
	N = dbe->bookFloat("N");
	Pt = dbe->bookFloat("Pt");
	PtBarrel = dbe->bookFloat("PtBarrel");
	PtEndcap = dbe->bookFloat("PtEndcap");
	PtForward = dbe->bookFloat("PtForward");
	Eta = dbe->bookFloat("Eta");
	Phi = dbe->bookFloat("Phi");
	EtaPhi = dbe->bookFloat("EtaPhi");
	N_L1 = dbe->bookFloat("N_L1");
	Pt_L1 = dbe->bookFloat("Pt_L1");
	PtBarrel_L1 = dbe->bookFloat("PtBarrel_L1");
	PtEndcap_L1 = dbe->bookFloat("PtEndcap_L1");
	PtForward_L1 = dbe->bookFloat("PtForward_L1");
	Eta_L1 = dbe->bookFloat("Eta_L1");
	Phi_L1 = dbe->bookFloat("Phi_L1");
	EtaPhi_L1 = dbe->bookFloat("EtaPhi_L1");

	N_HLT = dbe->bookFloat("N_HLT");
	Pt_HLT = dbe->bookFloat("Pt_HLT");
	PtBarrel_HLT = dbe->bookFloat("PtBarrel_HLT");
	PtEndcap_HLT = dbe->bookFloat("PtEndcap_HLT");
	PtForward_HLT = dbe->bookFloat("PtForward_HLT");
	Eta_HLT = dbe->bookFloat("Eta_HLT");
	Phi_HLT = dbe->bookFloat("Phi_HLT");
	EtaPhi_HLT = dbe->bookFloat("EtaPhi_HLT");

	PtResolution_L1HLT = dbe->bookFloat("PtResolution_L1HLT");
	EtaResolution_L1HLT = dbe->bookFloat("EtaResolution_L1HLT");
	PhiResolution_L1HLT = dbe->bookFloat("PhiResolution_L1HLT");
	PtResolution_HLTRecObj = dbe->bookFloat("PtResolution_HLTRecObj");
	EtaResolution_HLTRecObj = dbe->bookFloat("EtaResolution_HLTRecObj");
	PhiResolution_HLTRecObj = dbe->bookFloat("PhiResolution_HLTRecObj");
	PtCorrelation_L1HLT = dbe->bookFloat("PtCorrelation_L1HLT");
	EtaCorrelation_L1HLT = dbe->bookFloat("EtaCorrelation_L1HLT");
	PhiCorrelation_L1HLT = dbe->bookFloat("PhiCorrelation_L1HLT");
	PtCorrelation_HLTRecObj = dbe->bookFloat("PtCorrelation_HLTRecObj");
	EtaCorrelation_HLTRecObj = dbe->bookFloat("EtaCorrelation_HLTRecObj");
	PhiCorrelation_HLTRecObj = dbe->bookFloat("PhiCorrelation_HLTRecObj");

	jetAveragePt =  dbe->bookFloat("jetAveragePt");
	jetAverageEta = dbe->bookFloat("jetAverageEta");
	jetPhiDifference = dbe->bookFloat("jetPhiDifference");
	hltAveragePt = dbe->bookFloat("hltAveragePt");
	hltAverageEta = dbe->bookFloat("hltAverageEta");
	hltPhiDifference = dbe->bookFloat("hltPhiDifference");

	std::string labelname("ME");
	std::string histoname(labelname+"");
	std::string title(labelname+"");
	if(v->getObjectType() == trigger::TriggerJet)
	{  

	  histoname = labelname+"_recObjN";
	  title     = labelname+"_recObjN;Reco multiplicity()"+trigPath;
	  N = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
	  TH1 *h = N->getTH1();


	  histoname = labelname+"_recObjPt";
	  title = labelname+"_recObjPt; Reco Pt[GeV/c]"+trigPath;
	  Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = Pt->getTH1();

 
	  histoname = labelname+"_recObjPtBarrel";
	  title = labelname+"_recObjPtBarrel;Reco Pt[GeV/c]"+trigPath;
	  PtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtBarrel->getTH1();


	  histoname = labelname+"_recObjPtEndcap";
	  title = labelname+"_recObjPtEndcap;Reco Pt[GeV/c]"+trigPath;
	  PtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtEndcap->getTH1();


	  histoname = labelname+"_recObjPtForward";
	  title = labelname+"_recObjPtForward;Reco Pt[GeV/c]"+trigPath;
	  PtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtForward->getTH1();


	  histoname = labelname+"_recObjEta";
	  title = labelname+"_recObjEta;Reco #eta"+trigPath;
	  Eta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = Eta->getTH1();


	  histoname = labelname+"_recObjPhi";
	  title = labelname+"_recObjPhi;Reco #Phi"+trigPath;
	  Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi->getTH1();


	  histoname = labelname+"_recObjEtaPhi";
	  title = labelname+"_recObjEtaPhi;Reco #eta;Reco #Phi"+trigPath;
	  EtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = EtaPhi->getTH1();


  
	  histoname = labelname+"_l1ObjN";         
	  title     = labelname+"_l1ObjN;L1 multiplicity"+trigPath;
	  N_L1 = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
	  h = N_L1->getTH1();                                              


	  histoname = labelname+"_l1ObjPt";
	  title = labelname+"_l1ObjPt;L1 Pt[GeV/c]"+trigPath;
	  Pt_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = Pt_L1->getTH1();                                                            

                                                                            
	  histoname = labelname+"_l1ObjPtBarrel";                                    
	  title = labelname+"_l1ObjPtBarrel;L1 Pt[GeV/c]"+trigPath;                              
	  PtBarrel_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtBarrel_L1->getTH1();                                                            


	  histoname = labelname+"_l1ObjPtEndcap";
	  title = labelname+"_l1ObjPtEndcap;L1 Pt[GeV/c]"+trigPath;
	  PtEndcap_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtEndcap_L1->getTH1();                                                            


	  histoname = labelname+"_l1ObjPtForward";
	  title = labelname+"_l1ObjPtForward;L1 Pt[GeV/c]"+trigPath;
	  PtForward_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtForward_L1->getTH1();                                                            


	  histoname = labelname+"_l1ObjEta";
	  title = labelname+"_l1ObjEta;L1 #eta"+trigPath;
	  Eta_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = Eta_L1->getTH1();                                                               


	  histoname = labelname+"_l1ObjPhi";
	  title = labelname+"_l1ObjPhi;L1 #Phi"+trigPath;
	  Phi_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi_L1->getTH1();                                                               


	  histoname = labelname+"_l1ObjEtaPhi";
	  title = labelname+"_l1ObjEtaPhi;L1 #eta;L1 #Phi"+trigPath;
	  EtaPhi_L1 =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = EtaPhi_L1->getTH1();                                                                                        


	  histoname = labelname+"_hltObjN";         
	  title     = labelname+"_hltObjN;HLT multiplicity"+trigPath;
	  N_HLT = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
	  h = N_HLT->getTH1();                                              


	  histoname = labelname+"_hltObjPt";
	  title = labelname+"_hltObjPt;HLT Pt[GeV/c]"+trigPath;
	  Pt_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = Pt_HLT->getTH1();                                                            

                                                                            
	  histoname = labelname+"_hltObjPtBarrel";                                    
	  title = labelname+"_hltObjPtBarrel;HLT Pt[GeV/c]"+trigPath;                              
	  PtBarrel_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtBarrel_HLT->getTH1();                                                            


	  histoname = labelname+"_hltObjPtEndcap";
	  title = labelname+"_hltObjPtEndcap;HLT Pt[GeV/c]"+trigPath;
	  PtEndcap_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtEndcap_HLT->getTH1();                                                            


	  histoname = labelname+"_hltObjPtForward";
	  title = labelname+"_hltObjPtForward;HLT Pt[GeV/c]"+trigPath;
	  PtForward_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtForward_HLT->getTH1();                                                            


	  histoname = labelname+"_hltObjEta";
	  title = labelname+"_hltObjEta;HLT #eta"+trigPath;
	  Eta_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = Eta_HLT->getTH1();                                                               


	  histoname = labelname+"_hltObjPhi";
	  title = labelname+"_hltObjPhi;HLT #Phi"+trigPath;
	  Phi_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi_HLT->getTH1();                                                               


	  histoname = labelname+"_hltObjEtaPhi";
	  title = labelname+"_hltObjEtaPhi;HLT #eta;HLT #Phi"+trigPath;
	  EtaPhi_HLT =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = EtaPhi_HLT->getTH1();                                                                                        


	  histoname = labelname+"_l1HLTPtResolution";
	  title = labelname+"_l1HLTPtResolution;(Pt(L1)-Pt(HLT))/Pt(L1)"+trigPath;
	  PtResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PtResolution_L1HLT->getTH1();


	  histoname = labelname+"_l1HLTEtaResolution";
	  title = labelname+"_l1HLTEtaResolution;(#eta(L1)-#eta(HLT))/#eta(L1)"+trigPath;
	  EtaResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = EtaResolution_L1HLT->getTH1();


	  histoname = labelname+"_l1HLTPhiResolution";
	  title = labelname+"_l1HLTPhiResolution;(#Phi(L1)-#Phi(HLT))/#Phi(L1)"+trigPath;
	  PhiResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PhiResolution_L1HLT->getTH1();


	  histoname = labelname+"_l1HLTPtCorrelation";
	  title = labelname+"_l1HLTPtCorrelation;Pt(L1)[GeV/c];Pt(HLT)[GeV/c]"+trigPath;
	  PtCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
	  h = PtCorrelation_L1HLT->getTH1();


	  histoname = labelname+"_l1HLTEtaCorrelation";
	  title = labelname+"_l1HLTEtaCorrelation;#eta(L1);#eta(HLT)"+trigPath;
	  EtaCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
	  h = EtaCorrelation_L1HLT->getTH1();


	  histoname = labelname+"_l1HLTPhiCorrelation";
	  title = labelname+"_l1HLTPhiCorrelation;#Phi(L1);#Phi(HLT)"+trigPath;
	  PhiCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
	  h = PhiCorrelation_L1HLT->getTH1();


  
	  histoname = labelname+"_hltRecObjPtResolution";
	  title = labelname+"_hltRecObjPtResolution;(Pt(HLT)-Pt(Reco))/Pt(HLT)"+trigPath;
	  PtResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PtResolution_HLTRecObj->getTH1();


	  histoname = labelname+"_hltRecObjEtaResolution";
	  title = labelname+"_hltRecObjEtaResolution;(#eta(HLT)-#eta(Reco))/#eta(HLT)"+trigPath;
	  EtaResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = EtaResolution_HLTRecObj->getTH1();


	  histoname = labelname+"_hltRecObjPhiResolution";
	  title = labelname+"_hltRecObjPhiResolution;(#Phi(HLT)-#Phi(Reco))/#Phi(HLT)"+trigPath;
	  PhiResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PhiResolution_HLTRecObj->getTH1();



	  histoname = labelname+"_hltRecObjPtCorrelation";
	  title = labelname+"_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]"+trigPath;
	  PtCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
	  h = PtCorrelation_HLTRecObj->getTH1();


	  histoname = labelname+"_hltRecObjEtaCorrelation";
	  title = labelname+"_hltRecObjEtaCorrelation;#eta(HLT);#eta(Reco)"+trigPath;
	  EtaCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
	  h = EtaCorrelation_HLTRecObj->getTH1();


	  histoname = labelname+"_hltRecObjPhiCorrelation";
	  title = labelname+"_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)"+trigPath;
	  PhiCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
	  h = PhiCorrelation_HLTRecObj->getTH1();


	  if((v->getTriggerType().compare("DiJet_Trigger") == 0))
	  {
	    histoname = labelname+"_RecObjAveragePt";
	    title     = labelname+"_RecObjAveragePt;Reco Average Pt[GeV/c]"+trigPath;
	    jetAveragePt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	    h = jetAveragePt->getTH1();


	    histoname = labelname+"_RecObjAverageEta";
	    title     = labelname+"_RecObjAverageEta;Reco Average #eta"+trigPath;
	    jetAverageEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	    h = jetAverageEta->getTH1();


	    histoname = labelname+"_RecObjPhiDifference";
	    title     = labelname+"_RecObjPhiDifference;Reco #Delta#Phi"+trigPath;
	    jetPhiDifference = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	    h = jetPhiDifference->getTH1();


	    histoname = labelname+"_hltObjAveragePt";
	    title     = labelname+"_hltObjAveragePt;HLT Average Pt[GeV/c]"+trigPath;
	    hltAveragePt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	    h = hltAveragePt->getTH1();


	    histoname = labelname+"_hltObjAverageEta";
	    title     = labelname+"_hltObjAverageEta;HLT Average #eta"+trigPath;
	    hltAverageEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	    h = hltAverageEta->getTH1();


	    histoname = labelname+"_hltObjPhiDifference";
	    title     = labelname+"_hltObjPhiDifference;Reco #Delta#Phi"+trigPath;
	    hltPhiDifference = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	    h = hltPhiDifference->getTH1();


	  }

	}// histos for Jet Triggers


	if(v->getObjectType() == trigger::TriggerMET)
	{   
	  histoname = labelname+"_recObjPt";
	  title = labelname+"_recObjPt;Reco Pt[GeV/c]"+trigPath;
	  Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 *h = Pt->getTH1();

 

	  histoname = labelname+"_recObjPhi";
	  title = labelname+"_recObjPhi;Reco #Phi"+trigPath;
	  Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi->getTH1();


	  histoname = labelname+"_l1ObjPt";
	  title = labelname+"_l1ObjPt;L1 Pt[GeV/c]"+trigPath;
	  Pt_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = Pt_L1->getTH1();                                                            

                                                                            

	  histoname = labelname+"_l1ObjPhi";
	  title = labelname+"_l1ObjPhi;L1 #Phi"+trigPath;
	  Phi_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi_L1->getTH1();                                                               



	  histoname = labelname+"_hltObjPt";
	  title = labelname+"_hltObjPt;HLT Pt[GeV/c]"+trigPath;
	  Pt_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = Pt_HLT->getTH1();                                                            

                                                                            

	  histoname = labelname+"_hltObjPhi";
	  title = labelname+"_hltObjPhi;HLT #Phi"+trigPath;
	  Phi_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi_HLT->getTH1();                                                               



	  histoname = labelname+"_l1HLTPtResolution";
	  title = labelname+"_l1HLTPtResolution;(Pt(L1)-Pt(HLT))/Pt(L1)"+trigPath;
	  PtResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PtResolution_L1HLT->getTH1();



	  histoname = labelname+"_l1HLTPhiResolution";
	  title = labelname+"_l1HLTPhiResolution;(#Phi(L1)-#Phi(HLT))/#Phi(L1)"+trigPath;
	  PhiResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PhiResolution_L1HLT->getTH1();




	  histoname = labelname+"_l1HLTPtCorrelation";
	  title = labelname+"_l1HLTPtCorrelation;Pt(L1)[GeV/c];Pt(HLT)[GeV/c]"+trigPath;
	  PtCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
	  h = PtCorrelation_L1HLT->getTH1();



	  histoname = labelname+"_l1HLTPhiCorrelation";
	  title = labelname+"_l1HLTPhiCorrelation;#Phi(L1);#Phi(HLT)"+trigPath;
	  PhiCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
	  h = PhiCorrelation_L1HLT->getTH1();



	  histoname = labelname+"_hltRecObjPtResolution";
	  title = labelname+"_hltRecObjPtResolution;(Pt(HLT)-Pt(Reco))/Pt(HLT)"+trigPath;
	  PtResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PtResolution_HLTRecObj->getTH1();



	  histoname = labelname+"_hltRecObjPhiResolution";
	  title = labelname+"_hltRecObjPhiResolution;(#Phi(HLT)-#Phi(Reco))/#Phi(HLT)"+trigPath;
	  PhiResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PhiResolution_HLTRecObj->getTH1();




	  histoname = labelname+"_hltRecObjPtCorrelation";
	  title = labelname+"_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]"+trigPath;
	  PtCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
	  h = PtCorrelation_HLTRecObj->getTH1();



	  histoname = labelname+"_hltRecObjPhiCorrelation";
	  title = labelname+"_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)"+trigPath;
	  PhiCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
	  h = PhiCorrelation_HLTRecObj->getTH1();

   

	}// histos for MET Triggers 

	v->setHistos(  N, Pt,  PtBarrel, PtEndcap, PtForward, Eta, Phi, EtaPhi,
		       N_L1, Pt_L1,  PtBarrel_L1, PtEndcap_L1, PtForward_L1, Eta_L1, Phi_L1, EtaPhi_L1,
		       N_HLT, Pt_HLT,  PtBarrel_HLT, PtEndcap_HLT, PtForward_HLT, Eta_HLT, Phi_HLT, EtaPhi_HLT,
		       PtResolution_L1HLT, EtaResolution_L1HLT,PhiResolution_L1HLT,
		       PtResolution_HLTRecObj,EtaResolution_HLTRecObj,PhiResolution_HLTRecObj,
		       PtCorrelation_L1HLT,EtaCorrelation_L1HLT,PhiCorrelation_L1HLT,
		       PtCorrelation_HLTRecObj,EtaCorrelation_HLTRecObj,PhiCorrelation_HLTRecObj,
		       jetAveragePt, jetAverageEta, jetPhiDifference, hltAveragePt, hltAverageEta, hltPhiDifference,
		       dummy, dummy, dummy);
 
      }

    }
    if(plotAllwrtMu_)
    {
      int Nbins_       = 10;
      int Nmin_        = -0.5;
      int Nmax_        = 9.5;
      int Ptbins_      = 40;
      int Etabins_     = 40;
      int Phibins_     = 35;
      int Resbins_     = 30;
      double PtMin_    = 0.;
      double PtMax_    = 200.;
      double EtaMin_   = -5.;
      double EtaMax_   =  5.;
      double PhiMin_   = -3.14159;
      double PhiMax_   =  3.14159;
      double ResMin_   =  -1.5;
      double ResMax_   =   1.5;
      // Now define histos wrt Muon trigger
      std::string dirName = dirname_ + "/MonitorAllTriggersWrtMuonTrigger/";
      for(PathInfoCollection::iterator v = hltPathsAllWrtMu_.begin(); v!= hltPathsAllWrtMu_.end(); ++v ){
  
	std::string subdirName = dirName + v->getPath();
	std::string trigPath = "("+v->getPath()+")";
	dbe->setCurrentFolder(subdirName);             
                                                   
  
	MonitorElement *dummy, *N, *Pt,  *PtBarrel, *PtEndcap, *PtForward, *Eta, *Phi, *EtaPhi,
	  *N_L1, *Pt_L1,  *PtBarrel_L1, *PtEndcap_L1, *PtForward_L1, *Eta_L1, *Phi_L1, *EtaPhi_L1,
	  *N_HLT, *Pt_HLT,  *PtBarrel_HLT, *PtEndcap_HLT, *PtForward_HLT, *Eta_HLT, *Phi_HLT, *EtaPhi_HLT,
	  *PtResolution_L1HLT, *EtaResolution_L1HLT,*PhiResolution_L1HLT,
	  *PtResolution_HLTRecObj, *EtaResolution_HLTRecObj,*PhiResolution_HLTRecObj,
	  *PtCorrelation_L1HLT,*EtaCorrelation_L1HLT,*PhiCorrelation_L1HLT,
	  *PtCorrelation_HLTRecObj,*EtaCorrelation_HLTRecObj,*PhiCorrelation_HLTRecObj,
	  *jetAveragePt, *jetAverageEta, *jetPhiDifference, *hltAveragePt, *hltAverageEta, *hltPhiDifference;

	dummy =  dbe->bookFloat("dummy");
	N = dbe->bookFloat("N");
	Pt = dbe->bookFloat("Pt");
	PtBarrel = dbe->bookFloat("PtBarrel");
	PtEndcap = dbe->bookFloat("PtEndcap");
	PtForward = dbe->bookFloat("PtForward");
	Eta = dbe->bookFloat("Eta");
	Phi = dbe->bookFloat("Phi");
	EtaPhi = dbe->bookFloat("EtaPhi");
	N_L1 = dbe->bookFloat("N_L1");
	Pt_L1 = dbe->bookFloat("Pt_L1");
	PtBarrel_L1 = dbe->bookFloat("PtBarrel_L1");
	PtEndcap_L1 = dbe->bookFloat("PtEndcap_L1");
	PtForward_L1 = dbe->bookFloat("PtForward_L1");
	Eta_L1 = dbe->bookFloat("Eta_L1");
	Phi_L1 = dbe->bookFloat("Phi_L1");
	EtaPhi_L1 = dbe->bookFloat("EtaPhi_L1");

	N_HLT = dbe->bookFloat("N_HLT");
	Pt_HLT = dbe->bookFloat("Pt_HLT");
	PtBarrel_HLT = dbe->bookFloat("PtBarrel_HLT");
	PtEndcap_HLT = dbe->bookFloat("PtEndcap_HLT");
	PtForward_HLT = dbe->bookFloat("PtForward_HLT");
	Eta_HLT = dbe->bookFloat("Eta_HLT");
	Phi_HLT = dbe->bookFloat("Phi_HLT");
	EtaPhi_HLT = dbe->bookFloat("EtaPhi_HLT");

	PtResolution_L1HLT = dbe->bookFloat("PtResolution_L1HLT");
	EtaResolution_L1HLT = dbe->bookFloat("EtaResolution_L1HLT");
	PhiResolution_L1HLT = dbe->bookFloat("PhiResolution_L1HLT");
	PtResolution_HLTRecObj = dbe->bookFloat("PtResolution_HLTRecObj");
	EtaResolution_HLTRecObj = dbe->bookFloat("EtaResolution_HLTRecObj");
	PhiResolution_HLTRecObj = dbe->bookFloat("PhiResolution_HLTRecObj");
	PtCorrelation_L1HLT = dbe->bookFloat("PtCorrelation_L1HLT");
	EtaCorrelation_L1HLT = dbe->bookFloat("EtaCorrelation_L1HLT");
	PhiCorrelation_L1HLT = dbe->bookFloat("PhiCorrelation_L1HLT");
	PtCorrelation_HLTRecObj = dbe->bookFloat("PtCorrelation_HLTRecObj");
	EtaCorrelation_HLTRecObj = dbe->bookFloat("EtaCorrelation_HLTRecObj");
	PhiCorrelation_HLTRecObj = dbe->bookFloat("PhiCorrelation_HLTRecObj");

	jetAveragePt =  dbe->bookFloat("jetAveragePt");
	jetAverageEta = dbe->bookFloat("jetAverageEta");
	jetPhiDifference = dbe->bookFloat("jetPhiDifference");
	hltAveragePt = dbe->bookFloat("hltAveragePt");
	hltAverageEta = dbe->bookFloat("hltAverageEta");
	hltPhiDifference = dbe->bookFloat("hltPhiDifference");
	std::string labelname("ME");
	std::string histoname(labelname+"");
	std::string title(labelname+"");
	if(v->getObjectType() == trigger::TriggerJet)
	{  

	  histoname = labelname+"_recObjN";
	  title     = labelname+"_recObjN;Reco multiplicity()"+trigPath;
	  N = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
	  TH1 *h = N->getTH1();


	  histoname = labelname+"_recObjPt";
	  title = labelname+"_recObjPt; Reco Pt[GeV/c]"+trigPath;
	  Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = Pt->getTH1();

 
	  histoname = labelname+"_recObjPtBarrel";
	  title = labelname+"_recObjPtBarrel;Reco Pt[GeV/c]"+trigPath;
	  PtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtBarrel->getTH1();


	  histoname = labelname+"_recObjPtEndcap";
	  title = labelname+"_recObjPtEndcap;Reco Pt[GeV/c]"+trigPath;
	  PtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtEndcap->getTH1();


	  histoname = labelname+"_recObjPtForward";
	  title = labelname+"_recObjPtForward;Reco Pt[GeV/c]"+trigPath;
	  PtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtForward->getTH1();


	  histoname = labelname+"_recObjEta";
	  title = labelname+"_recObjEta;Reco #eta"+trigPath;
	  Eta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = Eta->getTH1();


	  histoname = labelname+"_recObjPhi";
	  title = labelname+"_recObjPhi;Reco #Phi"+trigPath;
	  Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi->getTH1();


	  histoname = labelname+"_recObjEtaPhi";
	  title = labelname+"_recObjEtaPhi;Reco #eta;Reco #Phi"+trigPath;
	  EtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = EtaPhi->getTH1();


  
	  histoname = labelname+"_l1ObjN";         
	  title     = labelname+"_l1ObjN;L1 multiplicity"+trigPath;
	  N_L1 = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
	  h = N_L1->getTH1();                                              


	  histoname = labelname+"_l1ObjPt";
	  title = labelname+"_l1ObjPt;L1 Pt[GeV/c]"+trigPath;
	  Pt_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = Pt_L1->getTH1();                                                            

                                                                            
	  histoname = labelname+"_l1ObjPtBarrel";                                    
	  title = labelname+"_l1ObjPtBarrel;L1 Pt[GeV/c]"+trigPath;                              
	  PtBarrel_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtBarrel_L1->getTH1();                                                            


	  histoname = labelname+"_l1ObjPtEndcap";
	  title = labelname+"_l1ObjPtEndcap;L1 Pt[GeV/c]"+trigPath;
	  PtEndcap_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtEndcap_L1->getTH1();                                                            


	  histoname = labelname+"_l1ObjPtForward";
	  title = labelname+"_l1ObjPtForward;L1 Pt[GeV/c]"+trigPath;
	  PtForward_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtForward_L1->getTH1();                                                            


	  histoname = labelname+"_l1ObjEta";
	  title = labelname+"_l1ObjEta;L1 #eta"+trigPath;
	  Eta_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = Eta_L1->getTH1();                                                               


	  histoname = labelname+"_l1ObjPhi";
	  title = labelname+"_l1ObjPhi;L1 #Phi"+trigPath;
	  Phi_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi_L1->getTH1();                                                               


	  histoname = labelname+"_l1ObjEtaPhi";
	  title = labelname+"_l1ObjEtaPhi;L1 #eta;L1 #Phi"+trigPath;
	  EtaPhi_L1 =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = EtaPhi_L1->getTH1();                                                                                        


	  histoname = labelname+"_hltObjN";         
	  title     = labelname+"_hltObjN;HLT multiplicity"+trigPath;
	  N_HLT = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
	  h = N_HLT->getTH1();                                              


	  histoname = labelname+"_hltObjPt";
	  title = labelname+"_hltObjPt;HLT Pt[GeV/c]"+trigPath;
	  Pt_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = Pt_HLT->getTH1();                                                            

                                                                            
	  histoname = labelname+"_hltObjPtBarrel";                                    
	  title = labelname+"_hltObjPtBarrel;HLT Pt[GeV/c]"+trigPath;                              
	  PtBarrel_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtBarrel_HLT->getTH1();                                                            


	  histoname = labelname+"_hltObjPtEndcap";
	  title = labelname+"_hltObjPtEndcap;HLT Pt[GeV/c]"+trigPath;
	  PtEndcap_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtEndcap_HLT->getTH1();                                                            


	  histoname = labelname+"_hltObjPtForward";
	  title = labelname+"_hltObjPtForward;HLT Pt[GeV/c]"+trigPath;
	  PtForward_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = PtForward_HLT->getTH1();                                                            


	  histoname = labelname+"_hltObjEta";
	  title = labelname+"_hltObjEta;HLT #eta"+trigPath;
	  Eta_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = Eta_HLT->getTH1();                                                               


	  histoname = labelname+"_hltObjPhi";
	  title = labelname+"_hltObjPhi;HLT #Phi"+trigPath;
	  Phi_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi_HLT->getTH1();                                                               


	  histoname = labelname+"_hltObjEtaPhi";
	  title = labelname+"_hltObjEtaPhi;HLT #eta;HLT #Phi"+trigPath;
	  EtaPhi_HLT =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = EtaPhi_HLT->getTH1();                                                                                        


	  histoname = labelname+"_l1HLTPtResolution";
	  title = labelname+"_l1HLTPtResolution;(Pt(L1)-Pt(HLT))/Pt(L1)"+trigPath;
	  PtResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PtResolution_L1HLT->getTH1();


	  histoname = labelname+"_l1HLTEtaResolution";
	  title = labelname+"_l1HLTEtaResolution;(#eta(L1)-#eta(HLT))/#eta(L1)"+trigPath;
	  EtaResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = EtaResolution_L1HLT->getTH1();


	  histoname = labelname+"_l1HLTPhiResolution";
	  title = labelname+"_l1HLTPhiResolution;(#Phi(L1)-#Phi(HLT))/#Phi(L1)"+trigPath;
	  PhiResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PhiResolution_L1HLT->getTH1();


	  histoname = labelname+"_l1HLTPtCorrelation";
	  title = labelname+"_l1HLTPtCorrelation;Pt(L1)[GeV/c];Pt(HLT)[GeV/c]"+trigPath;
	  PtCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
	  h = PtCorrelation_L1HLT->getTH1();


	  histoname = labelname+"_l1HLTEtaCorrelation";
	  title = labelname+"_l1HLTEtaCorrelation;#eta(L1);#eta(HLT)"+trigPath;
	  EtaCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
	  h = EtaCorrelation_L1HLT->getTH1();


	  histoname = labelname+"_l1HLTPhiCorrelation";
	  title = labelname+"_l1HLTPhiCorrelation;#Phi(L1);#Phi(HLT)"+trigPath;
	  PhiCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
	  h = PhiCorrelation_L1HLT->getTH1();


  
	  histoname = labelname+"_hltRecObjPtResolution";
	  title = labelname+"_hltRecObjPtResolution;(Pt(HLT)-Pt(Reco))/Pt(HLT)"+trigPath;
	  PtResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PtResolution_HLTRecObj->getTH1();


	  histoname = labelname+"_hltRecObjEtaResolution";
	  title = labelname+"_hltRecObjEtaResolution;(#eta(HLT)-#eta(Reco))/#eta(HLT)"+trigPath;
	  EtaResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = EtaResolution_HLTRecObj->getTH1();


	  histoname = labelname+"_hltRecObjPhiResolution";
	  title = labelname+"_hltRecObjPhiResolution;(#Phi(HLT)-#Phi(Reco))/#Phi(HLT)"+trigPath;
	  PhiResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PhiResolution_HLTRecObj->getTH1();



	  histoname = labelname+"_hltRecObjPtCorrelation";
	  title = labelname+"_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]"+trigPath;
	  PtCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
	  h = PtCorrelation_HLTRecObj->getTH1();


	  histoname = labelname+"_hltRecObjEtaCorrelation";
	  title = labelname+"_hltRecObjEtaCorrelation;#eta(HLT);#eta(Reco)"+trigPath;
	  EtaCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
	  h = EtaCorrelation_HLTRecObj->getTH1();


	  histoname = labelname+"_hltRecObjPhiCorrelation";
	  title = labelname+"_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)"+trigPath;
	  PhiCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
	  h = PhiCorrelation_HLTRecObj->getTH1();


	  if((v->getTriggerType().compare("DiJet_Trigger") == 0))
	  {
	    histoname = labelname+"_RecObjAveragePt";
	    title     = labelname+"_RecObjAveragePt;Reco Average Pt[GeV/c]"+trigPath;
	    jetAveragePt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	    h = jetAveragePt->getTH1();


	    histoname = labelname+"_RecObjAverageEta";
	    title     = labelname+"_RecObjAverageEta;Reco Average #eta"+trigPath;
	    jetAverageEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	    h = jetAverageEta->getTH1();


	    histoname = labelname+"_RecObjPhiDifference";
	    title     = labelname+"_RecObjPhiDifference;Reco #Delta#Phi"+trigPath;
	    jetPhiDifference = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	    h = jetPhiDifference->getTH1();


	    histoname = labelname+"_hltObjAveragePt";
	    title     = labelname+"_hltObjAveragePt;HLT Average Pt[GeV/c]"+trigPath;
	    hltAveragePt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	    h = hltAveragePt->getTH1();


	    histoname = labelname+"_hltObjAverageEta";
	    title     = labelname+"_hltObjAverageEta;HLT Average #eta"+trigPath;
	    hltAverageEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	    h = hltAverageEta->getTH1();


	    histoname = labelname+"_hltObjPhiDifference";
	    title     = labelname+"_hltObjPhiDifference;Reco #Delta#Phi"+trigPath;
	    hltPhiDifference = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	    h = hltPhiDifference->getTH1();


	  }

	}// histos for Jet Triggers
 

	if(v->getObjectType() == trigger::TriggerMET)
	{   
	  histoname = labelname+"_recObjPt";
	  title = labelname+"_recObjPt;Reco Pt[GeV/c]"+trigPath;
	  Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 *h = Pt->getTH1();

 

	  histoname = labelname+"_recObjPhi";
	  title = labelname+"_recObjPhi;Reco #Phi"+trigPath;
	  Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi->getTH1();


	  histoname = labelname+"_l1ObjPt";
	  title = labelname+"_l1ObjPt;L1 Pt[GeV/c]"+trigPath;
	  Pt_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = Pt_L1->getTH1();                                                            

                                                                            

	  histoname = labelname+"_l1ObjPhi";
	  title = labelname+"_l1ObjPhi;L1 #Phi"+trigPath;
	  Phi_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi_L1->getTH1();                                                               



	  histoname = labelname+"_hltObjPt";
	  title = labelname+"_hltObjPt;HLT Pt[GeV/c]"+trigPath;
	  Pt_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = Pt_HLT->getTH1();                                                            

                                                                            

	  histoname = labelname+"_hltObjPhi";
	  title = labelname+"_hltObjPhi;HLT #Phi"+trigPath;
	  Phi_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = Phi_HLT->getTH1();                                                               



	  histoname = labelname+"_l1HLTPtResolution";
	  title = labelname+"_l1HLTPtResolution;(Pt(L1)-Pt(HLT))/Pt(L1)"+trigPath;
	  PtResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PtResolution_L1HLT->getTH1();



	  histoname = labelname+"_l1HLTPhiResolution";
	  title = labelname+"_l1HLTPhiResolution;(#Phi(L1)-#Phi(HLT))/#Phi(L1)"+trigPath;
	  PhiResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PhiResolution_L1HLT->getTH1();




	  histoname = labelname+"_l1HLTPtCorrelation";
	  title = labelname+"_l1HLTPtCorrelation;Pt(L1)[GeV/c];Pt(HLT)[GeV/c]"+trigPath;
	  PtCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
	  h = PtCorrelation_L1HLT->getTH1();



	  histoname = labelname+"_l1HLTPhiCorrelation";
	  title = labelname+"_l1HLTPhiCorrelation;#Phi(L1);#Phi(HLT)"+trigPath;
	  PhiCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
	  h = PhiCorrelation_L1HLT->getTH1();



	  histoname = labelname+"_hltRecObjPtResolution";
	  title = labelname+"_hltRecObjPtResolution;(Pt(HLT)-Pt(Reco))/Pt(HLT)"+trigPath;
	  PtResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PtResolution_HLTRecObj->getTH1();



	  histoname = labelname+"_hltRecObjPhiResolution";
	  title = labelname+"_hltRecObjPhiResolution;(#Phi(HLT)-#Phi(Reco))/#Phi(HLT)"+trigPath;
	  PhiResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
	  h = PhiResolution_HLTRecObj->getTH1();




	  histoname = labelname+"_hltRecObjPtCorrelation";
	  title = labelname+"_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]"+trigPath;
	  PtCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
	  h = PtCorrelation_HLTRecObj->getTH1();



	  histoname = labelname+"_hltRecObjPhiCorrelation";
	  title = labelname+"_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)"+trigPath;
	  PhiCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
	  h = PhiCorrelation_HLTRecObj->getTH1();

   

	}// histos for MET Triggers 
	v->setHistos(  N, Pt,  PtBarrel, PtEndcap, PtForward, Eta, Phi, EtaPhi,
		       N_L1, Pt_L1,  PtBarrel_L1, PtEndcap_L1, PtForward_L1, Eta_L1, Phi_L1, EtaPhi_L1,
		       N_HLT, Pt_HLT,  PtBarrel_HLT, PtEndcap_HLT, PtForward_HLT, Eta_HLT, Phi_HLT, EtaPhi_HLT,
		       PtResolution_L1HLT, EtaResolution_L1HLT,PhiResolution_L1HLT,
		       PtResolution_HLTRecObj,EtaResolution_HLTRecObj,PhiResolution_HLTRecObj,
		       PtCorrelation_L1HLT,EtaCorrelation_L1HLT,PhiCorrelation_L1HLT,
		       PtCorrelation_HLTRecObj,EtaCorrelation_HLTRecObj,PhiCorrelation_HLTRecObj,
		       jetAveragePt, jetAverageEta, jetPhiDifference, hltAveragePt, hltAverageEta, hltPhiDifference,
		       dummy, dummy, dummy);
 
      }

    }
    //-------Now Efficiency histos--------
    if(plotEff_)
    {
      int Ptbins_      = 100;
      int Etabins_     = 40;
      int Phibins_     = 35;
      double PtMin_    = 0.;
      double PtMax_    = 200.;
      double EtaMin_   = -5.;
      double EtaMax_   =  5.;
      double PhiMin_   = -3.14159;
      double PhiMax_   =  3.14159;
      // Now define histos wrt lower threshold trigger
      std::string dirName1 = dirname_ + "/RelativeTriggerEff/";
      for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v ){
	std::string labelname("ME") ;
	std::string subdirName = dirName1 + v->getPath() + "_wrt_" + v->getDenomPath();
	dbe->setCurrentFolder(subdirName);
	std::string histoname(labelname+"");
	std::string title(labelname+"");
   
	MonitorElement *dummy;
	dummy =  dbe->bookFloat("dummy");   
    
	if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType().compare("SingleJet_Trigger") == 0))
	{ 
	  histoname = labelname+"_NumeratorPt";
	  title     = labelname+"NumeratorPt;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 * h = NumeratorPt->getTH1();


	  histoname = labelname+"_NumeratorPtBarrel";
	  title     = labelname+"NumeratorPtBarrel;Calo Pt[GeV/c] ";
	  MonitorElement * NumeratorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = NumeratorPtBarrel->getTH1();


	  histoname = labelname+"_NumeratorPtEndcap";
	  title     = labelname+"NumeratorPtEndcap;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = NumeratorPtEndcap->getTH1();


	  histoname = labelname+"_NumeratorPtForward";
	  title     = labelname+"NumeratorPtForward;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = NumeratorPtForward->getTH1();


	  histoname = labelname+"_NumeratorEta";
	  title     = labelname+"NumeratorEta;Calo #eta ";
	  MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = NumeratorEta->getTH1();


	  histoname = labelname+"_NumeratorPhi";
	  title     = labelname+"NumeratorPhi;Calo #Phi";
	  MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = NumeratorPhi->getTH1();


	  histoname = labelname+"_NumeratorEtaPhi";
	  title     = labelname+"NumeratorEtaPhi;Calo #eta;Calo #Phi";
	  MonitorElement * NumeratorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = NumeratorEtaPhi->getTH1();


	  histoname = labelname+"_DenominatorPt";
	  title     = labelname+"DenominatorPt;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPt->getTH1();


	  histoname = labelname+"_DenominatorPtBarrel";
	  title     = labelname+"DenominatorPtBarrel;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPtBarrel->getTH1();


	  histoname = labelname+"_DenominatorPtEndcap";
	  title     = labelname+"DenominatorPtEndcap;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPtEndcap->getTH1();


	  histoname = labelname+"_DenominatorPtForward";
	  title     = labelname+"DenominatorPtForward;Calo Pt[GeV/c] ";
	  MonitorElement * DenominatorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPtForward->getTH1();


	  histoname = labelname+"_DenominatorEta";
	  title     = labelname+"DenominatorEta;Calo #eta ";
	  MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = DenominatorEta->getTH1();


	  histoname = labelname+"_DenominatorPhi";
	  title     = labelname+"DenominatorPhi;Calo #Phi";
	  MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = DenominatorPhi->getTH1();


	  histoname = labelname+"_DenominatorEtaPhi";
	  title     = labelname+"DenominatorEtaPhi;Calo #eta; Calo #Phi";
	  MonitorElement * DenominatorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = DenominatorEtaPhi->getTH1();



	  v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
			    DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi);
	}// Loop over Jet Trigger

	if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType().compare("DiJet_Trigger") == 0))
	{

	  histoname = labelname+"_NumeratorAvrgPt";
	  title     = labelname+"NumeratorAvrgPt;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 * h = NumeratorPt->getTH1();


	  histoname = labelname+"_NumeratorAvrgEta";
	  title     = labelname+"NumeratorAvrgEta;Calo #eta";
	  MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = NumeratorEta->getTH1();


	  histoname = labelname+"_DenominatorAvrgPt";
	  title     = labelname+"DenominatorAvrgPt;Calo Pt[GeV/c] ";
	  MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPt->getTH1();


	  histoname = labelname+"_DenominatorAvrgEta";
	  title     = labelname+"DenominatorAvrgEta;Calo #eta";
	  MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = DenominatorEta->getTH1();


	  v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, NumeratorEta, dummy, dummy,
			    DenominatorPt,  dummy, dummy, dummy, DenominatorEta, dummy, dummy);
   

	}

	if(v->getObjectType() == trigger::TriggerMET)
	{
	  histoname = labelname+"_NumeratorPt";
	  title     = labelname+"NumeratorPt;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 * h = NumeratorPt->getTH1();



	  histoname = labelname+"_NumeratorPhi";
	  title     = labelname+"NumeratorPhi;Calo #Phi";
	  MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = NumeratorPhi->getTH1();



	  histoname = labelname+"_DenominatorPt";
	  title     = labelname+"DenominatorPt;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPt->getTH1();


   
	  histoname = labelname+"_DenominatorPhi";
	  title     = labelname+"DenominatorPhi;Calo #Phi";
	  MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = DenominatorPhi->getTH1();

   

	  v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, dummy, NumeratorPhi, dummy,
			    DenominatorPt, dummy, dummy, dummy, dummy, DenominatorPhi, dummy);


	}// Loop over MET trigger
      }

      //------Efficiency wrt Muon trigger-----------------------
      std::string dirName2 = dirname_ + "/EffWrtMuonTrigger/";
      for(PathInfoCollection::iterator v = hltPathsEffWrtMu_.begin(); v!= hltPathsEffWrtMu_.end(); ++v ){
	std::string labelname("ME") ;
	std::string subdirName = dirName2 + v->getPath();
	std::string histoname(labelname+"");
	std::string title(labelname+"");
	dbe->setCurrentFolder(subdirName);

	MonitorElement *dummy;
	dummy =  dbe->bookFloat("dummy");
	if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType().compare("SingleJet_Trigger") == 0))
	{ 
	  histoname = labelname+"_NumeratorPt";
	  title     = labelname+"NumeratorPt;Pt[GeV/c]";
	  MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 * h = NumeratorPt->getTH1();


	  histoname = labelname+"_NumeratorPtBarrel";
	  title     = labelname+"NumeratorPtBarrel;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = NumeratorPtBarrel->getTH1();


	  histoname = labelname+"_NumeratorPtEndcap";
	  title     = labelname+"NumeratorPtEndcap;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = NumeratorPtEndcap->getTH1();


	  histoname = labelname+"_NumeratorPtForward";
	  title     = labelname+"NumeratorPtForward;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = NumeratorPtForward->getTH1();


	  histoname = labelname+"_NumeratorEta";
	  title     = labelname+"NumeratorEta;Calo #eta ";
	  MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = NumeratorEta->getTH1();


	  histoname = labelname+"_NumeratorPhi";
	  title     = labelname+"NumeratorPhi;Calo #Phi";
	  MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = NumeratorPhi->getTH1();


	  histoname = labelname+"_NumeratorEtaPhi";
	  title     = labelname+"NumeratorEtaPhi;Calo #eta;Calo #Phi";
	  MonitorElement * NumeratorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = NumeratorEtaPhi->getTH1();


	  histoname = labelname+"_DenominatorPt";
	  title     = labelname+"DenominatorPt;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPt->getTH1();


	  histoname = labelname+"_DenominatorPtBarrel";
	  title     = labelname+"DenominatorPtBarrel;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPtBarrel->getTH1();


	  histoname = labelname+"_DenominatorPtEndcap";
	  title     = labelname+"DenominatorPtEndcap;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPtEndcap->getTH1();


	  histoname = labelname+"_DenominatorPtForward";
	  title     = labelname+"DenominatorPtForward;Calo Pt[GeV/c] ";
	  MonitorElement * DenominatorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPtForward->getTH1();


	  histoname = labelname+"_DenominatorEta";
	  title     = labelname+"DenominatorEta;Calo #eta";
	  MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = DenominatorEta->getTH1();


	  histoname = labelname+"_DenominatorPhi";
	  title     = labelname+"DenominatorPhi;Calo #Phi";
	  MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = DenominatorPhi->getTH1();


	  histoname = labelname+"_DenominatorEtaPhi";
	  title     = labelname+"DenominatorEtaPhi;Calo #eta (IC5);Calo #Phi ";
	  MonitorElement * DenominatorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = DenominatorEtaPhi->getTH1();



	  v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
			    DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi);
	}// Loop over Jet Trigger

	if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType().compare("DiJet_Trigger") == 0))
	{
	  histoname = labelname+"_NumeratorAvrgPt";
	  title     = labelname+"NumeratorAvrgPt;Calo Pt[GeV/c] ";
	  MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 * h = NumeratorPt->getTH1();


	  histoname = labelname+"_NumeratorAvrgEta";
	  title     = labelname+"NumeratorAvrgEta;Calo #eta";
	  MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = NumeratorEta->getTH1();


	  histoname = labelname+"_DenominatorAvrgPt";
	  title     = labelname+"DenominatorAvrgPt;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPt->getTH1();


	  histoname = labelname+"_DenominatorAvrgEta";
	  title     = labelname+"DenominatorAvrgEta;Calo #eta ";
	  MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = DenominatorEta->getTH1();


	  v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, NumeratorEta, dummy, dummy,
			    DenominatorPt,  dummy, dummy, dummy, DenominatorEta, dummy, dummy);
   

	}
	if(v->getObjectType() == trigger::TriggerMET)
	{
	  histoname = labelname+"_NumeratorPt";
	  title     = labelname+"NumeratorPt;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 * h = NumeratorPt->getTH1();



	  histoname = labelname+"_NumeratorPhi";
	  title     = labelname+"NumeratorPhi;Calo #Phi";
	  MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = NumeratorPhi->getTH1();



	  histoname = labelname+"_DenominatorPt";
	  title     = labelname+"DenominatorPt;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPt->getTH1();


   
	  histoname = labelname+"_DenominatorPhi";
	  title     = labelname+"DenominatorPhi;Calo #Phi";
	  MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = DenominatorPhi->getTH1();

   

	  v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, dummy, NumeratorPhi, dummy,
			    DenominatorPt, dummy, dummy, dummy, dummy, DenominatorPhi, dummy);


	}// Loop over MET trigger 
      }
      //--------Efficiency  wrt MiniBias trigger---------
      std::string dirName3  = dirname_ + "/EffWrtMBTrigger/";
      for(PathInfoCollection::iterator v = hltPathsEffWrtMB_.begin(); v!= hltPathsEffWrtMB_.end(); ++v ){
	std::string labelname("ME") ;
	std::string subdirName = dirName3 + v->getPath() ;
	std::string histoname(labelname+"");
	std::string title(labelname+"");
	dbe->setCurrentFolder(subdirName);
	MonitorElement *dummy;
	dummy =  dbe->bookFloat("dummy");   
	if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType().compare("SingleJet_Trigger") == 0))
	{ 
	  histoname = labelname+"_NumeratorPt";
	  title     = labelname+"NumeratorPt;Calo Pt[GeV/c] ";
	  MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 * h = NumeratorPt->getTH1();


	  histoname = labelname+"_NumeratorPtBarrel";
	  title     = labelname+"NumeratorPtBarrel;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = NumeratorPtBarrel->getTH1();


	  histoname = labelname+"_NumeratorPtEndcap";
	  title     = labelname+"NumeratorPtEndcap; Calo Pt[GeV/c] ";
	  MonitorElement * NumeratorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = NumeratorPtEndcap->getTH1();


	  histoname = labelname+"_NumeratorPtForward";
	  title     = labelname+"NumeratorPtForward;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = NumeratorPtForward->getTH1();


	  histoname = labelname+"_NumeratorEta";
	  title     = labelname+"NumeratorEta;Calo #eta ";
	  MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = NumeratorEta->getTH1();


	  histoname = labelname+"_NumeratorPhi";
	  title     = labelname+"NumeratorPhi;Calo #Phi";
	  MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = NumeratorPhi->getTH1();


	  histoname = labelname+"_NumeratorEtaPhi";
	  title     = labelname+"NumeratorEtaPhi;Calo #eta;Calo #Phi ";
	  MonitorElement * NumeratorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = NumeratorEtaPhi->getTH1();


	  histoname = labelname+"_DenominatorPt";
	  title     = labelname+"DenominatorPt;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPt->getTH1();


	  histoname = labelname+"_DenominatorPtBarrel";
	  title     = labelname+"DenominatorPtBarrel;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPtBarrel->getTH1();


	  histoname = labelname+"_DenominatorPtEndcap";
	  title     = labelname+"DenominatorPtEndcap;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPtEndcap->getTH1();


	  histoname = labelname+"_DenominatorPtForward";
	  title     = labelname+"DenominatorPtForward;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPtForward->getTH1();


	  histoname = labelname+"_DenominatorEta";
	  title     = labelname+"DenominatorEta;Calo #eta ";
	  MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = DenominatorEta->getTH1();


	  histoname = labelname+"_DenominatorPhi";
	  title     = labelname+"DenominatorPhi;Calo #Phi";
	  MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = DenominatorPhi->getTH1();


	  histoname = labelname+"_DenominatorEtaPhi";
	  title     = labelname+"DenominatorEtaPhi;Calo #eta ;Calo #Phi ";
	  MonitorElement * DenominatorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
	  h = DenominatorEtaPhi->getTH1();



	  v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
			    DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi);
	}// Loop over Jet Trigger


	if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType().compare("DiJet_Trigger") == 0))
	{

	  histoname = labelname+"_NumeratorAvrgPt";
	  title     = labelname+"NumeratorAvrgPt;Calo Pt[GeV/c] ";
	  MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 * h = NumeratorPt->getTH1();


	  histoname = labelname+"_NumeratorAvrgEta";
	  title     = labelname+"NumeratorAvrgEta;Calo #eta ";
	  MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = NumeratorEta->getTH1();


	  histoname = labelname+"_DenominatorAvrgPt";
	  title     = labelname+"DenominatorAvrgPt;Calo Pt[GeV/c] ";
	  MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPt->getTH1();


	  histoname = labelname+"_DenominatorAvrgEta";
	  title     = labelname+"DenominatorAvrgEta;Calo #eta ";
	  MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	  h = DenominatorEta->getTH1();


	  v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, NumeratorEta, dummy, dummy,
			    DenominatorPt,  dummy, dummy, dummy, DenominatorEta, dummy, dummy);

    

	}
	if(v->getObjectType() == trigger::TriggerMET)
	{
	  histoname = labelname+"_NumeratorPt";
	  title     = labelname+"NumeratorPt;Calo Pt[GeV/c]";
	  MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  TH1 * h = NumeratorPt->getTH1();



	  histoname = labelname+"_NumeratorPhi";
	  title     = labelname+"NumeratorPhi;Calo #Phi";
	  MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = NumeratorPhi->getTH1();



	  histoname = labelname+"_DenominatorPt";
	  title     = labelname+"DenominatorPt;Calo Pt[GeV/c]";
	  MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	  h = DenominatorPt->getTH1();


   
	  histoname = labelname+"_DenominatorPhi";
	  title     = labelname+"DenominatorPhi;Calo #Phi";
	  MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	  h = DenominatorPhi->getTH1();

   



	  v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, dummy, NumeratorPhi, dummy,
			    DenominatorPt, dummy, dummy, dummy, dummy, DenominatorPhi, dummy);


	}// Loop over MET trigger
      }


    }// This is loop over all efficiency plots
    //--------Histos to see WHY trigger is NOT fired----------
    int Nbins_       = 10;
    int Nmin_        = 0;
    int Nmax_        = 10;
    int Ptbins_      = 100;
    int Etabins_     = 40;
    int Phibins_     = 35;
    double PtMin_    = 0.;
    double PtMax_    = 200.;
    double EtaMin_   = -5.;
    double EtaMax_   =  5.;
    double PhiMin_   = -3.14159;
    double PhiMax_   =  3.14159;

    std::string dirName4_ = dirname_ + "/TriggerNotFired/";
    for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){

      MonitorElement *dummy;
      dummy =  dbe->bookFloat("dummy");
 

      std::string labelname("ME") ;
      std::string histoname(labelname+"");
      std::string title(labelname+"");
      dbe->setCurrentFolder(dirName4_ + v->getPath());
  
      histoname = labelname+"_TriggerSummary";
      title     = labelname+"Summary of trigger levels"; 
      MonitorElement * TriggerSummary = dbe->book1D(histoname.c_str(),title.c_str(),7, -0.5,6.5);

      std::vector<std::string> trigger;
      trigger.push_back("Nevt");
      trigger.push_back("L1 failed");
      trigger.push_back("L1 & HLT failed");
      trigger.push_back("L1 failed but not HLT");
      trigger.push_back("L1 passed");
      trigger.push_back("L1 & HLT passed");
      trigger.push_back("L1 passed but not HLT");
      for(unsigned int i =0; i < trigger.size(); i++)TriggerSummary->setBinLabel(i+1, trigger[i]);

      if((v->getTriggerType().compare("SingleJet_Trigger") == 0))
      {
	histoname = labelname+"_JetPt"; 
	title     = labelname+"Leading jet pT;Pt[GeV/c]";
	MonitorElement * JetPt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	TH1 * h = JetPt->getTH1();


	histoname = labelname+"_JetEtaVsPt";
	title     = labelname+"Leading jet #eta vs pT;#eta;Pt[GeV/c]";
	MonitorElement * JetEtaVsPt = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Ptbins_,PtMin_,PtMax_);
	h = JetEtaVsPt->getTH1();


	histoname = labelname+"_JetPhiVsPt";
	title     = labelname+"Leading jet #Phi vs pT;#Phi;Pt[GeV/c]";
	MonitorElement * JetPhiVsPt = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Ptbins_,PtMin_,PtMax_);
	h = JetPhiVsPt->getTH1();


   
   
	v->setDgnsHistos( TriggerSummary, dummy, JetPt, JetEtaVsPt, JetPhiVsPt, dummy, dummy, dummy, dummy, dummy, dummy); 
      }// single Jet trigger  

      if((v->getTriggerType().compare("DiJet_Trigger") == 0))
      {
	histoname = labelname+"_JetSize"; 
	title     = labelname+"Jet Size;multiplicity";
	MonitorElement * JetSize = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
	TH1 * h = JetSize->getTH1();



	histoname = labelname+"_AvergPt";
	title     = labelname+"Average Pt;Pt[GeV/c]";
	MonitorElement * Pt12 = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	h = Pt12->getTH1();


	histoname = labelname+"_AvergEta";
	title     = labelname+"Average Eta;#eta";
	MonitorElement * Eta12 = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
	h = Eta12->getTH1();


	histoname = labelname+"_PhiDifference";
	title     = labelname+"#Delta#Phi;#Delta#Phi";
	MonitorElement * Phi12 = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
	h = Phi12->getTH1();


	histoname = labelname+"_Pt3Jet";
	title     = labelname+"Pt of 3rd Jet;Pt[GeV/c]";
	MonitorElement * Pt3 = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	h = Pt3->getTH1();


	histoname = labelname+"_Pt12VsPt3Jet";
	title     = labelname+"Pt of 3rd Jet vs Average Pt of leading jets;Avergage Pt[GeV/c]; Pt of 3rd Jet [GeV/c]";
	MonitorElement * Pt12Pt3 = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
	h = Pt12Pt3->getTH1();


	histoname = labelname+"_Pt12VsPhi12";
	title     = labelname+"Average Pt of leading jets vs #Delta#Phi between leading jets;Avergage Pt[GeV/c]; #Delta#Phi";
	MonitorElement * Pt12Phi12 = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Phibins_,PhiMin_,PhiMax_);
	h = Pt12Phi12->getTH1();


	v->setDgnsHistos( TriggerSummary, JetSize, dummy, dummy, dummy, Pt12, Eta12, Phi12, Pt3, Pt12Pt3, Pt12Phi12);

      }// Dijet Jet trigger

      if((v->getTriggerType().compare("MET_Trigger") == 0))
      {
	histoname = labelname+"_MET";
	title     = labelname+"MET;Pt[GeV/c]";
	MonitorElement * MET = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	//	TH1 * h = MET->getTH1();

	v->setDgnsHistos(TriggerSummary, dummy, MET, dummy, dummy, dummy, dummy, dummy,dummy,dummy,dummy);
      } // MET trigger  


    }

  }  
}

//--------------------------------------------------------
void JetMETHLTOfflineSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
						  const EventSetup& context) {
}
//--------------------------------------------------------
void JetMETHLTOfflineSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
						const EventSetup& context) {
}
// - method called once each job just after ending the event loop  ------------
void 
JetMETHLTOfflineSource::endJob() {
  delete jetID;
}

/// EndRun
void JetMETHLTOfflineSource::endRun(const edm::Run& run, const edm::EventSetup& c){
  if (verbose_) std::cout << "endRun, run " << run.id() << std::endl;
}

bool JetMETHLTOfflineSource::isBarrel(double eta){
  bool output = false;
  if (fabs(eta)<=1.3) output=true;
  return output;
}


bool JetMETHLTOfflineSource::isEndCap(double eta){
  bool output = false;
  if (fabs(eta)<=3.0 && fabs(eta)>1.3) output=true;
  return output;
}


bool JetMETHLTOfflineSource::isForward(double eta){
  bool output = false;
  if (fabs(eta)>3.0) output=true;
  return output;
}


bool JetMETHLTOfflineSource::validPathHLT(std::string pathname){
  // hltConfig_ has to be defined first before calling this method
  bool output=false;
  for (unsigned int j=0; j!=hltConfig_.size(); ++j) {
    if (hltConfig_.triggerName(j) == pathname )
      output=true;
  }
  return output;
}

bool JetMETHLTOfflineSource::isHLTPathAccepted(std::string pathName){
  // triggerResults_, triggerNames_ has to be defined first before calling this method
  bool output=false;
  if(&triggerResults_) {
    unsigned index = triggerNames_.triggerIndex(pathName);
    if(index < triggerNames_.size() && triggerResults_->accept(index)) output = true;
  }
  return output;
}
// This returns the position of trigger name defined in summary histograms
double JetMETHLTOfflineSource::TriggerPosition(std::string trigName){
  int nbins = rate_All->getTH1()->GetNbinsX();
  double binVal = -100;
  for(int ibin=1; ibin<nbins+1; ibin++)
  {
    const char * binLabel = rate_All->getTH1()->GetXaxis()->GetBinLabel(ibin);
    if(binLabel[0]=='\0')continue;
    //       std::string binLabel_str = string(binLabel);
    //       if(binLabel_str.compare(trigName)!=0)continue;
    if(trigName.compare(binLabel)!=0)continue;

    if(trigName.compare(binLabel)==0){
      binVal = rate_All->getTH1()->GetBinCenter(ibin);
      break;
    }
  }
  return binVal;
}
bool JetMETHLTOfflineSource::isTriggerObjectFound(std::string objectName){
  // processname_, triggerObj_ has to be defined before calling this method
  bool output=false;
  edm::InputTag testTag(objectName,"",processname_);
  const int index = triggerObj_->filterIndex(testTag);
  if ( index >= triggerObj_->sizeFilters() ) {    
    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";
  } else {       
    const trigger::Keys & k = triggerObj_->filterKeys(index);
    if (k.size()) output=true;
  }
  return output;
}



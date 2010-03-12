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
#include "FWCore/Framework/interface/TriggerNames.h"
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
using namespace std;

  
JetMETHLTOfflineSource::JetMETHLTOfflineSource(const edm::ParameterSet& iConfig)
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
  // plotting paramters
  MuonTrigPaths_ = iConfig.getUntrackedParameter<vector<std::string> >("pathnameMuon");
  MBTrigPaths_ = iConfig.getUntrackedParameter<vector<std::string> >("pathnameMB");
  caloJetsTag_ = iConfig.getParameter<edm::InputTag>("CaloJetCollectionLabel");
  caloMETTag_ = iConfig.getParameter<edm::InputTag>("CaloMETCollectionLabel"); 
  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");

  


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
    
    triggerNames_.init(*(triggerResults_.product()));

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
  if(plotAll_)fillMEforMonAllTrigger();
  if(plotAllwrtMu_)fillMEforMonAllTriggerwrtMuonTrigger();
  if(plotEff_)
  {
  fillMEforEffAllTrigger(); 
  fillMEforEffWrtMuTrigger();
  fillMEforEffWrtMBTrigger();
  }
  fillMEforTriggerNTfired();
}


void JetMETHLTOfflineSource::fillMEforMonTriggerSummary(){
  // Trigger summary for all paths
    int trignum_1 = -1;
    for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v )
        {
         trignum_1++;
         if(isHLTPathAccepted(v->getPath()))rate_All->Fill(trignum_1);
         if(isHLTPathAccepted(v->getPath()))correlation_All->Fill(trignum_1,trignum_1);
         int trignum_2 = trignum_1;
         for(PathInfoCollection::iterator w = v+1; w!= hltPathsAll_.end(); ++w )
           {
           trignum_2++;
           if(isHLTPathAccepted(w->getPath()) && isHLTPathAccepted(v->getPath()))correlation_All->Fill(trignum_1,trignum_2);
           if(!isHLTPathAccepted(w->getPath()) && isHLTPathAccepted(v->getPath()))correlation_All->Fill(trignum_2,trignum_1); 
           }
        }
  // Trigger summary for all paths wrt Muon trigger
    bool muTrig = false;
    for(size_t i=0;i<MuonTrigPaths_.size();++i)if(isHLTPathAccepted(MuonTrigPaths_[i]))muTrig = true;  
    if(muTrig )
    { 
    trignum_1 = -1;
    for(PathInfoCollection::iterator v = hltPathsWrtMu_.begin(); v!= hltPathsWrtMu_.end(); ++v )
        {
         trignum_1++;
         if(isHLTPathAccepted(v->getPath()))rate_AllWrtMu->Fill(trignum_1);
         if(isHLTPathAccepted(v->getPath()))correlation_AllWrtMu->Fill(trignum_1,trignum_1);
         int trignum_2 = trignum_1;
         for(PathInfoCollection::iterator w = v+1; w!= hltPathsWrtMu_.end(); ++w )
           {
           trignum_2++;
           if(isHLTPathAccepted(w->getPath()) && isHLTPathAccepted(v->getPath()))correlation_AllWrtMu->Fill(trignum_1,trignum_2);
           if(!isHLTPathAccepted(w->getPath()) && isHLTPathAccepted(v->getPath()))correlation_AllWrtMu->Fill(trignum_2,trignum_1);
           }
        } 
  }// Muon trigger fired

 // Trigger summary for all paths wrt MB trigger
    bool mbTrig = false;
    for(size_t i=0;i<MBTrigPaths_.size();++i)if(isHLTPathAccepted(MBTrigPaths_[i]))mbTrig = true;
    if(mbTrig )
    {
    trignum_1 = -1;
    for(PathInfoCollection::iterator v = hltPathsEffWrtMB_.begin(); v!= hltPathsEffWrtMB_.end(); ++v )
        {
         trignum_1++;
         if(isHLTPathAccepted(v->getPath()))rate_AllWrtMB->Fill(trignum_1);
         if(isHLTPathAccepted(v->getPath()))correlation_AllWrtMB->Fill(trignum_1,trignum_1);
         int trignum_2 = trignum_1;
         for(PathInfoCollection::iterator w = v+1; w!= hltPathsEffWrtMB_.end(); ++w )
           {
           trignum_2++;
           if(isHLTPathAccepted(w->getPath()) && isHLTPathAccepted(v->getPath()))correlation_AllWrtMB->Fill(trignum_1,trignum_2);
           if(!isHLTPathAccepted(w->getPath()) && isHLTPathAccepted(v->getPath()))correlation_AllWrtMB->Fill(trignum_2,trignum_1);
           }
        }
  }// MB trigger fired

}

void JetMETHLTOfflineSource::fillMEforTriggerNTfired(){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  for(PathInfoCollection::iterator v = hltPathsNTfired_.begin(); v!= hltPathsNTfired_.end(); ++v )
    {
     for(int i = 0; i < npath; ++i) {
       if (triggerNames_.triggerName(i).find(v->getPath()) != std::string::npos && !(triggerResults_->accept(i)))
             {
             if(v->getTriggerType() == "SingleJet_Trigger" && calojetColl_.isValid())
              {
              CaloJetCollection::const_iterator jet = calojet.begin();
              v->getMEhisto_JetPt()->Fill(jet->pt());
              v->getMEhisto_JetEta()->Fill(jet->eta());      
              v->getMEhisto_EtavsPt1()->Fill(jet->eta(),jet->pt());
                 
              }// single jet trigger is not fired

            if(v->getTriggerType() == "DiJet_Trigger" && calojetColl_.isValid())
              {
              v->getMEhisto_JetSize()->Fill(calojet.size()) ;
              if (calojet.size()>=2){
               CaloJetCollection::const_iterator jet = calojet.begin();
               CaloJetCollection::const_iterator jet2= calojet.begin(); jet2++;
               v->getMEhisto_EtavsPt1()->Fill(jet->eta(),jet->pt());
               v->getMEhisto_EtavsPt2()->Fill(jet2->eta(),jet2->pt());
               v->getMEhisto_Pt12()->Fill(jet->pt(),jet2->pt());
               v->getMEhisto_Eta12()->Fill(jet->eta(),jet2->eta());
              }
             }// di jet trigger is not fired 

           if(v->getTriggerType() == "MET_Trigger" && calometColl_.isValid())
              {
              const CaloMETCollection *calometcol = calometColl_.product();
              const CaloMET met = calometcol->front();
              v->getMEhisto_JetPt()->Fill(met.pt());
          }//MET trigger is not fired   

       }// trigger not fired
      }//All paths
     }//path collections  
 


}


void JetMETHLTOfflineSource::fillMEforMonAllTrigger(){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects()); 
  for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v )
    {
     for(int i = 0; i < npath; ++i) {
       if (triggerNames_.triggerName(i).find(v->getPath()) != std::string::npos && triggerResults_->accept(i))
             {
            edm::InputTag l1Tag(v->getl1Path(),"",processname_);
            const int l1Index = triggerObj_->filterIndex(l1Tag);
            edm::InputTag hltTag(v->getLabel(),"",processname_);
            const int hltIndex = triggerObj_->filterIndex(hltTag);
            bool l1TrigBool = false;
            bool hltTrigBool = false;
            int jetsize = 0;
            if ( l1Index >= triggerObj_->sizeFilters() ) {
            edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< l1Index << " of that name "<<l1Tag;
            } else {
             l1TrigBool = true;
             const trigger::Keys & kl1 = triggerObj_->filterKeys(l1Index);
             std::string l1name = triggerObj_->filterTag(l1Index).encode();
             if(v->getObjectType() == trigger::TriggerJet)v->getMEhisto_N_L1()->Fill(kl1.size());
             for( trigger::Keys::const_iterator ki = kl1.begin(); ki != kl1.end(); ++ki)
              {
               double l1TrigEta = 0;
               double l1TrigPhi = 0;
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
               
                if ( hltIndex >= triggerObj_->sizeFilters() ) {
                edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt"<< hltIndex << " of that name ";
                } else {
                const trigger::Keys & khlt = triggerObj_->filterKeys(hltIndex);
                std::string hltname = triggerObj_->filterTag(hltIndex).encode();
                 if(v->getObjectType() == trigger::TriggerJet)v->getMEhisto_N_HLT()->Fill(khlt.size());
                 for(trigger::Keys::const_iterator kj = khlt.begin();kj != khlt.end(); ++kj)
                  {
                  hltTrigBool = true;
                  double hltTrigEta = 0.;
                  double hltTrigPhi = 0.;
                  if(v->getObjectType() == trigger::TriggerJet)
                   {
                   hltTrigEta = toc[*kj].eta();
                   hltTrigPhi = toc[*kj].phi();
                   
                   if((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi)) < 0.4)
                    {
                    v->getMEhisto_Pt_HLT()->Fill(toc[*kj].pt());
                    if (isBarrel(toc[*kj].eta()))  v->getMEhisto_PtBarrel_HLT()->Fill(toc[*kj].pt());
                    if (isEndCap(toc[*kj].eta()))  v->getMEhisto_PtEndcap_HLT()->Fill(toc[*kj].pt());
                    if (isForward(toc[*kj].eta())) v->getMEhisto_PtForward_HLT()->Fill(toc[*kj].pt());
                    v->getMEhisto_Eta_HLT()->Fill(toc[*kj].eta());
                    v->getMEhisto_Phi_HLT()->Fill(toc[*kj].phi());
                    v->getMEhisto_EtaPhi_HLT()->Fill(toc[*kj].eta(),toc[*kj].phi());
                    
                    v->getMEhisto_PtCorrelation_L1HLT()->Fill(toc[*ki].pt(),toc[*kj].pt());
                    v->getMEhisto_EtaCorrelation_L1HLT()->Fill(toc[*ki].eta(),toc[*kj].eta());  
                    v->getMEhisto_PhiCorrelation_L1HLT()->Fill(toc[*ki].phi(),toc[*kj].phi());
                     
                    v->getMEhisto_PtResolution_L1HLT()->Fill((toc[*ki].pt()-toc[*kj].pt())/(toc[*ki].pt()));
                    v->getMEhisto_EtaResolution_L1HLT()->Fill((toc[*ki].eta()-toc[*kj].eta())/(toc[*ki].eta()));
                    v->getMEhisto_PhiResolution_L1HLT()->Fill((toc[*ki].phi()-toc[*kj].phi())/(toc[*ki].phi()));
                    } 
                    if(v->getObjectType() == trigger::TriggerMET)
                    {
                    v->getMEhisto_Pt_HLT()->Fill(toc[*kj].pt());
                    v->getMEhisto_Phi_HLT()->Fill(toc[*kj].phi());
                    v->getMEhisto_PtCorrelation_L1HLT()->Fill(toc[*ki].pt(),toc[*kj].pt());
                    v->getMEhisto_PhiCorrelation_L1HLT()->Fill(toc[*ki].phi(),toc[*kj].phi()); 
                    v->getMEhisto_PtResolution_L1HLT()->Fill((toc[*ki].pt()-toc[*kj].pt())/(toc[*ki].pt()));
                    v->getMEhisto_PhiResolution_L1HLT()->Fill((toc[*ki].phi()-toc[*kj].phi())/(toc[*ki].phi()));
                    }
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
                         v->getMEhisto_PtCorrelation_L1RecObj()->Fill(toc[*ki].pt(),jet->pt());
                         v->getMEhisto_EtaCorrelation_L1RecObj()->Fill(toc[*ki].eta(),jet->eta());
                         v->getMEhisto_PhiCorrelation_L1RecObj()->Fill(toc[*ki].phi(),jet->phi());

                         v->getMEhisto_PtResolution_L1RecObj()->Fill((toc[*ki].pt()-jet->pt())/(toc[*ki].pt()));
                         v->getMEhisto_EtaResolution_L1RecObj()->Fill((toc[*ki].eta()-jet->eta())/(toc[*ki].eta()));
                         v->getMEhisto_PhiResolution_L1RecObj()->Fill((toc[*ki].phi()-jet->phi())/(toc[*ki].phi()));

                         v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(),jet->pt());
                         v->getMEhisto_EtaCorrelation_HLTRecObj()->Fill(toc[*kj].eta(),jet->eta());
                         v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(),jet->phi());

                         v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt()-jet->pt())/(toc[*kj].pt()));
                         v->getMEhisto_EtaResolution_HLTRecObj()->Fill((toc[*kj].eta()-jet->eta())/(toc[*kj].eta()));
                         v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*kj].phi()-jet->phi())/(toc[*kj].phi()));

                        }// matching jet
                                    
                     }// Jet Loop
                    v->getMEhisto_N()->Fill(jetsize);
                    }// valid jet collection

                    if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
                    const CaloMETCollection *calometcol = calometColl_.product();
                    const CaloMET met = calometcol->front();
                    v->getMEhisto_Pt()->Fill(met.pt()); 
                    v->getMEhisto_Phi()->Fill(met.phi());
                    v->getMEhisto_PtCorrelation_L1RecObj()->Fill(toc[*ki].pt(),met.pt());
                    v->getMEhisto_PhiCorrelation_L1RecObj()->Fill(toc[*ki].phi(),met.phi());
                    v->getMEhisto_PtResolution_L1RecObj()->Fill((toc[*ki].pt()-met.pt())/(toc[*ki].pt()));
                    v->getMEhisto_PhiResolution_L1RecObj()->Fill((toc[*ki].phi()-met.phi())/(toc[*ki].phi())); 
     
                    v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(),met.pt());
                    v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(),met.phi());
                    v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt()-met.pt())/(toc[*kj].pt()));
                    v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*kj].phi()-met.phi())/(toc[*kj].phi())); 
                    }// valid MET Collection 
                    }// matching  HLT candidate     

                 }//Loop over HLT trigger candidates

           }// valid hlt trigger object

         }// Loop over L1 objects
        } // trigger name exists/not  
          
       if(l1TrigBool && !hltTrigBool)
         {

         if ( l1Index >= triggerObj_->sizeFilters() ) {
          edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< l1Index << " of that name "<<l1Tag;
           } else {
           l1TrigBool = true;
           const trigger::Keys & kl1 = triggerObj_->filterKeys(l1Index);
           std::string l1name = triggerObj_->filterTag(l1Index).encode();
            for( trigger::Keys::const_iterator ki = kl1.begin(); ki != kl1.end(); ++ki)
              {
              double l1TrigEta = toc[*ki].eta();
              double l1TrigPhi = toc[*ki].phi();
              if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
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
                         v->getMEhisto_PtCorrelation_L1RecObj()->Fill(toc[*ki].pt(),jet->pt());
                         v->getMEhisto_EtaCorrelation_L1RecObj()->Fill(toc[*ki].eta(),jet->eta());
                         v->getMEhisto_PhiCorrelation_L1RecObj()->Fill(toc[*ki].phi(),jet->phi());

                         v->getMEhisto_PtResolution_L1RecObj()->Fill((toc[*ki].pt()-jet->pt())/(toc[*ki].pt()));
                         v->getMEhisto_EtaResolution_L1RecObj()->Fill((toc[*ki].eta()-jet->eta())/(toc[*ki].eta()));
                         v->getMEhisto_PhiResolution_L1RecObj()->Fill((toc[*ki].phi()-jet->phi())/(toc[*ki].phi()));


                        }// matching jet
                                    
                     }// Jet Loop
                    v->getMEhisto_N()->Fill(jetsize);
                   }// valid Jet collection

                    if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
                    const CaloMETCollection *calometcol = calometColl_.product();
                    const CaloMET met = calometcol->front();
                    v->getMEhisto_Pt()->Fill(met.pt()); 
                    v->getMEhisto_Phi()->Fill(met.phi());
                    v->getMEhisto_PtCorrelation_L1RecObj()->Fill(toc[*ki].pt(),met.pt());
                    v->getMEhisto_PhiCorrelation_L1RecObj()->Fill(toc[*ki].phi(),met.phi());
                    v->getMEhisto_PtResolution_L1RecObj()->Fill((toc[*ki].pt()-met.pt())/(toc[*ki].pt()));
                    v->getMEhisto_PhiResolution_L1RecObj()->Fill((toc[*ki].phi()-met.phi())/(toc[*ki].phi())); 
     
                    }// valid MET Collection         
             

              }// Loop over keys
             }// valid object

         }// L1 is fired but not HLT       
      
      }//HLT is fired
     }// trigger under study
    }//Loop over all trigger paths

}

//-------------plots wrt Muon Trigger------------
void JetMETHLTOfflineSource::fillMEforMonAllTriggerwrtMuonTrigger(){

    int npath;
    if(&triggerResults_) {
     npath = triggerResults_->size();
     } else {
       return;
     }

    const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());
    bool muTrig = false;
    for(size_t i=0;i<MuonTrigPaths_.size();++i)if(isHLTPathAccepted(MuonTrigPaths_[i]))muTrig = true;
    if(muTrig)
    {
    for(PathInfoCollection::iterator v = hltPathsWrtMu_.begin(); v!= hltPathsWrtMu_.end(); ++v )
    {
    for(int i = 0; i < npath; ++i) {
       if (triggerNames_.triggerName(i).find(v->getPath()) != std::string::npos && triggerResults_->accept(i))
             {
            edm::InputTag l1Tag(v->getl1Path(),"",processname_);
            const int l1Index = triggerObj_->filterIndex(l1Tag);
            edm::InputTag hltTag(v->getLabel(),"",processname_);
            const int hltIndex = triggerObj_->filterIndex(hltTag);
            bool l1TrigBool = false;
            bool hltTrigBool = false;
            int jetsize = 0;
            if ( l1Index >= triggerObj_->sizeFilters() ) {
            edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< l1Index << " of that name "<<l1Tag;
            } else {
             l1TrigBool = true;
             const trigger::Keys & kl1 = triggerObj_->filterKeys(l1Index);
             std::string l1name = triggerObj_->filterTag(l1Index).encode();
             v->getMEhisto_N_L1()->Fill(kl1.size());
             for( trigger::Keys::const_iterator ki = kl1.begin(); ki != kl1.end(); ++ki)
              {
               double l1TrigEta = 0;
               double l1TrigPhi = 0;
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
               
                if ( hltIndex >= triggerObj_->sizeFilters() ) {
                edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt"<< hltIndex << " of that name ";
                } else {
                const trigger::Keys & khlt = triggerObj_->filterKeys(hltIndex);
                std::string hltname = triggerObj_->filterTag(hltIndex).encode();
                 v->getMEhisto_N_HLT()->Fill(khlt.size());
                 for(trigger::Keys::const_iterator kj = khlt.begin();kj != khlt.end(); ++kj)
                  {
                  hltTrigBool = true;
                  double hltTrigEta = 0.;
                  double hltTrigPhi = 0.;
                  if(v->getObjectType() == trigger::TriggerJet)
                   {
                   hltTrigEta = toc[*kj].eta();
                   hltTrigPhi = toc[*kj].phi();
                   
                   if((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi)) < 0.4)
                    {
                    v->getMEhisto_Pt_HLT()->Fill(toc[*kj].pt());
                    if (isBarrel(toc[*kj].eta()))  v->getMEhisto_PtBarrel_HLT()->Fill(toc[*kj].pt());
                    if (isEndCap(toc[*kj].eta()))  v->getMEhisto_PtEndcap_HLT()->Fill(toc[*kj].pt());
                    if (isForward(toc[*kj].eta())) v->getMEhisto_PtForward_HLT()->Fill(toc[*kj].pt());
                    v->getMEhisto_Eta_HLT()->Fill(toc[*kj].eta());
                    v->getMEhisto_Phi_HLT()->Fill(toc[*kj].phi());
                    v->getMEhisto_EtaPhi_HLT()->Fill(toc[*kj].eta(),toc[*kj].phi());
                    
                    v->getMEhisto_PtCorrelation_L1HLT()->Fill(toc[*ki].pt(),toc[*kj].pt());
                    v->getMEhisto_EtaCorrelation_L1HLT()->Fill(toc[*ki].eta(),toc[*kj].eta());  
                    v->getMEhisto_PhiCorrelation_L1HLT()->Fill(toc[*ki].phi(),toc[*kj].phi());
                     
                    v->getMEhisto_PtResolution_L1HLT()->Fill((toc[*ki].pt()-toc[*kj].pt())/(toc[*ki].pt()));
                    v->getMEhisto_EtaResolution_L1HLT()->Fill((toc[*ki].eta()-toc[*kj].eta())/(toc[*ki].eta()));
                    v->getMEhisto_PhiResolution_L1HLT()->Fill((toc[*ki].phi()-toc[*kj].phi())/(toc[*ki].phi()));
                    } 
                    if(v->getObjectType() == trigger::TriggerMET)
                    {
                    v->getMEhisto_Pt_HLT()->Fill(toc[*kj].pt());
                    v->getMEhisto_Phi_HLT()->Fill(toc[*kj].phi());
                    v->getMEhisto_PtCorrelation_L1HLT()->Fill(toc[*ki].pt(),toc[*kj].pt());
                    v->getMEhisto_PhiCorrelation_L1HLT()->Fill(toc[*ki].phi(),toc[*kj].phi()); 
                    v->getMEhisto_PtResolution_L1HLT()->Fill((toc[*ki].pt()-toc[*kj].pt())/(toc[*ki].pt()));
                    v->getMEhisto_PhiResolution_L1HLT()->Fill((toc[*ki].phi()-toc[*kj].phi())/(toc[*ki].phi()));
                    }
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
                         v->getMEhisto_PtCorrelation_L1RecObj()->Fill(toc[*ki].pt(),jet->pt());
                         v->getMEhisto_EtaCorrelation_L1RecObj()->Fill(toc[*ki].eta(),jet->eta());
                         v->getMEhisto_PhiCorrelation_L1RecObj()->Fill(toc[*ki].phi(),jet->phi());

                         v->getMEhisto_PtResolution_L1RecObj()->Fill((toc[*ki].pt()-jet->pt())/(toc[*ki].pt()));
                         v->getMEhisto_EtaResolution_L1RecObj()->Fill((toc[*ki].eta()-jet->eta())/(toc[*ki].eta()));
                         v->getMEhisto_PhiResolution_L1RecObj()->Fill((toc[*ki].phi()-jet->phi())/(toc[*ki].phi()));

                         v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(),jet->pt());
                         v->getMEhisto_EtaCorrelation_HLTRecObj()->Fill(toc[*kj].eta(),jet->eta());
                         v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(),jet->phi());

                         v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt()-jet->pt())/(toc[*kj].pt()));
                         v->getMEhisto_EtaResolution_HLTRecObj()->Fill((toc[*kj].eta()-jet->eta())/(toc[*kj].eta()));
                         v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*kj].phi()-jet->phi())/(toc[*kj].phi()));

                        }// matching jet
                                    
                     }// Jet Loop
                    v->getMEhisto_N()->Fill(jetsize);
                    }// valid jet collection

                    if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
                    const CaloMETCollection *calometcol = calometColl_.product();
                    const CaloMET met = calometcol->front();
                    v->getMEhisto_Pt()->Fill(met.pt()); 
                    v->getMEhisto_Phi()->Fill(met.phi());
                    v->getMEhisto_PtCorrelation_L1RecObj()->Fill(toc[*ki].pt(),met.pt());
                    v->getMEhisto_PhiCorrelation_L1RecObj()->Fill(toc[*ki].phi(),met.phi());
                    v->getMEhisto_PtResolution_L1RecObj()->Fill((toc[*ki].pt()-met.pt())/(toc[*ki].pt()));
                    v->getMEhisto_PhiResolution_L1RecObj()->Fill((toc[*ki].phi()-met.phi())/(toc[*ki].phi())); 
     
                    v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(),met.pt());
                    v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(),met.phi());
                    v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt()-met.pt())/(toc[*kj].pt()));
                    v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*kj].phi()-met.phi())/(toc[*kj].phi())); 
                    }// valid MET Collection 
                    }// matching  HLT candidate     

                 }//Loop over HLT trigger candidates

           }// valid hlt trigger object

         }// Loop over L1 objects
        } // trigger name exists/not  
          
       if(l1TrigBool && !hltTrigBool)
         {

         if ( l1Index >= triggerObj_->sizeFilters() ) {
          edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< l1Index << " of that name "<<l1Tag;
           } else {
           l1TrigBool = true;
           const trigger::Keys & kl1 = triggerObj_->filterKeys(l1Index);
           std::string l1name = triggerObj_->filterTag(l1Index).encode();
            for( trigger::Keys::const_iterator ki = kl1.begin(); ki != kl1.end(); ++ki)
              {
              double l1TrigEta = toc[*ki].eta();
              double l1TrigPhi = toc[*ki].phi();
              if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
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
                         v->getMEhisto_PtCorrelation_L1RecObj()->Fill(toc[*ki].pt(),jet->pt());
                         v->getMEhisto_EtaCorrelation_L1RecObj()->Fill(toc[*ki].eta(),jet->eta());
                         v->getMEhisto_PhiCorrelation_L1RecObj()->Fill(toc[*ki].phi(),jet->phi());

                         v->getMEhisto_PtResolution_L1RecObj()->Fill((toc[*ki].pt()-jet->pt())/(toc[*ki].pt()));
                         v->getMEhisto_EtaResolution_L1RecObj()->Fill((toc[*ki].eta()-jet->eta())/(toc[*ki].eta()));
                         v->getMEhisto_PhiResolution_L1RecObj()->Fill((toc[*ki].phi()-jet->phi())/(toc[*ki].phi()));


                        }// matching jet
                                    
                     }// Jet Loop
                    v->getMEhisto_N()->Fill(jetsize);
                   }// valid Jet collection

                    if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
                    const CaloMETCollection *calometcol = calometColl_.product();
                    const CaloMET met = calometcol->front();
                    v->getMEhisto_Pt()->Fill(met.pt()); 
                    v->getMEhisto_Phi()->Fill(met.phi());
                    v->getMEhisto_PtCorrelation_L1RecObj()->Fill(toc[*ki].pt(),met.pt());
                    v->getMEhisto_PhiCorrelation_L1RecObj()->Fill(toc[*ki].phi(),met.phi());
                    v->getMEhisto_PtResolution_L1RecObj()->Fill((toc[*ki].pt()-met.pt())/(toc[*ki].pt()));
                    v->getMEhisto_PhiResolution_L1RecObj()->Fill((toc[*ki].phi()-met.phi())/(toc[*ki].phi())); 
     
                    }// valid MET Collection         
             

              }// Loop over keys
             }// valid object

         }// L1 is fired but not HLT       
      
      }//HLT is fired
     }// trigger under study

    }//Loop over all trigger paths
    }// Muon trigger fired


}

void JetMETHLTOfflineSource::fillMEforEffAllTrigger(){

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
     for(int i = 0; i < npath; ++i) {
       if (triggerNames_.triggerName(i).find(v->getDenomPath()) != std::string::npos && triggerResults_->accept(i))denompassed = true;
       if (triggerNames_.triggerName(i).find(v->getPath()) != std::string::npos && triggerResults_->accept(i))numpassed = true;
        }// Loop over paths  

             if(denompassed){
              rate_Denominator->Fill(denom); 
              if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
              CaloJetCollection::const_iterator jet = calojet.begin();
              v->getMEhisto_DenominatorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
              v->getMEhisto_DenominatorEta()->Fill(jet->eta());
              v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
              v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
            
       }// Jet trigger and valid jet collection
       if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
       const CaloMETCollection *calometcol = calometColl_.product();
       const CaloMET met = calometcol->front();
       v->getMEhisto_DenominatorPt()->Fill(met.pt());
       v->getMEhisto_DenominatorPhi()->Fill(met.phi());     

      }// MET trigger and valid MET collection 

      if (numpassed)
             {
              rate_Numerator->Fill(num);
              if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
              CaloJetCollection::const_iterator jet = calojet.begin();
              v->getMEhisto_NumeratorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
              v->getMEhisto_NumeratorEta()->Fill(jet->eta());
              v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
              v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());

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


void JetMETHLTOfflineSource::fillMEforEffWrtMuTrigger(){

int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  bool muTrig = false;
  bool denompassed = false;
  for(size_t i=0;i<MuonTrigPaths_.size();++i)if(isHLTPathAccepted(MuonTrigPaths_[i]))muTrig = true;
  for(PathInfoCollection::iterator v = hltPathsEffWrtMu_.begin(); v!= hltPathsEffWrtMu_.end(); ++v )
    {
     bool numpassed   = false; 
     for(int i = 0; i < npath; ++i) {
       
       if (triggerNames_.triggerName(i).find(v->getPath()) != std::string::npos && triggerResults_->accept(i))numpassed = true;
       if(muTrig)denompassed = true;
        }// Loop over paths  

             if(denompassed){
              if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
              CaloJetCollection::const_iterator jet = calojet.begin();
              v->getMEhisto_DenominatorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
              v->getMEhisto_DenominatorEta()->Fill(jet->eta());
              v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
              v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
            
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
              CaloJetCollection::const_iterator jet = calojet.begin();
              v->getMEhisto_NumeratorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
              v->getMEhisto_NumeratorEta()->Fill(jet->eta());
              v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
              v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());

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


void JetMETHLTOfflineSource::fillMEforEffWrtMBTrigger(){

int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  bool mbTrig = false;
  bool denompassed = false;
  for(size_t i=0;i<MBTrigPaths_.size();++i)if(isHLTPathAccepted(MBTrigPaths_[i]))mbTrig = true;
  for(PathInfoCollection::iterator v = hltPathsEffWrtMB_.begin(); v!= hltPathsEffWrtMB_.end(); ++v )
    {
     bool numpassed   = false; 
     for(int i = 0; i < npath; ++i) {
       
       if (triggerNames_.triggerName(i).find(v->getPath()) != std::string::npos && triggerResults_->accept(i))numpassed = true;
       if(mbTrig)denompassed = true;
        }// Loop over paths  

             if(denompassed){
              if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
              CaloJetCollection::const_iterator jet = calojet.begin();
              v->getMEhisto_DenominatorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
              v->getMEhisto_DenominatorEta()->Fill(jet->eta());
              v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
              v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
            
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
              CaloJetCollection::const_iterator jet = calojet.begin();
              v->getMEhisto_NumeratorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
              v->getMEhisto_NumeratorEta()->Fill(jet->eta());
              v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
              v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());

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
  if (hltConfig_.init(run,c,processname_,changed)) {
    if (changed) {
      //if(verbose_) 
      //hltConfig_.dump("Triggers");
      LogInfo("HLTJetMETDQMSource") 	<< "HLTJetMETDQMSource:analyze: The number of valid triggers has changed since beginning of job." << std::endl;
      //	<< "Processing of events halted for summary histograms"  << std::endl;
      //<< "Summary histograms do not support changing configurations." << std::endl
     
    }
  }
   
  if ( hltConfig_.size() <= 0 ) {
    LogError("HLTJetMETDQMSource") << "HLT config size error" << std::endl;
    return;
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
    std::string pathname = hltConfig_.triggerName(i);
    if(verbose_)cout<<"==pathname=="<<pathname<<endl;
    std::string dpathname = "";
    std::string l1pathname = "dummy";
    std::string denompathname = "";
    unsigned int usedPrescale = 1;
    unsigned int objectType = 0;
    std::string triggerType = "";
    std::string filtername("dummy");
    std::string Denomfiltername("denomdummy");

   if (pathname.find("Jet") != std::string::npos && !(pathname.find("Mu") != std::string::npos) && !(pathname.find("BTag") != std::string::npos) && !(pathname.find("Triple") != std::string::npos) && !(pathname.find("Quad") != std::string::npos) && !(pathname.find("DiJet") != std::string::npos)){
         triggerType = "SingleJet_Trigger"; 
         objectType = trigger::TriggerJet;
      }
  if (pathname.find("DiJet") != std::string::npos && !(pathname.find("Mu") != std::string::npos) && !(pathname.find("BTag") != std::string::npos) && !(pathname.find("Triple") != std::string::npos) && !(pathname.find("Quad") != std::string::npos) ){
         triggerType = "DiJet_Trigger";
         objectType = trigger::TriggerJet;
      }
    if (pathname.find("MET") != std::string::npos ){
         triggerType = "MET_Trigger";  
         objectType = trigger::TriggerMET;
      }
    
    if((pathname.find("Fwd") != std::string::npos) || (pathname.find("Quad") != std::string::npos))dpathname = "HLT_L1MuOpen";

     if(objectType == trigger::TriggerJet  && !(pathname.find("DiJet") != std::string::npos) && !(pathname.find("Quad") != std::string::npos) && !(pathname.find("Fwd") != std::string::npos))
        {
         singleJet++;
         if(singleJet > 1)dpathname = dpathname = hltConfig_.triggerName(i-1);
         if(singleJet == 1)dpathname = "HLT_L1MuOpen";
        }  

      if(objectType == trigger::TriggerJet  && (pathname.find("DiJet") != std::string::npos))
        {
         diJet++;
         if(diJet > 1)dpathname = dpathname = hltConfig_.triggerName(i-1);
         if(diJet == 1)dpathname = "HLT_L1MuOpen";
        } 
      if(objectType == trigger::TriggerMET  )
        {
         met++;
         if(met > 1)dpathname = dpathname = hltConfig_.triggerName(i-1);
         if(met == 1)dpathname = "HLT_L1MuOpen";
        }
   // find L1 condition for numpath with numpath objecttype 
      // find PSet for L1 global seed for numpath, 
      // list module labels for numpath
  
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
       }
   if(objectType != 0)
    {
    if(verbose_)cout<<"==pathname=="<<pathname<<"==denompath=="<<dpathname<<"==filtername=="<<filtername<<"==denomfiltername=="<<Denomfiltername<<"==l1pathname=="<<l1pathname<<"==objectType=="<<objectType<<endl;    
   
    hltPathsAll_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
    hltPathsWrtMu_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
    hltPathsEff_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
    hltPathsEffWrtMu_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
    hltPathsEffWrtMB_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
    hltPathsNTfired_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));    


    }
  } //Loop over paths

    //---book trigger summary histos
    std::string foldernm = "/TriggerSummary/";
    if (dbe)   {
      dbe->setCurrentFolder(dirname_ + foldernm);
    }

    std::string histonm="JetMET_TriggerRate";
    std::string histot="JetMET TriggerRate Summary";
     
    rate_All = dbe->book1D(histonm.c_str(),histot.c_str(),
                           hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5);


    histonm="JetMET_TriggerRate_Correlation";
    histot="JetMET TriggerRate Correlation Summary";

    correlation_All = dbe->book2D(histonm.c_str(),histot.c_str(),
                           hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5,hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5);


    histonm="JetMET_TriggerRate_WrtMuTrigger";
    histot="JetMET TriggerRate Summary Wrt Muon Trigger ";
    
    rate_AllWrtMu = dbe->book1D(histonm.c_str(),histot.c_str(),
                           hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5);


    histonm="JetMET_TriggerRate_Correlation_WrtMuTrigger";
    histot="JetMET TriggerRate Correlation Summary Wrt Muon Trigger";

    correlation_AllWrtMu = dbe->book2D(histonm.c_str(),histot.c_str(),
                           hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5,hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5);

    histonm="JetMET_TriggerRate_WrtMBTrigger";
    histot="JetMET TriggerRate Summary Wrt MB Trigger";

    rate_AllWrtMB = dbe->book1D(histonm.c_str(),histot.c_str(),
                           hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5);


    histonm="JetMET_TriggerRate_Correlation_WrtMBTrigger";
    histot="JetMET TriggerRate Correlation Wrt MB Trigger";

    correlation_AllWrtMB = dbe->book2D(histonm.c_str(),histot.c_str(),
                           hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5,hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5);
   

    rate_Denominator  = dbe->book1D("rate_Denominator","rate Denominator Triggers",
                           hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5);
    rate_Numerator  = dbe->book1D("rate_Numerator","rate of Numerator Triggers",
                           hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5); 
    rate_Eff  = dbe->book1D("rate_Eff","rate_Eff wrt Lower threshold trigger",
                           hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5);

    //---Set bin label
    unsigned int nname=0;
 
    for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){
      std::string labelnm("dummy");
      std::string denomlabelnm("dummy");
      labelnm = v->getPath(); 
      denomlabelnm = v->getDenomPath();
      rate_All->setBinLabel(nname+1,labelnm); 
      rate_AllWrtMu->setBinLabel(nname+1,labelnm);
      rate_AllWrtMB->setBinLabel(nname+1,labelnm);
      correlation_All->setBinLabel(nname+1,labelnm,1);
      correlation_AllWrtMu->setBinLabel(nname+1,labelnm,1);
      correlation_AllWrtMB->setBinLabel(nname+1,labelnm,1); 
      correlation_All->setBinLabel(nname+1,labelnm,2);
      correlation_AllWrtMu->setBinLabel(nname+1,labelnm,2);
      correlation_AllWrtMB->setBinLabel(nname+1,labelnm,2);
      rate_Denominator->setBinLabel(nname+1,denomlabelnm);
      rate_Numerator->setBinLabel(nname+1,labelnm);
      rate_Eff->setBinLabel(nname+1,labelnm);
      nname++;

    }

// Now define histos for All triggers
   if(plotAll_)
   {
   int Nbins_       = 10;
   int Nmin_        = 0;
   int Nmax_        = 10;
   int Ptbins_      = 40;
   int Etabins_     = 40;
   int Phibins_     = 20;
   int Resbins_     = 30;
   double PtMin_    = 0.;
   double PtMax_    = 200.;
   double EtaMin_   = -5.;
   double EtaMax_   =  5.;
   double PhiMin_   = -3.14159;
   double PhiMax_   =  3.14159;
   double ResMin_   =  -1.5;
   double ResMax_   =   1.5;
   
   for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){  
   foldernm = "/MonitorAllTriggers/";
    if (dbe)   {
      dbe->setCurrentFolder(dirname_ + foldernm);
    }
   
   MonitorElement *dummy;
   dummy =  dbe->bookFloat("dummy");    

   std::string labelname("dummy");
   labelname = v->getPath();
   std::string histoname(labelname+"");
   std::string title(labelname+"");
   if(v->getObjectType() == trigger::TriggerJet)
   {   
   histoname = labelname+"_recObjN";
   title     = labelname+"_recObjN;multiplicity";
   MonitorElement * N = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
   TH1 *h = N->getTH1();
   h->Sumw2();    

   histoname = labelname+"_recObjPt";
   title = labelname+"_recObjPt;Pt[GeV/c]";
   MonitorElement * Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt->getTH1();
   h->Sumw2();
 
   histoname = labelname+"_recObjPtBarrel";
   title = labelname+"_recObjPtBarrel;Pt[GeV/c]";
   MonitorElement * PtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtBarrel->getTH1();
   h->Sumw2();

   histoname = labelname+"_recObjPtEndcap";
   title = labelname+"_recObjPtEndcap;Pt[GeV/c]";
   MonitorElement * PtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtEndcap->getTH1();
   h->Sumw2();

   histoname = labelname+"_recObjPtForward";
   title = labelname+"_recObjPtForward;Pt[GeV/c]";
   MonitorElement * PtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtForward->getTH1();
   h->Sumw2();

   histoname = labelname+"_recObjEta";
   title = labelname+"_recObjEta;#eta";
   MonitorElement * Eta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = Eta->getTH1();
   h->Sumw2();

   histoname = labelname+"_recObjPhi";
   title = labelname+"_recObjPhi;#Phi";
   MonitorElement * Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi->getTH1();
   h->Sumw2();

   histoname = labelname+"_recObjEtaPhi";
   title = labelname+"_recObjEtaPhi;#eta;#Phi";
   MonitorElement * EtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = EtaPhi->getTH1();
   h->Sumw2();

  
   histoname = labelname+"_l1ObjN";         
   title     = labelname+"_l1ObjN;multiplicity";
   MonitorElement * N_L1 = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
   h = N_L1->getTH1();                                              
   h->Sumw2();                                               

   histoname = labelname+"_l1ObjPt";
   title = labelname+"_l1ObjPt;Pt[GeV/c]";
   MonitorElement * Pt_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt_L1->getTH1();                                                            
   h->Sumw2();                                                             
                                                                            
   histoname = labelname+"_l1ObjPtBarrel";                                    
   title = labelname+"_l1ObjPtBarrel;Pt[GeV/c]";                              
   MonitorElement * PtBarrel_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtBarrel_L1->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_l1ObjPtEndcap";
   title = labelname+"_l1ObjPtEndcap;Pt[GeV/c]";
   MonitorElement * PtEndcap_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtEndcap_L1->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_l1ObjPtForward";
   title = labelname+"_l1ObjPtForward;Pt[GeV/c]";
   MonitorElement * PtForward_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtForward_L1->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_l1ObjEta";
   title = labelname+"_l1ObjEta;#eta";
   MonitorElement * Eta_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = Eta_L1->getTH1();                                                               
   h->Sumw2();                                                                

   histoname = labelname+"_l1ObjPhi";
   title = labelname+"_l1ObjPhi;#Phi";
   MonitorElement * Phi_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi_L1->getTH1();                                                               
   h->Sumw2();                                                                

   histoname = labelname+"_l1ObjEtaPhi";
   title = labelname+"_l1ObjEtaPhi;#eta;#Phi";
   MonitorElement * EtaPhi_L1 =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = EtaPhi_L1->getTH1();                                                                                        
   h->Sumw2();                                

   histoname = labelname+"_hltObjN";         
   title     = labelname+"_hltObjN;multiplicity";
   MonitorElement * N_HLT = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
   h = N_HLT->getTH1();                                              
   h->Sumw2();                                               

   histoname = labelname+"_hltObjPt";
   title = labelname+"_hltObjPt;Pt[GeV/c]";
   MonitorElement * Pt_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt_HLT->getTH1();                                                            
   h->Sumw2();                                                             
                                                                            
   histoname = labelname+"_hltObjPtBarrel";                                    
   title = labelname+"_hltObjPtBarrel;Pt[GeV/c]";                              
   MonitorElement * PtBarrel_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtBarrel_HLT->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_hltObjPtEndcap";
   title = labelname+"_hltObjPtEndcap;Pt[GeV/c]";
   MonitorElement * PtEndcap_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtEndcap_HLT->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_hltObjPtForward";
   title = labelname+"_hltObjPtForward;Pt[GeV/c]";
   MonitorElement * PtForward_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtForward_HLT->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_hltObjEta";
   title = labelname+"_hltObjEta;#eta";
   MonitorElement * Eta_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = Eta_HLT->getTH1();                                                               
   h->Sumw2();                                                                

   histoname = labelname+"_hltObjPhi";
   title = labelname+"_hltObjPhi;#Phi";
   MonitorElement * Phi_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi_HLT->getTH1();                                                               
   h->Sumw2();                                                                

   histoname = labelname+"_hltObjEtaPhi";
   title = labelname+"_hltObjEtaPhi;#eta;#Phi";
   MonitorElement * EtaPhi_HLT =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = EtaPhi_HLT->getTH1();                                                                                        
   h->Sumw2();                                

   histoname = labelname+"_l1HLTPtResolution";
   title = labelname+"_l1HLTPtResolution;#frac{Pt(L1)-Pt(HLT)}{Pt(L1)}";
   MonitorElement * PtResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1HLTEtaResolution";
   title = labelname+"_l1HLTEtaResolution;#frac{#eta(L1)-#eta(HLT)}{#eta(L1)}";
   MonitorElement * EtaResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = EtaResolution_L1HLT->getTH1();
   h->Sumw2(); 

   histoname = labelname+"_l1HLTPhiResolution";
   title = labelname+"_l1HLTPhiResolution;#frac{#Phi(L1)-#Phi(HLT)}{#Phi(L1)}";
   MonitorElement * PhiResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPtResolution";
   title = labelname+"_l1RecObjPtResolution;#frac{Pt(L1)-Pt(Reco)}{Pt(L1)}";
   MonitorElement * PtResolution_L1RecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjEtaResolution";
   title = labelname+"_l1RecObjEtaResolution;#frac{#eta(L1)-#eta(Reco)}{#eta(L1)}";
   MonitorElement * EtaResolution_L1RecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = EtaResolution_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPhiResolution";
   title = labelname+"_l1RecObjPhiResolution;#frac{#Phi(L1)-#Phi(Reco)}{#Phi(L1)}";
   MonitorElement * PhiResolution_L1RecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPtResolution";
   title = labelname+"_hltRecObjPtResolution;#frac{Pt(HLT)-Pt(Reco)}{Pt(HLT)}";
   MonitorElement * PtResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjEtaResolution";
   title = labelname+"_hltRecObjEtaResolution;#frac{#eta(HLT)-#eta(Reco)}{#eta(HLT)}";
   MonitorElement * EtaResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = EtaResolution_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPhiResolution";
   title = labelname+"_hltRecObjPhiResolution;#frac{#Phi(HLT)-#Phi(Reco)}{#Phi(HLT)}";
   MonitorElement * PhiResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1HLTPtCorrelation";
   title = labelname+"_l1HLTPtCorrelation;Pt(L1)[GeV/c];Pt(HLT)[GeV/c]";
   MonitorElement * PtCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_L1HLT->getTH1();
   h->Sumw2();  

   histoname = labelname+"_l1HLTEtaCorrelation";
   title = labelname+"_l1HLTEtaCorrelation;#eta(L1);#eta(HLT)";
   MonitorElement * EtaCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
   h = EtaCorrelation_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1HLTPhiCorrelation";
   title = labelname+"_l1HLTPhiCorrelation;#Phi(L1);#Phi(HLT)";
   MonitorElement * PhiCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPtCorrelation";
   title = labelname+"_l1RecObjPtCorrelation;Pt(L1)[GeV/c];Pt(Reco)[GeV/c]";
   MonitorElement * PtCorrelation_L1RecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjEtaCorrelation";
   title = labelname+"_l1RecObjEtaCorrelation;#eta(L1);#eta(Reco)";
   MonitorElement * EtaCorrelation_L1RecObj = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
   h = EtaCorrelation_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPhiCorrelation";
   title = labelname+"_l1RecObjPhiCorrelation;#Phi(L1);#Phi(Reco)";
   MonitorElement * PhiCorrelation_L1RecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPtCorrelation";
   title = labelname+"_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]";
   MonitorElement * PtCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjEtaCorrelation";
   title = labelname+"_hltRecObjEtaCorrelation;#eta(HLT);#eta(Reco)";
   MonitorElement * EtaCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
   h = EtaCorrelation_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPhiCorrelation";
   title = labelname+"_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)";
   MonitorElement * PhiCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_HLTRecObj->getTH1();
   h->Sumw2();

   v->setHistos(  N, Pt,  PtBarrel, PtEndcap, PtForward, Eta, Phi, EtaPhi,
                  N_L1, Pt_L1,  PtBarrel_L1, PtEndcap_L1, PtForward_L1, Eta_L1, Phi_L1, EtaPhi_L1,
                  N_HLT, Pt_HLT,  PtBarrel_HLT, PtEndcap_HLT, PtForward_HLT, Eta_HLT, Phi_HLT, EtaPhi_HLT, 
                  PtResolution_L1HLT, EtaResolution_L1HLT,PhiResolution_L1HLT,
                  PtResolution_L1RecObj, EtaResolution_L1RecObj,PhiResolution_L1RecObj,
                  PtResolution_HLTRecObj,EtaResolution_HLTRecObj,PhiResolution_HLTRecObj,
                  PtCorrelation_L1HLT,EtaCorrelation_L1HLT,PhiCorrelation_L1HLT,
                  PtCorrelation_L1RecObj,EtaCorrelation_L1RecObj,PhiCorrelation_L1RecObj,
                  PtCorrelation_HLTRecObj,EtaCorrelation_HLTRecObj,PhiCorrelation_HLTRecObj);

   }// histos for Jet Triggers

 if(v->getObjectType() == trigger::TriggerMET)
   {   
   histoname = labelname+"_recObjPt";
   title = labelname+"_recObjPt;Pt[GeV/c]";
   MonitorElement * Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 *h = Pt->getTH1();
   h->Sumw2();
 

   histoname = labelname+"_recObjPhi";
   title = labelname+"_recObjPhi;#Phi";
   MonitorElement * Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1ObjPt";
   title = labelname+"_l1ObjPt;Pt[GeV/c]";
   MonitorElement * Pt_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt_L1->getTH1();                                                            
   h->Sumw2();                                                             
                                                                            

   histoname = labelname+"_l1ObjPhi";
   title = labelname+"_l1ObjPhi;#Phi";
   MonitorElement * Phi_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi_L1->getTH1();                                                               
   h->Sumw2();                                                                


   histoname = labelname+"_hltObjPt";
   title = labelname+"_hltObjPt;Pt[GeV/c]";
   MonitorElement * Pt_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt_HLT->getTH1();                                                            
   h->Sumw2();                                                             
                                                                            

   histoname = labelname+"_hltObjPhi";
   title = labelname+"_hltObjPhi;#Phi";
   MonitorElement * Phi_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi_HLT->getTH1();                                                               
   h->Sumw2();                                                                


   histoname = labelname+"_l1HLTPtResolution";
   title = labelname+"_l1HLTPtResolution;#frac{Pt(L1)-Pt(HLT)}{Pt(L1)}";
   MonitorElement * PtResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_L1HLT->getTH1();
   h->Sumw2();


   histoname = labelname+"_l1HLTPhiResolution";
   title = labelname+"_l1HLTPhiResolution;#frac{#Phi(L1)-#Phi(HLT)}{#Phi(L1)}";
   MonitorElement * PhiResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPtResolution";
   title = labelname+"_l1RecObjPtResolution;#frac{Pt(L1)-Pt(Reco)}{Pt(L1)}";
   MonitorElement * PtResolution_L1RecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPhiResolution";
   title = labelname+"_l1RecObjPhiResolution;#frac{#Phi(L1)-#Phi(Reco)}{#Phi(L1)}";
   MonitorElement * PhiResolution_L1RecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPtResolution";
   title = labelname+"_hltRecObjPtResolution;#frac{Pt(HLT)-Pt(Reco)}{Pt(HLT)}";
   MonitorElement * PtResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_HLTRecObj->getTH1();
   h->Sumw2();


   histoname = labelname+"_hltRecObjPhiResolution";
   title = labelname+"_hltRecObjPhiResolution;#frac{#Phi(HLT)-#Phi(Reco)}{#Phi(HLT)}";
   MonitorElement * PhiResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1HLTPtCorrelation";
   title = labelname+"_l1HLTPtCorrelation;Pt(L1)[GeV/c];Pt(HLT)[GeV/c]";
   MonitorElement * PtCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_L1HLT->getTH1();
   h->Sumw2();  


   histoname = labelname+"_l1HLTPhiCorrelation";
   title = labelname+"_l1HLTPhiCorrelation;#Phi(L1);#Phi(HLT)";
   MonitorElement * PhiCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPtCorrelation";
   title = labelname+"_l1RecObjPtCorrelation;Pt(L1)[GeV/c];Pt(Reco)[GeV/c]";
   MonitorElement * PtCorrelation_L1RecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_L1RecObj->getTH1();
   h->Sumw2();


   histoname = labelname+"_l1RecObjPhiCorrelation";
   title = labelname+"_l1RecObjPhiCorrelation;#Phi(L1);#Phi(Reco)";
   MonitorElement * PhiCorrelation_L1RecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPtCorrelation";
   title = labelname+"_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]";
   MonitorElement * PtCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_HLTRecObj->getTH1();
   h->Sumw2();


   histoname = labelname+"_hltRecObjPhiCorrelation";
   title = labelname+"_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)";
   MonitorElement * PhiCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_HLTRecObj->getTH1();
   h->Sumw2();

   v->setHistos(  dummy, Pt,  dummy, dummy, dummy, dummy, Phi, dummy,
                  dummy, Pt_L1,  dummy, dummy, dummy, dummy, Phi_L1, dummy,
                  dummy, Pt_HLT,  dummy, dummy, dummy, dummy, Phi_HLT, dummy, 
                  PtResolution_L1HLT, dummy,PhiResolution_L1HLT,
                  PtResolution_L1RecObj, dummy,PhiResolution_L1RecObj,
                  PtResolution_HLTRecObj,dummy,PhiResolution_HLTRecObj,
                  PtCorrelation_L1HLT,dummy,PhiCorrelation_L1HLT,
                  PtCorrelation_L1RecObj,dummy,PhiCorrelation_L1RecObj,
                  PtCorrelation_HLTRecObj,dummy,PhiCorrelation_HLTRecObj);

   }// histos for MET Triggers 
 
  }

 }
 if(plotAllwrtMu_)
  {
   int Nbins_       = 10;
   int Nmin_        = 0;
   int Nmax_        = 10;
   int Ptbins_      = 40;
   int Etabins_     = 40;
   int Phibins_     = 20;
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
  for(PathInfoCollection::iterator v = hltPathsWrtMu_.begin(); v!= hltPathsWrtMu_.end(); ++v ){
   std::string labelname("dummy");
   labelname = v->getPath();
   std::string histoname(labelname+"");
   std::string title(labelname+"");
   std::string foldernm1 = "/MonitorAllTriggerwrtMuonTrigger/";
    if (dbe)   {
      dbe->setCurrentFolder(dirname_ + foldernm1);
    }

   MonitorElement *dummy;
   dummy =  dbe->bookFloat("dummy");

  if(v->getObjectType() == trigger::TriggerJet)
   {   
   histoname = labelname+"_recObjN";
   title     = labelname+"_recObjN;multiplicity";
   MonitorElement * N = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
   TH1 *h = N->getTH1();
   h->Sumw2();    

   histoname = labelname+"_recObjPt";
   title = labelname+"_recObjPt;Pt[GeV/c]";
   MonitorElement * Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt->getTH1();
   h->Sumw2();
 
   histoname = labelname+"_recObjPtBarrel";
   title = labelname+"_recObjPtBarrel;Pt[GeV/c]";
   MonitorElement * PtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtBarrel->getTH1();
   h->Sumw2();

   histoname = labelname+"_recObjPtEndcap";
   title = labelname+"_recObjPtEndcap;Pt[GeV/c]";
   MonitorElement * PtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtEndcap->getTH1();
   h->Sumw2();

   histoname = labelname+"_recObjPtForward";
   title = labelname+"_recObjPtForward;Pt[GeV/c]";
   MonitorElement * PtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtForward->getTH1();
   h->Sumw2();

   histoname = labelname+"_recObjEta";
   title = labelname+"_recObjEta;#eta";
   MonitorElement * Eta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = Eta->getTH1();
   h->Sumw2();

   histoname = labelname+"_recObjPhi";
   title = labelname+"_recObjPhi;#Phi";
   MonitorElement * Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi->getTH1();
   h->Sumw2();

   histoname = labelname+"_recObjEtaPhi";
   title = labelname+"_recObjEtaPhi;#eta;#Phi";
   MonitorElement * EtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = EtaPhi->getTH1();
   h->Sumw2();

  
   histoname = labelname+"_l1ObjN";         
   title     = labelname+"_l1ObjN;multiplicity";
   MonitorElement * N_L1 = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
   h = N_L1->getTH1();                                              
   h->Sumw2();                                               

   histoname = labelname+"_l1ObjPt";
   title = labelname+"_l1ObjPt;Pt[GeV/c]";
   MonitorElement * Pt_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt_L1->getTH1();                                                            
   h->Sumw2();                                                             
                                                                            
   histoname = labelname+"_l1ObjPtBarrel";                                    
   title = labelname+"_l1ObjPtBarrel;Pt[GeV/c]";                              
   MonitorElement * PtBarrel_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtBarrel_L1->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_l1ObjPtEndcap";
   title = labelname+"_l1ObjPtEndcap;Pt[GeV/c]";
   MonitorElement * PtEndcap_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtEndcap_L1->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_l1ObjPtForward";
   title = labelname+"_l1ObjPtForward;Pt[GeV/c]";
   MonitorElement * PtForward_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtForward_L1->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_l1ObjEta";
   title = labelname+"_l1ObjEta;#eta";
   MonitorElement * Eta_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = Eta_L1->getTH1();                                                               
   h->Sumw2();                                                                

   histoname = labelname+"_l1ObjPhi";
   title = labelname+"_l1ObjPhi;#Phi";
   MonitorElement * Phi_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi_L1->getTH1();                                                               
   h->Sumw2();                                                                

   histoname = labelname+"_l1ObjEtaPhi";
   title = labelname+"_l1ObjEtaPhi;#eta;#Phi";
   MonitorElement * EtaPhi_L1 =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = EtaPhi_L1->getTH1();                                                                                        
   h->Sumw2();                                

   histoname = labelname+"_hltObjN";         
   title     = labelname+"_hltObjN;multiplicity";
   MonitorElement * N_HLT = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
   h = N_HLT->getTH1();                                              
   h->Sumw2();                                               

   histoname = labelname+"_hltObjPt";
   title = labelname+"_hltObjPt;Pt[GeV/c]";
   MonitorElement * Pt_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt_HLT->getTH1();                                                            
   h->Sumw2();                                                             
                                                                            
   histoname = labelname+"_hltObjPtBarrel";                                    
   title = labelname+"_hltObjPtBarrel;Pt[GeV/c]";                              
   MonitorElement * PtBarrel_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtBarrel_HLT->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_hltObjPtEndcap";
   title = labelname+"_hltObjPtEndcap;Pt[GeV/c]";
   MonitorElement * PtEndcap_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtEndcap_HLT->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_hltObjPtForward";
   title = labelname+"_hltObjPtForward;Pt[GeV/c]";
   MonitorElement * PtForward_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = PtForward_HLT->getTH1();                                                            
   h->Sumw2();                                                             

   histoname = labelname+"_hltObjEta";
   title = labelname+"_hltObjEta;#eta";
   MonitorElement * Eta_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = Eta_HLT->getTH1();                                                               
   h->Sumw2();                                                                

   histoname = labelname+"_hltObjPhi";
   title = labelname+"_hltObjPhi;#Phi";
   MonitorElement * Phi_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi_HLT->getTH1();                                                               
   h->Sumw2();                                                                

   histoname = labelname+"_hltObjEtaPhi";
   title = labelname+"_hltObjEtaPhi;#eta;#Phi";
   MonitorElement * EtaPhi_HLT =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = EtaPhi_HLT->getTH1();                                                                                        
   h->Sumw2();                                

   histoname = labelname+"_l1HLTPtResolution";
   title = labelname+"_l1HLTPtResolution;#frac{Pt(L1)-Pt(HLT)}{Pt(L1)}";
   MonitorElement * PtResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1HLTEtaResolution";
   title = labelname+"_l1HLTEtaResolution;#frac{#eta(L1)-#eta(HLT)}{#eta(L1)}";
   MonitorElement * EtaResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = EtaResolution_L1HLT->getTH1();
   h->Sumw2(); 

   histoname = labelname+"_l1HLTPhiResolution";
   title = labelname+"_l1HLTPhiResolution;#frac{#Phi(L1)-#Phi(HLT)}{#Phi(L1)}";
   MonitorElement * PhiResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPtResolution";
   title = labelname+"_l1RecObjPtResolution;#frac{Pt(L1)-Pt(Reco)}{Pt(L1)}";
   MonitorElement * PtResolution_L1RecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjEtaResolution";
   title = labelname+"_l1RecObjEtaResolution;#frac{#eta(L1)-#eta(Reco)}{#eta(L1)}";
   MonitorElement * EtaResolution_L1RecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = EtaResolution_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPhiResolution";
   title = labelname+"_l1RecObjPhiResolution;#frac{#Phi(L1)-#Phi(Reco)}{#Phi(L1)}";
   MonitorElement * PhiResolution_L1RecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPtResolution";
   title = labelname+"_hltRecObjPtResolution;#frac{Pt(HLT)-Pt(Reco)}{Pt(HLT)}";
   MonitorElement * PtResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjEtaResolution";
   title = labelname+"_hltRecObjEtaResolution;#frac{#eta(HLT)-#eta(Reco)}{#eta(HLT)}";
   MonitorElement * EtaResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = EtaResolution_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPhiResolution";
   title = labelname+"_hltRecObjPhiResolution;#frac{#Phi(HLT)-#Phi(Reco)}{#Phi(HLT)}";
   MonitorElement * PhiResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1HLTPtCorrelation";
   title = labelname+"_l1HLTPtCorrelation;Pt(L1)[GeV/c];Pt(HLT)[GeV/c]";
   MonitorElement * PtCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_L1HLT->getTH1();
   h->Sumw2();  

   histoname = labelname+"_l1HLTEtaCorrelation";
   title = labelname+"_l1HLTEtaCorrelation;#eta(L1);#eta(HLT)";
   MonitorElement * EtaCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
   h = EtaCorrelation_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1HLTPhiCorrelation";
   title = labelname+"_l1HLTPhiCorrelation;#Phi(L1);#Phi(HLT)";
   MonitorElement * PhiCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPtCorrelation";
   title = labelname+"_l1RecObjPtCorrelation;Pt(L1)[GeV/c];Pt(Reco)[GeV/c]";
   MonitorElement * PtCorrelation_L1RecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjEtaCorrelation";
   title = labelname+"_l1RecObjEtaCorrelation;#eta(L1);#eta(Reco)";
   MonitorElement * EtaCorrelation_L1RecObj = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
   h = EtaCorrelation_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPhiCorrelation";
   title = labelname+"_l1RecObjPhiCorrelation;#Phi(L1);#Phi(Reco)";
   MonitorElement * PhiCorrelation_L1RecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPtCorrelation";
   title = labelname+"_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]";
   MonitorElement * PtCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjEtaCorrelation";
   title = labelname+"_hltRecObjEtaCorrelation;#eta(HLT);#eta(Reco)";
   MonitorElement * EtaCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
   h = EtaCorrelation_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPhiCorrelation";
   title = labelname+"_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)";
   MonitorElement * PhiCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_HLTRecObj->getTH1();
   h->Sumw2();

   v->setHistos(  N, Pt,  PtBarrel, PtEndcap, PtForward, Eta, Phi, EtaPhi,
                  N_L1, Pt_L1,  PtBarrel_L1, PtEndcap_L1, PtForward_L1, Eta_L1, Phi_L1, EtaPhi_L1,
                  N_HLT, Pt_HLT,  PtBarrel_HLT, PtEndcap_HLT, PtForward_HLT, Eta_HLT, Phi_HLT, EtaPhi_HLT, 
                  PtResolution_L1HLT, EtaResolution_L1HLT,PhiResolution_L1HLT,
                  PtResolution_L1RecObj, EtaResolution_L1RecObj,PhiResolution_L1RecObj,
                  PtResolution_HLTRecObj,EtaResolution_HLTRecObj,PhiResolution_HLTRecObj,
                  PtCorrelation_L1HLT,EtaCorrelation_L1HLT,PhiCorrelation_L1HLT,
                  PtCorrelation_L1RecObj,EtaCorrelation_L1RecObj,PhiCorrelation_L1RecObj,
                  PtCorrelation_HLTRecObj,EtaCorrelation_HLTRecObj,PhiCorrelation_HLTRecObj);

   }// histos for Jet Triggers

 if(v->getObjectType() == trigger::TriggerMET)
   {   
   histoname = labelname+"_recObjPt";
   title = labelname+"_recObjPt;Pt[GeV/c]";
   MonitorElement * Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 *h = Pt->getTH1();
   h->Sumw2();
 

   histoname = labelname+"_recObjPhi";
   title = labelname+"_recObjPhi;#Phi";
   MonitorElement * Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1ObjPt";
   title = labelname+"_l1ObjPt;Pt[GeV/c]";
   MonitorElement * Pt_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt_L1->getTH1();                                                            
   h->Sumw2();                                                             
                                                                            

   histoname = labelname+"_l1ObjPhi";
   title = labelname+"_l1ObjPhi;#Phi";
   MonitorElement * Phi_L1 =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi_L1->getTH1();                                                               
   h->Sumw2();                                                                


   histoname = labelname+"_hltObjPt";
   title = labelname+"_hltObjPt;Pt[GeV/c]";
   MonitorElement * Pt_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt_HLT->getTH1();                                                            
   h->Sumw2();                                                             
                                                                            

   histoname = labelname+"_hltObjPhi";
   title = labelname+"_hltObjPhi;#Phi";
   MonitorElement * Phi_HLT =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi_HLT->getTH1();                                                               
   h->Sumw2();                                                                


   histoname = labelname+"_l1HLTPtResolution";
   title = labelname+"_l1HLTPtResolution;#frac{Pt(L1)-Pt(HLT)}{Pt(L1)}";
   MonitorElement * PtResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_L1HLT->getTH1();
   h->Sumw2();


   histoname = labelname+"_l1HLTPhiResolution";
   title = labelname+"_l1HLTPhiResolution;#frac{#Phi(L1)-#Phi(HLT)}{#Phi(L1)}";
   MonitorElement * PhiResolution_L1HLT = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPtResolution";
   title = labelname+"_l1RecObjPtResolution;#frac{Pt(L1)-Pt(Reco)}{Pt(L1)}";
   MonitorElement * PtResolution_L1RecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPhiResolution";
   title = labelname+"_l1RecObjPhiResolution;#frac{#Phi(L1)-#Phi(Reco)}{#Phi(L1)}";
   MonitorElement * PhiResolution_L1RecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPtResolution";
   title = labelname+"_hltRecObjPtResolution;#frac{Pt(HLT)-Pt(Reco)}{Pt(HLT)}";
   MonitorElement * PtResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PtResolution_HLTRecObj->getTH1();
   h->Sumw2();


   histoname = labelname+"_hltRecObjPhiResolution";
   title = labelname+"_hltRecObjPhiResolution;#frac{#Phi(HLT)-#Phi(Reco)}{#Phi(HLT)}";
   MonitorElement * PhiResolution_HLTRecObj = dbe->book1D(histoname.c_str(),title.c_str(),Resbins_,ResMin_,ResMax_);
   h = PhiResolution_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1HLTPtCorrelation";
   title = labelname+"_l1HLTPtCorrelation;Pt(L1)[GeV/c];Pt(HLT)[GeV/c]";
   MonitorElement * PtCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_L1HLT->getTH1();
   h->Sumw2();  


   histoname = labelname+"_l1HLTPhiCorrelation";
   title = labelname+"_l1HLTPhiCorrelation;#Phi(L1);#Phi(HLT)";
   MonitorElement * PhiCorrelation_L1HLT = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_L1HLT->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1RecObjPtCorrelation";
   title = labelname+"_l1RecObjPtCorrelation;Pt(L1)[GeV/c];Pt(Reco)[GeV/c]";
   MonitorElement * PtCorrelation_L1RecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_L1RecObj->getTH1();
   h->Sumw2();


   histoname = labelname+"_l1RecObjPhiCorrelation";
   title = labelname+"_l1RecObjPhiCorrelation;#Phi(L1);#Phi(Reco)";
   MonitorElement * PhiCorrelation_L1RecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_L1RecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltRecObjPtCorrelation";
   title = labelname+"_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]";
   MonitorElement * PtCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_HLTRecObj->getTH1();
   h->Sumw2();


   histoname = labelname+"_hltRecObjPhiCorrelation";
   title = labelname+"_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)";
   MonitorElement * PhiCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);
   h = PhiCorrelation_HLTRecObj->getTH1();
   h->Sumw2();

   v->setHistos(  dummy, Pt,  dummy, dummy, dummy, dummy, Phi, dummy,
                  dummy, Pt_L1,  dummy, dummy, dummy, dummy, Phi_L1, dummy,
                  dummy, Pt_HLT,  dummy, dummy, dummy, dummy, Phi_HLT, dummy, 
                  PtResolution_L1HLT, dummy,PhiResolution_L1HLT,
                  PtResolution_L1RecObj, dummy,PhiResolution_L1RecObj,
                  PtResolution_HLTRecObj,dummy,PhiResolution_HLTRecObj,
                  PtCorrelation_L1HLT,dummy,PhiCorrelation_L1HLT,
                  PtCorrelation_L1RecObj,dummy,PhiCorrelation_L1RecObj,
                  PtCorrelation_HLTRecObj,dummy,PhiCorrelation_HLTRecObj);

   }// histos for MET Triggers 

 }

}
//-------Now Efficiency histos--------
 if(plotEff_)
  {
   int Ptbins_      = 40;
   int Etabins_     = 40;
   int Phibins_     = 20;
   double PtMin_    = 0.;
   double PtMax_    = 200.;
   double EtaMin_   = -5.;
   double EtaMax_   =  5.;
   double PhiMin_   = -3.14159;
   double PhiMax_   =  3.14159;
// Now define histos wrt lower threshold trigger
  for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v ){
   std::string labelname("dummy") ;
   labelname = v->getPath() + "_wrt_" + v->getDenomPath();
   std::string histoname(labelname+"");
   std::string title(labelname+"");
   std::string foldernm1 = "/RelativeTriggerEff/";
    if (dbe)   {
      dbe->setCurrentFolder(dirname_ + foldernm1);
    }
    MonitorElement *dummy;
   dummy =  dbe->bookFloat("dummy");   
    
   if(v->getObjectType() == trigger::TriggerJet)
   { 
   histoname = labelname+"_NumeratorPt";
   title     = labelname+"NumeratorPt;Pt[GeV/c]";
   MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = NumeratorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorPtBarrel";
   title     = labelname+"NumeratorPtBarrel;Pt[GeV/c]";
   MonitorElement * NumeratorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = NumeratorPtBarrel->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorPtEndcap";
   title     = labelname+"NumeratorPtEndcap;Pt[GeV/c]";
   MonitorElement * NumeratorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = NumeratorPtEndcap->getTH1();
   h->Sumw2(); 

   histoname = labelname+"_NumeratorPtForward";
   title     = labelname+"NumeratorPtForward;Pt[GeV/c]";
   MonitorElement * NumeratorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = NumeratorPtForward->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorEta";
   title     = labelname+"NumeratorEta;#eta";
   MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = NumeratorEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorPhi";
   title     = labelname+"NumeratorPhi;#Phi";
   MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = NumeratorPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorEtaPhi";
   title     = labelname+"NumeratorEtaPhi;#eta;#Phi";
   MonitorElement * NumeratorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = NumeratorEtaPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPt";
   title     = labelname+"DenominatorPt;Pt[GeV/c]";
   MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPtBarrel";
   title     = labelname+"DenominatorPtBarrel;Pt[GeV/c]";
   MonitorElement * DenominatorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPtBarrel->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPtEndcap";
   title     = labelname+"DenominatorPtEndcap;Pt[GeV/c]";
   MonitorElement * DenominatorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPtEndcap->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPtForward";
   title     = labelname+"DenominatorPtForward;Pt[GeV/c]";
   MonitorElement * DenominatorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPtForward->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorEta";
   title     = labelname+"DenominatorEta;#eta";
   MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = DenominatorEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPhi";
   title     = labelname+"DenominatorPhi;#Phi";
   MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = DenominatorPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorEtaPhi";
   title     = labelname+"DenominatorEtaPhi;#eta;#Phi";
   MonitorElement * DenominatorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = DenominatorEtaPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_EffPt";
   title     = labelname+"EffPt;Pt[GeV/c]";
   MonitorElement * Eff_Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffPtBarrel";
   title     = labelname+"EffPtBarrel;Pt[GeV/c]";
   MonitorElement * Eff_PtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffPtEndcap";
   title     = labelname+"EffPtEndcap;Pt[GeV/c]";
   MonitorElement * Eff_PtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffPtForward";
   title     = labelname+"EffPtForward;Pt[GeV/c]";
   MonitorElement * Eff_PtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffEta";
   title     = labelname+"EffEta;#eta";
   MonitorElement * Eff_Eta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);

   histoname = labelname+"_EffPhi";
   title     = labelname+"EffPhi;#Phi";
   MonitorElement * Eff_Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);

   v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
                      DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi, Eff_Pt, Eff_PtBarrel, Eff_PtEndcap, Eff_PtForward, Eff_Eta, Eff_Phi);
  }// Loop over Jet Trigger

   if(v->getObjectType() == trigger::TriggerMET)
   {
   histoname = labelname+"_NumeratorPt";
   title     = labelname+"NumeratorPt;Pt[GeV/c]";
   MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = NumeratorPt->getTH1();
   h->Sumw2();


   histoname = labelname+"_NumeratorPhi";
   title     = labelname+"NumeratorPhi;#Phi";
   MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = NumeratorPhi->getTH1();
   h->Sumw2();


   histoname = labelname+"_DenominatorPt";
   title     = labelname+"DenominatorPt;Pt[GeV/c]";
   MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPt->getTH1();
   h->Sumw2();

   
   histoname = labelname+"_DenominatorPhi";
   title     = labelname+"DenominatorPhi;#Phi";
   MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = DenominatorPhi->getTH1();
   h->Sumw2();
   
   histoname = labelname+"_EffPt";
   title     = labelname+"EffPt;Pt[GeV/c]";
   MonitorElement * Eff_Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);


   histoname = labelname+"_EffPhi";
   title     = labelname+"EffPhi;#Phi";
   MonitorElement * Eff_Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);

   v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, dummy, NumeratorPhi, dummy,
                     DenominatorPt, dummy, dummy, dummy, dummy, DenominatorPhi, dummy,
                     Eff_Pt, dummy, dummy, dummy, dummy, Eff_Phi);


   }// Loop over MET trigger
}

//------Efficiency wrt Muon trigger-----------------------
for(PathInfoCollection::iterator v = hltPathsEffWrtMu_.begin(); v!= hltPathsEffWrtMu_.end(); ++v ){
   std::string labelname("dummy") ;
   labelname = "Eff_" + v->getPath() + "_wrt_MuTrigger";
   std::string histoname(labelname+"");
   std::string title(labelname+"");
   std::string foldernm1 = "/EffWrtMuonTrigger/";
    if (dbe)   {
      dbe->setCurrentFolder(dirname_ + foldernm1);
    }
    MonitorElement *dummy;
   dummy =  dbe->bookFloat("dummy");
   if(v->getObjectType() == trigger::TriggerJet)
   { 
   histoname = labelname+"_NumeratorPt";
   title     = labelname+"NumeratorPt;Pt[GeV/c]";
   MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = NumeratorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorPtBarrel";
   title     = labelname+"NumeratorPtBarrel;Pt[GeV/c]";
   MonitorElement * NumeratorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = NumeratorPtBarrel->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorPtEndcap";
   title     = labelname+"NumeratorPtEndcap;Pt[GeV/c]";
   MonitorElement * NumeratorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = NumeratorPtEndcap->getTH1();
   h->Sumw2(); 

   histoname = labelname+"_NumeratorPtForward";
   title     = labelname+"NumeratorPtForward;Pt[GeV/c]";
   MonitorElement * NumeratorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = NumeratorPtForward->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorEta";
   title     = labelname+"NumeratorEta;#eta";
   MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = NumeratorEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorPhi";
   title     = labelname+"NumeratorPhi;#Phi";
   MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = NumeratorPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorEtaPhi";
   title     = labelname+"NumeratorEtaPhi;#eta;#Phi";
   MonitorElement * NumeratorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = NumeratorEtaPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPt";
   title     = labelname+"DenominatorPt;Pt[GeV/c]";
   MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPtBarrel";
   title     = labelname+"DenominatorPtBarrel;Pt[GeV/c]";
   MonitorElement * DenominatorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPtBarrel->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPtEndcap";
   title     = labelname+"DenominatorPtEndcap;Pt[GeV/c]";
   MonitorElement * DenominatorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPtEndcap->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPtForward";
   title     = labelname+"DenominatorPtForward;Pt[GeV/c]";
   MonitorElement * DenominatorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPtForward->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorEta";
   title     = labelname+"DenominatorEta;#eta";
   MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = DenominatorEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPhi";
   title     = labelname+"DenominatorPhi;#Phi";
   MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = DenominatorPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorEtaPhi";
   title     = labelname+"DenominatorEtaPhi;#eta;#Phi";
   MonitorElement * DenominatorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = DenominatorEtaPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_EffPt";
   title     = labelname+"EffPt;Pt[GeV/c]";
   MonitorElement * Eff_Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffPtBarrel";
   title     = labelname+"EffPtBarrel;Pt[GeV/c]";
   MonitorElement * Eff_PtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffPtEndcap";
   title     = labelname+"EffPtEndcap;Pt[GeV/c]";
   MonitorElement * Eff_PtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffPtForward";
   title     = labelname+"EffPtForward;Pt[GeV/c]";
   MonitorElement * Eff_PtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffEta";
   title     = labelname+"EffEta;#eta";
   MonitorElement * Eff_Eta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);

   histoname = labelname+"_EffPhi";
   title     = labelname+"EffPhi;#Phi";
   MonitorElement * Eff_Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);

   v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
                    DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi, Eff_Pt, Eff_PtBarrel, Eff_PtEndcap, Eff_PtForward, Eff_Eta, Eff_Phi);
  }// Loop over Jet Trigger

   if(v->getObjectType() == trigger::TriggerMET)
   {
   histoname = labelname+"_NumeratorPt";
   title     = labelname+"NumeratorPt;Pt[GeV/c]";
   MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = NumeratorPt->getTH1();
   h->Sumw2();


   histoname = labelname+"_NumeratorPhi";
   title     = labelname+"NumeratorPhi;#Phi";
   MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = NumeratorPhi->getTH1();
   h->Sumw2();


   histoname = labelname+"_DenominatorPt";
   title     = labelname+"DenominatorPt;Pt[GeV/c]";
   MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPt->getTH1();
   h->Sumw2();

   
   histoname = labelname+"_DenominatorPhi";
   title     = labelname+"DenominatorPhi;#Phi";
   MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = DenominatorPhi->getTH1();
   h->Sumw2();
   
   histoname = labelname+"_EffPt";
   title     = labelname+"EffPt;Pt[GeV/c]";
   MonitorElement * Eff_Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);


   histoname = labelname+"_EffPhi";
   title     = labelname+"EffPhi;#Phi";
   MonitorElement * Eff_Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);

   v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, dummy, NumeratorPhi, dummy,
                     DenominatorPt, dummy, dummy, dummy, dummy, DenominatorPhi, dummy,
                     Eff_Pt, dummy, dummy, dummy, dummy, Eff_Phi);


   }// Loop over MET trigger 
}
//--------Efficiency  wrt MiniBias trigger---------
for(PathInfoCollection::iterator v = hltPathsEffWrtMB_.begin(); v!= hltPathsEffWrtMB_.end(); ++v ){
   std::string labelname("dummy") ;
   labelname = "Eff_" + v->getPath() + "_wrt_MBTrigger";
   std::string histoname(labelname+"");
   std::string title(labelname+"");
   std::string foldernm1 = "/EffWrtMBTrigger/";
    if (dbe)   {
      dbe->setCurrentFolder(dirname_ + foldernm1);
    }
   MonitorElement *dummy;
   dummy =  dbe->bookFloat("dummy");   
   if(v->getObjectType() == trigger::TriggerJet)
   { 
   histoname = labelname+"_NumeratorPt";
   title     = labelname+"NumeratorPt;Pt[GeV/c]";
   MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = NumeratorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorPtBarrel";
   title     = labelname+"NumeratorPtBarrel;Pt[GeV/c]";
   MonitorElement * NumeratorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = NumeratorPtBarrel->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorPtEndcap";
   title     = labelname+"NumeratorPtEndcap;Pt[GeV/c]";
   MonitorElement * NumeratorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = NumeratorPtEndcap->getTH1();
   h->Sumw2(); 

   histoname = labelname+"_NumeratorPtForward";
   title     = labelname+"NumeratorPtForward;Pt[GeV/c]";
   MonitorElement * NumeratorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = NumeratorPtForward->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorEta";
   title     = labelname+"NumeratorEta;#eta";
   MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = NumeratorEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorPhi";
   title     = labelname+"NumeratorPhi;#Phi";
   MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = NumeratorPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorEtaPhi";
   title     = labelname+"NumeratorEtaPhi;#eta;#Phi";
   MonitorElement * NumeratorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = NumeratorEtaPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPt";
   title     = labelname+"DenominatorPt;Pt[GeV/c]";
   MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPtBarrel";
   title     = labelname+"DenominatorPtBarrel;Pt[GeV/c]";
   MonitorElement * DenominatorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPtBarrel->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPtEndcap";
   title     = labelname+"DenominatorPtEndcap;Pt[GeV/c]";
   MonitorElement * DenominatorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPtEndcap->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPtForward";
   title     = labelname+"DenominatorPtForward;Pt[GeV/c]";
   MonitorElement * DenominatorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPtForward->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorEta";
   title     = labelname+"DenominatorEta;#eta";
   MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = DenominatorEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorPhi";
   title     = labelname+"DenominatorPhi;#Phi";
   MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = DenominatorPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorEtaPhi";
   title     = labelname+"DenominatorEtaPhi;#eta;#Phi";
   MonitorElement * DenominatorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
   h = DenominatorEtaPhi->getTH1();
   h->Sumw2();

   histoname = labelname+"_EffPt";
   title     = labelname+"EffPt;Pt[GeV/c]";
   MonitorElement * Eff_Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffPtBarrel";
   title     = labelname+"EffPtBarrel;Pt[GeV/c]";
   MonitorElement * Eff_PtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffPtEndcap";
   title     = labelname+"EffPtEndcap;Pt[GeV/c]";
   MonitorElement * Eff_PtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffPtForward";
   title     = labelname+"EffPtForward;Pt[GeV/c]";
   MonitorElement * Eff_PtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);

   histoname = labelname+"_EffEta";
   title     = labelname+"EffEta;#eta";
   MonitorElement * Eff_Eta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);

   histoname = labelname+"_EffPhi";
   title     = labelname+"EffPhi;#Phi";
   MonitorElement * Eff_Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);

   v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
                     DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi, Eff_Pt, Eff_PtBarrel, Eff_PtEndcap, Eff_PtForward, Eff_Eta, Eff_Phi);
  }// Loop over Jet Trigger

   if(v->getObjectType() == trigger::TriggerMET)
   {
   histoname = labelname+"_NumeratorPt";
   title     = labelname+"NumeratorPt;Pt[GeV/c]";
   MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = NumeratorPt->getTH1();
   h->Sumw2();


   histoname = labelname+"_NumeratorPhi";
   title     = labelname+"NumeratorPhi;#Phi";
   MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = NumeratorPhi->getTH1();
   h->Sumw2();


   histoname = labelname+"_DenominatorPt";
   title     = labelname+"DenominatorPt;Pt[GeV/c]";
   MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPt->getTH1();
   h->Sumw2();

   
   histoname = labelname+"_DenominatorPhi";
   title     = labelname+"DenominatorPhi;#Phi";
   MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = DenominatorPhi->getTH1();
   h->Sumw2();
   
   histoname = labelname+"_EffPt";
   title     = labelname+"EffPt;Pt[GeV/c]";
   MonitorElement * Eff_Pt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);


   histoname = labelname+"_EffPhi";
   title     = labelname+"EffPhi;#Phi";
   MonitorElement * Eff_Phi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);

   v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, dummy, NumeratorPhi, dummy,
                     DenominatorPt, dummy, dummy, dummy, dummy, DenominatorPhi, dummy,
                     Eff_Pt, dummy, dummy, dummy, dummy, Eff_Phi);


   }// Loop over MET trigger
}


}// This is loop over all efficiency plots
//--------Histos to see WHY trigger is NOT fired----------
   int Nbins_       = 10;
   int Nmin_        = 0;
   int Nmax_        = 10;
   int Ptbins_      = 40;
   int Etabins_     = 40;
   double PtMin_    = 0.;
   double PtMax_    = 200.;
   double EtaMin_   = -5.;
   double EtaMax_   =  5.;
for(PathInfoCollection::iterator v = hltPathsNTfired_.begin(); v!= hltPathsNTfired_.end(); ++v ){

    MonitorElement *dummy;
   dummy =  dbe->bookFloat("dummy");

   std::string labelname("dummy") ;
   labelname = v->getPath();
   std::string histoname(labelname+"");
   std::string title(labelname+"");
   std::string foldernm1 = "/TriggerNotFired/";
    if (dbe)   {
      dbe->setCurrentFolder(dirname_ + foldernm1);
    }
   
  if((v->getTriggerType())=="SingleJet_Trigger")
   {
   histoname = labelname+"_JetPt"; 
   title     = labelname+"Leading jet pT;Pt[GeV/c]";
   MonitorElement * JetPt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = JetPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_JetEta"; 
   title     = labelname+"Leading jet #eta;#eta";
   MonitorElement * JetEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = JetEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_JetEtaVsPt";
   title     = labelname+"Leading jet #eta vs pT;#eta;Pt[GeV/c]";
   MonitorElement * JetEtaVsPt = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Ptbins_,PtMin_,PtMax_);
   h = JetEtaVsPt->getTH1();
   h->Sumw2();
   
   v->setDgnsHistos( dummy, JetPt, JetEta, JetEtaVsPt, dummy, dummy, dummy);

   }// single Jet trigger  

  if((v->getTriggerType())=="DiJet_Trigger")
   {
   histoname = labelname+"_JetSize"; 
   title     = labelname+"Jet Size;multiplicity";
   MonitorElement * JetSize = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
   TH1 * h = JetSize->getTH1();
   h->Sumw2();


   histoname = labelname+"_1stJetEtaVsPt";
   title     = labelname+"Leading jet #eta vs pT;#eta;Pt[GeV/c]";
   MonitorElement * LeadingJet_EtaVsPt = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Ptbins_,PtMin_,PtMax_);
   h = LeadingJet_EtaVsPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_2ndJetEtaVsPt";
   title     = labelname+"Second Leading jet #eta vs pT;#eta;Pt[GeV/c]";
   MonitorElement * SecondLeadingJet_EtaVsPt = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Ptbins_,PtMin_,PtMax_);
   h = SecondLeadingJet_EtaVsPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_Pt12";
   title     = labelname+"Leading Jets Pt;Pt(first leading)[GeV/c];Pt(second leading)[GeV/c]";
   MonitorElement * Pt12 = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = Pt12->getTH1();
   h->Sumw2(); 

   histoname = labelname+"_Eta12";
   title     = labelname+"Leading Jets Eta;#eta(first leading);#eta(second leading)";
   MonitorElement * Eta12 = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_);
   h = Eta12->getTH1();
   h->Sumw2(); 

   v->setDgnsHistos( JetSize, dummy, dummy, LeadingJet_EtaVsPt, SecondLeadingJet_EtaVsPt, Pt12, Eta12);

   }// Dijet Jet trigger

  if((v->getTriggerType())=="MET_Trigger")
   {
   histoname = labelname+"_MET";
   title     = labelname+"MET;Pt[GeV/c]";
   MonitorElement * MET = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = MET->getTH1();
   h->Sumw2();
   v->setDgnsHistos(dummy, MET, dummy, dummy, dummy, dummy, dummy);
  } // MET trigger  


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
  LogInfo("JetMETHLTOfflineSource") << "analyzed " << nev_ << " events";
  return;
}

/// EndRun
void JetMETHLTOfflineSource::endRun(const edm::Run& run, const edm::EventSetup& c){
  if (verbose_) std::cout << "endRun, run " << run.id() << std::endl;
  if(plotEff_)
  {
  EffCalc(rate_Numerator, rate_Denominator, rate_Eff);
  for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v ) {
  if(v->getObjectType() == trigger::TriggerJet)
  {
  EffCalc(v->getMEhisto_NumeratorPt(),v->getMEhisto_DenominatorPt(),v->getMEhisto_Eff_Pt());
  EffCalc(v->getMEhisto_NumeratorPtBarrel(),v->getMEhisto_DenominatorPtBarrel(),v->getMEhisto_Eff_PtBarrel());
  EffCalc(v->getMEhisto_NumeratorPtEndcap(),v->getMEhisto_DenominatorPtEndcap(),v->getMEhisto_Eff_PtEndcap());
  EffCalc(v->getMEhisto_NumeratorPtForward(),v->getMEhisto_DenominatorPtForward(),v->getMEhisto_Eff_PtForward());
  EffCalc(v->getMEhisto_NumeratorEta(),v->getMEhisto_DenominatorEta(),v->getMEhisto_Eff_Eta());
  EffCalc(v->getMEhisto_NumeratorPhi(),v->getMEhisto_DenominatorPhi(),v->getMEhisto_Eff_Phi());
  }

  if(v->getObjectType() == trigger::TriggerMET)
  {
  EffCalc(v->getMEhisto_NumeratorPt(),v->getMEhisto_DenominatorPt(),v->getMEhisto_Eff_Pt());
  EffCalc(v->getMEhisto_NumeratorPhi(),v->getMEhisto_DenominatorPhi(),v->getMEhisto_Eff_Phi());
  }
  
   }// Loop over trigger path

 for(PathInfoCollection::iterator v = hltPathsEffWrtMu_.begin(); v!= hltPathsEffWrtMu_.end(); ++v ) {
  if(v->getObjectType() == trigger::TriggerJet)
  {
  EffCalc(v->getMEhisto_NumeratorPt(),v->getMEhisto_DenominatorPt(),v->getMEhisto_Eff_Pt());
  EffCalc(v->getMEhisto_NumeratorPtBarrel(),v->getMEhisto_DenominatorPtBarrel(),v->getMEhisto_Eff_PtBarrel());
  EffCalc(v->getMEhisto_NumeratorPtEndcap(),v->getMEhisto_DenominatorPtEndcap(),v->getMEhisto_Eff_PtEndcap());
  EffCalc(v->getMEhisto_NumeratorPtForward(),v->getMEhisto_DenominatorPtForward(),v->getMEhisto_Eff_PtForward());
  EffCalc(v->getMEhisto_NumeratorEta(),v->getMEhisto_DenominatorEta(),v->getMEhisto_Eff_Eta());
  EffCalc(v->getMEhisto_NumeratorPhi(),v->getMEhisto_DenominatorPhi(),v->getMEhisto_Eff_Phi());
  }
  if(v->getObjectType() == trigger::TriggerMET)
  {
  EffCalc(v->getMEhisto_NumeratorPt(),v->getMEhisto_DenominatorPt(),v->getMEhisto_Eff_Pt());
  EffCalc(v->getMEhisto_NumeratorPhi(),v->getMEhisto_DenominatorPhi(),v->getMEhisto_Eff_Phi());
  }
   }// Loop over trigger path

 for(PathInfoCollection::iterator v = hltPathsEffWrtMB_.begin(); v!= hltPathsEffWrtMB_.end(); ++v ) {
  if(v->getObjectType() == trigger::TriggerJet)
  {
  EffCalc(v->getMEhisto_NumeratorPt(),v->getMEhisto_DenominatorPt(),v->getMEhisto_Eff_Pt());
  EffCalc(v->getMEhisto_NumeratorPtBarrel(),v->getMEhisto_DenominatorPtBarrel(),v->getMEhisto_Eff_PtBarrel());
  EffCalc(v->getMEhisto_NumeratorPtEndcap(),v->getMEhisto_DenominatorPtEndcap(),v->getMEhisto_Eff_PtEndcap());
  EffCalc(v->getMEhisto_NumeratorPtForward(),v->getMEhisto_DenominatorPtForward(),v->getMEhisto_Eff_PtForward());
  EffCalc(v->getMEhisto_NumeratorEta(),v->getMEhisto_DenominatorEta(),v->getMEhisto_Eff_Eta());
  EffCalc(v->getMEhisto_NumeratorPhi(),v->getMEhisto_DenominatorPhi(),v->getMEhisto_Eff_Phi());
  }
  if(v->getObjectType() == trigger::TriggerMET)
  {
  EffCalc(v->getMEhisto_NumeratorPt(),v->getMEhisto_DenominatorPt(),v->getMEhisto_Eff_Pt());
  EffCalc(v->getMEhisto_NumeratorPhi(),v->getMEhisto_DenominatorPhi(),v->getMEhisto_Eff_Phi());
  }
   }// Loop over trigger path
 }
}

void JetMETHLTOfflineSource::EffCalc(MonitorElement *numME, MonitorElement *denomME, MonitorElement *effME)
{
  TH1F *Numerator = numME->getTH1F();
  TH1F *Denominator = denomME->getTH1F();
  if ((Numerator->Integral() != 0.)  && (Denominator->Integral() != 0.) ) {
     for(int j=1; j <= Numerator->GetXaxis()->GetNbins();j++ ){
      double y1 = Numerator->GetBinContent(j);
      double y2 = Denominator->GetBinContent(j);
      double eff = y2 > 0. ? y1/y2 : 0.;
      effME->setBinContent(j, eff); 

      double efferr = 0.0;
      if (y2 && y1  > 0.) efferr =  sqrt(eff*(1-eff)/y2) ;
      effME->setBinError(j, efferr);

     }// Loop over bins
  }// Non zero integral

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
    int npath = triggerResults_->size();
    for(int i = 0; i < npath; ++i) {
      if (triggerNames_.triggerName(i).find(pathName) != std::string::npos
          && triggerResults_->accept(i))
        { output = true; break; }
    }
  }
  return output;
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



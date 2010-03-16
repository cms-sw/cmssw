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
       if (triggerNames_.triggerName(i).find(v->getPath()) != std::string::npos )
             {
             v->getMEhisto_TriggerSummary()->Fill(0.);
             edm::InputTag l1Tag(v->getl1Path(),"",processname_);
             const int l1Index = triggerObj_->filterIndex(l1Tag);
             bool l1found = false;
             if ( l1Index < triggerObj_->sizeFilters() ) l1found = true;
             if(!l1found)v->getMEhisto_TriggerSummary()->Fill(1.);
             if(!l1found && !(triggerResults_->accept(i)))v->getMEhisto_TriggerSummary()->Fill(2.);
             if(!l1found && (triggerResults_->accept(i)))v->getMEhisto_TriggerSummary()->Fill(3.);
             if(l1found)v->getMEhisto_TriggerSummary()->Fill(4.);
             if(l1found && (triggerResults_->accept(i)))v->getMEhisto_TriggerSummary()->Fill(5.); 
             if(l1found && !(triggerResults_->accept(i)))v->getMEhisto_TriggerSummary()->Fill(6.);
             if(!(triggerResults_->accept(i)))
             { 
             if(l1found)
             {
             if(v->getTriggerType() == "SingleJet_Trigger" && calojetColl_.isValid())
              {
              CaloJetCollection::const_iterator jet = calojet.begin();
              v->getMEhisto_JetPt()->Fill(jet->pt());
              v->getMEhisto_EtavsPt()->Fill(jet->eta(),jet->pt());
              v->getMEhisto_PhivsPt()->Fill(jet->phi(),jet->pt());
                 
              }// single jet trigger is not fired

            if(v->getTriggerType() == "DiJet_Trigger" && calojetColl_.isValid())
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

           if(v->getTriggerType() == "MET_Trigger" && calometColl_.isValid())
              {
              const CaloMETCollection *calometcol = calometColl_.product();
              const CaloMET met = calometcol->front();
              v->getMEhisto_JetPt()->Fill(met.pt());
          }//MET trigger is not fired   
         } // L1 is fired
       }//
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
            std::vector<double>jetPtVec;
            std::vector<double>jetPhiVec; 
            std::vector<double>jetEtaVec;
            
            std::vector<double>hltPtVec;
            std::vector<double>hltPhiVec;
            std::vector<double>hltEtaVec;

            std::vector<double>l1PtVec;
            std::vector<double>l1PhiVec;
            std::vector<double>l1EtaVec; 

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

                         v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(),jet->pt());
                         v->getMEhisto_EtaCorrelation_HLTRecObj()->Fill(toc[*kj].eta(),jet->eta());
                         v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(),jet->phi());

                         v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt()-jet->pt())/(toc[*kj].pt()));
                         v->getMEhisto_EtaResolution_HLTRecObj()->Fill((toc[*kj].eta()-jet->eta())/(toc[*kj].eta()));
                         v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*kj].phi()-jet->phi())/(toc[*kj].phi()));
                         if(v->getTriggerType() == "DiJet_Trigger")
                           {
                            jetPhiVec.push_back(jet->phi());
                            jetPtVec.push_back(jet->pt());
                            jetEtaVec.push_back(jet->eta());         

                            hltPhiVec.push_back(toc[*kj].phi());
                            hltPtVec.push_back(toc[*kj].pt());
                            hltEtaVec.push_back(toc[*kj].eta());
 
                            l1PhiVec.push_back(toc[*ki].phi());
                            l1PtVec.push_back(toc[*ki].pt());
                            l1EtaVec.push_back(toc[*ki].eta());
                           }
                          

                        }// matching jet
                                    
                     }// Jet Loop
                    v->getMEhisto_N()->Fill(jetsize);
                    }// valid jet collection

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
                    }// matching  HLT candidate     

                 }//Loop over HLT trigger candidates

           }// valid hlt trigger object

         }// Loop over L1 objects
           if(v->getTriggerType() == "DiJet_Trigger")
             {
               double AveJetPt = (jetPtVec[0] + jetPtVec[1])/2;
               double AveJetEta = (jetEtaVec[0] + jetEtaVec[1])/2;               
               double JetDelPhi = deltaPhi(jetPhiVec[0],jetPhiVec[1]);
               double AveHLTPt = (hltPtVec[0] + hltPtVec[1])/2;
               double AveHLTEta = (hltEtaVec[0] + hltEtaVec[1])/2;
               double HLTDelPhi = deltaPhi(hltPhiVec[0],hltPhiVec[1]);
               double AveL1Pt = (l1PtVec[0] + l1PtVec[1])/2;
               double AveL1Eta = (l1EtaVec[0] + l1EtaVec[1])/2;
               double L1DelPhi = deltaPhi(l1PhiVec[0],l1PhiVec[1]);
 
               v->getMEhisto_AveragePt_RecObj()->Fill(AveJetPt);
               v->getMEhisto_AverageEta_RecObj()->Fill(AveJetEta);
               v->getMEhisto_DeltaPhi_RecObj()->Fill(JetDelPhi);
 
               v->getMEhisto_AveragePt_HLTObj()->Fill(AveHLTPt);
               v->getMEhisto_AverageEta_HLTObj()->Fill(AveHLTEta);
               v->getMEhisto_DeltaPhi_HLTObj()->Fill(HLTDelPhi);       

               v->getMEhisto_AveragePt_L1Obj()->Fill(AveL1Pt);
               v->getMEhisto_AverageEta_L1Obj()->Fill(AveL1Eta);
               v->getMEhisto_DeltaPhi_L1Obj()->Fill(L1DelPhi); 
 
             }
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


                        }// matching jet
                                    
                     }// Jet Loop
                    v->getMEhisto_N()->Fill(jetsize);
                   }// valid Jet collection

                    if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
                    const CaloMETCollection *calometcol = calometColl_.product();
                    const CaloMET met = calometcol->front();
                    v->getMEhisto_Pt()->Fill(met.pt()); 
                    v->getMEhisto_Phi()->Fill(met.phi());
     
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
              std::vector<double>jetPtVec;                                                                      
            std::vector<double>jetPhiVec;                                                                     
            std::vector<double>jetEtaVec;                                                                     
                                                                                                              
            std::vector<double>hltPtVec;                                                                      
            std::vector<double>hltPhiVec;                                                                     
            std::vector<double>hltEtaVec;                                                                     

            std::vector<double>l1PtVec;
            std::vector<double>l1PhiVec;
            std::vector<double>l1EtaVec; 

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

                         v->getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(),jet->pt());
                         v->getMEhisto_EtaCorrelation_HLTRecObj()->Fill(toc[*kj].eta(),jet->eta());
                         v->getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(),jet->phi());

                         v->getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt()-jet->pt())/(toc[*kj].pt()));
                         v->getMEhisto_EtaResolution_HLTRecObj()->Fill((toc[*kj].eta()-jet->eta())/(toc[*kj].eta()));
                         v->getMEhisto_PhiResolution_HLTRecObj()->Fill((toc[*kj].phi()-jet->phi())/(toc[*kj].phi()));
                         if(v->getTriggerType() == "DiJet_Trigger")                                                  
                           {                                                                                         
                            jetPhiVec.push_back(jet->phi());                                                         
                            jetPtVec.push_back(jet->pt());                                                           
                            jetEtaVec.push_back(jet->eta());                                                         

                            hltPhiVec.push_back(toc[*kj].phi());
                            hltPtVec.push_back(toc[*kj].pt());  
                            hltEtaVec.push_back(toc[*kj].eta());
                                                                
                            l1PhiVec.push_back(toc[*ki].phi()); 
                            l1PtVec.push_back(toc[*ki].pt());   
                            l1EtaVec.push_back(toc[*ki].eta()); 
                           }                                    
                                                                

                        }// matching jet
                                        
                     }// Jet Loop       
                    v->getMEhisto_N()->Fill(jetsize);
                    }// valid jet collection         

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
                    }// matching  HLT candidate                                                                 

                 }//Loop over HLT trigger candidates

           }// valid hlt trigger object

         }// Loop over L1 objects
           if(v->getTriggerType() == "DiJet_Trigger")
             {                                       
               double AveJetPt = (jetPtVec[0] + jetPtVec[1])/2;
               double AveJetEta = (jetEtaVec[0] + jetEtaVec[1])/2;               
               double JetDelPhi = deltaPhi(jetPhiVec[0],jetPhiVec[1]);           
               double AveHLTPt = (hltPtVec[0] + hltPtVec[1])/2;                  
               double AveHLTEta = (hltEtaVec[0] + hltEtaVec[1])/2;               
               double HLTDelPhi = deltaPhi(hltPhiVec[0],hltPhiVec[1]);           
               double AveL1Pt = (l1PtVec[0] + l1PtVec[1])/2;                     
               double AveL1Eta = (l1EtaVec[0] + l1EtaVec[1])/2;                  
               double L1DelPhi = deltaPhi(l1PhiVec[0],l1PhiVec[1]);              
                                                                                 
               v->getMEhisto_AveragePt_RecObj()->Fill(AveJetPt);                 
               v->getMEhisto_AverageEta_RecObj()->Fill(AveJetEta);               
               v->getMEhisto_DeltaPhi_RecObj()->Fill(JetDelPhi);                  
 
               v->getMEhisto_AveragePt_HLTObj()->Fill(AveHLTPt);
               v->getMEhisto_AverageEta_HLTObj()->Fill(AveHLTEta);
               v->getMEhisto_DeltaPhi_HLTObj()->Fill(HLTDelPhi);       

               v->getMEhisto_AveragePt_L1Obj()->Fill(AveL1Pt);
               v->getMEhisto_AverageEta_L1Obj()->Fill(AveL1Eta);
               v->getMEhisto_DeltaPhi_L1Obj()->Fill(L1DelPhi);   
                                                                
             }                                                  
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

                        }// matching jet
                                        
                     }// Jet Loop       
                    v->getMEhisto_N()->Fill(jetsize);
                   }// valid Jet collection          

                    if(calometColl_.isValid() && (v->getObjectType() == trigger::TriggerMET)){
                    const CaloMETCollection *calometcol = calometColl_.product();             
                    const CaloMET met = calometcol->front();                                  
                    v->getMEhisto_Pt()->Fill(met.pt());                                       
                    v->getMEhisto_Phi()->Fill(met.phi());                                     
                                                                                                               
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
              ME_Denominator_rate->Fill(denom); 
              if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
              CaloJetCollection::const_iterator jet = calojet.begin();
              if(v->getTriggerType() == "SingleJet_Trigger")
              {
              v->getMEhisto_DenominatorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
              v->getMEhisto_DenominatorEta()->Fill(jet->eta());
              v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
              v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
              }
              if(v->getTriggerType() == "DiJet_Trigger")
              {
              CaloJetCollection::const_iterator jet2 = jet++;
              v->getMEhisto_DenominatorPt()->Fill((jet->pt() + jet2->pt())/2.);
              v->getMEhisto_DenominatorEta()->Fill((jet->eta() + jet2->eta())/2.);
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
              ME_Numerator_rate->Fill(num);
              if(calojetColl_.isValid() && (v->getObjectType() == trigger::TriggerJet)){
              CaloJetCollection::const_iterator jet = calojet.begin();
              if(v->getTriggerType() == "SingleJet_Trigger")
              {
              v->getMEhisto_NumeratorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
              v->getMEhisto_NumeratorEta()->Fill(jet->eta());
              v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
              v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());
              }
              if(v->getTriggerType() == "DiJet_Trigger")
              {
              CaloJetCollection::const_iterator jet2 = jet++;
              v->getMEhisto_NumeratorPt()->Fill((jet->pt() + jet2->pt())/2.);
              v->getMEhisto_NumeratorEta()->Fill((jet->eta() + jet2->eta())/2.);
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
              if(v->getTriggerType() == "SingleJet_Trigger")
              {
              v->getMEhisto_DenominatorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
              v->getMEhisto_DenominatorEta()->Fill(jet->eta());
              v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
              v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
              }
              if(v->getTriggerType() == "DiJet_Trigger")
              {
              CaloJetCollection::const_iterator jet2 = jet++;
              v->getMEhisto_DenominatorPt()->Fill((jet->pt() + jet2->pt())/2.);
              v->getMEhisto_DenominatorEta()->Fill((jet->eta() + jet2->eta())/2.);
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
              CaloJetCollection::const_iterator jet = calojet.begin();
              if(v->getTriggerType() == "SingleJet_Trigger")
              {
              v->getMEhisto_NumeratorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
              v->getMEhisto_NumeratorEta()->Fill(jet->eta());
              v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
              v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());
              }
              if(v->getTriggerType() == "DiJet_Trigger")     
              {
              CaloJetCollection::const_iterator jet2 = jet++; 
              v->getMEhisto_NumeratorPt()->Fill((jet->pt() + jet2->pt())/2.);  
              v->getMEhisto_NumeratorEta()->Fill((jet->eta() + jet2->eta())/2.);
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
              if(v->getTriggerType() == "SingleJet_Trigger" ) 
              {
              v->getMEhisto_DenominatorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
              v->getMEhisto_DenominatorEta()->Fill(jet->eta());
              v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
              v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
              }
              if(v->getTriggerType() == "DiJet_Trigger" ) 
              {
              CaloJetCollection::const_iterator jet2 = jet++;
              v->getMEhisto_DenominatorPt()->Fill((jet->pt() + jet2->pt())/2.);  
              v->getMEhisto_DenominatorEta()->Fill((jet->eta() + jet2->eta())/2.);

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
              CaloJetCollection::const_iterator jet = calojet.begin();
              if(v->getTriggerType() == "SingleJet_Trigger" )
              {
              v->getMEhisto_NumeratorPt()->Fill(jet->pt());
              if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
              v->getMEhisto_NumeratorEta()->Fill(jet->eta());
              v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
              v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());
              }
              if(v->getTriggerType() == "DiJet_Trigger" )
              {
              CaloJetCollection::const_iterator jet2 = jet++;
              v->getMEhisto_NumeratorPt()->Fill((jet->pt() + jet2->pt())/2.);
              v->getMEhisto_NumeratorEta()->Fill((jet->eta() + jet2->eta())/2.);

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
   

    ME_Denominator_rate  = dbe->book1D("ME_Denominator_rate","rate Denominator Triggers",
                           hltPathsAll_.size()+1,-0.5,hltPathsAll_.size()+1-0.5);
    ME_Numerator_rate  = dbe->book1D("ME_Numerator_rate","rate of Numerator Triggers",
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
      ME_Denominator_rate->setBinLabel(nname+1,denomlabelnm);
      ME_Numerator_rate->setBinLabel(nname+1,labelnm);
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
   std::string dirName = dirname_ + "/MonitorAllTriggers/";
   for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){  

   std::string subdirName = dirName + v->getPath();
   dbe->setCurrentFolder(subdirName);  
 
   MonitorElement *dummy;
   dummy =  dbe->bookFloat("dummy");    

   std::string labelname("ME");
   std::string histoname(labelname+"");
   std::string title(labelname+"");
   if(v->getObjectType() == trigger::TriggerJet && v->getTriggerType() == "SingleJet_Trigger")
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
                  PtResolution_HLTRecObj,EtaResolution_HLTRecObj,PhiResolution_HLTRecObj,
                  PtCorrelation_L1HLT,EtaCorrelation_L1HLT,PhiCorrelation_L1HLT,
                  PtCorrelation_HLTRecObj,EtaCorrelation_HLTRecObj,PhiCorrelation_HLTRecObj,
                  dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy);

   }// histos for Jet Triggers

  if(v->getObjectType() == trigger::TriggerJet && v->getTriggerType() == "DiJet_Trigger")
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


   histoname = labelname+"_hltRecObjPtCorrelation";
   title = labelname+"_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]";
   MonitorElement * PtCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_HLTRecObj->getTH1();                                                                                              
   h->Sumw2();                                                                                                                         

   histoname = labelname+"_hltRecObjEtaCorrelation";
   title = labelname+"_hltRecObjEtaCorrelation;#eta(HLT);#eta(Reco)";
   MonitorElement * EtaCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_
);                                                                                                                                          
   h = EtaCorrelation_HLTRecObj->getTH1();                                                                                                  
   h->Sumw2();

   histoname = labelname+"_hltRecObjPhiCorrelation";
   title = labelname+"_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)";
   MonitorElement * PhiCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_
);
   h = PhiCorrelation_HLTRecObj->getTH1();
   h->Sumw2();

   histoname = labelname+"_RecObjAveragePt";
   title     = labelname+"_RecObjAveragePt;Pt[GeV/c]";
   MonitorElement * jetAveragePt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = jetAveragePt->getTH1();
   h->Sumw2();

   histoname = labelname+"_RecObjAverageEta";
   title     = labelname+"_RecObjAverageEta;#eta";
   MonitorElement * jetAverageEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = jetAverageEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_RecObjPhiDifference";
   title     = labelname+"_RecObjPhiDifference;#Delta#Phi";
   MonitorElement * jetPhiDifference = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = jetPhiDifference->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltObjAveragePt";
   title     = labelname+"_hltObjAveragePt;Pt[GeV/c]";
   MonitorElement * hltAveragePt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = hltAveragePt->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltObjAverageEta";
   title     = labelname+"_hltObjAverageEta;#eta";
   MonitorElement * hltAverageEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = hltAverageEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_hltObjPhiDifference";
   title     = labelname+"_hltObjPhiDifference;#Delta#Phi";
   MonitorElement * hltPhiDifference = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = hltPhiDifference->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1ObjAveragePt";
   title     = labelname+"_l1ObjAveragePt;Pt[GeV/c]";
   MonitorElement * l1AveragePt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = l1AveragePt->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1ObjAverageEta";
   title     = labelname+"_l1ObjAverageEta;#eta";
   MonitorElement * l1AverageEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = l1AverageEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_l1ObjPhiDifference";
   title     = labelname+"_l1ObjPhiDifference;#Delta#Phi";
   MonitorElement * l1PhiDifference = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = l1PhiDifference->getTH1();
   h->Sumw2();
  

   v->setHistos(  N, Pt,  PtBarrel, PtEndcap, PtForward, Eta, Phi, EtaPhi,
                  N_L1, Pt_L1,  PtBarrel_L1, PtEndcap_L1, PtForward_L1, Eta_L1, Phi_L1, EtaPhi_L1,
                  N_HLT, Pt_HLT,  PtBarrel_HLT, PtEndcap_HLT, PtForward_HLT, Eta_HLT, Phi_HLT, EtaPhi_HLT,
                  PtResolution_L1HLT, EtaResolution_L1HLT,PhiResolution_L1HLT,
                  PtResolution_HLTRecObj,EtaResolution_HLTRecObj,PhiResolution_HLTRecObj,
                  PtCorrelation_L1HLT,EtaCorrelation_L1HLT,PhiCorrelation_L1HLT,
                  PtCorrelation_HLTRecObj,EtaCorrelation_HLTRecObj,PhiCorrelation_HLTRecObj,
                  jetAveragePt, jetAverageEta, jetPhiDifference, hltAveragePt, hltAverageEta, hltPhiDifference,
                  l1AveragePt, l1AverageEta, l1PhiDifference);

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
                  PtResolution_HLTRecObj,dummy,PhiResolution_HLTRecObj,
                  PtCorrelation_L1HLT,dummy,PhiCorrelation_L1HLT,
                  PtCorrelation_HLTRecObj,dummy,PhiCorrelation_HLTRecObj,
                  dummy, dummy, dummy, dummy,dummy, dummy, dummy, dummy,dummy);

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
  std::string dirName = dirname_ + "/MonitorAllTriggersWrtMuonTrigger/";
  for(PathInfoCollection::iterator v = hltPathsWrtMu_.begin(); v!= hltPathsWrtMu_.end(); ++v ){
  
  std::string subdirName = dirName + v->getPath();
  dbe->setCurrentFolder(subdirName);             
                                                   
   MonitorElement *dummy;                          
   dummy =  dbe->bookFloat("dummy");               

   std::string labelname("ME");
   std::string histoname(labelname+"");
   std::string title(labelname+"");    
   if(v->getObjectType() == trigger::TriggerJet && v->getTriggerType() == "SingleJet_Trigger")
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
                  PtResolution_HLTRecObj,EtaResolution_HLTRecObj,PhiResolution_HLTRecObj,                  
                  PtCorrelation_L1HLT,EtaCorrelation_L1HLT,PhiCorrelation_L1HLT,                           
                  PtCorrelation_HLTRecObj,EtaCorrelation_HLTRecObj,PhiCorrelation_HLTRecObj,               
                  dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy);                          

   }// histos for Jet Triggers

  if(v->getObjectType() == trigger::TriggerJet && v->getTriggerType() == "DiJet_Trigger")
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


   histoname = labelname+"_hltRecObjPtCorrelation";
   title = labelname+"_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]";
   MonitorElement * PtCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = PtCorrelation_HLTRecObj->getTH1();                                                                                              
   h->Sumw2();                                                                                                                         

   histoname = labelname+"_hltRecObjEtaCorrelation";
   title = labelname+"_hltRecObjEtaCorrelation;#eta(HLT);#eta(Reco)";
   MonitorElement * EtaCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Etabins_,EtaMin_,EtaMax_ );                                                                                                                                          
   h = EtaCorrelation_HLTRecObj->getTH1();                                                                                                  
   h->Sumw2();                                                                                                                              

   histoname = labelname+"_hltRecObjPhiCorrelation";
   title = labelname+"_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)";
   MonitorElement * PhiCorrelation_HLTRecObj = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Phibins_,PhiMin_,PhiMax_);                                                                                                                                          
   h = PhiCorrelation_HLTRecObj->getTH1();                                                                                                  
   h->Sumw2();                                                                                                                              

   histoname = labelname+"_RecObjAveragePt";
   title     = labelname+"_RecObjAveragePt;Pt[GeV/c]";
   MonitorElement * jetAveragePt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = jetAveragePt->getTH1();                                                                        
   h->Sumw2();                                                                                        

   histoname = labelname+"_RecObjAverageEta";
   title     = labelname+"_RecObjAverageEta;#eta";
   MonitorElement * jetAverageEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = jetAverageEta->getTH1();                                                                           
   h->Sumw2();                                                                                            

   histoname = labelname+"_RecObjPhiDifference";
   title     = labelname+"_RecObjPhiDifference;#Delta#Phi";
   MonitorElement * jetPhiDifference = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = jetPhiDifference->getTH1();                                                                           
   h->Sumw2();                                                                                               

   histoname = labelname+"_hltObjAveragePt";
   title     = labelname+"_hltObjAveragePt;Pt[GeV/c]";
   MonitorElement * hltAveragePt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = hltAveragePt->getTH1();                                                                        
   h->Sumw2();                                                                                        

   histoname = labelname+"_hltObjAverageEta";
   title     = labelname+"_hltObjAverageEta;#eta";
   MonitorElement * hltAverageEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = hltAverageEta->getTH1();                                                                           
   h->Sumw2();                                                                                            

   histoname = labelname+"_hltObjPhiDifference";
   title     = labelname+"_hltObjPhiDifference;#Delta#Phi";
   MonitorElement * hltPhiDifference = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = hltPhiDifference->getTH1();                                                                           
   h->Sumw2();                                                                                               

   histoname = labelname+"_l1ObjAveragePt";
   title     = labelname+"_l1ObjAveragePt;Pt[GeV/c]";
   MonitorElement * l1AveragePt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = l1AveragePt->getTH1();                                                                        
   h->Sumw2();                                                                                       

   histoname = labelname+"_l1ObjAverageEta";
   title     = labelname+"_l1ObjAverageEta;#eta";
   MonitorElement * l1AverageEta = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = l1AverageEta->getTH1();                                                                           
   h->Sumw2();                                                                                           

   histoname = labelname+"_l1ObjPhiDifference";
   title     = labelname+"_l1ObjPhiDifference;#Delta#Phi";
   MonitorElement * l1PhiDifference = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = l1PhiDifference->getTH1();
   h->Sumw2();


   v->setHistos(  N, Pt,  PtBarrel, PtEndcap, PtForward, Eta, Phi, EtaPhi,
                  N_L1, Pt_L1,  PtBarrel_L1, PtEndcap_L1, PtForward_L1, Eta_L1, Phi_L1, EtaPhi_L1,
                  N_HLT, Pt_HLT,  PtBarrel_HLT, PtEndcap_HLT, PtForward_HLT, Eta_HLT, Phi_HLT, EtaPhi_HLT,
                  PtResolution_L1HLT, EtaResolution_L1HLT,PhiResolution_L1HLT,
                  PtResolution_HLTRecObj,EtaResolution_HLTRecObj,PhiResolution_HLTRecObj,
                  PtCorrelation_L1HLT,EtaCorrelation_L1HLT,PhiCorrelation_L1HLT,
                  PtCorrelation_HLTRecObj,EtaCorrelation_HLTRecObj,PhiCorrelation_HLTRecObj,
                  jetAveragePt, jetAverageEta, jetPhiDifference, hltAveragePt, hltAverageEta, hltPhiDifference,
                  l1AveragePt, l1AverageEta, l1PhiDifference);

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
                  PtResolution_HLTRecObj,dummy,PhiResolution_HLTRecObj,
                  PtCorrelation_L1HLT,dummy,PhiCorrelation_L1HLT,
                  PtCorrelation_HLTRecObj,dummy,PhiCorrelation_HLTRecObj,
                  dummy, dummy, dummy, dummy,dummy, dummy, dummy, dummy,dummy);

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
  std::string dirName1 = dirname_ + "/RelativeTriggerEff/";
  for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v ){
   std::string labelname("ME") ;
   std::string subdirName = dirName1 + v->getPath() + "_wrt_" + v->getDenomPath();
   dbe->setCurrentFolder(subdirName);
   std::string histoname(labelname+"");
   std::string title(labelname+"");
   
   MonitorElement *dummy;
   dummy =  dbe->bookFloat("dummy");   
    
   if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType() == "SingleJet_Trigger"))
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


   v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
                      DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi);
  }// Loop over Jet Trigger

   if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType() == "DiJet_Trigger"))
   {
   histoname = labelname+"_NumeratorAvrgPt";
   title     = labelname+"NumeratorAvrgPt;Pt[GeV/c]";
   MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = NumeratorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorAvrgEta";
   title     = labelname+"NumeratorAvrgEta;#eta";
   MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = NumeratorEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorAvrgPt";
   title     = labelname+"DenominatorAvrgPt;Pt[GeV/c]";
   MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorAvrgEta";
   title     = labelname+"DenominatorAvrgEta;#eta";
   MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = DenominatorEta->getTH1();
   h->Sumw2();

   v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, NumeratorEta, dummy, dummy,
               DenominatorPt,  dummy, dummy, dummy, DenominatorEta, dummy, dummy);

   }

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
   if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType() == "SingleJet_Trigger"))
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


   v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
                    DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi);
  }// Loop over Jet Trigger

  if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType() == "DiJet_Trigger"))
   {
   histoname = labelname+"_NumeratorAvrgPt";
   title     = labelname+"NumeratorAvrgPt;Pt[GeV/c]";
   MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = NumeratorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorAvrgEta";
   title     = labelname+"NumeratorAvrgEta;#eta";
   MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = NumeratorEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorAvrgPt";
   title     = labelname+"DenominatorAvrgPt;Pt[GeV/c]";
   MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorAvrgEta";
   title     = labelname+"DenominatorAvrgEta;#eta";
   MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = DenominatorEta->getTH1();
   h->Sumw2();

   v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, NumeratorEta, dummy, dummy,
               DenominatorPt,  dummy, dummy, dummy, DenominatorEta, dummy, dummy);

   }
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
   if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType() == "SingleJet_Trigger"))
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


   v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
                     DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi);
  }// Loop over Jet Trigger


  if((v->getObjectType() == trigger::TriggerJet) && (v->getTriggerType() == "DiJet_Trigger"))
   {
   histoname = labelname+"_NumeratorAvrgPt";
   title     = labelname+"NumeratorAvrgPt;Pt[GeV/c]";
   MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = NumeratorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_NumeratorAvrgEta";
   title     = labelname+"NumeratorAvrgEta;#eta";
   MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = NumeratorEta->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorAvrgPt";
   title     = labelname+"DenominatorAvrgPt;Pt[GeV/c]";
   MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = DenominatorPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_DenominatorAvrgEta";
   title     = labelname+"DenominatorAvrgEta;#eta";
   MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = DenominatorEta->getTH1();
   h->Sumw2();

   v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, NumeratorEta, dummy, dummy,
               DenominatorPt,  dummy, dummy, dummy, DenominatorEta, dummy, dummy);

   }
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
   



   v->setEffHistos(  NumeratorPt,  dummy, dummy, dummy, dummy, NumeratorPhi, dummy,
                     DenominatorPt, dummy, dummy, dummy, dummy, DenominatorPhi, dummy);


   }// Loop over MET trigger
}


}// This is loop over all efficiency plots
//--------Histos to see WHY trigger is NOT fired----------
   int Nbins_       = 10;
   int Nmin_        = 0;
   int Nmax_        = 10;
   int Ptbins_      = 40;
   int Etabins_     = 40;
   int Phibins_     = 20;
   double PtMin_    = 0.;
   double PtMax_    = 200.;
   double EtaMin_   = -5.;
   double EtaMax_   =  5.;
   double PhiMin_   = -3.14159;
   double PhiMax_   =  3.14159;

   std::string dirName4_ = dirname_ + "/TriggerNotFired/";
   for(PathInfoCollection::iterator v = hltPathsNTfired_.begin(); v!= hltPathsNTfired_.end(); ++v ){

    MonitorElement *dummy;
    dummy =  dbe->bookFloat("dummy");
 

   std::string labelname("ME") ;
   std::string histoname(labelname+"");
   std::string title(labelname+"");
   dbe->setCurrentFolder(dirName4_ + v->getPath());
  
   histoname = labelname+"_TriggerSummary";
   title     = labelname+"Summary of trigger levels"; 
   MonitorElement * TriggerSummary = dbe->book1D(histoname.c_str(),title.c_str(),8, -0.5,7.5);
   std::string trigger[7] = {"Nevt","L1 failed", "L1 & HLT failed", "L1 failed but not HLT","L1 passed", "L1 & HLT passed","L1 passed but not HLT"};
   for(int i =0; i < 8; i++)TriggerSummary->setBinLabel(i+1, trigger[i]);
  if((v->getTriggerType())=="SingleJet_Trigger")
   {
   histoname = labelname+"_JetPt"; 
   title     = labelname+"Leading jet pT;Pt[GeV/c]";
   MonitorElement * JetPt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = JetPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_JetEtaVsPt";
   title     = labelname+"Leading jet #eta vs pT;#eta;Pt[GeV/c]";
   MonitorElement * JetEtaVsPt = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Ptbins_,PtMin_,PtMax_);
   h = JetEtaVsPt->getTH1();
   h->Sumw2();

   histoname = labelname+"_JetPhiVsPt";
   title     = labelname+"Leading jet #Phi vs pT;#Phi;Pt[GeV/c]";
   MonitorElement * JetPhiVsPt = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Ptbins_,PtMin_,PtMax_);
   h = JetPhiVsPt->getTH1();
   h->Sumw2();

   
   
   v->setDgnsHistos( TriggerSummary, dummy, JetPt, JetEtaVsPt, JetPhiVsPt, dummy, dummy, dummy, dummy, dummy, dummy); 
   }// single Jet trigger  

  if((v->getTriggerType())=="DiJet_Trigger")
   {
   histoname = labelname+"_JetSize"; 
   title     = labelname+"Jet Size;multiplicity";
   MonitorElement * JetSize = dbe->book1D(histoname.c_str(),title.c_str(),Nbins_,Nmin_,Nmax_);
   TH1 * h = JetSize->getTH1();
   h->Sumw2();


   histoname = labelname+"_AvergPt";
   title     = labelname+"Average Pt;Pt[GeV/c]";
   MonitorElement * Pt12 = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt12->getTH1();
   h->Sumw2(); 

   histoname = labelname+"_AvergEta";
   title     = labelname+"Average Eta;#eta";
   MonitorElement * Eta12 = dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
   h = Eta12->getTH1();
   h->Sumw2();

   histoname = labelname+"_PhiDifference";
   title     = labelname+"#Delta#Phi;#Delta#Phi";
   MonitorElement * Phi12 = dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
   h = Phi12->getTH1();
   h->Sumw2(); 

   histoname = labelname+"_Pt3Jet";
   title     = labelname+"Pt of 3rd Jet;Pt[GeV/c]";
   MonitorElement * Pt3 = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   h = Pt3->getTH1();
   h->Sumw2();

   histoname = labelname+"_Pt12VsPt3Jet";
   title     = labelname+"Pt of 3rd Jet vs Average Pt of leading jets;Avergage Pt[GeV/c]; Pt of 3rd Jet [GeV/c]";
   MonitorElement * Pt12Pt3 = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Ptbins_,PtMin_,PtMax_);
   h = Pt12Pt3->getTH1();
   h->Sumw2();

   histoname = labelname+"_Pt12VsPhi12";
   title     = labelname+"Average Pt of leading jets vs #Delta#Phi between leading jets;Avergage Pt[GeV/c]; #Delta#Phi";
   MonitorElement * Pt12Phi12 = dbe->book2D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_,Phibins_,PhiMin_,PhiMax_);
   h = Pt12Phi12->getTH1();
   h->Sumw2();

   v->setDgnsHistos( TriggerSummary, JetSize, dummy, dummy, dummy, Pt12, Eta12, Phi12, Pt3, Pt12Pt3, Pt12Phi12);

   }// Dijet Jet trigger

  if((v->getTriggerType())=="MET_Trigger")
   {
   histoname = labelname+"_MET";
   title     = labelname+"MET;Pt[GeV/c]";
   MonitorElement * MET = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
   TH1 * h = MET->getTH1();
   h->Sumw2();
   v->setDgnsHistos(TriggerSummary, dummy, MET, dummy, dummy, dummy, dummy, dummy,dummy,dummy,dummy);
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



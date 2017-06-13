#include <math.h>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/TrackingMonitor/interface/GetLumi.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "DataFormats/Common/interface/Handle.h"


#include "DQMOffline/Trigger/plugins/MssmHbbBtagTriggerMonitor.h"

#include "TLorentzVector.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

MssmHbbBtagTriggerMonitor::MssmHbbBtagTriggerMonitor( const edm::ParameterSet& iConfig )
{
  folderName_             = iConfig.getParameter<std::string>("dirname");
  processname_            = iConfig.getParameter<std::string>("processname");
  pathname_               = iConfig.getParameter<std::string>("pathname");
  triggerobjbtag_         = iConfig.getParameter<std::string>("triggerobjbtag");
  jetPtmin_               = iConfig.getParameter<double>("jetPtMin");
  jetEtamax_              = iConfig.getParameter<double>("jetEtaMax");
  tagBtagmin_             = iConfig.getParameter<double>("tagBtagMin");
  probeBtagmin_           = iConfig.getParameter<double>("probeBtagMin");
  triggerSummaryLabel_    = iConfig.getParameter<edm::InputTag>("triggerSummary");
  triggerResultsLabel_    = iConfig.getParameter<edm::InputTag>("triggerResults");
  triggerSummaryToken_    = consumes <trigger::TriggerEvent> (triggerSummaryLabel_);
  triggerResultsToken_    = consumes <edm::TriggerResults>   (triggerResultsLabel_);
  offlineCSVPFToken_      = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("offlineCSVPF"));

}

MssmHbbBtagTriggerMonitor::~MssmHbbBtagTriggerMonitor()
{

}

void MssmHbbBtagTriggerMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
   
  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());
  
  pt_jet1_ = ibooker.book1D("pt_jet1","pt_jet1",60,0,600);
  pt_jet2_ = ibooker.book1D("pt_jet2","pt_jet2",60,0,600);
  
  pt_probe_ = ibooker.book1D("pt_probe","pt_probe",60,0,600);
  pt_probe_match_ = ibooker.book1D("pt_probe_match","pt_probe_match",60,0,600);
  

  discr_offline_btagcsv_jet1_ = ibooker.book1D("discr_offline_btagcsv_jet1","discr_offline_btagcsv_jet1",20,0,10);
  discr_offline_btagcsv_jet2_ = ibooker.book1D("discr_offline_btagcsv_jet2","discr_offline_btagcsv_jet2",20,0,10);
  
  // Initialize the GenericTriggerEventFlag
}

void MssmHbbBtagTriggerMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)
{
   edm::Handle<edm::TriggerResults> triggerResultsHandler;
   iEvent.getByToken(triggerResultsToken_, triggerResultsHandler);
   const edm::TriggerResults & triggers = *(triggerResultsHandler.product());
   
   bool accept = false;
   
   for ( size_t j = 0 ; j < hltConfig_.size() ; ++j )
   {
      if ( hltConfig_.triggerName(j).find(pathname_) == 0 )
      {
         if ( triggers.accept(j) ) 
         {
            accept = true;
            break;
         }
      }
   }
   
   if ( accept )
   {
      bool match1 = false;
      bool match2 = false;
      
      edm::Handle<reco::JetTagCollection> offlineJetTagPFHandler;
      iEvent.getByToken(offlineCSVPFToken_, offlineJetTagPFHandler);
      
      if(offlineJetTagPFHandler.isValid())
      {
         reco::JetTagCollection jettags = *(offlineJetTagPFHandler.product());
         if ( jettags.size() > 1 )
         {
            const reco::Jet * jet1 = jettags.key(0).get();
            const reco::Jet * jet2 = jettags.key(1).get();
            TLorentzVector p4_jet1;
            p4_jet1.SetPtEtaPhiM(jet1->pt(),jet1->eta(),jet1->phi(),0);
            TLorentzVector p4_jet2;
            p4_jet2.SetPtEtaPhiM(jet2->pt(),jet2->eta(),jet2->phi(),0);
            float btag1 = jettags.value(0);
            float btag2 = jettags.value(1);
               
            if ( jet1->pt() > jetPtmin_ && jet2->pt() > jetPtmin_ && fabs(jet1->eta()) < jetEtamax_ && fabs(jet2->eta()) < jetEtamax_ )
            {
               if ( btag1 > tagBtagmin_ && btag2 > probeBtagmin_ )
               {
                  pt_jet1_ -> Fill(jet1->pt());
                  pt_jet2_ -> Fill(jet2->pt());
                  discr_offline_btagcsv_jet1_ -> Fill(-log(1-btag1));
                  discr_offline_btagcsv_jet2_ -> Fill(-log(1-btag2));
                  
// trigger objects matching  
                  std::vector<trigger::TriggerObject> onlinebtags;           
                  edm::Handle<trigger::TriggerEvent> triggerEventHandler;
                  iEvent.getByToken(triggerSummaryToken_, triggerEventHandler);
                  const unsigned int filterIndex(triggerEventHandler->filterIndex(edm::InputTag(triggerobjbtag_,"",processname_)));
                  if ( filterIndex < triggerEventHandler->sizeFilters() )
                  {
                     const trigger::Keys& keys(triggerEventHandler->filterKeys(filterIndex));
                     const trigger::TriggerObjectCollection & triggerObjects = triggerEventHandler->getObjects();
                     for ( auto & key : keys )
                     {
                        onlinebtags.push_back(triggerObjects[key]);
                     }
                  }
                  for ( auto & to : onlinebtags )
                  {
                     TLorentzVector p4_to;
                     p4_to.SetPtEtaPhiM(to.pt(),to.eta(),to.phi(),0);
                     
                     if ( p4_jet1.DeltaR(p4_to) ) match1 = true;
                     if ( p4_jet2.DeltaR(p4_to) ) match2 = true;
                  }
                  if ( match1 ) // jet1 is the tag
                  {
                     pt_probe_ -> Fill(jet2->pt());
                     if ( match2 ) // jet2 is the probe
                     {
                        pt_probe_match_ -> Fill(jet2->pt());
                     }
                  }
               } // offline jets btag
            } // offline jets kinematic selection
         } // at least two offline jets
      } // offline jettag is valid
   } // accept trigger
   
  // Filter out events if Trigger Filtering is requested
//  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

}
void MssmHbbBtagTriggerMonitor::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
    bool changed(true);
    if (!hltConfig_.init(iRun, iSetup, processname_, changed))
    {
       LogDebug("MssmHbbBtagTriggerMonitor") << "HLTConfigProvider failed to initialize.";
    }
}


// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MssmHbbBtagTriggerMonitor);

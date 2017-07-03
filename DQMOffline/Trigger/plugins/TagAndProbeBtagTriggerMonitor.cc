#include <math.h>

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"


#include "DQMOffline/Trigger/plugins/TagAndProbeBtagTriggerMonitor.h"

#include "TLorentzVector.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

TagAndProbeBtagTriggerMonitor::TagAndProbeBtagTriggerMonitor( const edm::ParameterSet& iConfig )
{
  folderName_             = iConfig.getParameter<std::string>("dirname");
  processname_            = iConfig.getParameter<std::string>("processname");
  triggerobjbtag_         = iConfig.getParameter<std::string>("triggerobjbtag");
  jetPtmin_               = iConfig.getParameter<double>("jetPtMin");
  jetEtamax_              = iConfig.getParameter<double>("jetEtaMax");
  tagBtagmin_             = iConfig.getParameter<double>("tagBtagMin");
  probeBtagmin_           = iConfig.getParameter<double>("probeBtagMin");
  triggerSummaryLabel_    = iConfig.getParameter<edm::InputTag>("triggerSummary");
  triggerSummaryToken_    = consumes <trigger::TriggerEvent> (triggerSummaryLabel_);
  offlineBtagToken_       = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("offlineBtag"));
  genTriggerEventFlag_    = new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("genericTriggerEventPSet"),consumesCollector(), *this);

  jetPtbins_              = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetPt"));
  jetEtabins_             = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetEta"));
  jetPhibins_             = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetPhi"));
  jetBtagbins_            = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetBtag"));
  
}

TagAndProbeBtagTriggerMonitor::~TagAndProbeBtagTriggerMonitor()
{
  if (genTriggerEventFlag_) delete genTriggerEventFlag_;

}

Binning TagAndProbeBtagTriggerMonitor::getHistoPSet(edm::ParameterSet pset)
{
   return Binning
   {
      pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
   };
}



void TagAndProbeBtagTriggerMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
   
  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());
  
  pt_jet1_  = ibooker.book1D("pt_jet1","pt_jet1",jetPtbins_.nbins,jetPtbins_.xmin,jetPtbins_.xmax);
  pt_jet2_  = ibooker.book1D("pt_jet2","pt_jet2",jetPtbins_.nbins,jetPtbins_.xmin,jetPtbins_.xmax);
  eta_jet1_ = ibooker.book1D("eta_jet1","eta_jet1",jetEtabins_.nbins,jetEtabins_.xmin,jetEtabins_.xmax);
  eta_jet2_ = ibooker.book1D("eta_jet2","eta_jet2",jetEtabins_.nbins,jetEtabins_.xmin,jetEtabins_.xmax);
  phi_jet1_ = ibooker.book1D("phi_jet1","phi_jet1",jetPhibins_.nbins,jetPhibins_.xmin,jetPhibins_.xmax);
  phi_jet2_ = ibooker.book1D("phi_jet2","phi_jet2",jetPhibins_.nbins,jetPhibins_.xmin,jetPhibins_.xmax);
  eta_phi_jet1_ = ibooker.book2D("eta_phi_jet1","eta_phi_jet1",jetEtabins_.nbins,jetEtabins_.xmin,jetEtabins_.xmax,jetPhibins_.nbins,jetPhibins_.xmin,jetPhibins_.xmax);
  eta_phi_jet2_ = ibooker.book2D("eta_phi_jet2","eta_phi_jet2",jetEtabins_.nbins,jetEtabins_.xmin,jetEtabins_.xmax,jetPhibins_.nbins,jetPhibins_.xmin,jetPhibins_.xmax);
  
  pt_probe_        = ibooker.book1D("pt_probe","pt_probe",jetPtbins_.nbins,jetPtbins_.xmin,jetPtbins_.xmax);
  pt_probe_match_  = ibooker.book1D("pt_probe_match","pt_probe_match",jetPtbins_.nbins,jetPtbins_.xmin,jetPtbins_.xmax);
  eta_probe_       = ibooker.book1D("eta_probe","eta_probe",jetEtabins_.nbins,jetEtabins_.xmin,jetEtabins_.xmax);
  eta_probe_match_ = ibooker.book1D("eta_probe_match","eta_probe_match",jetEtabins_.nbins,jetEtabins_.xmin,jetEtabins_.xmax);
  phi_probe_       = ibooker.book1D("phi_probe","phi_probe",jetPhibins_.nbins,jetPhibins_.xmin,jetPhibins_.xmax);
  phi_probe_match_ = ibooker.book1D("phi_probe_match","phi_probe_match",jetPhibins_.nbins,jetPhibins_.xmin,jetPhibins_.xmax);
  eta_phi_probe_       = ibooker.book2D("eta_phi_probe","eta_phi_probe",jetEtabins_.nbins,jetEtabins_.xmin,jetEtabins_.xmax,jetPhibins_.nbins,jetPhibins_.xmin,jetPhibins_.xmax);
  eta_phi_probe_match_ = ibooker.book2D("eta_phi_probe_match","eta_phi_match",jetEtabins_.nbins,jetEtabins_.xmin,jetEtabins_.xmax,jetPhibins_.nbins,jetPhibins_.xmin,jetPhibins_.xmax);

  discr_offline_btag_jet1_ = ibooker.book1D("discr_offline_btag_jet1","discr_offline_btag_jet1",jetBtagbins_.nbins,jetBtagbins_.xmin,jetBtagbins_.xmax);
  discr_offline_btag_jet2_ = ibooker.book1D("discr_offline_btag_jet2","discr_offline_btag_jet2",jetBtagbins_.nbins,jetBtagbins_.xmin,jetBtagbins_.xmax);
  
  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_ && genTriggerEventFlag_->on() ) genTriggerEventFlag_->initRun( iRun, iSetup );
}

void TagAndProbeBtagTriggerMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)
{

//    bool accept = false;
   bool match1 = false;
   bool match2 = false;
   
   edm::Handle<reco::JetTagCollection> offlineJetTagPFHandler;
   iEvent.getByToken(offlineBtagToken_, offlineJetTagPFHandler);
      
   if( ! offlineJetTagPFHandler.isValid()) return;
   
   // applying selection for event; tag & probe -> selection  for all events
   if (genTriggerEventFlag_->on() && ! genTriggerEventFlag_->accept( iEvent, iSetup) )
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
               pt_jet1_  -> Fill(jet1->pt());
               pt_jet2_  -> Fill(jet2->pt());
               eta_jet1_ -> Fill(jet1->eta());
               eta_jet2_ -> Fill(jet2->eta());
               phi_jet1_ -> Fill(jet1->phi());
               phi_jet2_ -> Fill(jet2->phi());
               eta_phi_jet1_ -> Fill(jet1->eta(),jet1->phi());
               eta_phi_jet2_ -> Fill(jet2->eta(),jet2->phi());
               if ( btag1 < 0 ) btag1 = -0.5;
               if ( btag2 < 0 ) btag2 = -0.5;
               discr_offline_btag_jet1_ -> Fill(btag1);
               discr_offline_btag_jet2_ -> Fill(btag2);
               
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
                  pt_probe_  -> Fill(jet2->pt());
                  eta_probe_ -> Fill(jet2->eta());
                  phi_probe_ -> Fill(jet2->phi());
                  eta_phi_probe_ -> Fill(jet2->eta(),jet2->phi());
                  if ( match2 ) // jet2 is the probe
                  {
                     pt_probe_match_  -> Fill(jet2->pt());
                     eta_probe_match_ -> Fill(jet2->eta());
                     phi_probe_match_ -> Fill(jet2->phi());
                     eta_phi_probe_match_ -> Fill(jet2->eta(),jet2->phi());
                  }
               }
            } // offline jets btag
         } // offline jets kinematic selection
      } // at least two offline jets
   } // accept trigger
   
}
void TagAndProbeBtagTriggerMonitor::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
}


// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TagAndProbeBtagTriggerMonitor);

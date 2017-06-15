#include <math.h>

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"


#include "DQMOffline/Trigger/plugins/MssmHbbMonitor.h"

#include "TLorentzVector.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

MssmHbbMonitor::MssmHbbMonitor( const edm::ParameterSet& iConfig )
{
  folderName_             = iConfig.getParameter<std::string>("dirname");
  processname_            = iConfig.getParameter<std::string>("processname");
  pathname_               = iConfig.getParameter<std::string>("pathname");
  jetPtmin_               = iConfig.getParameter<double>("jetPtMin");
  jetEtamax_              = iConfig.getParameter<double>("jetEtaMax");
  triggerResultsToken_    = consumes <edm::TriggerResults>   (iConfig.getParameter<edm::InputTag>("triggerResults"));
  muonsToken_             = consumes<reco::MuonCollection>   (iConfig.getParameter<edm::InputTag>("muons"));
  offlineBtagToken_       = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("offlineBtag"));

  jetPtbins_              = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetPt"));
  jetEtabins_             = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetEta"));
  jetPhibins_             = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetPhi"));
  jetDRbins_              = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetsDR"));
  jetBtagbins_            = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetBtag"));
  muonPtbins_             = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonPt"));
  muonEtabins_            = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonEta"));
  muonPhibins_            = getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("muonPhi"));
  
}

MssmHbbMonitor::~MssmHbbMonitor()
{

}

Binning MssmHbbMonitor::getHistoPSet(edm::ParameterSet pset)
{
   return Binning
   {
      pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
   };
}



void MssmHbbMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
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
  
  deta_jet12_ = ibooker.book1D("deta_jet12","deta_jet12",jetEtabins_.nbins,jetEtabins_.xmin,jetEtabins_.xmax);
  dphi_jet12_ = ibooker.book1D("dphi_jet12","dphi_jet12",jetPhibins_.nbins,jetPhibins_.xmin,jetPhibins_.xmax);
  dr_jet12_   = ibooker.book1D("dr_jet12","dr_jet12",jetDRbins_.nbins,jetDRbins_.xmin,jetDRbins_.xmax);

  discr_offline_btag_jet1_ = ibooker.book1D("discr_offline_btag_jet1","discr_offline_btag_jet1",jetBtagbins_.nbins,jetBtagbins_.xmin,jetBtagbins_.xmax);
  discr_offline_btag_jet2_ = ibooker.book1D("discr_offline_btag_jet2","discr_offline_btag_jet2",jetBtagbins_.nbins,jetBtagbins_.xmin,jetBtagbins_.xmax);

  pt_muon_  = ibooker.book1D("pt_muon","pt_muon",muonPtbins_.nbins,muonPtbins_.xmin,muonPtbins_.xmax);
  eta_muon_ = ibooker.book1D("eta_muon","eta_muon",muonEtabins_.nbins,muonEtabins_.xmin,muonEtabins_.xmax);
  phi_muon_ = ibooker.book1D("phi_muon","phi_muon",muonPhibins_.nbins,muonPhibins_.xmin,muonPhibins_.xmax);
    
  // Initialize the GenericTriggerEventFlag
}

void MssmHbbMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)
{

   bool accept = false;
   
   edm::Handle<edm::TriggerResults> triggerResultsHandler;
   iEvent.getByToken(triggerResultsToken_, triggerResultsHandler);
   if ( ! triggerResultsHandler.isValid() )  return;
      
   edm::Handle<reco::JetTagCollection> offlineJetTagPFHandler;
   iEvent.getByToken(offlineBtagToken_, offlineJetTagPFHandler);
      
   if( ! offlineJetTagPFHandler.isValid()) return;
   
   const edm::TriggerResults & triggers = *(triggerResultsHandler.product());
   
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
         if ( btag1 < 0 ) btag1 = -0.5;
         if ( btag2 < 0 ) btag2 = -0.5;
         
         float deta12 = p4_jet1.Eta() - p4_jet2.Eta();
         float dphi12 = p4_jet1.DeltaPhi(p4_jet2);
         float dr12   = p4_jet1.DeltaR(p4_jet2);
            
         if ( jet1->pt() > jetPtmin_ && jet2->pt() > jetPtmin_ && fabs(jet1->eta()) < jetEtamax_ && fabs(jet2->eta()) < jetEtamax_ )
         {
            pt_jet1_  -> Fill(jet1->pt());
            pt_jet2_  -> Fill(jet2->pt());
            eta_jet1_ -> Fill(jet1->eta());
            eta_jet2_ -> Fill(jet2->eta());
            phi_jet1_ -> Fill(jet1->phi());
            phi_jet2_ -> Fill(jet2->phi());
            deta_jet12_ -> Fill(deta12);
            dphi_jet12_ -> Fill(dphi12);
            dr_jet12_   -> Fill(dr12);
            eta_phi_jet1_ -> Fill(jet1->eta(),jet1->phi());
            eta_phi_jet2_ -> Fill(jet2->eta(),jet2->phi());
            
            discr_offline_btag_jet1_ -> Fill(btag1);
            discr_offline_btag_jet2_ -> Fill(btag2);
         } // offline jets kinematic selection
      } // at least two offline jets
   } // accept trigger
   
  // Filter out events if Trigger Filtering is requested
//  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

}
void MssmHbbMonitor::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
    bool changed(true);
    if (!hltConfig_.init(iRun, iSetup, processname_, changed))
    {
       LogDebug("MssmHbbMonitor") << "HLTConfigProvider failed to initialize.";
    }
}


// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MssmHbbMonitor);

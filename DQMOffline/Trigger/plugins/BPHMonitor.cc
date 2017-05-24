#include "DQMOffline/Trigger/plugins/BPHMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

BPHMonitor::BPHMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , muoToken_             ( mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )  
  , bsToken_              ( mayConsume<reco::BeamSpot>             (iConfig.getParameter<edm::InputTag>("beamSpot")))
  , trToken_              ( mayConsume<reco::TrackCollection>             (iConfig.getParameter<edm::InputTag>("tracks")))
  , phi_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("phiPSet")    ) )
  , pt_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("ptPSet")    ) )
  , eta_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("etaPSet")    ) )
  , d0_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("d0PSet")     ) )
  , z0_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("z0PSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , muoSelection_ref ( iConfig.getParameter<std::string>("muoSelection_ref") )
  , muoSelection_tag ( iConfig.getParameter<std::string>("muoSelection_tag") )
  , muoSelection_probe ( iConfig.getParameter<std::string>("muoSelection_probe") )
  , nmuons_     ( iConfig.getParameter<int>("nmuons" )     )
  , tnp_     ( iConfig.getParameter<int>("tnp" )     )
  , trOrMu_     ( iConfig.getParameter<int>("trOrMu" )     )
  , trSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , trSelection_ref ( iConfig.getParameter<std::string>("muoSelection_ref") )

{

  muPhi_.numerator   = nullptr;
  muPhi_.denominator = nullptr;
  muEta_.numerator   = nullptr;
  muEta_.denominator = nullptr;
  muPt_.numerator   = nullptr;
  muPt_.denominator = nullptr;
  mud0_.numerator   = nullptr;
  mud0_.denominator   = nullptr;
  muz0_.numerator   = nullptr;
  muz0_.denominator = nullptr;

}

BPHMonitor::~BPHMonitor()
{
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

MEbinning BPHMonitor::getHistoPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

MEbinning BPHMonitor::getHistoLSPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      0.,
      double(pset.getParameter<int32_t>("nbins"))
      };
}

void BPHMonitor::setMETitle(METME& me, std::string titleX, std::string titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, std::string& histname, std::string& histtitle, int& nbins, double& min, double& max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, std::string& histname, std::string& histtitle, std::vector<double> binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, double& ymin, double& ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, int& nbinsY, double& ymin, double& ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY)
{
  int nbinsX = binningX.size()-1;
  std::vector<float> fbinningX(binningX.begin(),binningX.end());
  float* arrX = &fbinningX[0];
  int nbinsY = binningY.size()-1;
  std::vector<float> fbinningY(binningY.begin(),binningY.end());
  float* arrY = &fbinningY[0];

  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, arrX, nbinsY, arrY);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, arrX, nbinsY, arrY);
}

void BPHMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle, istnp, trMu;

  if (tnp_) istnp = "Tag_and_Probe/"; else istnp = "";
  std::string currentFolder = folderName_ + istnp;
  ibooker.setCurrentFolder(currentFolder.c_str());
  if (trOrMu_) trMu = "tr";else trMu = "mu";
  histname = trMu+"Pt"; histtitle = trMu+"_P_{t}";
  bookME(ibooker,muPt_,histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
  setMETitle(muPt_,trMu+"_Pt[GeV]","events/1GeV");

  histname =trMu+"Phi"; histtitle =trMu+"Phi";
  bookME(ibooker,muPhi_,histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(muPhi_,trMu+"_#phi","events / 0.1 rad");

  histname =trMu+"Eta"; histtitle = trMu+"_Eta";
  bookME(ibooker,muEta_,histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
  setMETitle(muEta_,trMu+"_#eta","events/ ");

  histname =trMu+ "_d0"; histtitle =trMu+ "_d0";
  bookME(ibooker,mud0_,histname,histtitle, d0_binning_.nbins,d0_binning_.xmin, d0_binning_.xmax);
  setMETitle(mud0_,trMu+"_d0","events/bin ");

  histname = trMu+"_z0"; histtitle =trMu+"_z0";
  bookME(ibooker,muz0_,histname,histtitle, z0_binning_.nbins,z0_binning_.xmin, z0_binning_.xmax);
  setMETitle(muz0_,trMu+"_z0","events/bin ");

  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TLorentzVector.h"
void BPHMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken( bsToken_,  beamSpot);
  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken( muoToken_, muoHandle );
    

//  edm::Handle<reco::BeamSpot> const& beamSpot;
  // Filter out events if Trigger Filtering is requested
  if (tnp_>0) {
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;
  std::vector<reco::Muon> tagMuons;
  std::vector<reco::Muon> probeMuons;
  for ( auto const & m : *muoHandle ) {
  if ( muoSelection_tag( m ) ) tagMuons.push_back(m);
  }
  for (int i = 0; i<int(tagMuons.size());i++){
    for ( auto const & m : *muoHandle ) { 
      if ((tagMuons[i].pt() - m.pt())<=0.01)continue;//not the same  
      if ((tagMuons[i].p4()+m.p4()).M() >2.596&& (tagMuons[i].p4()+m.p4()).M() <3.596){//near to J/psi mass
      muPhi_.denominator->Fill(m.phi());
      muEta_.denominator->Fill(m.eta());
      muPt_.denominator ->Fill(m.pt());
      if (muoSelection_probe( m )){
        muPhi_.numerator->Fill(m.phi());
        muEta_.numerator->Fill(m.eta());
        muPt_.numerator ->Fill(m.pt());

        }

      }      

    }
      
  }
    

  }  
  else{
    if (trOrMu_){
      edm::Handle<reco::TrackCollection> trHandle;
      iEvent.getByToken( trToken_, trHandle );
  for (auto const & t : *trHandle) {
    if(!trSelection_ref(t))continue;
    muPhi_.denominator->Fill(t.phi());
    muEta_.denominator->Fill(t.eta());
    muPt_.denominator ->Fill(t.pt());
  }
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;
  for (auto const & t : *trHandle) {
    if(!trSelection_(t))continue;
    muPhi_.numerator->Fill(t.phi());
    muEta_.numerator->Fill(t.eta());
    muPt_.numerator ->Fill(t.pt());
  }

//  

  
    }

  else{
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

//  std::vector<reco::Muon> tagMuons;
//  for ( auto const & m : *muoHandle ) {
//    if ( muoSelection_tag( m ) ) tagMuons.push_back(m);
//  }


  for (auto const & m : *muoHandle) {
    if(!muoSelection_ref(m))continue;
    muPhi_.denominator->Fill(m.phi());
    muEta_.denominator->Fill(m.eta());
    muPt_.denominator ->Fill(m.pt());
    const reco::Track * track = 0;
    if (m.isTrackerMuon()) track = & * m.innerTrack();
    else if (m.isStandAloneMuon()) track = & * m.outerTrack();
    if (track) {
      double d0 = track->dxy(beamSpot->position());
      double z0 = track->dz(beamSpot->position());
      mud0_.denominator ->Fill(d0);
      muz0_.denominator ->Fill(z0);
    }
  } 
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;
  for (auto const & m : *muoHandle) {
    if(!muoSelection_(m))continue;
    muPhi_.numerator->Fill(m.phi());
    muEta_.numerator->Fill(m.eta());
    muPt_.numerator ->Fill(m.pt());
    const reco::Track * track = 0;
    if (m.isTrackerMuon()) track = & * m.innerTrack();
    else if (m.isStandAloneMuon()) track = & * m.outerTrack();
    if (track) {
      double d0 = track->dxy(beamSpot->position());
      double z0 = track->dz(beamSpot->position());
      mud0_.numerator ->Fill(d0);
      muz0_.numerator ->Fill(z0);
    }
  }
}	
}
}
void BPHMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void BPHMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins", 2500);
}

void BPHMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/BPH/" );
  desc.add<edm::InputTag>( "tracks",  edm::InputTag("generalTracks") );
  desc.add<edm::InputTag>( "offlinePVs",     edm::InputTag("offlinePrimaryVertices") );
  desc.add<edm::InputTag>( "beamSpot",edm::InputTag("offlineBeamSpot") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<std::string>("muoSelection", "abs(eta)<1.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0");
  desc.add<std::string>("muoSelection_ref", "isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0");
  desc.add<std::string>("muoSelection_tag",  "isGlobalMuon && isPFMuon && isTrackerMuon && abs(eta) < 2.4 && innerTrack.hitPattern.numberOfValidPixelHits > 0 && innerTrack.hitPattern.trackerLayersWithMeasurement > 5 && globalTrack.hitPattern.numberOfValidMuonHits > 0 && globalTrack.normalizedChi2 < 10");//tight selection for tag muon
  desc.add<std::string>("muoSelection_probe", "isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0");
  desc.add<int>("nmuons",     1);
  desc.add<int>( "tnp", 0 );
  desc.add<int>( "trOrMu", 0 );//if =0, track param monitoring

  edm::ParameterSetDescription genericTriggerEventPSet;
  genericTriggerEventPSet.add<bool>("andOr");
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi") );
  genericTriggerEventPSet.add<std::vector<int> >("dcsPartitions",{});
  genericTriggerEventPSet.add<bool>("andOrDcs", false);
  genericTriggerEventPSet.add<bool>("errorReplyDcs", true);
  genericTriggerEventPSet.add<std::string>("dbLabel","");
  genericTriggerEventPSet.add<bool>("andOrHlt", true);
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT") );
  genericTriggerEventPSet.add<std::vector<std::string> >("hltPaths",{});
  genericTriggerEventPSet.add<std::string>("hltDBKey","");
  genericTriggerEventPSet.add<bool>("errorReplyHlt",false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel",0);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription phiPSet;
  edm::ParameterSetDescription etaPSet;
  edm::ParameterSetDescription ptPSet;
  edm::ParameterSetDescription d0PSet;
  edm::ParameterSetDescription z0PSet;
  fillHistoPSetDescription(phiPSet);
  fillHistoPSetDescription(ptPSet);
  fillHistoPSetDescription(etaPSet);
  fillHistoPSetDescription(z0PSet);
  fillHistoPSetDescription(d0PSet);
  histoPSet.add<edm::ParameterSetDescription>("d0PSet", d0PSet);
  histoPSet.add<edm::ParameterSetDescription>("etaPSet", etaPSet);
  histoPSet.add<edm::ParameterSetDescription>("phiPSet", phiPSet);
  histoPSet.add<edm::ParameterSetDescription>("ptPSet", ptPSet);
  histoPSet.add<edm::ParameterSetDescription>("z0PSet", z0PSet);
  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("bphMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BPHMonitor);

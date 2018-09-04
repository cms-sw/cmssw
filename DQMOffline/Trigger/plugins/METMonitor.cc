#include "DQMOffline/Trigger/plugins/METMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "DataFormats/Math/interface/deltaPhi.h"

// -----------------------------
//  constructors and destructor
// -----------------------------

METMonitor::METMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , metInputTag_          ( iConfig.getParameter<edm::InputTag>    ("met")          )
  , jetInputTag_          ( iConfig.getParameter<edm::InputTag>    ("jets")         )
  , eleInputTag_          ( iConfig.getParameter<edm::InputTag>    ("electrons")    ) 
  , muoInputTag_          ( iConfig.getParameter<edm::InputTag>    ("muons")        ) 
  , vtxInputTag_          ( iConfig.getParameter<edm::InputTag>    ("vertices")     ) 
  , metToken_             ( consumes<reco::PFMETCollection>        ( metInputTag_ ) )
  , jetToken_             ( mayConsume<reco::PFJetCollection>      ( jetInputTag_ ) )
  , eleToken_             ( mayConsume<reco::GsfElectronCollection>( eleInputTag_ ) )
  , muoToken_             ( mayConsume<reco::MuonCollection>       ( muoInputTag_ ) )
  , vtxToken_             ( mayConsume<reco::VertexCollection>     ( vtxInputTag_ ) )
  , met_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("metBinning") )
  , met_binning_          ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("metPSet")    ) )
  , ls_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , njets_      ( iConfig.getParameter<unsigned>("njets" )      )
  , nelectrons_ ( iConfig.getParameter<unsigned>("nelectrons" ) )
  , nmuons_     ( iConfig.getParameter<unsigned>("nmuons" )     )
{
    // this vector has to be alligned to the the number of Tokens accessed by this module
    warningPrinted4token_.push_back(false); // PFMETCollection
    warningPrinted4token_.push_back(false); // JetCollection
    warningPrinted4token_.push_back(false); // GsfElectronCollection
    warningPrinted4token_.push_back(false); // MuonCollection
    warningPrinted4token_.push_back(false); // VertexCollection
}

METMonitor::~METMonitor() = default;

METMonitor::MEbinning METMonitor::getHistoPSet(const edm::ParameterSet& pset)
{
  return METMonitor::MEbinning{
    pset.getParameter<unsigned>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

METMonitor::MEbinning METMonitor::getHistoLSPSet(const edm::ParameterSet& pset)
{
  return METMonitor::MEbinning{
    pset.getParameter<unsigned>("nbins"),
      0.,
      double(pset.getParameter<unsigned>("nbins"))
      };
}

void METMonitor::setMETitle(METME& me, const std::string& titleX, const std::string& titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void METMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void METMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void METMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void METMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void METMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
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

void METMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder);

  histname = "deltaphi_metjet1"; histtitle = "DPHI_METJ1";
  bookME(ibooker,deltaphimetj1ME_,histname,histtitle,phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(deltaphimetj1ME_,"delta phi (met, j1)","events / 0.1 rad");

  histname = "deltaphi_jet1jet2"; histtitle = "DPHI_J1J2";
  bookME(ibooker,deltaphij1j2ME_,histname,histtitle,phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(deltaphij1j2ME_,"delta phi (j1, j2)","events / 0.1 rad");

  histname = "met"; histtitle = "PFMET";
  bookME(ibooker,metME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
  setMETitle(metME_,"PF MET [GeV]","events / [GeV]");

  histname = "met_variable"; histtitle = "PFMET";
  bookME(ibooker,metME_variableBinning_,histname,histtitle,met_variable_binning_);
  setMETitle(metME_variableBinning_,"PF MET [GeV]","events / [GeV]");

  histname = "metVsLS"; histtitle = "PFMET vs LS";
  bookME(ibooker,metVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
  setMETitle(metVsLS_,"LS","PF MET [GeV]");

  histname = "metPhi"; histtitle = "PFMET phi";
  bookME(ibooker,metPhiME_,histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(metPhiME_,"PF MET #phi","events / 0.1 rad");

  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
void METMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {
  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;
  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken( metToken_, metHandle );
  if ( !metHandle.isValid() ) {
    if (!warningPrinted4token_[0]) {
      edm::LogWarning("METMonitor") << "skipping events because the collection " << metInputTag_.label().c_str() << " is not available";
      warningPrinted4token_[0] = true;
    }
    return;
  }
  reco::PFMET pfmet = metHandle->front();
  if ( ! metSelection_( pfmet ) ) return;

  float met = pfmet.pt();
  float phi = pfmet.phi();

  std::vector<reco::PFJet> jets;
  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken( jetToken_, jetHandle );
  if ( jetHandle.isValid() ) {
    if ( jetHandle->size() < njets_ ) return;
    for ( auto const & j : *jetHandle ) {
      if ( jetSelection_(j) ) {
	jets.push_back(j);
      }
    }
  } else {
    if (!warningPrinted4token_[1]) {
      if ( jetInputTag_.label().empty() )
	edm::LogWarning("METMonitor") << "JetCollection not set";
      else
	edm::LogWarning("METMonitor") << "skipping events because the collection " << jetInputTag_.label().c_str() << " is not available";
      warningPrinted4token_[1] = true;
    }
    // if Handle is not valid, because the InputTag has been mis-configured, then skip the event
    if ( !jetInputTag_.label().empty() ) return;
  }
  float deltaPhi_met_j1= 10.0;
  float deltaPhi_j1_j2 = 10.0;

  if (!jets.empty()   ) deltaPhi_met_j1 = fabs( deltaPhi( pfmet.phi(),  jets[0].phi() ));
  if (jets.size() >= 2) deltaPhi_j1_j2  = fabs( deltaPhi( jets[0].phi(),  jets[1].phi() ));

  std::vector<reco::GsfElectron> electrons;
  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken( eleToken_, eleHandle );
  if ( eleHandle.isValid() ) {
    if ( eleHandle->size() < nelectrons_ ) return;
    for ( auto const & e : *eleHandle ) {
      if ( eleSelection_( e ) ) electrons.push_back(e);
    }
    if (electrons.size() < nelectrons_ ) return;
  } else {
    if (!warningPrinted4token_[2]) {
      warningPrinted4token_[2] = true;
      if ( eleInputTag_.label().empty() )
	edm::LogWarning("METMonitor") << "GsfElectronCollection not set";
      else	
	edm::LogWarning("METMonitor") << "skipping events because the collection " << eleInputTag_.label().c_str() << " is not available";
    }   
    if ( !eleInputTag_.label().empty() ) return;
  }

  reco::Vertex vtx;
  edm::Handle<reco::VertexCollection> vtxHandle;
  iEvent.getByToken(vtxToken_, vtxHandle);
  if ( vtxHandle.isValid() ) {
    for (auto const & v : *vtxHandle) {
      bool isFake =  v.isFake() ;
    
      if (!isFake) {
	vtx = v;
	break;
      }
    }
  } else {
    if (!warningPrinted4token_[3]) {
      warningPrinted4token_[3] = true;
      if ( vtxInputTag_.label().empty() )
	edm::LogWarning("METMonitor") << "VertexCollection is not set";
      else
	edm::LogWarning("METMonitor") << "skipping events because the collection " << vtxInputTag_.label().c_str() << " is not available";
    }
    if ( !vtxInputTag_.label().empty() ) return;
  }


  std::vector<reco::Muon> muons;
  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken( muoToken_, muoHandle );
  if ( muoHandle.isValid() ) {
    if ( muoHandle->size() < nmuons_ ) return;
    for ( auto const & m : *muoHandle ) {
      bool pass = m.isGlobalMuon() && m.isPFMuon() && m.globalTrack()->normalizedChi2() < 10. && m.globalTrack()->hitPattern().numberOfValidMuonHits() > 0 && m.numberOfMatchedStations() > 1 && fabs(m.muonBestTrack()->dxy(vtx.position())) < 0.2 && fabs(m.muonBestTrack()->dz(vtx.position())) < 0.5 && m.innerTrack()->hitPattern().numberOfValidPixelHits() > 0 && m.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5;
      if ( muoSelection_( m ) && pass ) muons.push_back(m);
    }
    if ( muons.size() < nmuons_ ) return;
  } else {
    if (!warningPrinted4token_[4]) {
      warningPrinted4token_[4] = true;
      if ( muoInputTag_.label().empty() )
	edm::LogWarning("METMonitor") << "MuonCollection not set";
      else
	edm::LogWarning("METMonitor") << "skipping events because the collection " << muoInputTag_.label().c_str() << " is not available";
    }
    if ( !muoInputTag_.label().empty() ) return;
  }

  // filling histograms (denominator)  
  metME_.denominator -> Fill(met);
  metME_variableBinning_.denominator -> Fill(met);
  metPhiME_.denominator -> Fill(phi);
  deltaphimetj1ME_.denominator -> Fill(deltaPhi_met_j1);
  deltaphij1j2ME_.denominator -> Fill(deltaPhi_j1_j2);

  int ls = iEvent.id().luminosityBlock();
  metVsLS_.denominator -> Fill(ls, met);
  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  // filling histograms (num_genTriggerEventFlag_)  
  metME_.numerator -> Fill(met);
  metME_variableBinning_.numerator -> Fill(met);
  metPhiME_.numerator -> Fill(phi);
  metVsLS_.numerator -> Fill(ls, met);
  deltaphimetj1ME_.numerator  -> Fill(deltaPhi_met_j1); 
  deltaphij1j2ME_.numerator  -> Fill(deltaPhi_j1_j2); 
}

void METMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void METMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins", 2500 );
  pset.add<double>         ( "xmin",     0.);
  pset.add<double>         ( "xmax",  2500.);
}

void METMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/MET" );

  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<edm::InputTag>( "vertices",edm::InputTag("offlinePrimaryVertices") );
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<unsigned>("njets",      0);
  desc.add<unsigned>("nelectrons", 0);
  desc.add<unsigned>("nmuons",     0);

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
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel",1);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription metPSet;
  fillHistoPSetDescription(metPSet);
  histoPSet.add<edm::ParameterSetDescription>("metPSet", metPSet);
  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.};
  histoPSet.add<std::vector<double> >("metBinning", bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("metMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(METMonitor);

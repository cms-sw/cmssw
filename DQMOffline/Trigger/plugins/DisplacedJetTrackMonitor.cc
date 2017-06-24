#include "DQMOffline/Trigger/plugins/DisplacedJetTrackMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

DisplacedJetTrackMonitor::DisplacedJetTrackMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , calojetToken_         ( mayConsume<reco::CaloJetCollection>    (iConfig.getParameter<edm::InputTag>("calojets")  ) )
  , tracksToken_          ( mayConsume<reco::TrackCollection>      (iConfig.getParameter<edm::InputTag>("tracks")    ) ) 
  , eleToken_             ( mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons") ) )   
  , muoToken_             ( mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )   
  , ntrack_binning_       ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("ntrackPSet") ) )
  , ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))

  , calojetSelection_ ( iConfig.getParameter<std::string>("calojetSelection"))
  , trackSelection_ ( iConfig.getParameter<std::string>("trackSelection"))
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )

  , ncalojets_  ( iConfig.getParameter<unsigned int>("ncalojets")   )
  , nelectrons_ ( iConfig.getParameter<unsigned int>("nelectrons" ) )
  , nmuons_     ( iConfig.getParameter<unsigned int>("nmuons" )     )
{

  nprompttrksjet1ME_.numerator = nullptr;
  nprompttrksjet1ME_.denominator = nullptr;
  nprompttrksjet2ME_.numerator = nullptr;
  nprompttrksjet2ME_.denominator = nullptr;

  nprompttrksjet1VsLS_.numerator = nullptr;
  nprompttrksjet1VsLS_.denominator = nullptr;
  nprompttrksjet2VsLS_.numerator = nullptr;
  nprompttrksjet2VsLS_.denominator = nullptr;
  ndisplacedtrksjet1ME_.numerator = nullptr;
  ndisplacedtrksjet1ME_.denominator = nullptr;

  ndisplacedtrksjet2ME_.numerator = nullptr;
  ndisplacedtrksjet2ME_.denominator = nullptr;
  ndisplacedtrksjet1VsLS_.numerator = nullptr;
  ndisplacedtrksjet1VsLS_.denominator = nullptr;
  ndisplacedtrksjet2VsLS_.numerator = nullptr;              
  ndisplacedtrksjet2VsLS_.denominator = nullptr; 
 
  
}

DisplacedJetTrackMonitor::~DisplacedJetTrackMonitor()
{
}

MEbinning DisplacedJetTrackMonitor::getHistoPSet(edm::ParameterSet const& pset)
{
  return MEbinning{
    pset.getParameter<unsigned int>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

MEbinning DisplacedJetTrackMonitor::getHistoLSPSet(edm::ParameterSet const& pset)
{
  return MEbinning{
    pset.getParameter<unsigned int>("nbins"),
      0.,
      double(pset.getParameter<unsigned int>("nbins"))
      };
}

void DisplacedJetTrackMonitor::setMETitle(DJME& me, const std::string& titleX, const std::string& titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void DisplacedJetTrackMonitor::bookME(DQMStore::IBooker &ibooker, DJME& me, const std::string& histname, const std::string& histtitle, unsigned int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void DisplacedJetTrackMonitor::bookME(DQMStore::IBooker &ibooker, DJME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void DisplacedJetTrackMonitor::bookME(DQMStore::IBooker &ibooker, DJME& me, const std::string& histname, const std::string& histtitle, unsigned int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void DisplacedJetTrackMonitor::bookME(DQMStore::IBooker &ibooker, DJME& me, const std::string& histname, const std::string& histtitle, unsigned int nbinsX, double xmin, double xmax, unsigned int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void DisplacedJetTrackMonitor::bookME(DQMStore::IBooker &ibooker, DJME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
{
  unsigned int nbinsX = binningX.size()-1;
  std::vector<float> fbinningX(binningX.begin(),binningX.end());
  float* arrX = &fbinningX[0];
  unsigned int nbinsY = binningY.size()-1;
  std::vector<float> fbinningY(binningY.begin(),binningY.end());
  float* arrY = &fbinningY[0];

  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, arrX, nbinsY, arrY);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, arrX, nbinsY, arrY);
}

void DisplacedJetTrackMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());

  histname = "nprompttrksjet1"; histtitle = "nprompttrksjet1";
  bookME(ibooker, nprompttrksjet1ME_, histname, histtitle, ntrack_binning_.nbins, ntrack_binning_.xmin, ntrack_binning_.xmax);
  setMETitle(nprompttrksjet1ME_, "Prompt Tracks in Jet 1", "events");

  histname = "nprompttrksjet2"; histtitle = "nprompttrksjet2";
  bookME(ibooker, nprompttrksjet2ME_, histname, histtitle, ntrack_binning_.nbins, ntrack_binning_.xmin, ntrack_binning_.xmax);
  setMETitle(nprompttrksjet2ME_, "Prompt Tracks in Jet 2", "events");
 
  histname = "nprompttrksjet1VsLS"; histtitle = "nprompttrksjet1 vs LS";
  bookME(ibooker, nprompttrksjet1VsLS_, histname, histtitle, ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, ntrack_binning_.xmin, ntrack_binning_.xmax);
  setMETitle(nprompttrksjet1VsLS_, "LS", "Prompt Tracks in Jet 1");
 
  histname = "nprompttrksjet2VsLS"; histtitle = "nprompttrksjet2 vs LS";
  bookME(ibooker, nprompttrksjet2VsLS_, histname, histtitle, ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, ntrack_binning_.xmin, ntrack_binning_.xmax);
  setMETitle(nprompttrksjet2VsLS_, "LS", "Prompt Tracks in Jet 2");

  histname = "ndisplacedtrksjet1"; histtitle = "ndisplacedtrksjet1";
  bookME(ibooker, ndisplacedtrksjet1ME_, histname, histtitle, ntrack_binning_.nbins, ntrack_binning_.xmin, ntrack_binning_.xmax);
  setMETitle(ndisplacedtrksjet1ME_, "Dispalced Tracks in Jet 1", "events");
  
  histname = "ndisplacedtrksjet2"; histtitle = "ndisplacedtrksjet2";
  bookME(ibooker, ndisplacedtrksjet2ME_, histname, histtitle, ntrack_binning_.nbins, ntrack_binning_.xmin, ntrack_binning_.xmax);
  setMETitle(ndisplacedtrksjet2ME_, "Dispalced Tracks in Jet 2", "events");

  histname = "ndisplacedtrksjet1VsLS"; histtitle = "ndisplacedtrksjet1 vs LS";
  bookME(ibooker, ndisplacedtrksjet1VsLS_, histname, histtitle, ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, ntrack_binning_.xmin, ntrack_binning_.xmax);
  setMETitle(ndisplacedtrksjet1VsLS_, "LS", "Displaced Tracks in Jet 1");

  histname = "ndisplacedtrksjet2VsLS"; histtitle = "ndisplacedtrksjet2 vs LS";
  bookME(ibooker, ndisplacedtrksjet2VsLS_, histname, histtitle, ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, ntrack_binning_.xmin, ntrack_binning_.xmax);
  setMETitle(ndisplacedtrksjet2VsLS_, "LS", "Displaced Tracks in Jet 2");
 
  
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
void DisplacedJetTrackMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  edm::Handle<reco::TrackCollection>  trackHandle;
  iEvent.getByToken(tracksToken_, trackHandle);
  std::vector<reco::Track> tracks;
  
  for (auto const & tk : *trackHandle){
      if (trackSelection_ (tk) ) tracks.push_back(tk);
  } 

  edm::Handle<reco::CaloJetCollection> calojetHandle;
  iEvent.getByToken( calojetToken_, calojetHandle);
  std::vector<reco::CaloJet> calojets;
  std::vector<int> nPromptTracks;
  std::vector<int> nDisplacedTracks;

  if ( calojetHandle->size()< ncalojets_) return;
  int idx = 0;
  int min_nptrk1 = 1000;
  int min_nptrk2 = 1000;
  int max_ndisptrk1 = -1;
  int max_ndisptrk2 = -1;
  for (auto const & cj : *calojetHandle) {
    if (calojetSelection_( cj ) ) {
        calojets.push_back(cj);
        int nptrks = 0;
        int ndisptrks = 0;
        for (size_t itrk = 0; itrk<tracks.size(); itrk++){
            float trk_eta = tracks.at(itrk).eta();
            float trk_phi = tracks.at(itrk).phi();
            if (deltaR(trk_eta, trk_phi, cj.eta(), cj.phi())<0.4){
                if(tracks.at(itrk).dxy()<0.5) nptrks++;
                else if (tracks.at(itrk).dxy()/tracks.at(itrk).d0Error()>2) ndisptrks++;
            }
        }
        if(nptrks < min_nptrk1){
            min_nptrk1 = nptrks;
        }
        else if(nptrks<min_nptrk2){
            min_nptrk2 = nptrks;
        }
        nPromptTracks.push_back(nptrks);
        nDisplacedTracks.push_back(ndisptrks);
        idx++;
    }

  }
  if (calojets.size() < ncalojets_) return;

  for (size_t ijet = 0; ijet < nPromptTracks.size(); ijet++){
      if (nPromptTracks.at(ijet)<=2){
          if(nDisplacedTracks.at(ijet) > max_ndisptrk1) max_ndisptrk1 = nDisplacedTracks.at(ijet);
          else if ( nDisplacedTracks.at(ijet) >= max_ndisptrk2) max_ndisptrk2 = nDisplacedTracks.at(ijet);

      }

  }

    
  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken( eleToken_, eleHandle );
  std::vector<reco::GsfElectron> electrons;
  if ( eleHandle->size() < nelectrons_ ) return;
  for ( auto const & e : *eleHandle ) {
    if ( eleSelection_( e ) ) electrons.push_back(e);
  }
  if ( electrons.size() < nelectrons_ ) return;
  
  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken( muoToken_, muoHandle );
  if ( muoHandle->size() < nmuons_ ) return;
  std::vector<reco::Muon> muons;
  for ( auto const & m : *muoHandle ) {
    if ( muoSelection_( m ) ) muons.push_back(m);
  }
  if ( muons.size() < nmuons_ ) return;

  nprompttrksjet1ME_.denominator->Fill(min_nptrk1);
  nprompttrksjet2ME_.denominator->Fill(min_nptrk2);

  ndisplacedtrksjet1ME_.denominator->Fill(max_ndisptrk1);
  ndisplacedtrksjet2ME_.denominator->Fill(max_ndisptrk2);
  
    

  int ls = iEvent.id().luminosityBlock();
  nprompttrksjet1VsLS_.denominator->Fill(ls, min_nptrk1);
  nprompttrksjet2VsLS_.denominator->Fill(ls, min_nptrk2);
 
  ndisplacedtrksjet1VsLS_.denominator->Fill(ls, max_ndisptrk1); 
  ndisplacedtrksjet2VsLS_.denominator->Fill(ls, max_ndisptrk2); 
 
  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  nprompttrksjet1ME_.numerator->Fill(min_nptrk1);
  nprompttrksjet2ME_.numerator->Fill(min_nptrk2);

  ndisplacedtrksjet1ME_.numerator->Fill(max_ndisptrk1);
  ndisplacedtrksjet2ME_.numerator->Fill(max_ndisptrk2);
  nprompttrksjet1VsLS_.numerator->Fill(ls, min_nptrk1);
  nprompttrksjet2VsLS_.numerator->Fill(ls, min_nptrk2);

  ndisplacedtrksjet1VsLS_.numerator->Fill(ls, max_ndisptrk1); 
  ndisplacedtrksjet2VsLS_.numerator->Fill(ls, max_ndisptrk2); 
}

void DisplacedJetTrackMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void DisplacedJetTrackMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins", 2500);
}

void DisplacedJetTrackMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/DisplacedJet" );

  desc.add<edm::InputTag>( "calojets",  edm::InputTag("ak4CaloJets") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<edm::InputTag>( "tracks",   edm::InputTag("generalTracks") );
  desc.add<std::string>("calojetSelection", "pt > 0");
  desc.add<std::string>("trackSelection", "pt > 0");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<unsigned int>("ncalojets",  0);    
  desc.add<unsigned int>("nelectrons", 0);
  desc.add<unsigned int>("nmuons",     0);

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
  edm::ParameterSetDescription ntrackPSet;
  fillHistoPSetDescription(ntrackPSet);
  histoPSet.add<edm::ParameterSetDescription>("ntrackPSet", ntrackPSet);
  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("DisplacedJetTrackMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DisplacedJetTrackMonitor);

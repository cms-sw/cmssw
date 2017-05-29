#include "DQMOffline/Trigger/plugins/PhotonMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

double MAX_PHI1 = 3.2;
int N_PHI1 = 64;
const MEbinning phi_binning_1{
  N_PHI1, -MAX_PHI1, MAX_PHI1
    };

// -----------------------------
//  constructors and destructor
// -----------------------------

PhotonMonitor::PhotonMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , metToken_             ( consumes<reco::PFMETCollection>      (iConfig.getParameter<edm::InputTag>("met")       ) )   
  , jetToken_             ( mayConsume<reco::PFJetCollection>      (iConfig.getParameter<edm::InputTag>("jets")      ) )   
  , eleToken_             ( mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons") ) )   
  , photonToken_             ( mayConsume<reco::PhotonCollection>      (iConfig.getParameter<edm::InputTag>("photons")      ) )   
  , photon_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("photonBinning") )
  , diphoton_mass_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("massBinning") )
  , photon_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("photonPSet")    ) )
  , ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , photonSelection_ ( iConfig.getParameter<std::string>("photonSelection") )
  , njets_      ( iConfig.getParameter<int>("njets" )      )
  , nphotons_      ( iConfig.getParameter<int>("nphotons" )      )
  , nelectrons_ ( iConfig.getParameter<int>("nelectrons" ) )
{

  photonME_.numerator   = nullptr;
  photonME_.denominator = nullptr;
  photonME_variableBinning_.numerator   = nullptr;
  photonME_variableBinning_.denominator = nullptr;
  diphotonMassME_.numerator   = nullptr;
  diphotonMassME_.denominator = nullptr;
  photonVsLS_.numerator   = nullptr;
  photonVsLS_.denominator = nullptr;

  
}
PhotonMonitor::~PhotonMonitor()
{
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

MEbinning PhotonMonitor::getHistoPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

MEbinning PhotonMonitor::getHistoLSPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      0.,
      double(pset.getParameter<int32_t>("nbins"))
      };
}

void PhotonMonitor::setTitle(PhotonME& me, std::string titleX, std::string titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void PhotonMonitor::bookME(DQMStore::IBooker &ibooker, PhotonME& me, const std::string& histname, const std::string& histtitle, int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void PhotonMonitor::bookME(DQMStore::IBooker &ibooker, PhotonME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void PhotonMonitor::bookME(DQMStore::IBooker &ibooker, PhotonME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void PhotonMonitor::bookME(DQMStore::IBooker &ibooker, PhotonME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void PhotonMonitor::bookME(DQMStore::IBooker &ibooker, PhotonME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
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

void PhotonMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				   edm::Run const        & iRun,
				   edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());

  histname = "photon_pt"; histtitle = "photon PT";
  bookME(ibooker,photonME_,histname,histtitle,photon_binning_.nbins,photon_binning_.xmin, photon_binning_.xmax);
  setTitle(photonME_,"Photon pT [GeV]","events / [GeV]");

  histname = "photon_pt_variable"; histtitle = "photon PT";
  bookME(ibooker,photonME_variableBinning_,histname,histtitle,photon_variable_binning_);
  setTitle(photonME_variableBinning_,"Photon pT [GeV]","events / [GeV]");

  histname = "photon_eta"; histtitle = "Photon eta";
  bookME(ibooker,photonEtaME_,histname,histtitle, phi_binning_1.nbins, phi_binning_1.xmin, phi_binning_1.xmax);
  setTitle(photonEtaME_,"Photon #eta","events / 0.1");

  histname = "photonVsLS"; histtitle = "photon pt vs LS";
  bookME(ibooker,photonVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,photon_binning_.xmin, photon_binning_.xmax);
  setTitle(photonVsLS_,"LS","Photon pT [GeV]");

  //for diphotons

  histname = "diphoton_mass"; histtitle = "Diphoton mass";
  bookME(ibooker,diphotonMassME_,histname,histtitle, diphoton_mass_binning_);
  setTitle(diphotonMassME_,"Diphoton mass","events / 0.1");

  histname = "subphoton_pt"; histtitle = "subphoton PT";
  bookME(ibooker,subphotonME_,histname,histtitle,photon_binning_.nbins,photon_binning_.xmin, photon_binning_.xmax);
  setTitle(subphotonME_,"subPhoton pT [GeV]","events / [GeV]");

  histname = "subphoton_eta"; histtitle = "subPhoton eta";
  bookME(ibooker,subphotonEtaME_,histname,histtitle, phi_binning_1.nbins, phi_binning_1.xmin, phi_binning_1.xmax);
  setTitle(subphotonEtaME_,"subPhoton #eta","events / 0.1");


  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
void PhotonMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken( metToken_, metHandle );
  reco::PFMET pfmet = metHandle->front();
  if ( ! metSelection_( pfmet ) ) return;
  
  //  float met = pfmet.pt();
  //  float phi = pfmet.phi();

  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken( jetToken_, jetHandle );
  std::vector<reco::PFJet> jets;
  if ( int(jetHandle->size()) < njets_ ) return;
  for ( auto const & j : *jetHandle ) {
    if ( jetSelection_( j ) ) jets.push_back(j);
  }
  if ( int(jets.size()) < njets_ ) return;
  
  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken( eleToken_, eleHandle );
  std::vector<reco::GsfElectron> electrons;
  if ( int(eleHandle->size()) < nelectrons_ ) return;
  for ( auto const & e : *eleHandle ) {
    if ( eleSelection_( e ) ) electrons.push_back(e);
  }
  if ( int(electrons.size()) < nelectrons_ ) return;

  edm::Handle<reco::PhotonCollection> photonHandle;
  iEvent.getByToken( photonToken_, photonHandle );
  std::vector<reco::Photon> photons;
  photons.clear();
  if ( int(photonHandle->size()) < nphotons_ ) return;
  for ( auto const & p : *photonHandle ) {
    if ( photonSelection_( p ) ) photons.push_back(p);
  }
  if ( int(photons.size()) < nphotons_ ) return;
  
  // filling histograms (denominator)  

  photonME_.denominator -> Fill(photons[0].pt());
  photonEtaME_.denominator -> Fill(photons[0].eta());
  photonME_variableBinning_.denominator -> Fill(photons[0].pt());

  if(nphotons_>1) 
    //filling diphoton histograms
    {
      subphotonME_.denominator -> Fill(photons[1].pt());
      subphotonEtaME_.denominator -> Fill(photons[1].eta());
      diphotonMassME_.denominator -> Fill(sqrt(2*photons[0].pt()*photons[1].pt()*(cosh(photons[0].eta()-photons[1].eta())-cos(photons[0].phi()-photons[1].phi()))));
    }
  
  int ls = iEvent.id().luminosityBlock();
  photonVsLS_.denominator -> Fill(ls, photons[0].pt());
  
  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  // filling histograms (num_genTriggerEventFlag_)  
  photonME_.numerator -> Fill(photons[0].pt());
  photonEtaME_.numerator -> Fill(photons[0].eta());
  photonME_variableBinning_.numerator -> Fill(photons[0].pt());
  if(nphotons_>1) 
    //filling diphoton histograms
    {
      subphotonME_.numerator -> Fill(photons[1].pt());
      subphotonEtaME_.numerator -> Fill(photons[1].eta());
      diphotonMassME_.numerator -> Fill(sqrt(2*photons[0].pt()*photons[1].pt()*(cosh(photons[0].eta()-photons[1].eta())-cos(photons[0].phi()-photons[1].phi()))));
    }

  
}

void PhotonMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void PhotonMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins", 2500);
}

void PhotonMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/Photon" );
  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "photons",edm::InputTag("gedPhotons") );
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("photonSelection", "pt > 145 && eta<1.4442 && hadTowOverEm<0.0597 && full5x5_sigmaIetaIeta()<0.01031 && chargedHadronIso<1.295");
  desc.add<int>("njets",      0);
  desc.add<int>("nelectrons", 0);
  desc.add<int>("nphotons",     0);

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
  
  histoPSet.add<edm::ParameterSetDescription>("photonPSet", metPSet);
  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.};
  histoPSet.add<std::vector<double> >("photonBinning", bins);
  std::vector<double> massbins = {90.,91.,92.,93.,94.,95.,96.,97.,98.,99.,100.,101.,102.,103.,104.,105.,106.,107.,108.,109.,110.,115.,120.,130.,150.,200.};
  histoPSet.add<std::vector<double> >("massBinning", massbins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("photonMonitoring", desc);
}


// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PhotonMonitor);

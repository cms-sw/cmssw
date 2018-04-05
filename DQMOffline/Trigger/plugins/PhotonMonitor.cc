#include "DQMOffline/Trigger/plugins/PhotonMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

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
  , ls_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , photonSelection_ ( iConfig.getParameter<std::string>("photonSelection") )
  , njets_      ( iConfig.getParameter<unsigned int>("njets" )      )
  , nphotons_      ( iConfig.getParameter<unsigned int>("nphotons" )      )
  , nelectrons_ ( iConfig.getParameter<unsigned int>("nelectrons" ) )
{

  photonME_.numerator   = nullptr;
  photonME_.denominator = nullptr;
  photonME_variableBinning_.numerator   = nullptr;
  photonME_variableBinning_.denominator = nullptr;
  photonVsLS_.numerator   = nullptr;
  photonVsLS_.denominator = nullptr;
  photonEtaME_.numerator   = nullptr;
  photonEtaME_.denominator = nullptr;
  photonPhiME_.numerator   = nullptr;
  photonPhiME_.denominator = nullptr;
  photonEtaPhiME_.numerator   = nullptr;       
  photonEtaPhiME_.denominator = nullptr;
  photonr9ME_.numerator   = nullptr;       
  photonr9ME_.denominator = nullptr;
  photonHoverEME_.numerator   = nullptr;       
  photonHoverEME_.denominator = nullptr;

  diphotonMassME_.numerator   = nullptr;
  diphotonMassME_.denominator = nullptr;

  subphotonME_.numerator   = nullptr;
  subphotonME_.denominator = nullptr;
  subphotonME_variableBinning_.numerator   = nullptr;
  subphotonME_variableBinning_.denominator = nullptr;
  subphotonEtaME_.numerator   = nullptr;
  subphotonEtaME_.denominator = nullptr;
  subphotonPhiME_.numerator   = nullptr;
  subphotonPhiME_.denominator = nullptr;
  subphotonEtaPhiME_.numerator   = nullptr;       
  subphotonEtaPhiME_.denominator = nullptr;
  subphotonr9ME_.numerator   = nullptr;       
  subphotonr9ME_.denominator = nullptr;
  subphotonHoverEME_.numerator   = nullptr;       
  subphotonHoverEME_.denominator = nullptr;
  
}
PhotonMonitor::~PhotonMonitor() = default;

MEbinning PhotonMonitor::getHistoPSet(edm::ParameterSet const& pset)
{
  return MEbinning{
    pset.getParameter<unsigned int>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

MEbinning PhotonMonitor::getHistoLSPSet(edm::ParameterSet const& pset)
{
  return MEbinning{
    pset.getParameter<unsigned int>("nbins"),
      0.,
      double(pset.getParameter<unsigned int>("nbins"))
      };
}

void PhotonMonitor::setTitle(PhotonME& me, const std::string& titleX, const std::string& titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void PhotonMonitor::bookME(DQMStore::IBooker &ibooker, PhotonME& me, const std::string& histname, const std::string& histtitle, unsigned int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void PhotonMonitor::bookME(DQMStore::IBooker &ibooker, PhotonME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  unsigned int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  //  float* arr = &fbinning[0];
  float* arr = fbinning.data();
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void PhotonMonitor::bookME(DQMStore::IBooker &ibooker, PhotonME& me, const std::string& histname, const std::string& histtitle, unsigned int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void PhotonMonitor::bookME(DQMStore::IBooker &ibooker, PhotonME& me, const std::string& histname, const std::string& histtitle, unsigned int nbinsX, double xmin, double xmax, unsigned int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void PhotonMonitor::bookME(DQMStore::IBooker &ibooker, PhotonME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
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

void PhotonMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder);

  histname = "photon_pt"; histtitle = "photon PT";
  bookME(ibooker,photonME_,histname,histtitle,photon_binning_.nbins,photon_binning_.xmin, photon_binning_.xmax);
  setTitle(photonME_,"Photon pT [GeV]","events / [GeV]");

  histname = "photon_pt_variable"; histtitle = "photon PT";
  bookME(ibooker,photonME_variableBinning_,histname,histtitle,photon_variable_binning_);
  setTitle(photonME_variableBinning_,"Photon pT [GeV]","events / [GeV]");

  histname = "photonVsLS"; histtitle = "photon pt vs LS";
  bookME(ibooker,photonVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,photon_binning_.xmin, photon_binning_.xmax);
  setTitle(photonVsLS_,"LS","Photon pT [GeV]");
  
  histname = "photon_phi"; histtitle = "Photon phi";
  bookME(ibooker,photonPhiME_,histname,histtitle, phi_binning_1.nbins, phi_binning_1.xmin, phi_binning_1.xmax);
  setTitle(photonPhiME_,"Photon #phi","events / 0.1 rad");


  histname = "photon_eta"; histtitle = "Photon eta";
  bookME(ibooker,photonEtaME_,histname,histtitle, eta_binning_.nbins, eta_binning_.xmin,eta_binning_.xmax);
  setTitle(photonEtaME_,"Photon #eta","events");

  histname = "photon_r9"; histtitle = "Photon r9";
  bookME(ibooker,photonr9ME_,histname,histtitle, r9_binning_.nbins, r9_binning_.xmin, r9_binning_.xmax);
  setTitle(photonr9ME_,"Photon r9","events");


  histname = "photon_hoE"; histtitle = "Photon hoverE";
  bookME(ibooker,photonHoverEME_,histname,histtitle, hoe_binning_.nbins, hoe_binning_.xmin, hoe_binning_.xmax);
  setTitle(photonHoverEME_,"Photon hoE","events");

  histname = "photon_etaphi"; histtitle = "Photon eta-phi"; 
  bookME(ibooker,photonEtaPhiME_,histname,histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax,phi_binning_1.nbins, phi_binning_1.xmin, phi_binning_1.xmax);
  setTitle(photonEtaPhiME_,"#eta","#phi"); 

  //for diphotons
  if(nphotons_>1)
    {
      histname = "diphoton_mass"; histtitle = "Diphoton mass";
      bookME(ibooker,diphotonMassME_,histname,histtitle, diphoton_mass_binning_);
      setTitle(diphotonMassME_,"Diphoton mass","events / 0.1");
      
      histname = "subphoton_pt"; histtitle = "subphoton PT";
      bookME(ibooker,subphotonME_,histname,histtitle,photon_binning_.nbins,photon_binning_.xmin, photon_binning_.xmax);
      setTitle(subphotonME_,"subPhoton pT [GeV]","events / [GeV]");
      
      histname = "subphoton_eta"; histtitle = "subPhoton eta";
      bookME(ibooker,subphotonEtaME_,histname,histtitle, eta_binning_.nbins, eta_binning_.xmin,eta_binning_.xmax);
      setTitle(subphotonEtaME_,"subPhoton #eta","events / 0.1");
      
      histname = "subphoton_phi"; histtitle = "subPhoton phi";
      bookME(ibooker,subphotonPhiME_,histname,histtitle, phi_binning_1.nbins, phi_binning_1.xmin, phi_binning_1.xmax);
      setTitle(subphotonPhiME_,"subPhoton #phi","events / 0.1 rad");
      
      histname = "subphoton_r9"; histtitle = "subPhoton r9";
      bookME(ibooker,subphotonr9ME_,histname,histtitle, r9_binning_.nbins, r9_binning_.xmin, r9_binning_.xmax);
      setTitle(subphotonr9ME_,"subPhoton r9","events");
      
      histname = "subphoton_hoE"; histtitle = "subPhoton hoverE";
      bookME(ibooker,subphotonHoverEME_,histname,histtitle, hoe_binning_.nbins, hoe_binning_.xmin, hoe_binning_.xmax);
      setTitle(subphotonHoverEME_,"subPhoton hoE","events");
      
      histname = "subphoton_etaphi"; histtitle = "subPhoton eta-phi"; 
      bookME(ibooker,subphotonEtaPhiME_,histname,histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax,phi_binning_1.nbins, phi_binning_1.xmin, phi_binning_1.xmax);
      setTitle(subphotonEtaPhiME_,"#eta","#phi"); 
    }
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
  
  //float met = pfmet.pt();
  //  float phi = pfmet.phi();

  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken( jetToken_, jetHandle );
  std::vector<reco::PFJet> jets;
  jets.clear();
  if ( jetHandle->size() < njets_ ) return;
  for ( auto const & j : *jetHandle ) {
    if ( jetSelection_( j ) ) jets.push_back(j);
  }
  if ( jets.size() < njets_ ) return;
  
  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken( eleToken_, eleHandle );
  std::vector<reco::GsfElectron> electrons;
  if ( eleHandle->size() < nelectrons_ ) return;
  for ( auto const & e : *eleHandle ) {
    if ( eleSelection_( e ) ) electrons.push_back(e);
  }
  if ( electrons.size() < nelectrons_ ) return;

  edm::Handle<reco::PhotonCollection> photonHandle;
  iEvent.getByToken( photonToken_, photonHandle );
  std::vector<reco::Photon> photons;
  photons.clear();
  
  if ( photonHandle->size() < nphotons_ ) return;
  for ( auto const & p : *photonHandle ) {
    if ( photonSelection_( p ) ) photons.push_back(p);
  }
  if ( photons.size() < nphotons_ ) return;
  

  // filling histograms (denominator)  
      int ls = iEvent.id().luminosityBlock();
      if(!(photons.empty()))
	
	{
	  photonME_.denominator -> Fill(photons[0].pt());
	  photonME_variableBinning_.denominator -> Fill(photons[0].pt());
	  photonPhiME_.denominator->Fill(photons[0].phi());
	  photonEtaME_.denominator->Fill(photons[0].eta());
	  photonVsLS_.denominator -> Fill(ls, photons[0].pt());
	  photonEtaPhiME_.denominator -> Fill(photons[0].eta(), photons[0].phi()); 
	  photonr9ME_.denominator->Fill(photons[0].r9());
	  photonHoverEME_.denominator->Fill(photons[0].hadTowOverEm());
	}
  
  if(nphotons_>1) 
    //filling diphoton histograms
    {
      subphotonME_.denominator -> Fill(photons[1].pt());
      subphotonEtaME_.denominator -> Fill(photons[1].eta());
      subphotonPhiME_.denominator -> Fill(photons[1].phi());
      subphotonEtaPhiME_.denominator -> Fill(photons[1].eta(), photons[1].phi());
      subphotonr9ME_.denominator->Fill(photons[1].r9());
      subphotonHoverEME_.denominator->Fill(photons[1].hadTowOverEm());
      diphotonMassME_.denominator -> Fill(sqrt(2*photons[0].pt()*photons[1].pt()*(cosh(photons[0].eta()-photons[1].eta())-cos(photons[0].phi()-photons[1].phi()))));
    }

  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  // filling histograms (num_genTriggerEventFlag_)  
  if(!(photons.empty()))
    {
      photonME_.numerator -> Fill(photons[0].pt());
      photonME_variableBinning_.numerator -> Fill(photons[0].pt());
      photonPhiME_.numerator->Fill(photons[0].phi());
      photonEtaME_.numerator->Fill(photons[0].eta());
      photonVsLS_.numerator -> Fill(ls, photons[0].pt());
      photonEtaPhiME_.numerator -> Fill(photons[0].eta(), photons[0].phi());
      photonr9ME_.numerator->Fill(photons[0].r9());
      photonHoverEME_.numerator->Fill(photons[0].hadTowOverEm());
    }
  if(nphotons_>1) 
    //filling diphoton histograms
    {
      subphotonME_.numerator -> Fill(photons[1].pt());
      subphotonEtaME_.numerator -> Fill(photons[1].eta());
      subphotonPhiME_.numerator -> Fill(photons[1].phi());
      subphotonEtaPhiME_.numerator -> Fill(photons[1].eta(), photons[1].phi());
      subphotonr9ME_.numerator->Fill(photons[1].r9());
      subphotonHoverEME_.numerator->Fill(photons[1].hadTowOverEm());
      diphotonMassME_.numerator -> Fill(sqrt(2*photons[0].pt()*photons[1].pt()*(cosh(photons[0].eta()-photons[1].eta())-cos(photons[0].phi()-photons[1].phi()))));
    }
}

void PhotonMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void PhotonMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins", 2500 );
  pset.add<double>         ( "xmin",     0.);
  pset.add<double>         ( "xmax",  2500.);
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
  //desc.add<std::string>("photonSelection", "pt > 145");
  desc.add<unsigned int>("njets",      0);
  desc.add<unsigned int>("nelectrons", 0);
  desc.add<unsigned int>("nphotons",     0);

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

#include "DQMOffline/Trigger/plugins/DisplacedJetHTMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"



DisplacedJetHTMonitor::DisplacedJetHTMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , calojetToken_         ( mayConsume<reco::CaloJetCollection>    (iConfig.getParameter<edm::InputTag>("calojets")  ) )
  , eleToken_             ( mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons") ) )
  , muoToken_             ( mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )
  , caloht_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("calohtBinning") )
  , caloht_binning_       ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("calohtPSet") ) )
  , ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , calojetSelection_ ( iConfig.getParameter<std::string>("calojetSelection"))
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , ncalojets_  ( iConfig.getParameter<unsigned int>("ncalojets")   )
  , nelectrons_ ( iConfig.getParameter<unsigned int>("nelectrons" ) )
  , nmuons_     ( iConfig.getParameter<unsigned int>("nmuons")     )



{

  caloHTME_.numerator = nullptr;
  caloHTME_.denominator = nullptr;
  caloHTME_variableBinning_.numerator = nullptr;
  caloHTME_variableBinning_.denominator = nullptr;
  caloHTVsLS_.numerator = nullptr;
  caloHTVsLS_.denominator = nullptr;
  
}

DisplacedJetHTMonitor::~DisplacedJetHTMonitor()
{
  //if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  //if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

MEbinning DisplacedJetHTMonitor::getHistoPSet(edm::ParameterSet const& pset)
{
  return MEbinning{
    pset.getParameter<unsigned int>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

MEbinning DisplacedJetHTMonitor::getHistoLSPSet(edm::ParameterSet const& pset)
{
  return MEbinning{
    pset.getParameter<unsigned int>("nbins"),
      0.,
      double(pset.getParameter<unsigned int>("nbins"))
      };
}

void DisplacedJetHTMonitor::setMETitle(DJME& me, const std::string& titleX, const std::string& titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void DisplacedJetHTMonitor::bookME(DQMStore::IBooker &ibooker, DJME& me, const std::string& histname, const std::string& histtitle, unsigned int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void DisplacedJetHTMonitor::bookME(DQMStore::IBooker &ibooker, DJME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  unsigned int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void DisplacedJetHTMonitor::bookME(DQMStore::IBooker &ibooker, DJME& me, const std::string& histname, const std::string& histtitle, unsigned int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void DisplacedJetHTMonitor::bookME(DQMStore::IBooker &ibooker, DJME& me, const std::string& histname, const std::string& histtitle, unsigned int nbinsX, double xmin, double xmax, unsigned int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void DisplacedJetHTMonitor::bookME(DQMStore::IBooker &ibooker, DJME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
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

void DisplacedJetHTMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());

  histname = "caloHT"; histtitle = "caloHT";
  bookME(ibooker, caloHTME_, histname, histtitle,  caloht_binning_.nbins, caloht_binning_.xmin, caloht_binning_.xmax);
  setMETitle(caloHTME_, "calo HT [GeV]", "events / [GeV]");

  histname = "caloHT_variable"; histtitle = "caloHT_variable";
  bookME(ibooker, caloHTME_variableBinning_, histname, histtitle, caloht_variable_binning_);
  setMETitle(caloHTME_variableBinning_, "calo HT [GeV]", "events / [GeV]");


  histname = "caloHTVsLS"; histtitle = "caloHT vs LS";
  bookME(ibooker, caloHTVsLS_, histname, histtitle, ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, caloht_binning_.xmin, caloht_binning_.xmax);
  setMETitle(caloHTVsLS_, "LS", "calo HT [GeV]");
  
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
void DisplacedJetHTMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  float calo_HT = 0;

  edm::Handle<reco::CaloJetCollection> calojetHandle;
  iEvent.getByToken( calojetToken_, calojetHandle);
  std::vector<reco::CaloJet> calojets;

  if ( calojetHandle->size()< ncalojets_) return;
  for (auto const & cj : *calojetHandle) {
    if (calojetSelection_( cj ) ) calojets.push_back(cj);
    if (cj.pt()>30 && fabs(cj.eta())<3.0) calo_HT+=cj.pt();

  }
  if (calojets.size() < ncalojets_) return;


    
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

  caloHTME_.denominator->Fill(calo_HT);
  caloHTME_variableBinning_.denominator->Fill(calo_HT);

  int ls = iEvent.id().luminosityBlock();
  caloHTVsLS_.denominator -> Fill(ls, calo_HT);
  
  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  caloHTME_.numerator->Fill(calo_HT);
  caloHTME_variableBinning_.numerator->Fill(calo_HT);
  caloHTVsLS_.numerator -> Fill(ls, calo_HT);
}

void DisplacedJetHTMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void DisplacedJetHTMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins", 2500);
}

void DisplacedJetHTMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/DisplacedJet" );

  desc.add<edm::InputTag>( "calojets",  edm::InputTag("ak4CaloJets") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<std::string>( "calojetSelection", "pt > 0");
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
  edm::ParameterSetDescription calohtPSet;
  fillHistoPSetDescription(calohtPSet);
  histoPSet.add<edm::ParameterSetDescription>("calohtPSet", calohtPSet);
  std::vector<double> htbins = {0., 100., 200., 300., 350., 360., 370., 380., 390., 400., 410., 420., 430., 440., 450., 460., 470., 480., 490., 500., 550., 600., 650., 700., 750., 800., 850., 900.};
  histoPSet.add<std::vector<double> >("calohtBinning", htbins);  

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("DisplacedJetHTMonitoring", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DisplacedJetHTMonitor);

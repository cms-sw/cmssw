#include "DQMOffline/Trigger/plugins/SOSMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "TLorentzVector.h"

double sosMAX_PHI = 3.2;
int sosN_PHI = 64;
const SOSbinning phi_binning_{
  sosN_PHI, -sosMAX_PHI, sosMAX_PHI
};
double sosMAX_ETA = 2.1;
int sosN_ETA = 64;
const SOSbinning eta_binning_{
 sosN_ETA,-sosMAX_ETA,sosMAX_ETA
};
// -----------------------------
//  constructors and destructor
// -----------------------------

SOSMonitor::SOSMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , metToken_             ( consumes<reco::PFMETCollection>      (iConfig.getParameter<edm::InputTag>("met")       ) )   
  , jetToken_             ( mayConsume<reco::PFJetCollection>      (iConfig.getParameter<edm::InputTag>("jets")      ) )   
  , eleToken_             ( mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons") ) )   
  , muoToken_             ( mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )   
  , sos_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("sosBinning") )
  , sos_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("sosPSet")    ) )
  , ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , njets_      ( iConfig.getParameter<int>("njets" )      )
  , nelectrons_ ( iConfig.getParameter<int>("nelectrons" ) )
  , nmuons_     ( iConfig.getParameter<int>("nmuons" )     )
  , turn_on_    (iConfig.getParameter<std::string>("turn_on"))
  , met_pt_cut_ (iConfig.getParameter<int>("met_pt_cut"))
  , mu2_pt_cut_ (iConfig.getParameter<int>("mu2_pt_cut"))
{

  sosMON_.numerator   = nullptr;
  sosMON_.denominator = nullptr;
  sosMON_variableBinning_.numerator   = nullptr;
  sosMON_variableBinning_.denominator = nullptr;
  sosVsLS_.numerator   = nullptr;
  sosVsLS_.denominator = nullptr;
  sosPhiEtaMON_.numerator   = nullptr;
  sosPhiEtaMON_.denominator = nullptr;
  sosPhiMON_.numerator   = nullptr;
  sosPhiMON_.denominator = nullptr;
 sosHTMON_.numerator   = nullptr;
 sosHTMON_.denominator = nullptr;
}

SOSMonitor::~SOSMonitor()
{
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

SOSbinning SOSMonitor::getHistoPSet(edm::ParameterSet pset)
{
  return SOSbinning{
    pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

SOSbinning SOSMonitor::getHistoLSPSet(edm::ParameterSet pset)
{
  return SOSbinning{
    pset.getParameter<int32_t>("nbins"),
      0.,
      double(pset.getParameter<int32_t>("nbins"))
      };
}

void SOSMonitor::setTitle(MON& mon, std::string titleX, std::string titleY)
{
  mon.numerator->setAxisTitle(titleX,1);
  mon.numerator->setAxisTitle(titleY,2);
  mon.denominator->setAxisTitle(titleX,1);
  mon.denominator->setAxisTitle(titleY,2);

}

void SOSMonitor::bookME(DQMStore::IBooker &ibooker, MON& mon, const std::string& histname, const std::string& histtitle, int nbins, double min, double max)
{
  mon.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  mon.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void SOSMonitor::bookME(DQMStore::IBooker &ibooker, MON& mon, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  mon.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  mon.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void SOSMonitor::bookME(DQMStore::IBooker &ibooker, MON& mon, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  mon.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  mon.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void SOSMonitor::bookME(DQMStore::IBooker &ibooker, MON& mon, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax)
{
  mon.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  mon.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void SOSMonitor::bookME(DQMStore::IBooker &ibooker, MON& mon, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
{
  int nbinsX = binningX.size()-1;
  std::vector<float> fbinningX(binningX.begin(),binningX.end());
  float* arrX = &fbinningX[0];
  int nbinsY = binningY.size()-1;
  std::vector<float> fbinningY(binningY.begin(),binningY.end());
  float* arrY = &fbinningY[0];

  mon.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, arrX, nbinsY, arrY);
  mon.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, arrX, nbinsY, arrY);
}

void SOSMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle, histtitle2;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());
  if (turn_on_=="met") histtitle2="PFMET ";
  else histtitle2="Pt(mu2)";
  histname = turn_on_ ; histtitle=histtitle2;
  bookME(ibooker,sosMON_,histname,histtitle,sos_binning_.nbins,sos_binning_.xmin, sos_binning_.xmax);
  setTitle(sosMON_,histtitle2+"[GeV]","events / [GeV]");

  histname = turn_on_+"_variable"; histtitle=histtitle2;
  bookME(ibooker,sosMON_variableBinning_,histname,histtitle,sos_variable_binning_);
  setTitle(sosMON_variableBinning_,histtitle2+"[GeV]","events / [GeV]");

  histname = turn_on_+"VsLS"; histtitle = histtitle2+"vs LS";
  bookME(ibooker,sosVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,sos_binning_.xmin, sos_binning_.xmax);
  setTitle(sosVsLS_,"LS",histtitle2+"[GeV]");

   histname = turn_on_+"_Phi"; histtitle = histtitle2+"phi";
  bookME(ibooker,sosPhiMON_,histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setTitle(sosPhiMON_,histtitle2+"#phi","events / 0.1 rad");
  
histname = turn_on_+"_PhiEta"; histtitle = histtitle2+"phiEta";
  bookME(ibooker,sosPhiEtaMON_,histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax,eta_binning_.nbins,eta_binning_.xmin,eta_binning_.xmax);
  setTitle(sosPhiEtaMON_,histtitle2+" #phi",histtitle2+" #eta");
  
 histname = turn_on_+"ht" ; histtitle=turn_on_+"ht";
  bookME(ibooker,sosHTMON_,histname,histtitle,sos_binning_.nbins,sos_binning_.xmin, sos_binning_.xmax);
  setTitle(sosHTMON_,"HT [GeV]","events / [GeV]");


  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
void SOSMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken( metToken_, metHandle );
  reco::PFMET pfmet = metHandle->front();
  if ( ! metSelection_( pfmet ) ) return;
  
  float met = pfmet.pt();
  float phi = pfmet.phi();
  int jet60=0;
  int jet30=0;
  ht=0;
  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken( jetToken_, jetHandle );
  std::vector<reco::PFJet> jets;
  if ( int(jetHandle->size()) < njets_ ) return;
  for ( auto const & j : *jetHandle ) {
    if ( jetSelection_( j ) ) jets.push_back(j);
   
  }
  if ( int(jets.size()) < njets_ ) return;
  //float ht=0;
  for (unsigned int j=0; j<jets.size(); j++) {
    if (jets[j].pt()>30) jet30+=1;
    if (jets[j].pt()>60) jet60+=1;  
    if (jets[j].pt()>0)   ht+=jets[j].pt();
  }
  if (jet30<2 && jet60<1) return;
  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken( eleToken_, eleHandle );
  std::vector<reco::GsfElectron> electrons;
  if ( int(eleHandle->size()) < nelectrons_ ) return;
  for ( auto const & e : *eleHandle ) {
    if ( eleSelection_( e ) ) electrons.push_back(e);
  }
  if ( int(electrons.size()) < nelectrons_ ) return;
  

  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken( muoToken_, muoHandle );
  if ( int(muoHandle->size()) < nmuons_ ) return;
  std::vector<reco::Muon> muons;
  for ( auto const & m : *muoHandle ) {
    if ( muoSelection_( m ) ) muons.push_back(m);
  }
  if ( int(muons.size()) < nmuons_ ) return;
if (nmuons_!=2) return;
TLorentzVector mu1,mu2;
mu1.SetPtEtaPhiM(muons[0].pt(),muons[0].eta(),muons[0].phi(),0.1);
mu2.SetPtEtaPhiM(muons[1].pt(),muons[1].eta(),muons[1].phi(),0.1);
if ((mu1+mu2).M()<60) return;
if (muons[0].pt()<mu2_pt_cut_ || muons[1].pt()<mu2_pt_cut_) return;
if (met<met_pt_cut_) return;
  // filling histograms (denominator)  
  float var=met,var_eta=0,var_phi=phi; 
 if (turn_on_=="met"){ var=met; var_eta=0; var_phi=phi;}
  else{ var=muons[1].pt(); var_eta=muons[1].eta(); var_phi=muons[1].phi();}
  sosMON_.denominator -> Fill(var);
  sosMON_variableBinning_.denominator -> Fill(var);
  sosPhiEtaMON_.denominator -> Fill(var_phi,var_eta);
  sosPhiMON_.denominator -> Fill(var_phi);
  sosHTMON_.denominator->Fill(ht);
  int ls = iEvent.id().luminosityBlock();
  sosVsLS_.denominator -> Fill(ls, var);
  
  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  // filling histograms (num_genTriggerEventFlag_)  
  sosMON_.numerator -> Fill(var);
  sosMON_variableBinning_.numerator -> Fill(var);
  sosPhiEtaMON_.numerator -> Fill(var_phi,var_eta);
  sosPhiMON_.numerator -> Fill(var_phi);
  sosVsLS_.numerator -> Fill(ls, var);
  sosHTMON_.numerator->Fill(ht);
 }

void SOSMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void SOSMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins", 2500);
}

void SOSMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/SOS" );

  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<int>("njets",      0);
  desc.add<int>("nelectrons", 0);
  desc.add<int>("nmuons",     0);
  desc.add<std::string>("turn_on","met");
  desc.add<int>("met_pt_cut",0);
  desc.add<int>("mu2_pt_cut",0);

  edm::ParameterSetDescription genericTriggerEventPSet;
 
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi") );
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT") );
  genericTriggerEventPSet.add<std::vector<std::string> >("hltPaths",{});

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription sosPSet;
  fillHistoPSetDescription(sosPSet);
  histoPSet.add<edm::ParameterSetDescription>("sosPSet", sosPSet);
  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.};
  histoPSet.add<std::vector<double> >("sosBinning", bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("sosMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SOSMonitor);

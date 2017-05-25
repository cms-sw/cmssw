#include "DQMOffline/Trigger/plugins/TopMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

TopMonitor::TopMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , metToken_             ( consumes<reco::PFMETCollection>      (iConfig.getParameter<edm::InputTag>("met")       ) )   
  , jetToken_             ( mayConsume<reco::PFJetCollection>      (iConfig.getParameter<edm::InputTag>("jets")      ) )   
  , eleToken_             ( mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons") ) )   
  , muoToken_             ( mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )   
  , met_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("metBinning") )
  , met_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("metPSet")    ) )
  , ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , phi_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("phiPSet")    ) )
  , pt_binning_           ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("ptPSet")    ) )
  , eta_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("etaPSet")    ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , njets_      ( iConfig.getParameter<int>("njets" )      )
  , nelectrons_ ( iConfig.getParameter<int>("nelectrons" ) )
  , nmuons_     ( iConfig.getParameter<int>("nmuons" )     )
{

  metME_.numerator   = nullptr;
  metME_.denominator = nullptr;
  metME_variableBinning_.numerator   = nullptr;
  metME_variableBinning_.denominator = nullptr;
  metVsLS_.numerator   = nullptr;
  metVsLS_.denominator = nullptr;
  metPhiME_.numerator   = nullptr;
  metPhiME_.denominator = nullptr;

  muPhi_.numerator   = nullptr;
  muEta_.numerator   = nullptr;
  muPt_.numerator   = nullptr;
  elePhi_.numerator   = nullptr;
  eleEta_.numerator   = nullptr;
  elePt_.numerator   = nullptr;
  jetPhi_.numerator   = nullptr;
  jetEta_.numerator   = nullptr;
  jetPt_.numerator   = nullptr; 

  muPhi_.denominator   = nullptr;
  muEta_.denominator   = nullptr;
  muPt_.denominator   = nullptr;
  elePhi_.denominator   = nullptr;
  eleEta_.denominator   = nullptr;
  elePt_.denominator   = nullptr;
  jetPhi_.denominator   = nullptr;
  jetEta_.denominator   = nullptr;
  jetPt_.denominator   = nullptr;
 
}

TopMonitor::~TopMonitor()
{
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

MEbinning TopMonitor::getHistoPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

MEbinning TopMonitor::getHistoLSPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      0.,
      double(pset.getParameter<int32_t>("nbins"))
      };
}

void TopMonitor::setMETitle(METME& me, std::string titleX, std::string titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
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

void TopMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());

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

  histname = "muPt"; histtitle = "muon p_{T}";
  bookME(ibooker,muPt_,histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
  setMETitle(muPt_,"muon p_{T} [GeV]","events");

  histname = "muPhi"; histtitle = "muon #phi";
  bookME(ibooker,muPhi_,histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(muPhi_," muon #phi","events");

  histname = "muEta"; histtitle = "muon #eta";
  bookME(ibooker,muEta_,histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
  setMETitle(muEta_," muon #eta","events");

  histname = "elePt"; histtitle = "electron p_{T}";
  bookME(ibooker,elePt_,histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
  setMETitle(elePt_,"electron p_{T} [GeV]","events");

  histname = "elePhi"; histtitle = "electron #phi";
  bookME(ibooker,elePhi_,histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(elePhi_," electron #phi","events");

  histname = "eleEta"; histtitle = "electron #eta";
  bookME(ibooker,eleEta_,histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
  setMETitle(eleEta_," electron #eta","events");

  histname = "jetPt"; histtitle = "jet p_{T}";
  bookME(ibooker,jetPt_,histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
  setMETitle(jetPt_,"jet p_{T} [GeV]","events");

  histname = "jetPhi"; histtitle = "jet #phi";
  bookME(ibooker,jetPhi_,histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(jetPhi_," jet #phi","events");

  histname = "jetEta"; histtitle = "jet #eta";
  bookME(ibooker,jetEta_,histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
  setMETitle(jetEta_," jet #eta","events");


  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
void TopMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken( metToken_, metHandle );
  reco::PFMET pfmet = metHandle->front();
  if ( ! metSelection_( pfmet ) ) return;
  
  float met = pfmet.pt();
  float phi = pfmet.phi();

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

  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken( jetToken_, jetHandle );
  std::vector<reco::PFJet> jets;
  if ( int(jetHandle->size()) < njets_ ) return;
  for ( auto const & j : *jetHandle ) {
      if ( jetSelection_( j ) ){
          bool isJetOverlappedWithLepton = false;
          TLorentzVector jp4; jp4.SetPtEtaPhiE(j.pt(),j.eta(),j.phi(),j.energy());
          for (auto const m : muons){
              TLorentzVector mp4; mp4.SetPtEtaPhiE(m.pt(),m.eta(),m.phi(),m.energy());
              if (mp4.DeltaR(jp4)<0.4){
                  isJetOverlappedWithLepton=true;
                  break;
              }
          }
          if (isJetOverlappedWithLepton) continue;
          for (auto const e : electrons){
              TLorentzVector ep4; ep4.SetPtEtaPhiE(e.pt(),e.eta(),e.phi(),e.energy());
              if (ep4.DeltaR(jp4)<0.4){
                  isJetOverlappedWithLepton=true;
                  break;
              }
          }
          if (isJetOverlappedWithLepton) continue;
          jets.push_back(j);
      }

  }
  if ( int(jets.size()) < njets_ ) return;

  // filling histograms (denominator)  
  metME_.denominator -> Fill(met);
  metME_variableBinning_.denominator -> Fill(met);
  metPhiME_.denominator -> Fill(phi);

  int ls = iEvent.id().luminosityBlock();
  metVsLS_.denominator -> Fill(ls, met);

  if (muons.size()>0 && nmuons_>0){
      muPhi_.denominator  -> Fill(muons.at(0).phi());
      muEta_.denominator  -> Fill(muons.at(0).eta());
      muPt_.denominator   -> Fill(muons.at(0).pt() );
  }
  if (electrons.size()>0 && nelectrons_>0){
      elePhi_.denominator -> Fill(electrons.at(0).phi());
      eleEta_.denominator -> Fill(electrons.at(0).eta());
      elePt_.denominator  -> Fill(electrons.at(0).pt() );
  }
  if (jets.size()>0 && njets_>0){
      jetPhi_.denominator -> Fill(jets.at(0).phi());
      jetEta_.denominator -> Fill(jets.at(0).eta());
      jetPt_.denominator  -> Fill(jets.at(0).pt()) ;
  }

  
  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  // filling histograms (num_genTriggerEventFlag_)  
  metME_.numerator -> Fill(met);
  metME_variableBinning_.numerator -> Fill(met);
  metPhiME_.numerator -> Fill(phi);
  metVsLS_.numerator -> Fill(ls, met);

  if (muons.size()>0 && nmuons_>0){
      muPhi_.numerator  -> Fill(muons.at(0).phi());
      muEta_.numerator  -> Fill(muons.at(0).eta());
      muPt_.numerator   -> Fill(muons.at(0).pt() );
  }
  if (electrons.size()>0 && nelectrons_>0){
      elePhi_.numerator -> Fill(electrons.at(0).phi());
      eleEta_.numerator -> Fill(electrons.at(0).eta());
      elePt_.numerator  -> Fill(electrons.at(0).pt() );
  }
  if (jets.size()>0 && njets_>0){
      jetPhi_.numerator -> Fill(jets.at(0).phi());
      jetEta_.numerator -> Fill(jets.at(0).eta());
      jetPt_.numerator  -> Fill(jets.at(0).pt()) ;
  }

}

void TopMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void TopMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins", 2500);
}

void TopMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/TOP" );

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

  descriptions.add("topMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TopMonitor);

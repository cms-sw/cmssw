#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DQM/TrackingMonitor/interface/GetLumi.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DQMOffline/Trigger/plugins/TopMonitor.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

TopMonitor::TopMonitor( const edm::ParameterSet& iConfig ) :
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , metToken_             ( consumes<reco::PFMETCollection>      (iConfig.getParameter<edm::InputTag>("met")       ) )
  , jetToken_             ( mayConsume<reco::PFJetCollection>      (iConfig.getParameter<edm::InputTag>("jets")      ) )
  , eleToken_             ( mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("electrons") ) )
  , muoToken_             ( mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )
  // Marina
  , jetTagToken_          ( mayConsume<reco::JetTagCollection>     (iConfig.getParameter<edm::InputTag>("btagalgo") ))
  //Suvankar
  , vtxToken_             ( mayConsume<reco::VertexCollection> (iConfig.getParameter<edm::InputTag>("vertices") ) )
  , met_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("metPSet")    ) )
  , ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , phi_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("phiPSet")    ) )
  , pt_binning_           ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("ptPSet")    ) )
  , eta_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("etaPSet")    ) )
  , HT_binning_           ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("htPSet")    ) )
  , DR_binning_           ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("DRPSet")    ) )
  // Marina
  , csv_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet> ("csvPSet")))
  //george
  , invMass_mumu_binning_  ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet> ("invMassPSet")))
  , MHT_binning_           ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("MHTPSet")    ) )

  , met_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("metBinning") )
  , HT_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("HTBinning") )
  , jetPt_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetPtBinning") )
  , muPt_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muPtBinning") )
  , elePt_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("elePtBinning") )
  , jetEta_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetEtaBinning") )
  , muEta_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muEtaBinning") )
  , eleEta_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("eleEtaBinning") )

 //george
 , invMass_mumu_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("invMassVariableBinning") )
 , MHT_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("MHTVariableBinning") )
  , HT_variable_binning_2D_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("HTBinning2D") )
  , jetPt_variable_binning_2D_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetPtBinning2D") )
  , muPt_variable_binning_2D_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muPtBinning2D") )
  , elePt_variable_binning_2D_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("elePtBinning2D") )
  , jetEta_variable_binning_2D_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetEtaBinning2D") )
  , muEta_variable_binning_2D_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("muEtaBinning2D") )
  , eleEta_variable_binning_2D_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("eleEtaBinning2D") )
  , phi_variable_binning_2D_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("phiBinning2D") )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , HTdefinition_ ( iConfig.getParameter<std::string>("HTdefinition") )
  , vtxSelection_ ( iConfig.getParameter<std::string>("vertexSelection") )
  , bjetSelection_( iConfig.getParameter<std::string>("bjetSelection"))
  , njets_      ( iConfig.getParameter<unsigned int>("njets" )      )
  , nelectrons_ ( iConfig.getParameter<unsigned int>("nelectrons" ) )
  , nmuons_     ( iConfig.getParameter<unsigned int>("nmuons" )     )
  , leptJetDeltaRmin_     ( iConfig.getParameter<double>("leptJetDeltaRmin" )     )
  , bJetMuDeltaRmax_     ( iConfig.getParameter<double>("bJetMuDeltaRmax" )     )
  , bJetDeltaEtaMax_     ( iConfig.getParameter<double>("bJetDeltaEtaMax" )     )
  , HTcut_     ( iConfig.getParameter<double>("HTcut" )     )
  // Marina
  , nbjets_    ( iConfig.getParameter<unsigned int>("nbjets"))
  , workingpoint_(iConfig.getParameter<double>("workingpoint"))
  //Suvankar
  , usePVcuts_ ( iConfig.getParameter<bool>("applyleptonPVcuts") )
  //george
  , invMassUppercut_ (iConfig.getParameter<double>("invMassUppercut"))
  , invMassLowercut_ (iConfig.getParameter<double>("invMassLowercut"))
  , opsign_ (iConfig.getParameter<bool>("oppositeSignMuons"))
  , MHTdefinition_ ( iConfig.getParameter<std::string>("MHTdefinition") )
  , MHTcut_     ( iConfig.getParameter<double>("MHTcut" )     )
{
    METME empty;
    empty.numerator = nullptr;
    empty.denominator = nullptr;

    metME_ = empty ;
    metME_variableBinning_ = empty ;
    metVsLS_ = empty ;
    metPhiME_ = empty ;
    eventHT_ = empty ;
    jetVsLS_ = empty ;
    muVsLS_ = empty ;
    eleVsLS_ = empty ;
    // Marina
    bjetVsLS_ = empty ;
    htVsLS_ = empty ;
    jetEtaPhi_HEP17_ = empty; // for HEP17 monitoring

    jetMulti_ = empty;
    eleMulti_ = empty;
    muMulti_ = empty;
    // Marina
    bjetMulti_ = empty;

    elePt_jetPt_ = empty;
    elePt_eventHT_ = empty;

    ele1Pt_ele2Pt_ = empty;
    ele1Eta_ele2Eta_ = empty;
    mu1Pt_mu2Pt_ = empty;
    mu1Eta_mu2Eta_ = empty;
    elePt_muPt_ = empty;
    eleEta_muEta_ = empty;
    //george
    invMass_mumu_=empty;
    eventMHT_=empty;
    invMass_mumu_variableBinning_=empty;
    eventMHT_variableBinning_=empty;



    //BTV
    DeltaR_jet_Mu_ = empty;

    muPhi_= std::vector<METME> (nmuons_,empty);
    muEta_= std::vector<METME> (nmuons_,empty);
    muPt_= std::vector<METME> (nmuons_,empty);
    muEta_variableBinning_= std::vector<METME> (nmuons_,empty);
    muPt_variableBinning_= std::vector<METME> (nmuons_,empty);
    muPtEta_= std::vector<METME> (nmuons_,empty);
    muEtaPhi_= std::vector<METME> (nmuons_,empty);

    elePhi_= std::vector<METME> (nelectrons_,empty);
    eleEta_= std::vector<METME> (nelectrons_,empty);
    elePt_= std::vector<METME> (nelectrons_,empty);
    eleEta_variableBinning_= std::vector<METME> (nelectrons_,empty);
    elePt_variableBinning_= std::vector<METME> (nelectrons_,empty);
    elePtEta_= std::vector<METME> (nelectrons_,empty);
    eleEtaPhi_= std::vector<METME> (nelectrons_,empty);


    jetPhi_= std::vector<METME> (njets_,empty);
    jetEta_= std::vector<METME> (njets_,empty);
    jetPt_= std::vector<METME> (njets_,empty);
    jetEta_variableBinning_= std::vector<METME> (njets_,empty);
    jetPt_variableBinning_= std::vector<METME> (njets_,empty);
    jetPtEta_= std::vector<METME> (njets_,empty);
    jetEtaPhi_= std::vector<METME> (njets_,empty);

    // Marina
    bjetPhi_= std::vector<METME> (nbjets_,empty);
    bjetEta_= std::vector<METME> (nbjets_,empty);
    bjetPt_= std::vector<METME> (nbjets_,empty);
    bjetCSV_= std::vector<METME> (nbjets_,empty);
    bjetEta_variableBinning_= std::vector<METME> (nbjets_,empty);
    bjetPt_variableBinning_= std::vector<METME> (nbjets_,empty);
    bjetPtEta_= std::vector<METME> (nbjets_,empty);
    bjetEtaPhi_= std::vector<METME> (nbjets_,empty);
    bjetCSVHT_= std::vector<METME> (nbjets_,empty);

  //Suvankar
  lepPVcuts_.dxy = (iConfig.getParameter<edm::ParameterSet>("leptonPVcuts")).getParameter<double>("dxy");
  lepPVcuts_.dz  = (iConfig.getParameter<edm::ParameterSet>("leptonPVcuts")).getParameter<double>("dz");
}

TopMonitor::~TopMonitor()
{
    if (num_genTriggerEventFlag_) num_genTriggerEventFlag_.reset();
    if (den_genTriggerEventFlag_) den_genTriggerEventFlag_.reset();
}

MEbinning TopMonitor::getHistoPSet(const edm::ParameterSet& pset)
{
  return MEbinning{
    pset.getParameter<uint32_t>("nbins"),
    pset.getParameter<double>("xmin"),
    pset.getParameter<double>("xmax"),
  };
}

MEbinning TopMonitor::getHistoLSPSet(const edm::ParameterSet& pset)
{
  return MEbinning{
    pset.getParameter<uint32_t>("nbins"),
    0.,
    double(pset.getParameter<uint32_t>("nbins"))
  };
}

void TopMonitor::setMETitle(METME& me, const std::string& titleX, const std::string& titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);
}

void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, unsigned int nbins, double min, double max)
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

void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, unsigned int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}

void TopMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, const std::string& histname, const std::string& histtitle, unsigned int nbinsX, double xmin, double xmax, unsigned int nbinsY, double ymin, double ymax)
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

  if (njets_ > 0){
      histname = "jetVsLS"; histtitle = "jet pt vs LS";
      bookME(ibooker,jetVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(jetVsLS_,"LS","jet pt [GeV]");
  }
  if (nmuons_ > 0){
      histname = "muVsLS"; histtitle = "muon pt vs LS";
      bookME(ibooker,muVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(muVsLS_,"LS","muon pt [GeV]");
  }
  if (nelectrons_ > 0){
      histname = "eleVsLS"; histtitle = "electron pt vs LS";
      bookME(ibooker,eleVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(eleVsLS_,"LS","electron pt [GeV]");
  }

  // Marina
  if (nbjets_ > 0){
    histname = "bjetVsLS"; histtitle = "b-jet pt vs LS";
    bookME(ibooker,bjetVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,pt_binning_.xmin, pt_binning_.xmax);
    setMETitle(bjetVsLS_,"LS","b-jet pt [GeV]");
  }

  if ( (nmuons_ > 0) && (nelectrons_ > 0)){
      histname = "elePt_muPt"; histtitle = "electron pt vs muon pt";
      bookME(ibooker,elePt_muPt_,histname,histtitle, elePt_variable_binning_2D_, muPt_variable_binning_2D_);
      setMETitle(elePt_muPt_,"electron pt [GeV]","muon pt [GeV]");

      histname = "eleEta_muEta"; histtitle = "electron #eta vs muon #eta";
      bookME(ibooker,eleEta_muEta_,histname,histtitle, eleEta_variable_binning_2D_, muEta_variable_binning_2D_);
      setMETitle(eleEta_muEta_,"electron #eta","muon #eta");
  }

  if ( nelectrons_ > 1 ){
      histname = "ele1Pt_ele2Pt"; histtitle = "electron-1 pt vs electron-2 pt";
      bookME(ibooker,ele1Pt_ele2Pt_,histname,histtitle, elePt_variable_binning_2D_, elePt_variable_binning_2D_);
      setMETitle(ele1Pt_ele2Pt_,"electron-1 pt [GeV]","electron-2 pt [GeV]");

      histname = "ele1Eta_ele2Eta"; histtitle = "electron-1 #eta vs electron-2 #eta";
      bookME(ibooker,ele1Eta_ele2Eta_,histname,histtitle, eleEta_variable_binning_2D_, eleEta_variable_binning_2D_);
      setMETitle(ele1Eta_ele2Eta_,"electron-1 #eta","electron-2 #eta");
  }

  if ( nmuons_ > 1 ) {
      histname = "mu1Pt_mu2Pt"; histtitle = "muon-1 pt vs muon-2 pt";
      bookME(ibooker,mu1Pt_mu2Pt_,histname,histtitle, muPt_variable_binning_2D_, muPt_variable_binning_2D_);
      setMETitle(mu1Pt_mu2Pt_,"muon-1 pt [GeV]","muon-2 pt [GeV]");

      histname = "mu1Eta_mu2Eta"; histtitle = "muon-1 #eta vs muon-2 #eta";
      bookME(ibooker,mu1Eta_mu2Eta_,histname,histtitle, muEta_variable_binning_2D_, muEta_variable_binning_2D_);
      setMETitle(mu1Eta_mu2Eta_,"muon-1 #eta","muon-2 #eta");
      //george
     histname = "invMass"; histtitle = "M mu1 mu2";
     bookME(ibooker,invMass_mumu_,histname,histtitle, invMass_mumu_binning_.nbins,invMass_mumu_binning_.xmin,invMass_mumu_binning_.xmax);
      setMETitle(invMass_mumu_,"M(mu1,mu2) [GeV]","events");
     histname = "invMass_variable"; histtitle = "M mu1 mu2 variable";
     bookME(ibooker,invMass_mumu_variableBinning_,histname,histtitle,invMass_mumu_variable_binning_);
     setMETitle(invMass_mumu_variableBinning_,"M(mu1,mu2) [GeV]","events / [GeV]");
  }

  if ( (njets_ > 0) && (nmuons_ > 0)){

      histname = "DeltaR_jet_Mu"; histtitle = "#DeltaR(jet,mu)";
      bookME(ibooker,DeltaR_jet_Mu_,histname,histtitle, DR_binning_.nbins, DR_binning_.xmin, DR_binning_.xmax );
      setMETitle(DeltaR_jet_Mu_,"#DeltaR(jet,mu)","events");
  }

  histname = "htVsLS"; histtitle = "event HT vs LS";
  bookME(ibooker,htVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,pt_binning_.xmin, pt_binning_.xmax);
  setMETitle(htVsLS_,"LS","event HT [GeV]");

  histname = "metPhi"; histtitle = "PFMET phi";
  bookME(ibooker,metPhiME_,histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(metPhiME_,"PF MET #phi","events / 0.1 rad");

  for (unsigned int iMu=0; iMu<nmuons_; ++iMu){
      std::string index = std::to_string(iMu+1);

      histname = "muPt_"; histtitle = "muon p_{T} - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,muPt_.at(iMu),histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(muPt_.at(iMu),"muon p_{T} [GeV]","events");
      histname.append("_variableBinning");
      bookME(ibooker,muPt_variableBinning_.at(iMu),histname,histtitle, muPt_variable_binning_);
      setMETitle(muPt_variableBinning_.at(iMu),"muon p_{T} [GeV]","events");

      histname = "muEta_"; histtitle = "muon #eta - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,muEta_.at(iMu),histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(muEta_.at(iMu)," muon #eta","events");
      histname.append("_variableBinning");
      bookME(ibooker,muEta_variableBinning_.at(iMu),histname,histtitle, muEta_variable_binning_);
      setMETitle(muEta_variableBinning_.at(iMu)," muon #eta","events");

      histname = "muPhi_"; histtitle = "muon #phi - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,muPhi_.at(iMu),histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(muPhi_.at(iMu)," muon #phi","events");

      histname = "muPtEta_"; histtitle = "muon p_{T} - #eta - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,muPtEta_.at(iMu),histname,histtitle, muPt_variable_binning_2D_, muEta_variable_binning_2D_);
      setMETitle(muPtEta_.at(iMu),"muon p_{T} [GeV]","muon #eta");

      histname = "muEtaPhi_"; histtitle = "muon #eta - #phi - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,muEtaPhi_.at(iMu),histname,histtitle, muEta_variable_binning_2D_, phi_variable_binning_2D_);
      setMETitle(muEtaPhi_.at(iMu),"muon #phi","muon #eta");


  }

  for (unsigned int iEle=0; iEle<nelectrons_; ++iEle){
      std::string index = std::to_string(iEle+1);

      histname = "elePt_"; histtitle = "electron p_{T} - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,elePt_.at(iEle),histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(elePt_.at(iEle),"electron p_{T} [GeV]","events");
      histname.append("_variableBinning");
      bookME(ibooker,elePt_variableBinning_.at(iEle),histname,histtitle, elePt_variable_binning_);
      setMETitle(elePt_variableBinning_.at(iEle),"electron p_{T} [GeV]","events");

      histname = "eleEta_"; histtitle = "electron #eta - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,eleEta_.at(iEle),histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(eleEta_.at(iEle)," electron #eta","events");
      histname.append("_variableBinning");
      bookME(ibooker,eleEta_variableBinning_.at(iEle),histname,histtitle, eleEta_variable_binning_);
      setMETitle(eleEta_variableBinning_.at(iEle),"electron #eta","events");

      histname = "elePhi_"; histtitle = "electron #phi - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,elePhi_.at(iEle),histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(elePhi_.at(iEle)," electron #phi","events");

      histname = "elePtEta_"; histtitle = "electron p_{T} - #eta - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,elePtEta_.at(iEle),histname,histtitle, elePt_variable_binning_2D_, eleEta_variable_binning_2D_);
      setMETitle(elePtEta_.at(iEle),"electron p_{T} [GeV]","electron #eta");

      histname = "eleEtaPhi_"; histtitle = "electron #eta - #phi - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,eleEtaPhi_.at(iEle),histname,histtitle, eleEta_variable_binning_2D_, phi_variable_binning_2D_);
      setMETitle(eleEtaPhi_.at(iEle),"electron #phi","electron #eta");


  }

  for (unsigned int iJet=0; iJet<njets_; ++iJet){
      std::string index = std::to_string(iJet+1);

      histname = "jetPt_"; histtitle = "jet p_{T} - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,jetPt_.at(iJet),histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
      setMETitle(jetPt_.at(iJet),"jet p_{T} [GeV]","events");
      histname.append("_variableBinning");
      bookME(ibooker,jetPt_variableBinning_.at(iJet),histname,histtitle, jetPt_variable_binning_);
      setMETitle(jetPt_variableBinning_.at(iJet),"jet p_{T} [GeV]","events");

      histname = "jetEta_"; histtitle = "jet #eta - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,jetEta_.at(iJet),histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(jetEta_.at(iJet)," jet #eta","events");
      histname.append("_variableBinning");
      bookME(ibooker,jetEta_variableBinning_.at(iJet),histname,histtitle, jetEta_variable_binning_);
      setMETitle(jetEta_variableBinning_.at(iJet),"jet #eta","events");

      histname = "jetPhi_"; histtitle = "jet #phi - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,jetPhi_.at(iJet),histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(jetPhi_.at(iJet)," jet #phi","events");

      histname = "jetPtEta_"; histtitle = "jet p_{T} - #eta - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,jetPtEta_.at(iJet),histname,histtitle, jetPt_variable_binning_2D_, jetEta_variable_binning_2D_);
      setMETitle(jetPtEta_.at(iJet),"jet p_{T} [GeV]","jet #eta");

      histname = "jetEtaPhi_"; histtitle = "jet #eta - #phi - ";
      histname.append(index); histtitle.append(index);
      bookME(ibooker,jetEtaPhi_.at(iJet),histname,histtitle, jetEta_variable_binning_2D_, phi_variable_binning_2D_);
      setMETitle(jetEtaPhi_.at(iJet),"#phi","jet #eta");



  }

  // Marina
  for (unsigned int iBJet=0; iBJet<nbjets_; ++iBJet){
    std::string index = std::to_string(iBJet+1);

    histname = "bjetPt_"; histtitle = "b-jet p_{T} - ";
    histname.append(index); histtitle.append(index);
    bookME(ibooker,bjetPt_.at(iBJet),histname,histtitle, pt_binning_.nbins, pt_binning_.xmin, pt_binning_.xmax);
    setMETitle(bjetPt_.at(iBJet)," b-jet p_{T} [GeV]","events");
    histname.append("_variableBinning");
    bookME(ibooker,bjetPt_variableBinning_.at(iBJet),histname,histtitle, jetPt_variable_binning_);
    setMETitle(bjetPt_variableBinning_.at(iBJet),"b-jet p_{T} [GeV]","events");

    histname = "bjetEta_"; histtitle = "b-jet #eta - ";
    histname.append(index); histtitle.append(index);
    bookME(ibooker,bjetEta_.at(iBJet),histname,histtitle, eta_binning_.nbins,eta_binning_.xmin, eta_binning_.xmax);
    setMETitle(bjetEta_.at(iBJet)," b-jet #eta","events");
    histname.append("_variableBinning");
    bookME(ibooker,bjetEta_variableBinning_.at(iBJet),histname,histtitle, jetEta_variable_binning_);
    setMETitle(bjetEta_variableBinning_.at(iBJet),"b-jet #eta","events");

    histname = "bjetPhi_"; histtitle = "b-jet #phi - ";
    histname.append(index); histtitle.append(index);
    bookME(ibooker,bjetPhi_.at(iBJet),histname,histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
    setMETitle(bjetPhi_.at(iBJet)," b-jet #phi","events");

    histname = "bjetCSV_"; histtitle = "b-jet CSV - ";
    histname.append(index); histtitle.append(index);
    bookME(ibooker,bjetCSV_.at(iBJet),histname,histtitle, csv_binning_.nbins, csv_binning_.xmin, csv_binning_.xmax);
    setMETitle(bjetCSV_.at(iBJet)," b-jet CSV","events");

    histname = "bjetPtEta_"; histtitle = "b-jet p_{T} - #eta - ";
    histname.append(index); histtitle.append(index);
    bookME(ibooker,bjetPtEta_.at(iBJet),histname,histtitle, jetPt_variable_binning_2D_, jetEta_variable_binning_2D_);
    setMETitle(bjetPtEta_.at(iBJet),"b-jet p_{T} [GeV]","b-jet #eta");

    histname = "bjetEtaPhi_"; histtitle = "b-jet #eta - #phi - ";
    histname.append(index); histtitle.append(index);
    bookME(ibooker,bjetEtaPhi_.at(iBJet),histname,histtitle, jetEta_variable_binning_2D_, phi_variable_binning_2D_);
    setMETitle(bjetEtaPhi_.at(iBJet),"b-jet #phi","b-jet #eta");

    histname = "bjetCSVHT_"; histtitle = "HT - b-jet CSV - ";
    histname.append(index); histtitle.append(index);
    bookME(ibooker,bjetCSVHT_.at(iBJet), histname, histtitle, csv_binning_.nbins, csv_binning_.xmin, csv_binning_.xmax, HT_binning_.nbins,HT_binning_.xmin, HT_binning_.xmax);
    setMETitle(bjetCSVHT_.at(iBJet),"b-jet CSV", "event HT [GeV]");
  }


  histname = "eventHT"; histtitle = "event HT";
  bookME(ibooker,eventHT_,histname,histtitle, HT_binning_.nbins,HT_binning_.xmin, HT_binning_.xmax);
  setMETitle(eventHT_," event HT [GeV]","events");
  histname.append("_variableBinning");
  bookME(ibooker,eventHT_variableBinning_,histname,histtitle, HT_variable_binning_);
  setMETitle(eventHT_variableBinning_,"event HT [GeV]","events");

  histname = "jetEtaPhi_HEP17"; histtitle = "jet #eta-#phi for HEP17";
  bookME(ibooker,jetEtaPhi_HEP17_,histname,histtitle,10,-2.5,2.5,18,-3.1415,3.1415); // for HEP17 monitoring
  setMETitle(eventHT_,"jet #eta","jet #phi");

  histname = "jetMulti"; histtitle = "jet multiplicity";
  bookME(ibooker,jetMulti_,histname,histtitle, 11,-.5, 10.5);
  setMETitle(jetMulti_,"jet multiplicity","events");

  histname = "eleMulti"; histtitle = "electron multiplicity";
  bookME(ibooker,eleMulti_,histname,histtitle, 6,-.5, 5.5);
  setMETitle(eleMulti_,"electron multiplicity","events");

  histname = "muMulti"; histtitle = "muon multiplicity";
  bookME(ibooker,muMulti_,histname,histtitle, 6,-.5, 5.5);
  setMETitle(muMulti_,"muon multiplicity","events");

  // Marina
  histname = "bjetMulti"; histtitle = "b-jet multiplicity";
  bookME(ibooker,bjetMulti_,histname,histtitle, 6,-.5, 5.5);
  setMETitle(bjetMulti_,"b-jet multiplicity","events");


  histname = "elePt_jetPt"; histtitle = "electron pt vs jet pt";
  bookME(ibooker,elePt_jetPt_,histname,histtitle, elePt_variable_binning_2D_, jetPt_variable_binning_2D_);
  setMETitle(elePt_jetPt_,"leading electron pt","leading jet pt");

  histname = "elePt_eventHT"; histtitle = "electron pt vs event HT";
  bookME(ibooker,elePt_eventHT_,histname,histtitle, elePt_variable_binning_2D_, HT_variable_binning_2D_);
  setMETitle(elePt_eventHT_,"leading electron pt","event HT");

  //george
  histname = "eventMHT"; histtitle = "event MHT";
  bookME(ibooker,eventMHT_,histname,histtitle, MHT_binning_.nbins,MHT_binning_.xmin, MHT_binning_.xmax);
  setMETitle(eventMHT_," event MHT [GeV]","events");
 histname = "eventMHT_variable"; histtitle = "event MHT variable";
     bookME(ibooker,eventMHT_variableBinning_,histname,histtitle,MHT_variable_binning_);
     setMETitle(eventMHT_variableBinning_,"event MHT [GeV]","events / [GeV]");




  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

void TopMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {
  mll=-2;
  sign=0;
  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  //Suvankar
  edm::Handle<reco::VertexCollection> primaryVertices;
  iEvent.getByToken(vtxToken_, primaryVertices);
  //Primary Vertex selection
  const reco::Vertex* pv = nullptr;
  for(auto const& v: *primaryVertices) {
    if ( !vtxSelection_( v ) )      continue;
    pv = &v;
    break;
  }
  if(usePVcuts_ && pv == nullptr)      return;

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken( metToken_, metHandle );
  if (!metHandle.isValid()){
      edm::LogWarning("TopMonitor") << "MET handle not valid \n";
      return;
  }
  reco::PFMET pfmet = metHandle->front();
  if ( ! metSelection_( pfmet ) ) return;

  float met = pfmet.pt();
  float phi = pfmet.phi();

  edm::Handle<edm::View<reco::GsfElectron> > eleHandle;
  iEvent.getByToken( eleToken_, eleHandle );
  if (!eleHandle.isValid()){
      edm::LogWarning("TopMonitor") << "Electron handle not valid \n";
      return;
  }
  std::vector<reco::GsfElectron> electrons;
  if ( eleHandle->size() < nelectrons_ ) return;
  for ( auto const & e : *eleHandle ) {
    if (eleSelection_(e)) electrons.push_back(e);
    //Suvankar
    if ( usePVcuts_ &&
         (std::fabs(e.gsfTrack()->dxy(pv->position())) >= lepPVcuts_.dxy || std::fabs(e.gsfTrack()->dz(pv->position())) >= lepPVcuts_.dz) ) continue;
  }
  if ( electrons.size() < nelectrons_ ) return;

  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken( muoToken_, muoHandle );
  if (!muoHandle.isValid()){
      edm::LogWarning("TopMonitor") << "Muon handle not valid \n";
      return;
  }
  if ( muoHandle->size() < nmuons_ ) return;
  std::vector<reco::Muon> muons;
  for ( auto const & m : *muoHandle ) {
    if ( muoSelection_( m ) ) muons.push_back(m);
    //Suvankar
    if ( usePVcuts_ &&
         (std::fabs(m.muonBestTrack()->dxy(pv->position())) >= lepPVcuts_.dxy || std::fabs(m.muonBestTrack()->dz(pv->position())) >= lepPVcuts_.dz) ) continue;
  }
  if ( muons.size() < nmuons_ ) return;

    //george

  if (nmuons_>1){
    mll = (muons[0].p4() + muons[1].p4()).M();
    sign = muons[0].charge() * muons[1].charge();
  }
  if (nmuons_>1 && invMassUppercut_>-1 && invMassLowercut_>-1 && (mll>invMassUppercut_ || mll<invMassLowercut_)) return;
  if (nmuons_>1 && opsign_ && sign==1) return;

  //cout<<" mll="<<mll<<"  invMasscut_="<<invMasscut_<<endl;
  double eventHT = 0.;
  math::XYZTLorentzVector eventMHT(0., 0., 0., 0.);

  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken( jetToken_, jetHandle );
  if (!jetHandle.isValid()) {
      edm::LogWarning("TopMonitor") << "Jet handle not valid \n";
      return;
  }
  std::vector<reco::PFJet> jets;
  if (jetHandle->size() < njets_) return;
  for (auto const & j : *jetHandle) {
    if (HTdefinition_(j)) {
      eventHT += j.pt();
    }
    if (MHTdefinition_(j)) {
      eventMHT += j.p4();
    }
    if (jetSelection_(j)) {
      bool isJetOverlappedWithLepton = false;
      if(nmuons_>0) {
        for (auto const& m : muons) {
          if (deltaR(j, m)<leptJetDeltaRmin_) {
            isJetOverlappedWithLepton=true;
            break;
          }
        }
      }
      if (isJetOverlappedWithLepton) continue;
      if(nelectrons_>0) {
        for (auto const & e: electrons) {
          if (deltaR(j, e)<leptJetDeltaRmin_) {
            isJetOverlappedWithLepton=true;
            break;
          }
        }
      }
      if (isJetOverlappedWithLepton) continue;
      jets.push_back(j);
    }

  }
  if (jets.size() < njets_) return;

  if (eventHT < HTcut_) return;
  if (MHTcut_>0 && eventMHT.pt()<MHTcut_) return;

  // Marina
  edm::Handle<reco::JetTagCollection> bjetHandle;
  iEvent.getByToken( jetTagToken_, bjetHandle );
  if (!bjetHandle.isValid()){
    edm::LogWarning("TopMonitor") << "B-Jet handle not valid \n";
    return;
  }

  const reco::JetTagCollection& bTags = *(bjetHandle.product());
  JetTagMap bjets;

  if (bTags.size() < nbjets_ ) return;
  for (unsigned int i=0; i!=bTags.size(); ++i){
      // Apply Selections
      if (!bjetSelection_(*dynamic_cast<const reco::Jet*>(bTags[i].first.get())) ) continue;
      if (bTags[i].second < workingpoint_                  ) continue;

      // Fill JetTag Map
      bjets.insert(JetTagMap::value_type(bTags[i].first, bTags[i].second));
  }

  if (bjets.size() < nbjets_ ) return;

  if (nbjets_ > 1){
      double deltaEta = std::abs(bjets.begin()->first->eta()-(++bjets.begin())->first->eta());
      if (deltaEta > bJetDeltaEtaMax_) return;
  }

  if ((nbjets_>0) && (nmuons_>0)){
      bool foundMuonInsideJet = false;
      for (auto const & bjet : bjets){
          for (auto const & mu : muons){
              double dR = deltaR(*bjet.first,mu);
              if (dR < bJetMuDeltaRmax_){
                  foundMuonInsideJet = true;
                  break;
              }
          }
          if(foundMuonInsideJet) break;
      }

      if (!foundMuonInsideJet) return;
  }

  // filling histograms (denominator)
  metME_.denominator -> Fill(met);
  metME_variableBinning_.denominator -> Fill(met);
  metPhiME_.denominator -> Fill(phi);
  eventHT_.denominator -> Fill(eventHT);
  eventHT_variableBinning_.denominator -> Fill(eventHT);
//george
  eventMHT_.denominator -> Fill(eventMHT.pt());
  eventMHT_variableBinning_.denominator -> Fill(eventMHT.pt());

  int ls = iEvent.id().luminosityBlock();
  metVsLS_.denominator -> Fill(ls, met);
  htVsLS_.denominator -> Fill(ls, eventHT);

  jetMulti_.denominator -> Fill(jets.size());
  eleMulti_.denominator -> Fill(electrons.size());
  muMulti_.denominator -> Fill(muons.size());

  // Marina
  bjetMulti_.denominator -> Fill(bjets.size());

  if (nmuons_ > 0){
      muVsLS_.denominator -> Fill(ls, muons.at(0).pt());
      if (nmuons_>1) {	
          mu1Pt_mu2Pt_.denominator->Fill(muons.at(0).pt(),muons.at(1).pt());
          mu1Eta_mu2Eta_.denominator->Fill(muons.at(0).eta(),muons.at(1).eta());
          invMass_mumu_.denominator->Fill(mll);
          invMass_mumu_variableBinning_.denominator->Fill(mll);
      }
      if(njets_>0){
          DeltaR_jet_Mu_.denominator -> Fill (deltaR(jets.at(0),muons.at(0)));
      }
  }


  if (njets_ > 0)      jetVsLS_.denominator -> Fill(ls, jets.at(0).pt());
  if (nelectrons_ > 0) {
      eleVsLS_.denominator -> Fill(ls, electrons.at(0).pt());
      elePt_eventHT_.denominator -> Fill (electrons.at(0).pt(), eventHT);
      if (njets_>0) elePt_jetPt_.denominator -> Fill (electrons.at(0).pt(), jets.at(0).pt());
      if (nmuons_>0) {
          elePt_muPt_.denominator->Fill(electrons.at(0).pt(),muons.at(0).pt());
          eleEta_muEta_.denominator->Fill(electrons.at(0).eta(),muons.at(0).eta());
      }
      if (nelectrons_>1) {
          ele1Pt_ele2Pt_.denominator->Fill(electrons.at(0).pt(),electrons.at(1).pt());
          ele1Eta_ele2Eta_.denominator->Fill(electrons.at(0).eta(),electrons.at(1).eta());
      }
  }


  // Marina
  if (nbjets_ > 0)     bjetVsLS_.denominator -> Fill(ls, bjets.begin()->first->pt());

  for (unsigned int iMu=0; iMu<muons.size(); ++iMu){
      if (iMu>=nmuons_) break;
      muPhi_.at(iMu).denominator  -> Fill(muons.at(iMu).phi());
      muEta_.at(iMu).denominator  -> Fill(muons.at(iMu).eta());
      muPt_.at(iMu).denominator   -> Fill(muons.at(iMu).pt() );
      muEta_variableBinning_.at(iMu).denominator  -> Fill(muons.at(iMu).eta());
      muPt_variableBinning_.at(iMu).denominator   -> Fill(muons.at(iMu).pt() );
      muPtEta_.at(iMu).denominator   -> Fill(muons.at(iMu).pt(), muons.at(iMu).eta() );
      muEtaPhi_.at(iMu).denominator   -> Fill(muons.at(iMu).eta(), muons.at(iMu).phi() );
  }
  for (unsigned int iEle=0; iEle<electrons.size(); ++iEle){
      if (iEle>=nelectrons_) break;
      elePhi_.at(iEle).denominator  -> Fill(electrons.at(iEle).phi());
      eleEta_.at(iEle).denominator  -> Fill(electrons.at(iEle).eta());
      elePt_.at(iEle).denominator   -> Fill(electrons.at(iEle).pt() );
      eleEta_variableBinning_.at(iEle).denominator  -> Fill(electrons.at(iEle).eta());
      elePt_variableBinning_.at(iEle).denominator   -> Fill(electrons.at(iEle).pt() );
      elePtEta_.at(iEle).denominator   -> Fill(electrons.at(iEle).pt(), electrons.at(iEle).eta() );
      eleEtaPhi_.at(iEle).denominator   -> Fill(electrons.at(iEle).eta(), electrons.at(iEle).phi() );
  }
  for (unsigned int iJet=0; iJet<jets.size(); ++iJet){
      if (iJet>=njets_) break;
      jetPhi_.at(iJet).denominator  -> Fill(jets.at(iJet).phi());
      jetEta_.at(iJet).denominator  -> Fill(jets.at(iJet).eta());
      jetPt_.at(iJet).denominator   -> Fill(jets.at(iJet).pt() );
      jetEta_variableBinning_.at(iJet).denominator  -> Fill(jets.at(iJet).eta());
      jetPt_variableBinning_.at(iJet).denominator   -> Fill(jets.at(iJet).pt() );
      jetPtEta_.at(iJet).denominator   -> Fill(jets.at(iJet).pt(), jets.at(iJet).eta() );
      jetEtaPhi_.at(iJet).denominator   -> Fill(jets.at(iJet).eta(), jets.at(iJet).phi() );
  }

  // Marina
  unsigned int iBJet = 0;
  for (auto & bjet: bjets){
    if (iBJet >=nbjets_) break;

    bjetPhi_.at(iBJet).denominator -> Fill(bjet.first->phi());
    bjetEta_.at(iBJet).denominator -> Fill(bjet.first->eta());
    bjetPt_.at(iBJet).denominator  -> Fill(bjet.first->pt());
    bjetCSV_.at(iBJet).denominator -> Fill(std::fmax(0.0, bjet.second));
    bjetEta_variableBinning_.at(iBJet).denominator -> Fill(bjet.first->eta());
    bjetPt_variableBinning_.at(iBJet).denominator  -> Fill(bjet.first->pt());
    bjetPtEta_.at(iBJet).denominator  -> Fill(bjet.first->pt(), bjet.first->eta());
    bjetEtaPhi_.at(iBJet).denominator -> Fill(bjet.first->eta(), bjet.first->phi());
    bjetCSVHT_.at(iBJet).denominator  -> Fill(std::fmax(0.0, bjet.second), eventHT);

    iBJet++;
  }



  if (jets.size() > 0){
      jetEtaPhi_HEP17_.denominator -> Fill (jets.at(0).eta(), jets.at(0).phi()); // for HEP17 monitorning
  }

  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  // filling histograms (num_genTriggerEventFlag_)
  metME_.numerator -> Fill(met);
  metME_variableBinning_.numerator -> Fill(met);
  metPhiME_.numerator -> Fill(phi);
  metVsLS_.numerator -> Fill(ls, met);
  htVsLS_.numerator -> Fill(ls, eventHT);
  eventHT_.numerator -> Fill(eventHT);
  eventHT_variableBinning_.numerator -> Fill(eventHT);
  eventMHT_.numerator -> Fill(eventMHT.pt());
  eventMHT_variableBinning_.numerator -> Fill(eventMHT.pt());


  if (nmuons_ > 0){
      muVsLS_.numerator -> Fill(ls, muons.at(0).pt());
      if (nmuons_>1) {
          mu1Pt_mu2Pt_.numerator->Fill(muons.at(0).pt(),muons.at(1).pt());
          mu1Eta_mu2Eta_.numerator->Fill(muons.at(0).eta(),muons.at(1).eta());
          invMass_mumu_.numerator->Fill(mll);
          invMass_mumu_variableBinning_.numerator->Fill(mll);
      }
      if(njets_>0){
          DeltaR_jet_Mu_.numerator -> Fill (deltaR(jets.at(0),muons.at(0)));
      }

  }
  if (njets_ > 0)      jetVsLS_.numerator -> Fill(ls, jets.at(0).pt());
  if (nelectrons_ > 0) {
      eleVsLS_.numerator -> Fill(ls, electrons.at(0).pt());
      elePt_eventHT_.numerator -> Fill (electrons.at(0).pt(), eventHT);
      if (njets_>0) elePt_jetPt_.numerator -> Fill (electrons.at(0).pt(), jets.at(0).pt());
      if (nmuons_>0) {
          elePt_muPt_.numerator->Fill(electrons.at(0).pt(),muons.at(0).pt());
          eleEta_muEta_.numerator->Fill(electrons.at(0).eta(),muons.at(0).eta());
      }
      if (nelectrons_>1) {
          ele1Pt_ele2Pt_.numerator->Fill(electrons.at(0).pt(),electrons.at(1).pt());
          ele1Eta_ele2Eta_.numerator->Fill(electrons.at(0).eta(),electrons.at(1).eta());
      }
  }

  // Marina
  if (nbjets_ > 0)     bjetVsLS_.numerator-> Fill(ls, bjets.begin()->first->pt());

  jetMulti_.numerator -> Fill(jets.size());
  eleMulti_.numerator -> Fill(electrons.size());
  muMulti_.numerator -> Fill(muons.size());
  // Marina
  bjetMulti_.numerator -> Fill(bjets.size());

  for (unsigned int iMu=0; iMu<muons.size(); ++iMu){
      if (iMu>=nmuons_) break;

      muPhi_.at(iMu).numerator  -> Fill(muons.at(iMu).phi());
      muEta_.at(iMu).numerator  -> Fill(muons.at(iMu).eta());
      muPt_.at(iMu).numerator   -> Fill(muons.at(iMu).pt() );
      muEta_variableBinning_.at(iMu).numerator  -> Fill(muons.at(iMu).eta());
      muPt_variableBinning_.at(iMu).numerator   -> Fill(muons.at(iMu).pt() );
      muPtEta_.at(iMu).numerator   -> Fill(muons.at(iMu).pt(), muons.at(iMu).eta() );
      muEtaPhi_.at(iMu).numerator   -> Fill(muons.at(iMu).eta(), muons.at(iMu).phi() );
  }
  for (unsigned int iEle=0; iEle<electrons.size(); ++iEle){
      if (iEle>=nelectrons_) break;
      elePhi_.at(iEle).numerator  -> Fill(electrons.at(iEle).phi());
      eleEta_.at(iEle).numerator  -> Fill(electrons.at(iEle).eta());
      elePt_.at(iEle).numerator   -> Fill(electrons.at(iEle).pt() );
      eleEta_variableBinning_.at(iEle).numerator  -> Fill(electrons.at(iEle).eta());
      elePt_variableBinning_.at(iEle).numerator   -> Fill(electrons.at(iEle).pt() );
      elePtEta_.at(iEle).numerator   -> Fill(electrons.at(iEle).pt(), electrons.at(iEle).eta() );
      eleEtaPhi_.at(iEle).numerator   -> Fill(electrons.at(iEle).eta(), electrons.at(iEle).phi() );
  }
  for (unsigned int iJet=0; iJet<jets.size(); ++iJet){
      if (iJet>=njets_) break;
      jetPhi_.at(iJet).numerator  -> Fill(jets.at(iJet).phi());
      jetEta_.at(iJet).numerator  -> Fill(jets.at(iJet).eta());
      jetPt_.at(iJet).numerator   -> Fill(jets.at(iJet).pt() );
      jetEta_variableBinning_.at(iJet).numerator  -> Fill(jets.at(iJet).eta());
      jetPt_variableBinning_.at(iJet).numerator   -> Fill(jets.at(iJet).pt() );
      jetPtEta_.at(iJet).numerator   -> Fill(jets.at(iJet).pt(), jets.at(iJet).eta() );
      jetEtaPhi_.at(iJet).numerator   -> Fill(jets.at(iJet).eta(), jets.at(iJet).phi() );
  }

  // Marina
  unsigned int j = 0;
  for (auto & bjet: bjets){
    if (j >=nbjets_) break;
    bjetPhi_.at(j).numerator -> Fill(bjet.first->pt());
    bjetEta_.at(j).numerator -> Fill(bjet.first->eta());
    bjetPt_.at(j).numerator  -> Fill(bjet.first->pt());
    bjetCSV_.at(j).numerator -> Fill(std::fmax(0.0,bjet.second));
    bjetEta_variableBinning_.at(j).numerator -> Fill(bjet.first->eta());
    bjetPt_variableBinning_.at(j).numerator  -> Fill(bjet.first->pt());
    bjetPtEta_.at(j).numerator  -> Fill(bjet.first->pt(), bjet.first->eta());
    bjetEtaPhi_.at(j).numerator -> Fill(bjet.first->eta(), bjet.first->phi());
    bjetCSVHT_.at(j).numerator  -> Fill(std::fmax(0.0,bjet.second), eventHT);

    j++;
  }


  if (jets.size() > 0){
      jetEtaPhi_HEP17_.numerator -> Fill (jets.at(0).eta(), jets.at(0).phi()); // for HEP17 monitorning
  }


}

void TopMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins",40);
  pset.add<double>( "xmin",0 );
  pset.add<double>( "xmax",100 );
}

void TopMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins", 2500);
}

void TopMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/TOP" );

  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  //Suvankar
  desc.add<edm::InputTag>( "vertices", edm::InputTag("offlinePrimaryVertices") );
  // Marina
  desc.add<edm::InputTag>( "btagalgo", edm::InputTag("pfCombinedSecondaryVertexV2BJetTags") );
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<std::string>("HTdefinition", "pt > 0");
  //Suvankar
  desc.add<std::string>("vertexSelection", "!isFake");
  desc.add<std::string>("bjetSelection","pt > 0");
  desc.add<unsigned int>("njets",      0);
  desc.add<unsigned int>("nelectrons", 0);
  desc.add<unsigned int>("nmuons",     0);
  desc.add<double>("leptJetDeltaRmin", 0);
  desc.add<double>("bJetMuDeltaRmax" , 9999.);
  desc.add<double>("bJetDeltaEtaMax" , 9999.);
  desc.add<double>("HTcut", 0);
  // Marina
  desc.add<unsigned int>("nbjets",     0);
  desc.add<double>("workingpoint",     0.8484); // medium CSV
  //Suvankar
  desc.add<bool>("applyleptonPVcuts", false);
  //george
  desc.add<double>("invMassUppercut",-1.0);
  desc.add<double>("invMassLowercut",-1.0);
  desc.add<bool>("oppositeSignMuons",false);
  desc.add<std::string>("MHTdefinition", "pt > 0");
  desc.add<double>("MHTcut", -1);

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
  edm::ParameterSetDescription phiPSet;
  edm::ParameterSetDescription etaPSet;
  edm::ParameterSetDescription ptPSet;
  edm::ParameterSetDescription htPSet;
  edm::ParameterSetDescription DRPSet;
  // Marina
  edm::ParameterSetDescription csvPSet;
  //george
 edm::ParameterSetDescription invMassPSet;
 edm::ParameterSetDescription MHTPSet;
  fillHistoPSetDescription(metPSet);
  fillHistoPSetDescription(phiPSet);
  fillHistoPSetDescription(ptPSet);
  fillHistoPSetDescription(etaPSet);
  fillHistoPSetDescription(htPSet);
  fillHistoPSetDescription(DRPSet);
  // Marina
  fillHistoPSetDescription(csvPSet);
  //george
  fillHistoPSetDescription(MHTPSet);
  fillHistoPSetDescription(invMassPSet);
  histoPSet.add<edm::ParameterSetDescription>("metPSet", metPSet);
  histoPSet.add<edm::ParameterSetDescription>("etaPSet", etaPSet);
  histoPSet.add<edm::ParameterSetDescription>("phiPSet", phiPSet);
  histoPSet.add<edm::ParameterSetDescription>("ptPSet", ptPSet);
  histoPSet.add<edm::ParameterSetDescription>("htPSet", htPSet);
  histoPSet.add<edm::ParameterSetDescription>("DRPSet", DRPSet);
  // Marina
  histoPSet.add<edm::ParameterSetDescription>("csvPSet", csvPSet);
  //george
  histoPSet.add<edm::ParameterSetDescription>("invMassPSet", invMassPSet);
  histoPSet.add<edm::ParameterSetDescription>("MHTPSet", MHTPSet);

  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.};
  std::vector<double> eta_bins = {-3.,-2.5,-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.,2.5,3.};
  histoPSet.add<std::vector<double> >("metBinning", bins);
  histoPSet.add<std::vector<double> >("HTBinning", bins);
  histoPSet.add<std::vector<double> >("jetPtBinning", bins);
  histoPSet.add<std::vector<double> >("elePtBinning", bins);
  histoPSet.add<std::vector<double> >("muPtBinning", bins);
  histoPSet.add<std::vector<double> >("jetEtaBinning", eta_bins);
  histoPSet.add<std::vector<double> >("eleEtaBinning", eta_bins);
  histoPSet.add<std::vector<double> >("muEtaBinning", eta_bins);
  //george
  histoPSet.add<std::vector<double> >("invMassVariableBinning", bins);
  histoPSet.add<std::vector<double> >("MHTVariableBinning", bins);

  std::vector<double> bins_2D = {0.,40.,80.,100.,120.,140.,160.,180.,200.,240.,280.,350.,450.,1000.};
  std::vector<double> eta_bins_2D = {-3.,-2.,-1.,0.,1.,2.,3.};
  std::vector<double> phi_bins_2D = {-3.1415,-2.5132,-1.8849,-1.2566,-0.6283,0,0.6283,1.2566,1.8849,2.5132,3.1415};
  histoPSet.add<std::vector<double> >("HTBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("jetPtBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("elePtBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("muPtBinning2D", bins_2D);
  histoPSet.add<std::vector<double> >("jetEtaBinning2D", eta_bins_2D);
  histoPSet.add<std::vector<double> >("eleEtaBinning2D", eta_bins_2D);
  histoPSet.add<std::vector<double> >("muEtaBinning2D", eta_bins_2D);
  histoPSet.add<std::vector<double> >("phiBinning2D", phi_bins_2D);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);
  //Suvankar
  edm::ParameterSetDescription lPVcutPSet;
  lPVcutPSet.add<double>( "dxy", 9999. );
  lPVcutPSet.add<double>( "dz",  9999. );
  desc.add<edm::ParameterSetDescription>("leptonPVcuts", lPVcutPSet);

  descriptions.add("topMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TopMonitor);


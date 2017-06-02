#include "DQMOffline/Trigger/plugins/JetMETMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


// Define Phi Bin //
double JetMET_MAX_PHI = 3.2;
int JetMET_N_PHI = 64;
MEbinning jetmet_phi_binning_{
  JetMET_N_PHI, -JetMET_MAX_PHI, JetMET_MAX_PHI
};
// Define Eta Bin //
double JetMET_MAX_ETA = 5;
int JetMET_N_ETA = 50;
MEbinning jetmet_eta_binning_{
  JetMET_N_ETA, -JetMET_MAX_ETA, JetMET_MAX_ETA
};



// -----------------------------
//  constructors and destructor
// -----------------------------

JetMETMonitor::JetMETMonitor( const edm::ParameterSet& iConfig ):
   metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , calojetSelection_ ( iConfig.getParameter<std::string>("calojetSelection") )
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") ) 
{
  folderName_            = iConfig.getParameter<std::string>("FolderName"); 
  metToken_              = consumes<reco::PFMETCollection>        (iConfig.getParameter<edm::InputTag>("met")       );
  pfjetToken_            = mayConsume<reco::PFJetCollection>      (iConfig.getParameter<edm::InputTag>("pfjets")    ); 
  calojetToken_          = mayConsume<reco::CaloJetCollection>    (iConfig.getParameter<edm::InputTag>("calojets")    ); 
  eleToken_              = mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons") );
  muoToken_              = mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     );     
  met_variable_binning_  = iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetptBinning");
  met_binning_           = getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("metPSet")    );
  ls_binning_            = getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     );

  num_genTriggerEventFlag_ = new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this);
  den_genTriggerEventFlag_ = new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this);

  njets_      = iConfig.getParameter<int>("njets" );
  nelectrons_ = iConfig.getParameter<int>("nelectrons" );
  nmuons_     = iConfig.getParameter<int>("nmuons" );
  ptcut_      = iConfig.getParameter<double>("ptcut" ); // for HLT Jet 
  isPFJetTrig    = iConfig.getParameter<bool>("ispfjettrg" );
  isCaloJetTrig  = iConfig.getParameter<bool>("iscalojettrg" );
  isMetTrig      = iConfig.getParameter<bool>("ismettrg" );
  jetmetME_.numerator   = nullptr;
  jetmetME_.denominator = nullptr;
  jetmetME_variableBinning_.numerator   = nullptr;
  jetmetME_variableBinning_.denominator = nullptr;
  jetmetVsLS_.numerator   = nullptr;
  jetmetVsLS_.denominator = nullptr;
  jetmetPhiME_.numerator   = nullptr;
  jetmetPhiME_.denominator = nullptr;
  if (isMetTrig == false) 
  {
     jetmetEtaME_.numerator   = nullptr;
     jetmetEtaME_.denominator = nullptr;
     
     jetmetEtaVsPhi_.numerator   = nullptr;
     jetmetEtaVsPhi_.denominator = nullptr;
     /// For Barrel
     jetmetHB_ME_.numerator   = nullptr;
     jetmetHB_ME_.denominator = nullptr;
     jetmetHB_ME_variableBinning_.numerator   = nullptr;
     jetmetHB_ME_variableBinning_.denominator = nullptr;
     jetmetHB_VsLS_.numerator   = nullptr;
     jetmetHB_VsLS_.denominator = nullptr;
     jetmetHB_PhiME_.numerator   = nullptr;
     jetmetHB_PhiME_.denominator = nullptr;
     jetmetHB_EtaME_.numerator   = nullptr;
     jetmetHB_EtaME_.denominator = nullptr;
     jetmetHB_EtaVsPhi_.numerator   = nullptr;
     jetmetHB_EtaVsPhi_.denominator = nullptr;
     
     /// For Endcap
     jetmetHE_ME_.numerator   = nullptr;
     jetmetHE_ME_.denominator = nullptr;
     jetmetHE_ME_variableBinning_.numerator   = nullptr;
     jetmetHE_ME_variableBinning_.denominator = nullptr;
     jetmetHE_VsLS_.numerator   = nullptr;
     jetmetHE_VsLS_.denominator = nullptr;
     jetmetHE_PhiME_.numerator   = nullptr;
     jetmetHE_PhiME_.denominator = nullptr;
     jetmetHE_EtaME_.numerator   = nullptr;
     jetmetHE_EtaME_.denominator = nullptr;
     jetmetHE_EtaVsPhi_.numerator   = nullptr;
     jetmetHE_EtaVsPhi_.denominator = nullptr;
     
     /// For Endcap_plus ///
     jetmetHE_p_ME_.numerator   = nullptr;
     jetmetHE_p_ME_.denominator = nullptr;
     jetmetHE_p_ME_variableBinning_.numerator   = nullptr;
     jetmetHE_p_ME_variableBinning_.denominator = nullptr;
     jetmetHE_p_VsLS_.numerator   = nullptr;
     jetmetHE_p_VsLS_.denominator = nullptr;
     jetmetHE_p_PhiME_.numerator   = nullptr;
     jetmetHE_p_PhiME_.denominator = nullptr;
     jetmetHE_p_EtaME_.numerator   = nullptr;
     jetmetHE_p_EtaME_.denominator = nullptr;
     jetmetHE_p_EtaVsPhi_.numerator   = nullptr;
     jetmetHE_p_EtaVsPhi_.denominator = nullptr;
     
     /// For Endcap_minus ///
     jetmetHE_m_ME_.numerator   = nullptr;
     jetmetHE_m_ME_.denominator = nullptr;
     jetmetHE_m_ME_variableBinning_.numerator   = nullptr;
     jetmetHE_m_ME_variableBinning_.denominator = nullptr;
     jetmetHE_m_VsLS_.numerator   = nullptr;
     jetmetHE_m_VsLS_.denominator = nullptr;
     jetmetHE_m_PhiME_.numerator   = nullptr;
     jetmetHE_m_PhiME_.denominator = nullptr;
     jetmetHE_m_EtaME_.numerator   = nullptr;
     jetmetHE_m_EtaME_.denominator = nullptr;
     jetmetHE_m_EtaVsPhi_.numerator   = nullptr;
     jetmetHE_m_EtaVsPhi_.denominator = nullptr;
     
     /// For Forward ///
     jetmetHF_ME_.numerator   = nullptr;
     jetmetHF_ME_.denominator = nullptr;
     jetmetHF_ME_variableBinning_.numerator   = nullptr;
     jetmetHF_ME_variableBinning_.denominator = nullptr;
     jetmetHF_VsLS_.numerator   = nullptr;
     jetmetHF_VsLS_.denominator = nullptr;
     jetmetHF_PhiME_.numerator   = nullptr;
     jetmetHF_PhiME_.denominator = nullptr;
     jetmetHF_EtaME_.numerator   = nullptr;
     jetmetHF_EtaME_.denominator = nullptr;
     jetmetHF_EtaVsPhi_.numerator   = nullptr;
     jetmetHF_EtaVsPhi_.denominator = nullptr;
  
     /// For HEP17///
     jetmetHEP17_ME_.numerator   = nullptr;
     jetmetHEP17_ME_.denominator = nullptr;
     jetmetHEP17_ME_variableBinning_.numerator   = nullptr;
     jetmetHEP17_ME_variableBinning_.denominator = nullptr;
     jetmetHEP17_VsLS_.numerator   = nullptr;
     jetmetHEP17_VsLS_.denominator = nullptr;
     jetmetHEP17_PhiME_.numerator   = nullptr;
     jetmetHEP17_PhiME_.denominator = nullptr;
     jetmetHEP17_EtaME_.numerator   = nullptr;
     jetmetHEP17_EtaME_.denominator = nullptr;
     jetmetHEP17_EtaVsPhi_.numerator   = nullptr;
     jetmetHEP17_EtaVsPhi_.denominator = nullptr;
  
     /// For HEM17///
     jetmetHEM17_ME_.numerator   = nullptr;
     jetmetHEM17_ME_.denominator = nullptr;
     jetmetHEM17_ME_variableBinning_.numerator   = nullptr;
     jetmetHEM17_ME_variableBinning_.denominator = nullptr;
     jetmetHEM17_VsLS_.numerator   = nullptr;
     jetmetHEM17_VsLS_.denominator = nullptr;
     jetmetHEM17_PhiME_.numerator   = nullptr;
     jetmetHEM17_PhiME_.denominator = nullptr;
     jetmetHEM17_EtaME_.numerator   = nullptr;
     jetmetHEM17_EtaME_.denominator = nullptr;
     jetmetHEM17_EtaVsPhi_.numerator   = nullptr;
     jetmetHEM17_EtaVsPhi_.denominator = nullptr;
  
     /// For HEP18///
     jetmetHEP18_ME_.numerator   = nullptr;
     jetmetHEP18_ME_.denominator = nullptr;
     jetmetHEP18_ME_variableBinning_.numerator   = nullptr;
     jetmetHEP18_ME_variableBinning_.denominator = nullptr;
     jetmetHEP18_VsLS_.numerator   = nullptr;
     jetmetHEP18_VsLS_.denominator = nullptr;
     jetmetHEP18_PhiME_.numerator   = nullptr;
     jetmetHEP18_PhiME_.denominator = nullptr;
     jetmetHEP18_EtaME_.numerator   = nullptr;
     jetmetHEP18_EtaME_.denominator = nullptr;
     jetmetHEP18_EtaVsPhi_.numerator   = nullptr;
     jetmetHEP18_EtaVsPhi_.denominator = nullptr;
  }
}

JetMETMonitor::~JetMETMonitor()
{
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

MEbinning JetMETMonitor::getHistoPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

MEbinning JetMETMonitor::getHistoLSPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      0.,
      double(pset.getParameter<int32_t>("nbins"))
      };
}

void JetMETMonitor::setMETitle(JetMETME& me, std::string titleX, std::string titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}
void JetMETMonitor::bookME(DQMStore::IBooker &ibooker, JetMETME& me, std::string& histname, std::string& histtitle, int& nbins, double& min, double& max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void JetMETMonitor::bookME(DQMStore::IBooker &ibooker, JetMETME& me, std::string& histname, std::string& histtitle, std::vector<double> binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void JetMETMonitor::bookME(DQMStore::IBooker &ibooker, JetMETME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, double& ymin, double& ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void JetMETMonitor::bookME(DQMStore::IBooker &ibooker, JetMETME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, int& nbinsY, double& ymin, double& ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void JetMETMonitor::bookME(DQMStore::IBooker &ibooker, JetMETME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY)
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

void JetMETMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;
  std::string hist_obtag = "";
  std::string histtitle_obtag = "";
  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());

  if (isPFJetTrig) {hist_obtag = "pfjet";          histtitle_obtag =  "PFJet";}
  else if (isCaloJetTrig) {hist_obtag = "calojet"; histtitle_obtag =  "CaloJet"; }
  else if (isMetTrig) {hist_obtag = "pfmet";       histtitle_obtag =  "PFMET"; }
  else {hist_obtag = "pfjet"; histtitle_obtag =  "PFJet"; } //default is pfjet 
  /// Histograms ///
  histname = hist_obtag + "pT"; histtitle = histtitle_obtag + " pT";
  bookME(ibooker,jetmetME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
  setMETitle(jetmetME_, histtitle_obtag +" pT [GeV]","events / [GeV]");
  
  histname = hist_obtag + "_variable"; histtitle = histtitle_obtag +" pT";
  bookME(ibooker,jetmetME_variableBinning_,histname,histtitle,met_variable_binning_);
  setMETitle(jetmetME_variableBinning_,histtitle_obtag + "pT [GeV]","events / [GeV]");
  
  histname = hist_obtag + "pTVsLS"; histtitle = histtitle_obtag + "PFJet vs LS";
  bookME(ibooker,jetmetVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
  setMETitle(jetmetVsLS_,"LS",histtitle_obtag + "pT [GeV]");
  
  histname = hist_obtag + "Phi"; histtitle = histtitle_obtag + " phi";
  bookME(ibooker,jetmetPhiME_,histname,histtitle, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
  setMETitle(jetmetPhiME_,histtitle_obtag +" #phi","events / 0.1 rad");
  
  if ( isMetTrig == false ){
     histname = hist_obtag +"Eta"; histtitle = histtitle_obtag + " eta";
     bookME(ibooker,jetmetEtaME_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax);
     setMETitle(jetmetEtaME_,histtitle_obtag + " #eta","events / #eta");
  
     histname = hist_obtag +"EtaVsPhi"; histtitle = histtitle_obtag + " eta Vs phi ";
     bookME(ibooker,jetmetEtaVsPhi_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetEtaVsPhi_,histtitle_obtag + " #eta","#phi");
  
     /// Histograms For Barrel ///
     histname = hist_obtag +"pT_HB"; histtitle = histtitle_obtag + " pT (HB)";
     bookME(ibooker,jetmetHB_ME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHB_ME_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname =  hist_obtag + "pT_variable_HB"; histtitle = histtitle_obtag + " pT (HB)";
     bookME(ibooker,jetmetHB_ME_variableBinning_,histname,histtitle,met_variable_binning_);
     setMETitle(jetmetHB_ME_variableBinning_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag + "pTVsLS_HB"; histtitle = histtitle_obtag + " vs LS (HB)";
     bookME(ibooker,jetmetHB_VsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHB_VsLS_,"LS",histtitle_obtag + " [GeV]");
     
     histname = hist_obtag + "phi_HB"; histtitle = histtitle_obtag + " phi (HB)";
     bookME(ibooker,jetmetHB_PhiME_,histname,histtitle, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHB_PhiME_,histtitle_obtag + " #phi","events / 0.1 rad");
     histname = hist_obtag + "Eta_HB"; histtitle = histtitle_obtag + " eta (HB)";
     bookME(ibooker,jetmetHB_EtaME_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax);
     setMETitle(jetmetHB_EtaME_,histtitle_obtag + " #eta","events / #eta");
 
     histname = hist_obtag +"EtaVsPhi_HB"; histtitle = histtitle_obtag + " eta Vs phi (HB)";
     bookME(ibooker,jetmetHB_EtaVsPhi_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHB_EtaVsPhi_,histtitle_obtag + " #eta","#phi");
 
     /// Histograms For Endcap ///
     histname = hist_obtag +"pT_HE"; histtitle = histtitle_obtag +" pT (HE)";
     bookME(ibooker,jetmetHE_ME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHE_ME_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pT_variable_HE"; histtitle = histtitle_obtag + " pT (HE)";
     bookME(ibooker,jetmetHE_ME_variableBinning_,histname,histtitle,met_variable_binning_);
     setMETitle(jetmetHE_ME_variableBinning_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pTVsLS_HE"; histtitle = histtitle_obtag + " vs LS (HE)";
     bookME(ibooker,jetmetHE_VsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHE_VsLS_,"LS",histtitle_obtag + " [GeV]");
     
     histname = hist_obtag +"Phi_HE"; histtitle = histtitle_obtag + " phi (HE)";
     bookME(ibooker,jetmetHE_PhiME_,histname,histtitle, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHE_PhiME_,histtitle_obtag + " #phi","events / 0.1 rad");
     
     histname = hist_obtag +"Eta_HE"; histtitle = histtitle_obtag + " eta (HE)";
     bookME(ibooker,jetmetHE_EtaME_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax);
     setMETitle(jetmetHE_EtaME_,histtitle_obtag + " #eta","events / #eta");
     
     histname = hist_obtag +"EtaVsPhi_HE"; histtitle = histtitle_obtag + " eta Vs phi (HE)";
     bookME(ibooker,jetmetHE_EtaVsPhi_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHE_EtaVsPhi_,histtitle_obtag + " #eta","#phi");

     /// Histograms For Endcap_plus ///
     histname = hist_obtag +"pT_HE_p"; histtitle = histtitle_obtag + " pT (HE+)";
     bookME(ibooker,jetmetHE_p_ME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHE_p_ME_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pT_variable_HE_p"; histtitle = histtitle_obtag + " pT (HE+)";
     bookME(ibooker,jetmetHE_p_ME_variableBinning_,histname,histtitle,met_variable_binning_);
     setMETitle(jetmetHE_p_ME_variableBinning_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pTVsLS_HE_p"; histtitle = histtitle_obtag + " vs LS (HE+)";
     bookME(ibooker,jetmetHE_p_VsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHE_p_VsLS_,"LS",histtitle_obtag + " [GeV]");
     
     histname = hist_obtag +"Phi_HE_p"; histtitle = histtitle_obtag + " phi (HE+)";
     bookME(ibooker,jetmetHE_p_PhiME_,histname,histtitle, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHE_p_PhiME_,histtitle_obtag + " #phi","events / 0.1 rad");
     
     histname = hist_obtag +"Eta_HE_p"; histtitle = histtitle_obtag + " eta (HE+)";
     bookME(ibooker,jetmetHE_p_EtaME_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax);
     setMETitle(jetmetHE_p_EtaME_,histtitle_obtag + " #eta","events / #eta");

     histname = hist_obtag +"EtaVsPhi_HE_p"; histtitle = histtitle_obtag + " eta Vs phi (HE+)";
     bookME(ibooker,jetmetHE_p_EtaVsPhi_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHE_p_EtaVsPhi_,histtitle_obtag + " #eta","#phi");

     /// Histograms For Endcap_minus ///
     histname = hist_obtag +"pT_HE_m"; histtitle = histtitle_obtag + " pT (HE-)";
     bookME(ibooker,jetmetHE_m_ME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHE_m_ME_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pT_variable_HE_m"; histtitle = histtitle_obtag + " pT (HE-)";
     bookME(ibooker,jetmetHE_m_ME_variableBinning_,histname,histtitle,met_variable_binning_);
     setMETitle(jetmetHE_m_ME_variableBinning_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pTVsLS_HE_m"; histtitle = histtitle_obtag + " vs LS (HE-)";
     bookME(ibooker,jetmetHE_m_VsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHE_m_VsLS_,"LS",histtitle_obtag + " [GeV]");
     
     histname = hist_obtag +"Phi_HE_m"; histtitle = histtitle_obtag + " phi (HE-)";
     bookME(ibooker,jetmetHE_m_PhiME_,histname,histtitle, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHE_m_PhiME_,histtitle_obtag + " #phi","events / 0.1 rad");
     
     histname = hist_obtag +"Eta_HE_m"; histtitle = histtitle_obtag + " eta (HE-)";
     bookME(ibooker,jetmetHE_m_EtaME_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax);
     setMETitle(jetmetHE_m_EtaME_,histtitle_obtag + " #eta","events / #eta");

     histname = hist_obtag +"EtaVsPhi_HE_m"; histtitle = histtitle_obtag + " eta Vs phi (HE-)";
     bookME(ibooker,jetmetHE_m_EtaVsPhi_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHE_m_EtaVsPhi_,histtitle_obtag + " #eta","#phi");

     /// Histograms For Forward ///
     histname = hist_obtag +"pT_HF"; histtitle = histtitle_obtag + " pT (HF)";
     bookME(ibooker,jetmetHF_ME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHF_ME_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pT_variable_HF"; histtitle = histtitle_obtag + " pT (HF)";
     bookME(ibooker,jetmetHF_ME_variableBinning_,histname,histtitle,met_variable_binning_);
     setMETitle(jetmetHF_ME_variableBinning_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pTVsLS_HF"; histtitle = histtitle_obtag + " vs LS (HF)";
     bookME(ibooker,jetmetHF_VsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHF_VsLS_,"LS",histtitle_obtag + " [GeV]");
     
     histname = hist_obtag +"Phi_HF"; histtitle = histtitle_obtag + " phi (HF)";
     bookME(ibooker,jetmetHF_PhiME_,histname,histtitle, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHF_PhiME_,histtitle_obtag + " #phi","events / 0.1 rad");
  
     histname = hist_obtag +"Eta_HF"; histtitle = histtitle_obtag + " eta (HF)";
     bookME(ibooker,jetmetHF_EtaME_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax);
     setMETitle(jetmetHF_EtaME_,histtitle_obtag + " #eta","events / #eta");
  
     histname = hist_obtag +"EtaVsPhi_HF"; histtitle = histtitle_obtag + " eta Vs phi (HF)";
     bookME(ibooker,jetmetHF_EtaVsPhi_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHF_EtaVsPhi_,histtitle_obtag + " #eta","#phi");

     /// Histograms For HEP17 ///
     histname = hist_obtag +"pT_HEP17"; histtitle = histtitle_obtag + " pT (HEP17)";
     bookME(ibooker,jetmetHEP17_ME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHEP17_ME_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pT_variable_HEP17"; histtitle = histtitle_obtag + " pT (HEP17)";
     bookME(ibooker,jetmetHEP17_ME_variableBinning_,histname,histtitle,met_variable_binning_);
     setMETitle(jetmetHEP17_ME_variableBinning_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pTVsLS_HEP17"; histtitle = histtitle_obtag + " vs LS (HEP17)";
     bookME(ibooker,jetmetHEP17_VsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHEP17_VsLS_,"LS",histtitle_obtag + " [GeV]");
     
     histname = hist_obtag +"Phi_HEP17"; histtitle = histtitle_obtag + " phi (HEP17)";
     bookME(ibooker,jetmetHEP17_PhiME_,histname,histtitle, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHEP17_PhiME_,histtitle_obtag + " #phi","events / 0.1 rad");
  
     histname = hist_obtag +"Eta_HEP17"; histtitle = histtitle_obtag + " eta (HEP17)";
     bookME(ibooker,jetmetHEP17_EtaME_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax);
     setMETitle(jetmetHEP17_EtaME_,histtitle_obtag + " #eta","events / #eta");
  
     histname = hist_obtag +"EtaVsPhi_HEP17"; histtitle = histtitle_obtag + " eta Vs phi (HEP17)";
     bookME(ibooker,jetmetHEP17_EtaVsPhi_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHEP17_EtaVsPhi_,histtitle_obtag + " #eta","#phi");

     /// Histograms For HEM17 ///
     histname = hist_obtag +"pT_HEM17"; histtitle = histtitle_obtag + " pT (HEM17)";
     bookME(ibooker,jetmetHEM17_ME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHEM17_ME_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pT_variable_HEM17"; histtitle = histtitle_obtag + " pT (HEM17)";
     bookME(ibooker,jetmetHEM17_ME_variableBinning_,histname,histtitle,met_variable_binning_);
     setMETitle(jetmetHEM17_ME_variableBinning_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pTVsLS_HEM17"; histtitle = histtitle_obtag + " vs LS (HEM17)";
     bookME(ibooker,jetmetHEM17_VsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHEM17_VsLS_,"LS",histtitle_obtag + " [GeV]");
     
     histname = hist_obtag +"Phi_HEM17"; histtitle = histtitle_obtag + " phi (HEM17)";
     bookME(ibooker,jetmetHEM17_PhiME_,histname,histtitle, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHEM17_PhiME_,histtitle_obtag + " #phi","events / 0.1 rad");
  
     histname = hist_obtag +"Eta_HEM17"; histtitle = histtitle_obtag + " eta (HEM17)";
     bookME(ibooker,jetmetHEM17_EtaME_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax);
     setMETitle(jetmetHEM17_EtaME_,histtitle_obtag + " #eta","events / #eta");
  
     histname = hist_obtag +"EtaVsPhi_HEM17"; histtitle = histtitle_obtag + " eta Vs phi (HEM17)";
     bookME(ibooker,jetmetHEM17_EtaVsPhi_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHEM17_EtaVsPhi_,histtitle_obtag + " #eta","#phi");

     /// Histograms For HEP18 ///
     histname = hist_obtag +"pT_HEP18"; histtitle = histtitle_obtag + " pT (HEP18)";
     bookME(ibooker,jetmetHEP18_ME_,histname,histtitle,met_binning_.nbins,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHEP18_ME_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pT_variable_HEP18"; histtitle = histtitle_obtag + " pT (HEP18)";
     bookME(ibooker,jetmetHEP18_ME_variableBinning_,histname,histtitle,met_variable_binning_);
     setMETitle(jetmetHEP18_ME_variableBinning_,histtitle_obtag + " pT [GeV]","events / [GeV]");
     
     histname = hist_obtag +"pTVsLS_HEP18"; histtitle = histtitle_obtag + " vs LS (HEP18)";
     bookME(ibooker,jetmetHEP18_VsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,met_binning_.xmin, met_binning_.xmax);
     setMETitle(jetmetHEP18_VsLS_,"LS",histtitle_obtag + " [GeV]");
     
     histname = hist_obtag +"Phi_HEP18"; histtitle = histtitle_obtag + " phi (HEP18)";
     bookME(ibooker,jetmetHEP18_PhiME_,histname,histtitle, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHEP18_PhiME_,histtitle_obtag + " #phi","events / 0.1 rad");
  
     histname = hist_obtag +"Eta_HEP18"; histtitle = histtitle_obtag + " eta (HEP18)";
     bookME(ibooker,jetmetHEP18_EtaME_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax);
     setMETitle(jetmetHEP18_EtaME_,histtitle_obtag + " #eta","events / #eta");

     histname = hist_obtag +"EtaVsPhi_HEP18"; histtitle = histtitle_obtag + " eta Vs phi (HEP18)";
     bookME(ibooker,jetmetHEP18_EtaVsPhi_,histname,histtitle, jetmet_eta_binning_.nbins, jetmet_eta_binning_.xmin, jetmet_eta_binning_.xmax, jetmet_phi_binning_.nbins, jetmet_phi_binning_.xmin, jetmet_phi_binning_.xmax);
     setMETitle(jetmetHEP18_EtaVsPhi_,histtitle_obtag + " #eta","#phi");

  }
  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/Math/interface/deltaR.h" // For Delta R
void JetMETMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {
  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  // PF Jet Trigger //
  if (isPFJetTrig) {
     edm::Handle<reco::PFJetCollection> jetHandle;
     iEvent.getByToken( pfjetToken_, jetHandle );
     std::vector<reco::PFJet> jets;
     if ( int(jetHandle->size()) < njets_ ) return;
     for ( auto const & jet : *jetHandle ) {
       jets.push_back(jet);
     }
     if ( int(jets.size()) < njets_ ) return;
   
     int ls = iEvent.id().luminosityBlock();
     // filling histograms (denominator)

     if (jets.size() < 1){return;}

     float jetpt_  = jets[0].pt(); 
     float jetphi_ = jets[0].phi(); 
     float jeteta_ = jets[0].eta();

     jetmetME_.denominator -> Fill(jetpt_);
     jetmetME_variableBinning_.denominator -> Fill(jetpt_);
     jetmetPhiME_.denominator -> Fill(jetphi_);
     jetmetEtaME_.denominator -> Fill(jeteta_);
     jetmetVsLS_.denominator -> Fill(ls, jetpt_);
     jetmetEtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);

     if (isBarrel( jeteta_ ) )
     {
        jetmetHB_ME_.denominator -> Fill(jetpt_);
        jetmetHB_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHB_PhiME_.denominator -> Fill(jetphi_);
        jetmetHB_EtaME_.denominator -> Fill(jeteta_);
        jetmetHB_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHB_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isEndCapP( jeteta_ ) )
     {
        jetmetHE_ME_.denominator -> Fill(jetpt_);
        jetmetHE_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHE_PhiME_.denominator -> Fill(jetphi_);
        jetmetHE_EtaME_.denominator -> Fill(jeteta_);
        jetmetHE_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHE_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
   
        jetmetHE_p_ME_.denominator -> Fill(jetpt_);
        jetmetHE_p_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHE_p_PhiME_.denominator -> Fill(jetphi_);
        jetmetHE_p_EtaME_.denominator -> Fill(jeteta_);
        jetmetHE_p_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHE_p_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isEndCapM( jeteta_ ) )
     {
        jetmetHE_ME_.denominator -> Fill(jetpt_);
        jetmetHE_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHE_PhiME_.denominator -> Fill(jetphi_);
        jetmetHE_EtaME_.denominator -> Fill(jeteta_);
        jetmetHE_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHE_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
   
        jetmetHE_m_ME_.denominator -> Fill(jetpt_);
        jetmetHE_m_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHE_m_PhiME_.denominator -> Fill(jetphi_);
        jetmetHE_m_EtaME_.denominator -> Fill(jeteta_);
        jetmetHE_m_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHE_m_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isForward( jeteta_ ) )
     {
   
        jetmetHF_ME_.denominator -> Fill(jetpt_);
        jetmetHF_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHF_PhiME_.denominator -> Fill(jetphi_);
        jetmetHF_EtaME_.denominator -> Fill(jeteta_);
        jetmetHF_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHF_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isHEP17( jeteta_, jetphi_ ) )
     {
        jetmetHEP17_ME_.denominator -> Fill(jetpt_);
        jetmetHEP17_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHEP17_PhiME_.denominator -> Fill(jetphi_);
        jetmetHEP17_EtaME_.denominator -> Fill(jeteta_);
        jetmetHEP17_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHEP17_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isHEM17( jeteta_, jetphi_ ) )
     {
        jetmetHEM17_ME_.denominator -> Fill(jetpt_);
        jetmetHEM17_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHEM17_PhiME_.denominator -> Fill(jetphi_);
        jetmetHEM17_EtaME_.denominator -> Fill(jeteta_);
        jetmetHEM17_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHEM17_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isHEP18( jeteta_, jetphi_ ) )
     {
        jetmetHEP18_ME_.denominator -> Fill(jetpt_);
        jetmetHEP18_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHEP18_PhiME_.denominator -> Fill(jetphi_);
        jetmetHEP18_EtaME_.denominator -> Fill(jeteta_);
        jetmetHEP18_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHEP18_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     } 

     if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return; // Require Numerator //
     jetmetME_.numerator -> Fill(jetpt_);
     jetmetME_variableBinning_.numerator -> Fill(jetpt_);
     jetmetPhiME_.numerator -> Fill(jetphi_);
     jetmetEtaME_.numerator -> Fill(jeteta_);
     jetmetVsLS_.numerator -> Fill(ls, jetpt_);
     jetmetEtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);

     if (isBarrel( jeteta_ ) )
     {
        jetmetHB_ME_.numerator -> Fill(jetpt_);
        jetmetHB_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHB_PhiME_.numerator -> Fill(jetphi_);
        jetmetHB_EtaME_.numerator -> Fill(jeteta_);
        jetmetHB_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHB_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isEndCapP( jeteta_ ) )
     {
        jetmetHE_ME_.numerator -> Fill(jetpt_);
        jetmetHE_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHE_PhiME_.numerator -> Fill(jetphi_);
        jetmetHE_EtaME_.numerator -> Fill(jeteta_);
        jetmetHE_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHE_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
   
        jetmetHE_p_ME_.numerator -> Fill(jetpt_);
        jetmetHE_p_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHE_p_PhiME_.numerator -> Fill(jetphi_);
        jetmetHE_p_EtaME_.numerator -> Fill(jeteta_);
        jetmetHE_p_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHE_p_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isEndCapM( jeteta_ ) )
     {
        jetmetHE_ME_.numerator -> Fill(jetpt_);
        jetmetHE_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHE_PhiME_.numerator -> Fill(jetphi_);
        jetmetHE_EtaME_.numerator -> Fill(jeteta_);
        jetmetHE_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHE_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
   
        jetmetHE_m_ME_.numerator -> Fill(jetpt_);
        jetmetHE_m_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHE_m_PhiME_.numerator -> Fill(jetphi_);
        jetmetHE_m_EtaME_.numerator -> Fill(jeteta_);
        jetmetHE_m_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHE_m_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isForward( jeteta_ ) )
     {
        jetmetHF_ME_.numerator -> Fill(jetpt_);
        jetmetHF_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHF_PhiME_.numerator -> Fill(jetphi_);
        jetmetHF_EtaME_.numerator -> Fill(jeteta_);
        jetmetHF_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHF_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isHEP17( jeteta_, jetphi_ ) )
     {
        jetmetHEP17_ME_.numerator -> Fill(jetpt_);
        jetmetHEP17_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHEP17_PhiME_.numerator -> Fill(jetphi_);
        jetmetHEP17_EtaME_.numerator -> Fill(jeteta_);
        jetmetHEP17_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHEP17_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isHEM17( jeteta_, jetphi_ ) )
     {
        jetmetHEM17_ME_.numerator -> Fill(jetpt_);
        jetmetHEM17_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHEM17_PhiME_.numerator -> Fill(jetphi_);
        jetmetHEM17_EtaME_.numerator -> Fill(jeteta_);
        jetmetHEM17_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHEM17_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isHEP18( jeteta_, jetphi_ ) )
     {
        jetmetHEP18_ME_.numerator -> Fill(jetpt_);
        jetmetHEP18_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHEP18_PhiME_.numerator -> Fill(jetphi_);
        jetmetHEP18_EtaME_.numerator -> Fill(jeteta_);
        jetmetHEP18_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHEP18_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     } 
  }// PFJet Trigger 
  // CaloJet Trigger
  if (isCaloJetTrig) {
     edm::Handle<reco::CaloJetCollection> jetHandle;
     iEvent.getByToken( calojetToken_, jetHandle );
     std::vector<reco::CaloJet> jets;
     if ( int(jetHandle->size()) < njets_ ) return;
     for ( auto const & jet : *jetHandle ) {
       jets.push_back(jet);
     }
     if ( int(jets.size()) < njets_ ) return;
   
     int ls = iEvent.id().luminosityBlock();
     // filling histograms (denominator)

     if (jets.size()<1){return;}

     float jetpt_  = jets[0].pt(); 
     float jetphi_ = jets[0].phi(); 
     float jeteta_ = jets[0].eta();
     jetmetME_.denominator -> Fill(jetpt_);
     jetmetME_variableBinning_.denominator -> Fill(jetpt_);
     jetmetPhiME_.denominator -> Fill(jetphi_);
     jetmetEtaME_.denominator -> Fill(jeteta_);
     jetmetVsLS_.denominator -> Fill(ls, jetpt_);
     jetmetEtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
   
     if (isBarrel( jeteta_ ) )
     {
        jetmetHB_ME_.denominator -> Fill(jetpt_);
        jetmetHB_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHB_PhiME_.denominator -> Fill(jetphi_);
        jetmetHB_EtaME_.denominator -> Fill(jeteta_);
        jetmetHB_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHB_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isEndCapP( jeteta_ ) )
     {
        jetmetHE_ME_.denominator -> Fill(jetpt_);
        jetmetHE_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHE_PhiME_.denominator -> Fill(jetphi_);
        jetmetHE_EtaME_.denominator -> Fill(jeteta_);
        jetmetHE_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHE_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
   
        jetmetHE_p_ME_.denominator -> Fill(jetpt_);
        jetmetHE_p_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHE_p_PhiME_.denominator -> Fill(jetphi_);
        jetmetHE_p_EtaME_.denominator -> Fill(jeteta_);
        jetmetHE_p_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHE_p_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isEndCapM( jeteta_ ) )
     {
        jetmetHE_ME_.denominator -> Fill(jetpt_);
        jetmetHE_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHE_PhiME_.denominator -> Fill(jetphi_);
        jetmetHE_EtaME_.denominator -> Fill(jeteta_);
        jetmetHE_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHE_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
   
        jetmetHE_m_ME_.denominator -> Fill(jetpt_);
        jetmetHE_m_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHE_m_PhiME_.denominator -> Fill(jetphi_);
        jetmetHE_m_EtaME_.denominator -> Fill(jeteta_);
        jetmetHE_m_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHE_m_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isForward( jeteta_ ) )
     {
   
        jetmetHF_ME_.denominator -> Fill(jetpt_);
        jetmetHF_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHF_PhiME_.denominator -> Fill(jetphi_);
        jetmetHF_EtaME_.denominator -> Fill(jeteta_);
        jetmetHF_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHF_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isHEP17( jeteta_, jetphi_ ) )
     {
        jetmetHEP17_ME_.denominator -> Fill(jetpt_);
        jetmetHEP17_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHEP17_PhiME_.denominator -> Fill(jetphi_);
        jetmetHEP17_EtaME_.denominator -> Fill(jeteta_);
        jetmetHEP17_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHEP17_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isHEM17( jeteta_, jetphi_ ) )
     {
        jetmetHEM17_ME_.denominator -> Fill(jetpt_);
        jetmetHEM17_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHEM17_PhiME_.denominator -> Fill(jetphi_);
        jetmetHEM17_EtaME_.denominator -> Fill(jeteta_);
        jetmetHEM17_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHEM17_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     }
     if (isHEP18( jeteta_, jetphi_ ) )
     {
        jetmetHEP18_ME_.denominator -> Fill(jetpt_);
        jetmetHEP18_ME_variableBinning_.denominator -> Fill(jetpt_);
        jetmetHEP18_PhiME_.denominator -> Fill(jetphi_);
        jetmetHEP18_EtaME_.denominator -> Fill(jeteta_);
        jetmetHEP18_VsLS_.denominator -> Fill(ls, jetpt_);
        jetmetHEP18_EtaVsPhi_.denominator -> Fill(jeteta_,jetphi_);
     } 
     // applying selection for numerator
     if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return; // Require Numerator //

     jetmetME_.numerator -> Fill(jetpt_);
     jetmetME_variableBinning_.numerator -> Fill(jetpt_);
     jetmetPhiME_.numerator -> Fill(jetphi_);
     jetmetEtaME_.numerator -> Fill(jeteta_);
     jetmetVsLS_.numerator -> Fill(ls, jetpt_);
     jetmetEtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
   
     if (isBarrel( jeteta_ ) )
     {
        jetmetHB_ME_.numerator -> Fill(jetpt_);
        jetmetHB_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHB_PhiME_.numerator -> Fill(jetphi_);
        jetmetHB_EtaME_.numerator -> Fill(jeteta_);
        jetmetHB_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHB_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isEndCapP( jeteta_ ) )
     {
        jetmetHE_ME_.numerator -> Fill(jetpt_);
        jetmetHE_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHE_PhiME_.numerator -> Fill(jetphi_);
        jetmetHE_EtaME_.numerator -> Fill(jeteta_);
        jetmetHE_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHE_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
   
        jetmetHE_p_ME_.numerator -> Fill(jetpt_);
        jetmetHE_p_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHE_p_PhiME_.numerator -> Fill(jetphi_);
        jetmetHE_p_EtaME_.numerator -> Fill(jeteta_);
        jetmetHE_p_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHE_p_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isEndCapM( jeteta_ ) )
     {
        jetmetHE_ME_.numerator -> Fill(jetpt_);
        jetmetHE_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHE_PhiME_.numerator -> Fill(jetphi_);
        jetmetHE_EtaME_.numerator -> Fill(jeteta_);
        jetmetHE_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHE_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
   
        jetmetHE_m_ME_.numerator -> Fill(jetpt_);
        jetmetHE_m_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHE_m_PhiME_.numerator -> Fill(jetphi_);
        jetmetHE_m_EtaME_.numerator -> Fill(jeteta_);
        jetmetHE_m_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHE_m_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isForward( jeteta_ ) )
     {
   
        jetmetHF_ME_.numerator -> Fill(jetpt_);
        jetmetHF_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHF_PhiME_.numerator -> Fill(jetphi_);
        jetmetHF_EtaME_.numerator -> Fill(jeteta_);
        jetmetHF_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHF_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isHEP17( jeteta_, jetphi_ ) )
     {
        jetmetHEP17_ME_.numerator -> Fill(jetpt_);
        jetmetHEP17_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHEP17_PhiME_.numerator -> Fill(jetphi_);
        jetmetHEP17_EtaME_.numerator -> Fill(jeteta_);
        jetmetHEP17_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHEP17_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isHEM17( jeteta_, jetphi_ ) )
     {
        jetmetHEM17_ME_.numerator -> Fill(jetpt_);
        jetmetHEM17_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHEM17_PhiME_.numerator -> Fill(jetphi_);
        jetmetHEM17_EtaME_.numerator -> Fill(jeteta_);
        jetmetHEM17_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHEM17_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     }
     if (isHEP18( jeteta_, jetphi_ ) )
     {
        jetmetHEP18_ME_.numerator -> Fill(jetpt_);
        jetmetHEP18_ME_variableBinning_.numerator -> Fill(jetpt_);
        jetmetHEP18_PhiME_.numerator -> Fill(jetphi_);
        jetmetHEP18_EtaME_.numerator -> Fill(jeteta_);
        jetmetHEP18_VsLS_.numerator -> Fill(ls, jetpt_);
        jetmetHEP18_EtaVsPhi_.numerator -> Fill(jeteta_,jetphi_);
     } 

  }
  if (isMetTrig) {
     edm::Handle<reco::PFMETCollection> metHandle;
     iEvent.getByToken( metToken_, metHandle );
     std::vector<reco::PFMET> mets;
     if ( int(metHandle->size()) < 0 ) return;
     for ( auto const & jet : *metHandle ) {
       mets.push_back(jet);
     }
     if ( int(mets.size()) < 0 ) return;
   
     int ls = iEvent.id().luminosityBlock();
     // filling histograms (denominator)
     std::vector <float> met_pt;
     std::vector <float> met_phi;
     met_pt.clear();
     met_phi.clear();
     for (unsigned int i =0; i < mets.size(); ++i )
     {
        float metpt_  = mets[i].pt(); 
        float metphi_ = mets[i].phi(); 
        if (metpt_ < ptcut_) {continue;}
        jetmetME_.denominator -> Fill(metpt_);
        jetmetME_variableBinning_.denominator -> Fill(metpt_);
        jetmetPhiME_.denominator -> Fill(metphi_);
        jetmetVsLS_.denominator -> Fill(ls, metpt_);
        met_pt.push_back(metpt_);
        met_phi.push_back(metphi_);
     }

     if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return; // Require Numerator //
   
     for (unsigned int i =0; i < met_pt.size(); ++i )
     {
        jetmetME_.numerator -> Fill(met_pt[i]);
        jetmetME_variableBinning_.numerator -> Fill(met_pt[i]);
        jetmetPhiME_.numerator -> Fill(met_phi[i]);
        jetmetVsLS_.numerator -> Fill(ls, met_pt[i]);
     }
  }
}

void JetMETMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void JetMETMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins", 2500);
}

void JetMETMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/JetMET" );

  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "pfjets",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "calojets",     edm::InputTag("ak4CaloJets") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<int>("njets",      0);
  desc.add<int>("nelectrons", 0);
  desc.add<int>("nmuons",     0);
  desc.add<double>("ptcut",   0);
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 20");
  desc.add<std::string>("calojetSelection", "pt > 20");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<bool>("ispfjettrg",    true);
  desc.add<bool>("iscalojettrg",  false);
  desc.add<bool>("ismettrg",      false);
  desc.add<bool>("isjetFrac", false);

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
//  genericTriggerEventPSet.add<std::string>("hltDBKey","");
  genericTriggerEventPSet.add<bool>("errorReplyHlt",false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel",1);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription metPSet;
  fillHistoPSetDescription(metPSet);
  histoPSet.add<edm::ParameterSetDescription>("metPSet", metPSet);  
  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.}; // Jet pT Binning
  histoPSet.add<std::vector<double> >("jetptBinning", bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("jetmetMonitoring", desc);
}

bool JetMETMonitor::isBarrel(double eta){
  bool output = false;
  if (fabs(eta)<=1.3) output=true;
  return output;
}

//------------------------------------------------------------------------//
bool JetMETMonitor::isEndCapM(double eta){
  bool output = false;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta/fabs(eta) < 0) ) output=true;
  return output;
}
/// For Hcal Endcap Plus Area
bool JetMETMonitor::isEndCapP(double eta){
  bool output = false;
  //if ( eta<=3.0 && eta >1.3) output=true;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta/fabs(eta) > 0) ) output=true;
  return output;
}
/// For Hcal Forward Plus Area
bool JetMETMonitor::isForward(double eta){
  bool output = false;
  if (fabs(eta)>3.0) output=true;
  return output;
}
/// For Hcal HEP17 Area
bool JetMETMonitor::isHEP17(double eta, double phi){
  bool output = false;
  // phi -0.87 to -0.52 
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta/fabs(eta) > 0) &&
      phi > -0.87 && phi <= -0.52 ) {output=true;}
  return output;
}
/// For Hcal HEM17 Area
bool JetMETMonitor::isHEM17(double eta, double phi){
  bool output = false;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta/fabs(eta) < 0) &&
      phi > -0.87 && phi <= -0.52 ) {output=true;}
  return output;
}
/// For Hcal HEP18 Area
bool JetMETMonitor::isHEP18(double eta, double phi){
  bool output = false;
  // phi -0.87 to -0.52 
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta/fabs(eta) > 0) &&
      phi > -0.52 && phi <= -0.17 ) {output=true;}
  return output;

}
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetMETMonitor);

#include "DQMOffline/Trigger/plugins/JetMonitor.h"

#include <utility>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"



// -----------------------------
//  constructors and destructor
// -----------------------------

JetMonitor::JetMonitor( const edm::ParameterSet& iConfig ):
num_genTriggerEventFlag_ ( new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
,den_genTriggerEventFlag_ ( new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
{
  folderName_            = iConfig.getParameter<std::string>("FolderName"); 
  jetSrc_  = mayConsume<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("jetSrc"));//jet 
  jetpT_variable_binning_  = iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetptBinning");
  jetpT_binning            = getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("jetPSet")    );
  jetptThr_binning_        = getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("jetPtThrPSet")    );
  ls_binning_              = getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     );


  ptcut_      = iConfig.getParameter<double>("ptcut" ); // for HLT Jet 
  isPFJetTrig    = iConfig.getParameter<bool>("ispfjettrg" );
  isCaloJetTrig  = iConfig.getParameter<bool>("iscalojettrg" );


}

JetMonitor::~JetMonitor() = default;

JetMonitor::MEbinning JetMonitor::getHistoPSet(const edm::ParameterSet& pset)
{
  return JetMonitor::MEbinning{
    pset.getParameter<unsigned int>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

JetMonitor::MEbinning JetMonitor::getHistoLSPSet(const edm::ParameterSet& pset)
{
  return JetMonitor::MEbinning{
    pset.getParameter<unsigned int>("nbins"),
      0.,
      double(pset.getParameter<unsigned int>("nbins"))
      };
}

void JetMonitor::setMETitle(JetME& me, const std::string& titleX, const std::string& titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);
}
void JetMonitor::bookME(DQMStore::IBooker &ibooker, JetME& me, std::string& histname, std::string& histtitle, unsigned int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void JetMonitor::bookME(DQMStore::IBooker &ibooker, JetME& me, std::string& histname, std::string& histtitle, std::vector<double> binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void JetMonitor::bookME(DQMStore::IBooker &ibooker, JetME& me, std::string& histname, std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void JetMonitor::bookME(DQMStore::IBooker &ibooker, JetME& me, std::string& histname, std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void JetMonitor::bookME(DQMStore::IBooker &ibooker, JetME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY)
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

void JetMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;
  std::string hist_obtag = "";
  std::string histtitle_obtag = "";
  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder);

  if (isPFJetTrig) {hist_obtag = "pfjet";          histtitle_obtag =  "PFJet";}
  else if (isCaloJetTrig) {hist_obtag = "calojet"; histtitle_obtag =  "CaloJet"; }
  else {hist_obtag = "pfjet"; histtitle_obtag =  "PFJet"; } //default is pfjet 

  bookMESub(ibooker,a_ME,sizeof(a_ME)/sizeof(a_ME[0]),hist_obtag,histtitle_obtag,"","");
  bookMESub(ibooker,a_ME_HB,sizeof(a_ME_HB)/sizeof(a_ME_HB[0]),hist_obtag,histtitle_obtag,"HB","(HB)",true, true, true, false); 
  bookMESub(ibooker,a_ME_HE,sizeof(a_ME_HE)/sizeof(a_ME_HE[0]),hist_obtag,histtitle_obtag,"HE","(HE)", true, true, true, false);
  bookMESub(ibooker,a_ME_HF,sizeof(a_ME_HF)/sizeof(a_ME_HF[0]),hist_obtag,histtitle_obtag,"HF","(HF)",true, true, true, false); 
  bookMESub(ibooker,a_ME_HE_p,sizeof(a_ME_HE_p)/sizeof(a_ME_HE_p[0]),hist_obtag,histtitle_obtag,"HE_p","(HE+)",true, true, true, false); 
  bookMESub(ibooker,a_ME_HE_m,sizeof(a_ME_HE_m)/sizeof(a_ME_HE_m[0]),hist_obtag,histtitle_obtag,"HE_m","(HE-)",true, true, true, false); 
  bookMESub(ibooker,a_ME_HEP17,sizeof(a_ME_HEP17)/sizeof(a_ME_HEP17[0]),hist_obtag,histtitle_obtag,"HEP17","(HEP17)", true, false, false, false);
  bookMESub(ibooker,a_ME_HEM17,sizeof(a_ME_HEM17)/sizeof(a_ME_HEM17[0]),hist_obtag,histtitle_obtag,"HEM17","(HEM17)", true, false, false, false);
  bookMESub(ibooker,a_ME_HEP18,sizeof(a_ME_HEP18)/sizeof(a_ME_HEP18[0]),hist_obtag,histtitle_obtag,"HEP18","(HEP18)", false, false, false, false);

  /*
    WE WOULD NEED TURNON CURVES TO BE COMPARED NOT JUST THE ZOOM OF A 2D MAP !!!

  histname = hist_obtag +"AbsEtaVsPhi_HEP17"; histtitle = histtitle_obtag + " |eta| Vs phi (HEP17) ";
  bookME(ibooker,jetHEP17_AbsEtaVsPhi_,histname,histtitle, eta_binning_hep17_.nbins, eta_binning_hep17_.xmin, eta_binning_hep17_.xmax, phi_binning_hep17_.nbins,phi_binning_hep17_.xmin,phi_binning_hep17_.xmax);
  setMETitle(jetHEP17_AbsEtaVsPhi_,histtitle_obtag + " |#eta|","#phi");

  histname = hist_obtag +"AbsEtaVsPhi_HEM17"; histtitle = histtitle_obtag + " |eta| Vs phi (HEM17) ";
  bookME(ibooker,jetHEM17_AbsEtaVsPhi_,histname,histtitle, eta_binning_hep17_.nbins, eta_binning_hep17_.xmin, eta_binning_hep17_.xmax, phi_binning_hep17_.nbins,phi_binning_hep17_.xmin,phi_binning_hep17_.xmax);
  setMETitle(jetHEM17_AbsEtaVsPhi_,histtitle_obtag + " |#eta|","#phi");
  */

  histname = hist_obtag +"abseta_HEP17"; histtitle = histtitle_obtag + " |#eta| (HEP17) ";
  bookME(ibooker,jetHEP17_AbsEta_,histname,histtitle, eta_binning_hep17_.nbins, eta_binning_hep17_.xmin, eta_binning_hep17_.xmax);
  setMETitle(jetHEP17_AbsEta_,histtitle_obtag + " |#eta|","events / |#eta|");

  histname = hist_obtag +"abseta_HEM17"; histtitle = histtitle_obtag + " |eta| (HEM17) ";
  bookME(ibooker,jetHEM17_AbsEta_,histname,histtitle, eta_binning_hep17_.nbins, eta_binning_hep17_.xmin, eta_binning_hep17_.xmax);
  setMETitle(jetHEM17_AbsEta_,histtitle_obtag + " |#eta|","events / |#eta|");

  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/Math/interface/deltaR.h" // For Delta R
void JetMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {
  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;
//  edm::Handle<reco::PFJetCollection> pfjetHandle;
//  iEvent.getByToken( pfjetToken_, pfjetHandle );

//  edm::Handle<reco::CaloJetCollection> calojetHandle;
//  iEvent.getByToken( calojetToken_, calojetHandle );

  int ls = iEvent.id().luminosityBlock();
  v_jetpt.clear();
  v_jeteta.clear();
  v_jetphi.clear();

  edm::Handle< edm::View<reco::Jet> > offjets;
  iEvent.getByToken( jetSrc_, offjets );
  if (!offjets.isValid()){
      edm::LogWarning("JetMonitor") << "Jet handle not valid \n";
      return;
  }
  for ( edm::View<reco::Jet>::const_iterator ibegin = offjets->begin(), iend = offjets->end(), ijet = ibegin; ijet != iend; ++ijet ) {
    //if (ijet->pt()< 20) {continue;}
    if (ijet->pt()< ptcut_) {continue;}
    v_jetpt.push_back(ijet->pt()); 
    v_jeteta.push_back(ijet->eta()); 
    v_jetphi.push_back(ijet->phi()); 
//    cout << "jetpt (view ) : " << ijet->pt() << endl;
  }

  if ( v_jetpt.empty() ) return;
  double jetpt_ = v_jetpt[0];
  double jeteta_ = v_jeteta[0];
  double jetphi_ = v_jetphi[0];

  FillME(a_ME,jetpt_,jetphi_,jeteta_,ls,"denominator"); 
  if (isBarrel( jeteta_ ) )
  {
    FillME(a_ME_HB,jetpt_,jetphi_,jeteta_,ls,"denominator",true, true, true, false); 
  }
  else if (isEndCapP( jeteta_ ) )
  {
    FillME(a_ME_HE,  jetpt_,jetphi_,jeteta_,ls,"denominator",true, true, true, false); 
    FillME(a_ME_HE_p,jetpt_,jetphi_,jeteta_,ls,"denominator",true, true, true, false); 
  }
  else if (isEndCapM( jeteta_ ) )
  {
    FillME(a_ME_HE,  jetpt_,jetphi_,jeteta_,ls,"denominator",true, true, true, false); 
    FillME(a_ME_HE_m,jetpt_,jetphi_,jeteta_,ls,"denominator",true, true, true, false); 
  }
  else if (isForward( jeteta_ ) )
  {
    FillME(a_ME_HF,jetpt_,jetphi_,jeteta_,ls,"denominator",true, true, true, false); 
  }

  if (isHEP17( jeteta_, jetphi_ ) )
  {
    FillME(a_ME_HEP17,jetpt_,jetphi_,jeteta_,ls,"denominator",true,false,false, false); // doPhi, doEta, doEtaPhi, doVsLS
    jetHEP17_AbsEta_.denominator->Fill(abs(jeteta_));
  }
  else if (isHEM17( jeteta_, jetphi_ ) )
  {
    FillME(a_ME_HEM17,jetpt_,jetphi_,jeteta_,ls,"denominator",true,false,false, false); // doPhi, doEta, doEtaPhi
     jetHEM17_AbsEta_.denominator->Fill(abs(jeteta_));
  }
  else if (isHEP18( jeteta_, jetphi_ ) )
  {
    FillME(a_ME_HEP18,jetpt_,jetphi_,jeteta_,ls,"denominator",false,false,false, false); // doPhi, doEta, doEtaPhi 
  }


  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return; // Require Numerator //

  FillME(a_ME,jetpt_,jetphi_,jeteta_,ls,"numerator"); 
  if (isBarrel( jeteta_ ) )
  {
    FillME(a_ME_HB,jetpt_,jetphi_,jeteta_,ls,"numerator",true, true, true, false); 
  }
  else if (isEndCapP( jeteta_ ) )
  {
    FillME(a_ME_HE,  jetpt_,jetphi_,jeteta_,ls,"numerator",true, true, true, false); 
    FillME(a_ME_HE_p,jetpt_,jetphi_,jeteta_,ls,"numerator",true, true, true, false); 
  }
  else if (isEndCapM( jeteta_ ) )
  {
    FillME(a_ME_HE,  jetpt_,jetphi_,jeteta_,ls,"numerator",true, true, true, false); 
    FillME(a_ME_HE_m,jetpt_,jetphi_,jeteta_,ls,"numerator",true, true, true, false); 
  }
  else if (isForward( jeteta_ ) )
  {
    FillME(a_ME_HF,jetpt_,jetphi_,jeteta_,ls,"numerator",true, true, true, false); 
  }

  if (isHEP17( jeteta_, jetphi_ ) )
  {
    FillME(a_ME_HEP17,jetpt_,jetphi_,jeteta_,ls,"numerator",true,false,false, false); // doPhi, doEta, doEtaPhi, doVsLS
    jetHEP17_AbsEta_.numerator->Fill(abs(jeteta_));
  }
  else if (isHEM17( jeteta_, jetphi_ ) )
  {
    FillME(a_ME_HEM17,jetpt_,jetphi_,jeteta_,ls,"numerator",true,false,false,false); // doPhi, doEta, doEtaPhi, doVsLS
    jetHEM17_AbsEta_.numerator->Fill(abs(jeteta_));
  }
  else if (isHEP18( jeteta_, jetphi_ ) )
  {
    FillME(a_ME_HEP18,jetpt_,jetphi_,jeteta_,ls,"numerator",false,false,false,false); // doPhi, doEta, doEtaPhi, doVsLS
  }

}

void JetMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void JetMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins", 2500);
  pset.add<double>         ( "xmin",     0.);
  pset.add<double>         ( "xmax",  2500.);
}

void JetMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/Jet" );

  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
//  desc.add<edm::InputTag>( "pfjets",     edm::InputTag("ak4PFJetsCHS") );
//  desc.add<edm::InputTag>( "calojets",     edm::InputTag("ak4CaloJets") );
  desc.add<edm::InputTag>( "jetSrc",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<int>("njets",      0);
  desc.add<int>("nelectrons", 0);
  desc.add<double>("ptcut",   20);
  desc.add<bool>("ispfjettrg",    true);
  desc.add<bool>("iscalojettrg",  false);

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
  edm::ParameterSetDescription jetPSet;
  edm::ParameterSetDescription jetPtThrPSet;
  fillHistoPSetDescription(jetPSet);
  fillHistoPSetDescription(jetPtThrPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetPSet", jetPSet);  
  histoPSet.add<edm::ParameterSetDescription>("jetPtThrPSet", jetPtThrPSet);  
  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.}; // Jet pT Binning
  histoPSet.add<std::vector<double> >("jetptBinning", bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("jetMonitoring", desc);
}

bool JetMonitor::isBarrel(double eta){
  bool output = false;
  if (fabs(eta)<=1.3) output=true;
  return output;
}

//------------------------------------------------------------------------//
bool JetMonitor::isEndCapM(double eta){
  bool output = false;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta < 0) ) output=true; // (mia) this magic number should come from some file in CMSSW !!!
  return output;
}
/// For Hcal Endcap Plus Area
bool JetMonitor::isEndCapP(double eta){
  bool output = false;
  //if ( eta<=3.0 && eta >1.3) output=true;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta > 0) ) output=true; // (mia) this magic number should come from some file in CMSSW !!!
  return output;
}
/// For Hcal Forward Plus Area
bool JetMonitor::isForward(double eta){
  bool output = false;
  if (fabs(eta)>3.0) output=true;
  return output;
}
/// For Hcal HEP17 Area
bool JetMonitor::isHEP17(double eta, double phi){
  bool output = false;
  // phi -0.87 to -0.52 
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta > 0) &&
      phi > -0.87 && phi <= -0.52 ) {output=true;} // (mia) this magic number should come from some file in CMSSW !!!
  return output;
}
/// For Hcal HEM17 Area
bool JetMonitor::isHEM17(double eta, double phi){
  bool output = false;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta < 0) &&
      phi > -0.87 && phi <= -0.52 ) {output=true;} // (mia) this magic number should come from some file in CMSSW !!!
  return output;
}
/// For Hcal HEP18 Area
bool JetMonitor::isHEP18(double eta, double phi){
  bool output = false;
  // phi -0.87 to -0.52 
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta > 0) &&
      phi > -0.52 && phi <= -0.17 ) {output=true;} // (mia) this magic number should come from some file in CMSSW !!!
  return output;

}
/*void JetMonitor::AutoNullPtr(JetME* a_me,const int len_){
   for (int i =0; i < len_; ++i)
   {
      a_me[i].denominator = nullptr;
      a_me[i].numerator = nullptr;
   }
}*/
void JetMonitor::FillME(JetME* a_me,double pt_, double phi_, double eta_, int ls_,const std::string& denu, bool doPhi, bool doEta, bool doEtaPhi, bool doVsLS){
   std::string isDeno = "";
   isDeno = denu;
   std::string DenoOrNume = "";
   DenoOrNume = denu;

   if (DenoOrNume == "denominator")
   {
      // index 0 = pt, 1 = ptThreshold , 2 = pt vs ls , 3 = phi, 4 = eta, 
      // 5 = eta vs phi, 6 = eta vs pt , 7 = abs(eta) , 8 = abs(eta) vs phi 
      a_me[0].denominator->Fill(pt_);// pt
      a_me[1].denominator->Fill(pt_);// jetpT Threshold binning for pt 
      if ( doVsLS   ) a_me[2].denominator->Fill(ls_,pt_);// pt vs ls
      if ( doPhi    ) a_me[3].denominator->Fill(phi_);// phi       
      if ( doEta    ) a_me[4].denominator->Fill(eta_);// eta
      if ( doEtaPhi ) a_me[5].denominator->Fill(eta_,phi_);// eta vs phi
      if ( doEta    ) a_me[6].denominator->Fill(eta_,pt_);// eta vs pT
   }
   else if (DenoOrNume == "numerator")
   {
      a_me[0].numerator->Fill(pt_);// pt 
      a_me[1].numerator->Fill(pt_);// jetpT Threshold binning for pt 
      if ( doVsLS   ) a_me[2].numerator->Fill(ls_,pt_);// pt vs ls
      if ( doPhi    ) a_me[3].numerator->Fill(phi_);// phi
      if ( doEta    ) a_me[4].numerator->Fill(eta_);// eat
      if ( doEtaPhi ) a_me[5].numerator->Fill(eta_,phi_);// eta vs phi
      if ( doEta    ) a_me[6].numerator->Fill(eta_,pt_);// eta vs pT 
   }
   else {
      edm::LogWarning("JetMonitor") << "CHECK OUT denu option in FillME !!! DenoOrNume ? : " << DenoOrNume << std::endl;
   }
}
void JetMonitor::bookMESub(DQMStore::IBooker & Ibooker , JetME* a_me,const int len_,const std::string& h_Name ,const std::string& h_Title, const std::string& h_subOptName , std::string h_suOptTitle, bool doPhi, bool doEta, bool doEtaPhi, bool doVsLS){

   std::string hName = h_Name;
   std::string hTitle = h_Title;
   std::string hSubN =""; 
   std::string hSubT =""; 
   hSubT = std::move(h_suOptTitle);

   int nbin_phi = jet_phi_binning_.nbins;
   double maxbin_phi = jet_phi_binning_.xmax;
   double minbin_phi = jet_phi_binning_.xmin;

   int nbin_eta = jet_eta_binning_.nbins;
   double maxbin_eta = jet_eta_binning_.xmax;
   double minbin_eta = jet_eta_binning_.xmin;

   if (h_subOptName != ""){
      hSubN = "_"+h_subOptName;
   }

   if (h_subOptName == "HEP17") {
      nbin_phi = phi_binning_hep17_.nbins;
      maxbin_phi = phi_binning_hep17_.xmax;
      minbin_phi = phi_binning_hep17_.xmin;

      nbin_eta = eta_binning_hep17_.nbins;
      maxbin_eta = eta_binning_hep17_.xmax;
      minbin_eta = eta_binning_hep17_.xmin;

   }
   if (h_subOptName == "HEM17") {
      nbin_phi = phi_binning_hep17_.nbins;
      maxbin_phi = phi_binning_hep17_.xmax;
      minbin_phi = phi_binning_hep17_.xmin;

      nbin_eta = eta_binning_hem17_.nbins;
      maxbin_eta = eta_binning_hem17_.xmax;
      minbin_eta = eta_binning_hem17_.xmin;

   }
   if (h_subOptName == "HEP18") {
      nbin_phi = phi_binning_hep18_.nbins;
      maxbin_phi = phi_binning_hep18_.xmax;
      minbin_phi = phi_binning_hep18_.xmin;

      nbin_eta = eta_binning_hep17_.nbins;
      maxbin_eta = eta_binning_hep17_.xmax;
      minbin_eta = eta_binning_hep17_.xmin;

   }
   hName = h_Name+"pT"+hSubN;
   hTitle = h_Title+" pT " + hSubT;
   bookME(Ibooker,a_me[0],hName,hTitle,jetpT_binning.nbins,jetpT_binning.xmin, jetpT_binning.xmax);
   setMETitle(a_me[0], h_Title +" pT [GeV]","events / [GeV]");
   
   hName = h_Name+ "pT_pTThresh" + hSubN;
   hTitle = h_Title+" pT " + hSubT;
   bookME(Ibooker,a_me[1],hName,hTitle,jetptThr_binning_.nbins,jetptThr_binning_.xmin, jetptThr_binning_.xmax);
   setMETitle(a_me[1],h_Title + "pT [GeV]","events / [GeV]");

   if ( doVsLS ) {
     hName = h_Name + "pTVsLS" + hSubN; 
     hTitle = h_Title+" vs LS " + hSubT;
     bookME(Ibooker,a_me[2],hName,hTitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,jetpT_binning.xmin, jetpT_binning.xmax);
     setMETitle(a_me[2],"LS",h_Title + "pT [GeV]");
   }

   if ( doPhi ) {
     hName = h_Name + "phi" + hSubN;
     hTitle = h_Title+" phi " + hSubT;
     bookME(Ibooker,a_me[3],hName,hTitle, nbin_phi, minbin_phi,maxbin_phi );
     setMETitle(a_me[3],h_Title +" #phi","events / 0.1 rad");
   }

   if ( doEta ) {
     hName = h_Name + "eta"+ hSubN;
     hTitle = h_Title+" eta " + hSubT;
     bookME(Ibooker,a_me[4],hName,hTitle, nbin_eta, minbin_eta, maxbin_eta);
     setMETitle(a_me[4],h_Title + " #eta","events / #eta");
   }

   if ( doEtaPhi ) {
     hName = h_Name + "EtaVsPhi"+hSubN;
     hTitle = h_Title+" eta Vs phi " + hSubT;
     bookME(Ibooker,a_me[5],hName,hTitle, nbin_eta, minbin_eta, maxbin_eta, nbin_phi, minbin_phi, maxbin_phi);
     setMETitle(a_me[5],h_Title + " #eta","#phi");
   }

   if ( doEta ) {
     hName = h_Name + "EtaVspT"+hSubN;
     hTitle = h_Title+" eta Vs pT " + hSubT;
     bookME(Ibooker,a_me[6],hName,hTitle, nbin_eta, minbin_eta, maxbin_eta, jetpT_binning.nbins,jetpT_binning.xmin, jetpT_binning.xmax);
     setMETitle(a_me[6],h_Title + " #eta","Leading Jet pT [GeV]");
   }

}
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetMonitor);

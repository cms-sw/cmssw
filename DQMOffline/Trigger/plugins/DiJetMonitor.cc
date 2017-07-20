#include "DQMOffline/Trigger/plugins/DiJetMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

DiJetMonitor::DiJetMonitor( const edm::ParameterSet& iConfig ):
num_genTriggerEventFlag_ ( new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
,den_genTriggerEventFlag_ ( new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
{
  folderName_            = iConfig.getParameter<std::string>("FolderName"); 
  dijetSrc_  = mayConsume<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("dijetSrc"));//jet 
  dijetpT_variable_binning_  = iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetptBinning");
  dijetpT_binning           = getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("dijetPSet")    );
  dijetptThr_binning_      = getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("dijetPtThrPSet")    );
  ls_binning_            = getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     );


  ptcut_      = iConfig.getParameter<double>("ptcut" ); // for HLT DiJet 
  isPFDiJetTrig    = iConfig.getParameter<bool>("ispfdijettrg" );
  isCaloDiJetTrig  = iConfig.getParameter<bool>("iscalodijettrg" );

  jetpt1ME_.histo = nullptr;
  jetpt2ME_.histo = nullptr;
  jetptAvgaME_.histo = nullptr;
  jetptAvgbME_.histo = nullptr;
  jetptTagME_.histo = nullptr;
  jetptPrbME_.histo = nullptr;
  jetptAsyME_.histo = nullptr;
  jetetaPrbME_.histo = nullptr;
  jetAsyEtaME_.histo = nullptr;
}


DiJetMonitor::~DiJetMonitor()
{
}

DiJetMonitor::MEbinning DiJetMonitor::getHistoPSet(edm::ParameterSet pset)
{
  return DiJetMonitor::MEbinning{
    pset.getParameter<unsigned int>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

DiJetMonitor::MEbinning DiJetMonitor::getHistoLSPSet(edm::ParameterSet pset)
{
  return DiJetMonitor::MEbinning{
    pset.getParameter<unsigned int>("nbins"),
      0.,
      double(pset.getParameter<unsigned int>("nbins"))
      };
}

void DiJetMonitor::setMETitle(DiJetME& me, std::string titleX, std::string titleY)
{
  me.histo->setAxisTitle(titleX,1);
  me.histo->setAxisTitle(titleY,2);
}
void DiJetMonitor::bookME(DQMStore::IBooker &ibooker, DiJetME& me, std::string& histname, std::string& histtitle, unsigned int nbins, double min, double max)
{
  me.histo = ibooker.book1D(histname, histtitle, nbins, min, max);
}
void DiJetMonitor::bookME(DQMStore::IBooker &ibooker, DiJetME& me, std::string& histname, std::string& histtitle, std::vector<double> binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.histo = ibooker.book1D(histname, histtitle, nbins, arr);
}
void DiJetMonitor::bookME(DQMStore::IBooker &ibooker, DiJetME& me, std::string& histname, std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.histo = ibooker.bookProfile(histname, histtitle, nbinsX, xmin, xmax, ymin, ymax);
}
void DiJetMonitor::bookME(DQMStore::IBooker &ibooker, DiJetME& me, std::string& histname, std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax)
{
  me.histo = ibooker.book2D(histname, histtitle, nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void DiJetMonitor::bookME(DQMStore::IBooker &ibooker, DiJetME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY)
{
  int nbinsX = binningX.size()-1;
  std::vector<float> fbinningX(binningX.begin(),binningX.end());
  float* arrX = &fbinningX[0];
  int nbinsY = binningY.size()-1;
  std::vector<float> fbinningY(binningY.begin(),binningY.end());
  float* arrY = &fbinningY[0];

  me.histo = ibooker.book2D(histname, histtitle, nbinsX, arrX, nbinsY, arrY);
}

void DiJetMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;
  std::string hist_obtag = "";
  std::string histtitle_obtag = "";
  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());


  histname = "pt_1"; histtitle = "leading Jet Pt";
  bookME(ibooker,jetpt1ME_,histname,histtitle,dijetpT_binning.nbins,dijetpT_binning.xmin, dijetpT_binning.xmax);
  setMETitle(jetpt1ME_,"Pt_1 [GeV]","events / [GeV]");

  histname = "pt_2"; histtitle = "second leading Jet Pt";
  bookME(ibooker,jetpt2ME_,histname,histtitle,dijetpT_binning.nbins,dijetpT_binning.xmin, dijetpT_binning.xmax);
  setMETitle(jetpt2ME_,"Pt_2 [GeV]","events / [GeV]");

  histname = "pt_avg_b"; histtitle = "Pt average before offline selection";
  bookME(ibooker,jetptAvgbME_,histname,histtitle,dijetpT_binning.nbins,dijetpT_binning.xmin, dijetpT_binning.xmax);
  setMETitle(jetptAvgbME_,"(pt_1 + pt_2)*0.5 [GeV]","events / [GeV]");

  histname = "pt_avg_a"; histtitle = "Pt average after offline selection";
  bookME(ibooker,jetptAvgaME_,histname,histtitle,dijetpT_binning.nbins,dijetpT_binning.xmin, dijetpT_binning.xmax);
  setMETitle(jetptAvgaME_,"(pt_1 + pt_2)*0.5 [GeV]","events / [GeV]");

  histname = "pt_tag"; histtitle = "Tag Jet Pt";
  bookME(ibooker,jetptTagME_,histname,histtitle,dijetpT_binning.nbins,dijetpT_binning.xmin, dijetpT_binning.xmax);
  setMETitle(jetptTagME_,"Pt_tag [GeV]","events / [GeV]");

  histname = "pt_prb"; histtitle = "Probe Jet Pt";
  bookME(ibooker,jetptPrbME_,histname,histtitle,dijetpT_binning.nbins,dijetpT_binning.xmin, dijetpT_binning.xmax);
  setMETitle(jetptPrbME_,"Pt_prb [GeV]","events / [GeV]");

  histname = "pt_asym"; histtitle = "Jet Pt Asymetry";
  bookME(ibooker,jetptAsyME_,histname,histtitle,asy_binning_.nbins,asy_binning_.xmin, asy_binning_.xmax);
  setMETitle(jetptAsyME_,"(pt_prb - pt_tag)/(pt_prb + pt_tag)","events");

  histname = "eta_prb"; histtitle = "Probe Jet eta";
  bookME(ibooker,jetetaPrbME_,histname,histtitle,dijet_eta_binning_.nbins,dijet_eta_binning_.xmin, dijet_eta_binning_.xmax);
  setMETitle(jetetaPrbME_,"Eta_probe","events");

  histname = "pt_Asym_VS_eta_prb"; histtitle = "Pt_Asym vs eta_ptb";
  bookME(ibooker,jetAsyEtaME_,histname,histtitle,asy_binning_.nbins, asy_binning_.xmin, asy_binning_.xmax, dijet_eta_binning_.nbins, dijet_eta_binning_.xmin,dijet_eta_binning_.xmax);
  setMETitle(jetAsyEtaME_,"(pt_prb - pt_tag)/(pt_prb + pt_tag)","Eta");

  // Initialize the GenericTriggerEventFlag
  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/Math/interface/deltaR.h" // For Delta R
void DiJetMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {
  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;
//  edm::Handle<reco::PFDiJetCollection> pfjetHandle;
//  iEvent.getByToken( pfjetToken_, pfjetHandle );

//  edm::Handle<reco::CaloDiJetCollection> calojetHandle;
//  iEvent.getByToken( calojetToken_, calojetHandle );

  int ls = iEvent.id().luminosityBlock();
  v_jetpt.clear();
  v_jeteta.clear();
  v_jetphi.clear();

//  edm::Handle< edm::View<reco::Jet> > offjets;
  edm::Handle<reco::PFJetCollection > offjets;
  iEvent.getByToken( dijetSrc_, offjets );
  if (!offjets.isValid()){
      edm::LogWarning("DiJetMonitor") << "DiJet handle not valid \n";
      return;
  }
  for ( reco::PFJetCollection::const_iterator ibegin = offjets->begin(), iend = offjets->end(), ijet = ibegin; ijet != iend; ++ijet ) {
    if (ijet->pt()< 20) {continue;}
//    if (ijet->pt()< ptcut_) {continue;}
    v_jetpt.push_back(ijet->pt()); 
    v_jeteta.push_back(ijet->eta()); 
    v_jetphi.push_back(ijet->phi()); 
//    cout << " ijet->pt() (view ) : " << ijet->pt() << endl;
  }

  if (v_jetpt.size() < 2) {return;}
//      edm::LogWarning("DiJetMonitor") << " v_jetps.size<2 is exist'  \n";
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return; 
//      edm::LogWarning("DiJetMonitor") << " events 'num_genTriggerEventFlag_->on'  \n";
  double pt_1 = v_jetpt[0];
  double eta_1 = v_jeteta[0];
  double phi_1 = v_jetphi[0];

  double pt_2 = v_jetpt[1];
  double eta_2 = v_jeteta[1];
  double phi_2 = v_jetphi[1];
//    cout << " TriggerPassed !! pt_1,2 (view ) :  pt_1 = " << pt_1 << ", pt_2  = "<<pt_2<< endl;
//    cout << " !!  pt_1,2 (view ) :  pt_1 = " << pt_1 << ", pt_2  = "<<pt_2<< endl;

  jetpt1ME_.histo -> Fill(pt_1);
  jetpt2ME_.histo -> Fill(pt_2);
  double pt_avg_b = (pt_1 + pt_2)*0.5;
  jetptAvgbME_.histo -> Fill(pt_avg_b);

  int tag_id = -999, probe_id = -999;
//_______Offline selection______//  
  if( dijet_selection(eta_1, phi_1, eta_2, phi_2, pt_1, pt_2,tag_id, probe_id ) == false ) return;
  
  if(tag_id == 0 && probe_id == 1) {
//    double pt_tag = pt_1;
//    double pt_prb = pt_2;
//    double eta_prb = eta_2;
//    double pt_asy = (pt_prb - pt_tag)/(pt_tag + pt_prb);
    double pt_asy = (pt_2 - pt_1)/(pt_1 + pt_2);
//    double pt_avg = (pt_tag + pt_prb)*0.5;
    double pt_avg = (pt_1 + pt_2)*0.5;
   jetptAvgaME_.histo -> Fill(pt_avg);
//   jetptTagME_.histo -> Fill(pt_tag);
//   jetptPrbME_.histo -> Fill(pt_prb);
   jetptTagME_.histo -> Fill(pt_1);
   jetptPrbME_.histo -> Fill(pt_2);
//   jetetaPrbME_.histo -> Fill(eta_prb);
   jetetaPrbME_.histo -> Fill(eta_2);
   jetptAsyME_.histo->Fill(pt_asy);
//   jetetaPrbME_.histo -> Fill(eta_prb);
//   jetAsyEtaME_.histo -> Fill(pt_asy,eta_prb);
   jetetaPrbME_.histo -> Fill(eta_2);
   jetAsyEtaME_.histo -> Fill(pt_asy,eta_2);
  }
  if(tag_id == 1 && probe_id == 0) {
//    double pt_tag = pt_2;
//    double pt_prb = pt_1;
//    double eta_prb = eta_1;
//    double pt_asy = (pt_prb - pt_tag)/(pt_tag + pt_prb);
//    double pt_avg = (pt_tag + pt_prb)*0.5;
    double pt_asy = (pt_1 - pt_2)/(pt_2 + pt_1);
    double pt_avg = (pt_2 + pt_1)*0.5;
   jetptAvgaME_.histo -> Fill(pt_avg);
//   jetptTagME_.histo -> Fill(pt_tag);
//   jetptPrbME_.histo -> Fill(pt_prb);
   jetptTagME_.histo -> Fill(pt_2);
   jetptPrbME_.histo -> Fill(pt_1);
//   jetetaPrbME_.histo -> Fill(eta_prb);
   jetetaPrbME_.histo -> Fill(eta_1);
   jetptAsyME_.histo->Fill(pt_asy);
//   jetetaPrbME_.histo -> Fill(eta_prb);
//   jetAsyEtaME_.histo -> Fill(pt_asy,eta_prb);
   jetetaPrbME_.histo -> Fill(eta_1);
   jetAsyEtaME_.histo -> Fill(pt_asy,eta_1);
  }
}





void DiJetMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins");
  pset.add<double>( "xmin" );
  pset.add<double>( "xmax" );
}

void DiJetMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ( "nbins", 2500);
}

void DiJetMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/Jet" );

  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "pfjets",     edm::InputTag("ak4PFDiJetsCHS") );
  desc.add<edm::InputTag>( "calojets",     edm::InputTag("ak4CaloDiJets") );
  desc.add<edm::InputTag>( "dijetSrc",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<int>("njets",      0);
  desc.add<int>("nelectrons", 0);
  desc.add<double>("ptcut",   20);
  desc.add<bool>("ispfdijettrg",    true);
  desc.add<bool>("iscalodijettrg",  false);

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
  edm::ParameterSetDescription dijetPSet;
  edm::ParameterSetDescription dijetPtThrPSet;
  fillHistoPSetDescription(dijetPSet);
  fillHistoPSetDescription(dijetPtThrPSet);
  histoPSet.add<edm::ParameterSetDescription>("dijetPSet", dijetPSet);  
  histoPSet.add<edm::ParameterSetDescription>("dijetPtThrPSet", dijetPtThrPSet);  
  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.}; // DiJet pT Binning
  histoPSet.add<std::vector<double> >("jetptBinning", bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("dijetMonitoring", desc);
}

bool DiJetMonitor::isBarrel(double eta){
  bool output = false;
  if (fabs(eta)<=1.3) output=true;
  return output;
}
//---- Additional DiJet offline selection------
bool DiJetMonitor::dijet_selection(double eta_1, double phi_1, double eta_2, double phi_2, double pt_1, double pt_2, int &tag_id, int &probe_id){
  bool passeta; //check that one of the jets in the barrel
  if (abs(eta_1)< 1.3 || abs(eta_2) < 1.3 ) 
    passeta=true;
  
  float delta_phi_1_2= (phi_1 - phi_2);
  bool other_cuts;//check that jets are back to back
  if (abs(delta_phi_1_2) >= 2.7)
    other_cuts=true;

   if(fabs(eta_1)<1.3 && fabs(eta_2)>1.3) {
    tag_id = 0; 
    probe_id = 1;
  }
  else 
    if(fabs(eta_2)<1.3 && fabs(eta_1)>1.3) {
      tag_id = 1; 
      probe_id = 0;
    }
    else 
      if(fabs(eta_2)<1.3 && fabs(eta_1)<1.3){
    int ran = rand();
    int numb = ran % 2 + 1;
       if(numb==1){
         tag_id = 0; 
         probe_id = 1;
       }
       if(numb==2){
         tag_id = 1; 
         probe_id = 0;
       }
     }
  if (passeta && other_cuts)
    return true;
  else
    return false;
}

//------------------------------------------------------------------------//
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DiJetMonitor);

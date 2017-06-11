// -----------------------------
// 
// Offline DQM for razor triggers. The razor inclusive analysis measures trigger efficiency
// in SingleElectron events (orthogonal to analysis), as a 2D function of the razor variables
// M_R and R^2. Also monitor dPhi_R, used offline for  QCD and/or detector-related MET tail 
// rejection.
// Based on DQMOffline/Trigger/plugins/METMonitor.*
//
// -----------------------------


#include "DQMOffline/Trigger/plugins/RazorMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

// -----------------------------
//  constructors and destructor
// -----------------------------

RazorMonitor::RazorMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , metToken_             ( consumes<reco::PFMETCollection>      (iConfig.getParameter<edm::InputTag>("met")       ) )   
  , jetToken_             ( mayConsume<reco::PFJetCollection>      (iConfig.getParameter<edm::InputTag>("jets")      ) )   
  , ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
  , rsq_binning_          ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >   ("rsqBins")    )
  , mr_binning_           ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >   ("mrBins")     )
  , dphiR_binning_        ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >   ("dphiRBins")  )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , tightJetSelection_ ( iConfig.getParameter<std::string>("tightJetSelection") )
  , njets_      ( iConfig.getParameter<int>("njets" )      )
  , rsqCut_     ( iConfig.getParameter<double>("rsqCut" )  )
  , mrCut_      ( iConfig.getParameter<double>("mrCut" )   )
{

  theHemispheres_ = consumes<std::vector<math::XYZTLorentzVector> >(iConfig.getParameter<edm::InputTag>("hemispheres"));

  MR_ME_.numerator = nullptr;
  MR_ME_.denominator = nullptr;
  MR_Tight_ME_.numerator = nullptr;
  MR_Tight_ME_.denominator = nullptr;
  MRVsLS_.numerator = nullptr;
  MRVsLS_.denominator = nullptr;
  Rsq_ME_.numerator = nullptr;
  Rsq_ME_.denominator = nullptr;
  Rsq_Tight_ME_.numerator = nullptr;
  Rsq_Tight_ME_.denominator = nullptr;
  RsqVsLS_.numerator = nullptr;
  RsqVsLS_.denominator = nullptr;
  dPhiR_ME_.numerator = nullptr;
  dPhiR_ME_.denominator = nullptr;
  dPhiRVsLS_.numerator = nullptr;
  dPhiRVsLS_.denominator = nullptr;

  MRVsRsq_ME_.numerator = nullptr;
  MRVsRsq_ME_.denominator = nullptr;
  MRVsRsq_Tight_ME_.numerator = nullptr;
  MRVsRsq_Tight_ME_.denominator = nullptr;

}

RazorMonitor::~RazorMonitor()
{
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

MEbinning RazorMonitor::getHistoLSPSet(edm::ParameterSet pset)
{
  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      0.,
      double(pset.getParameter<int32_t>("nbins"))
      };
}

void RazorMonitor::setMETitle(RazorME& me, std::string titleX, std::string titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void RazorMonitor::bookME(DQMStore::IBooker &ibooker, RazorME& me, const std::string& histname, const std::string& histtitle, int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void RazorMonitor::bookME(DQMStore::IBooker &ibooker, RazorME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void RazorMonitor::bookME(DQMStore::IBooker &ibooker, RazorME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void RazorMonitor::bookME(DQMStore::IBooker &ibooker, RazorME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void RazorMonitor::bookME(DQMStore::IBooker &ibooker, RazorME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
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

void RazorMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());

  // 1D hist, MR
  histname = "MR"; histtitle = "PF MR";
  bookME(ibooker,MR_ME_,histname,histtitle,mr_binning_);
  setMETitle(MR_ME_,"PF M_{R} [GeV]","events / [GeV]");

  // profile, MR vs LS
  histname = "MRVsLS"; histtitle = "PF MR vs LS";
  bookME(ibooker,MRVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, mr_binning_.front(),mr_binning_.back());
  setMETitle(MRVsLS_,"LS","PF M_{R} [GeV]");

  // 1D hist, MR, with tight jet requirement
  histname = "MR_Tight"; histtitle = "PF MR";
  bookME(ibooker,MR_Tight_ME_,histname,histtitle,mr_binning_);
  setMETitle(MR_Tight_ME_,"PF M_{R} [GeV]","events / [GeV]");

  // 1D hist, Rsq
  histname = "Rsq"; histtitle = "PF Rsq";
  bookME(ibooker,Rsq_ME_,histname,histtitle,rsq_binning_);
  setMETitle(Rsq_ME_,"PF R^{2}","events");

  // profile, Rsq vs LS
  histname = "RsqVsLS"; histtitle = "PF Rsq vs LS";
  bookME(ibooker,RsqVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, rsq_binning_.front(), rsq_binning_.back());
  setMETitle(RsqVsLS_,"LS","PF R^{2}");

  // 1D hist, Rsq, with tight jet requirement
  histname = "Rsq_Tight"; histtitle = "PF Rsq";
  bookME(ibooker,Rsq_Tight_ME_,histname,histtitle,rsq_binning_);
  setMETitle(Rsq_Tight_ME_,"PF R^{2}","events");

  // 1D hist, dPhiR
  histname = "dPhiR"; histtitle = "dPhiR";
  bookME(ibooker,dPhiR_ME_,histname,histtitle,dphiR_binning_);
  setMETitle(dPhiR_ME_,"dPhi_{R}","events");

  // profile, dPhiR vs LS
  histname = "dPhiRVsLS"; histtitle = "PF dPhiR vs LS";
  bookME(ibooker,dPhiRVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, dphiR_binning_.front(), dphiR_binning_.back());
  setMETitle(dPhiRVsLS_,"LS","dPhi_{R}");

  // 2D hist, MR & Rsq
  histname = "MRVsRsq"; histtitle = "PF MR vs PF Rsq";
  bookME(ibooker,MRVsRsq_ME_,histname,histtitle,mr_binning_, rsq_binning_);
  setMETitle(MRVsRsq_ME_,"M_{R} [GeV]","R^{2}");

  // 2D hist, MR & Rsq, with tight jet requirement
  histname = "MRVsRsq_Tight"; histtitle = "PF MR vs PF Rsq";
  bookME(ibooker,MRVsRsq_Tight_ME_,histname,histtitle,mr_binning_, rsq_binning_);
  setMETitle(MRVsRsq_Tight_ME_,"M_{R} [GeV]","R^{2}");

  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
void RazorMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  //met collection
  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken( metToken_, metHandle );
  reco::PFMET pfmet = metHandle->front();
  if ( ! metSelection_( pfmet ) ) return;

  `//jet collection, track # of jets for two working points
  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken( jetToken_, jetHandle );
  std::vector<reco::PFJet> jets;
  std::vector<reco::PFJet> tightJets;
  if ( int(jetHandle->size()) < njets_ ) return;
  for ( auto const & j : *jetHandle ) {
    if ( jetSelection_( j ) ) jets.push_back(j);
    if ( tightJetSelection_( j ) ) tightJets.push_back(j);
  }
  if ( int(jets.size()) < njets_ ) return;

  //razor hemisphere clustering from previous step
  edm::Handle< vector<math::XYZTLorentzVector> > hemispheres;
  iEvent.getByToken (theHemispheres_,hemispheres);

  if (not hemispheres.isValid()){
    return;
  }

  if(hemispheres->size() ==0){  // the Hemisphere Maker will produce an empty collection of hemispheres if # of jets is too high
    edm::LogError("DQM_HLT_Razor") << "Cannot calculate M_R and R^2 because there are too many jets! (trigger passed automatically without forming the hemispheres)" << endl;
    return;
  }

  if(hemispheres->size() != 0 && hemispheres->size() != 2 && hemispheres->size() != 5 && hemispheres->size() != 10){
    edm::LogError("DQM_HLT_Razor") << "Invalid hemisphere collection!  hemispheres->size() = " << hemispheres->size() << endl;
    return;
  }

  //define hemispheres
  TLorentzVector ja(hemispheres->at(0).x(),hemispheres->at(0).y(),hemispheres->at(0).z(),hemispheres->at(0).t());
  TLorentzVector jb(hemispheres->at(1).x(),hemispheres->at(1).y(),hemispheres->at(1).z(),hemispheres->at(1).t());

  //dummy vector (this trigger does not care about muons)
  std::vector<math::XYZTLorentzVector> muonVec;

  //calculate razor variables
  double MR = CalcMR(ja,jb);
  double R  = CalcR(MR,ja,jb,metHandle,muonVec);
  double Rsq = R*R;
  double dPhiR = abs(deltaPhi(ja.Phi(), jb.Phi()));

  //apply offline selection cuts
  if (Rsq<rsqCut_ && MR<mrCut_) return;

  int ls = iEvent.id().luminosityBlock();

  // applying selection for denominator
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  // filling histograms (denominator)
  if (Rsq>=rsqCut_) {
    MR_ME_.denominator -> Fill(MR);
    MRVsLS_.denominator -> Fill(ls, MR);
  }

  if (MR>=mrCut_) {
    Rsq_ME_.denominator -> Fill(Rsq);
    RsqVsLS_.denominator -> Fill(ls, Rsq);
  }

  dPhiR_ME_.denominator -> Fill(dPhiR);
  dPhiRVsLS_.denominator -> Fill(ls, dPhiR);

  MRVsRsq_ME_.denominator -> Fill(MR, Rsq);

  if ( int(tightJets.size()) >= njets_ ) {
    if (Rsq>=rsqCut_) MR_Tight_ME_.denominator -> Fill(MR);
    if (MR>=mrCut_)   Rsq_Tight_ME_.denominator -> Fill(Rsq);
      MRVsRsq_Tight_ME_.denominator -> Fill(MR, Rsq);
  }

  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  // filling histograms (numerator)
  if (Rsq>=rsqCut_) {
    MR_ME_.numerator -> Fill(MR);
    MRVsLS_.numerator -> Fill(ls, MR);
  }

  if (MR>=mrCut_) {
    Rsq_ME_.numerator -> Fill(Rsq);
    RsqVsLS_.numerator -> Fill(ls, Rsq);
  }

  dPhiR_ME_.numerator -> Fill(dPhiR);
  dPhiRVsLS_.numerator -> Fill(ls, dPhiR);

  MRVsRsq_ME_.numerator -> Fill(MR, Rsq);

  if ( int(tightJets.size()) >= njets_ ) {
    if (Rsq>=rsqCut_) MR_Tight_ME_.numerator -> Fill(MR);
    if (MR>=mrCut_)   Rsq_Tight_ME_.numerator -> Fill(Rsq);
    MRVsRsq_Tight_ME_.numerator -> Fill(MR, Rsq);
  }

}

void RazorMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int32_t>   ( "nbins", 2500);
}

void RazorMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/SUSY/Razor" );

  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>("hemispheres",edm::InputTag("hemispheres"))->setComment("hemisphere jets used to compute razor variables");
  desc.add<std::string>("metSelection", "pt > 0");

  // from 2016 offline selection
  desc.add<std::string>("jetSelection", "pt > 80");
  desc.add<std::string>("tightJetSelection", "pt > 120");
  desc.add<int>("njets",      2);
  desc.add<double>("mrCut",    300);
  desc.add<double>("rsqCut",   0.15);

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

  //binning from 2016 offline selection
  edm::ParameterSetDescription histoPSet;
  std::vector<double> mrbins = {0., 100., 200., 300., 400., 500., 575., 650., 750., 900., 1200., 1600., 2500., 4000.};
  histoPSet.add<std::vector<double> >("mrBins", mrbins);

  std::vector<double> rsqbins = {0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.41, 0.52, 0.64, 0.8, 1.5};
  histoPSet.add<std::vector<double> >("rsqBins", rsqbins);

  std::vector<double> dphirbins = {0., 0.5, 1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 3.2};
  histoPSet.add<std::vector<double> >("dphiRBins", dphirbins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("razorMonitoring", desc);
}

//CalcMR and CalcR borrowed from HLTRFilter.cc
double RazorMonitor::CalcMR(TLorentzVector ja, TLorentzVector jb){
  if(ja.Pt()<=0.1) return -1;

  ja.SetPtEtaPhiM(ja.Pt(),ja.Eta(),ja.Phi(),0.0);
  jb.SetPtEtaPhiM(jb.Pt(),jb.Eta(),jb.Phi(),0.0);

  if(ja.Pt() > jb.Pt()){
    TLorentzVector temp = ja;
    ja = jb;
    jb = temp;
  }

  double A = ja.P();
  double B = jb.P();
  double az = ja.Pz();
  double bz = jb.Pz();
  TVector3 jaT, jbT;
  jaT.SetXYZ(ja.Px(),ja.Py(),0.0);
  jbT.SetXYZ(jb.Px(),jb.Py(),0.0);
  double ATBT = (jaT+jbT).Mag2();

  double MR = sqrt((A+B)*(A+B)-(az+bz)*(az+bz)-
                   (jbT.Dot(jbT)-jaT.Dot(jaT))*(jbT.Dot(jbT)-jaT.Dot(jaT))/(jaT+jbT).Mag2());
  double mybeta = (jbT.Dot(jbT)-jaT.Dot(jaT))/
    sqrt(ATBT*((A+B)*(A+B)-(az+bz)*(az+bz)));

  double mygamma = 1./sqrt(1.-mybeta*mybeta);

  //use gamma times MRstar
  return MR*mygamma;
}

double RazorMonitor::CalcR(double MR, TLorentzVector ja, TLorentzVector jb, edm::Handle<std::vector<reco::PFMET> > inputMet, const std::vector<math::XYZTLorentzVector>& muons){
  //now we can calculate MTR
  TVector3 met;
  met.SetPtEtaPhi((inputMet->front()).pt(),0.0,(inputMet->front()).phi());

  std::vector<math::XYZTLorentzVector>::const_iterator muonIt;
  for(muonIt = muons.begin(); muonIt!=muons.end(); muonIt++){
    TVector3 tmp;
    tmp.SetPtEtaPhi(muonIt->pt(),0,muonIt->phi());
    met-=tmp;
  }

  double MTR = sqrt(0.5*(met.Mag()*(ja.Pt()+jb.Pt()) - met.Dot(ja.Vect()+jb.Vect())));

  //filter events
  return float(MTR)/float(MR); //R
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RazorMonitor);

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
  , eleToken_             ( mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons") ) )   
  , muoToken_             ( mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) ) 
  //, ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("lsPSet")     ) )
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

  theHemispheres_ = consumes<std::vector<math::XYZTLorentzVector> >(iConfig.getParameter<edm::InputTag>("hemispheres"));

  ls_binning_.nbins = 10;
  ls_binning_.xmin = 0;
  ls_binning_.xmax = 100000;
  mr_binning_.nbins = 10;
  mr_binning_.xmin = 0;
  mr_binning_.xmax = 1000;
  rsq_binning_.nbins=10;
  rsq_binning_.xmin=0;
  rsq_binning_.xmax=1;
  dphiR_binning_.nbins=10;
  dphiR_binning_.xmin=-3.2;
  dphiR_binning_.xmax=3.2;

  MR_ME_.numerator = nullptr;
  MR_ME_.denominator = nullptr;
  MR_Jet120_ME_.numerator = nullptr;
  MR_Jet120_ME_.denominator = nullptr;
  MRVsLS_.numerator = nullptr;
  MRVsLS_.denominator = nullptr;
  Rsq_ME_.numerator = nullptr;
  Rsq_ME_.denominator = nullptr;
  Rsq_Jet120_ME_.numerator = nullptr;
  Rsq_Jet120_ME_.denominator = nullptr;
  RsqVsLS_.numerator = nullptr;
  RsqVsLS_.denominator = nullptr;
  dPhiR_ME_.numerator = nullptr;
  dPhiR_ME_.denominator = nullptr;
  dPhiRVsLS_.numerator = nullptr;
  dPhiRVsLS_.denominator = nullptr;

  MRVsRsq_ME_.numerator = nullptr;
  MRVsRsq_ME_.denominator = nullptr;
  MRVsRsq_Jet120_ME_.numerator = nullptr;
  MRVsRsq_Jet120_ME_.denominator = nullptr;


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

  histname = "MR"; histtitle = "PF MR";
  bookME(ibooker,MR_ME_,histname,histtitle,mr_binning_.nbins,mr_binning_.xmin, mr_binning_.xmax);
  setMETitle(MR_ME_,"PF M_R [GeV]","events / [GeV]");

  histname = "MRVsLS"; histtitle = "PF MR vs LS";
  bookME(ibooker,MRVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, mr_binning_.xmin, mr_binning_.xmax);
  setMETitle(MRVsLS_,"LS","PF M_R [GeV]");

  histname = "MR_Jet120"; histtitle = "PF MR";
  bookME(ibooker,MR_Jet120_ME_,histname,histtitle,mr_binning_.nbins,mr_binning_.xmin, mr_binning_.xmax);
  setMETitle(MR_Jet120_ME_,"PF M_R [GeV]","events / [GeV]");

  histname = "Rsq"; histtitle = "PF Rsq";
  bookME(ibooker,Rsq_ME_,histname,histtitle,rsq_binning_.nbins,rsq_binning_.xmin, rsq_binning_.xmax);
  setMETitle(Rsq_ME_,"PF R^2","events");

  histname = "RsqVsLS"; histtitle = "PF Rsq vs LS";
  bookME(ibooker,RsqVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, rsq_binning_.xmin, rsq_binning_.xmax);
  setMETitle(RsqVsLS_,"LS","PF R^2");

  histname = "Rsq_Jet120"; histtitle = "PF Rsq";
  bookME(ibooker,Rsq_Jet120_ME_,histname,histtitle,rsq_binning_.nbins,rsq_binning_.xmin, rsq_binning_.xmax);
  setMETitle(Rsq_Jet120_ME_,"PF R^2","events");

  histname = "dPhiR"; histtitle = "dPhiR";
  bookME(ibooker,dPhiR_ME_,histname,histtitle,dphiR_binning_.nbins,dphiR_binning_.xmin, dphiR_binning_.xmax);
  setMETitle(dPhiR_ME_,"dPhi_R","events");

  histname = "dPhiRVsLS"; histtitle = "PF dPhiR vs LS";
  bookME(ibooker,dPhiRVsLS_,histname,histtitle,ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, dphiR_binning_.xmin, dphiR_binning_.xmax);
  setMETitle(dPhiRVsLS_,"LS","dPhi_R");

  histname = "MRVsRsq"; histtitle = "PF MR vs PF Rsq";
  bookME(ibooker,MRVsRsq_ME_,histname,histtitle,mr_binning_.nbins, mr_binning_.xmin, mr_binning_.xmax, rsq_binning_.nbins, rsq_binning_.xmin, rsq_binning_.xmax);
  setMETitle(MRVsRsq_ME_,"M_R [GeV]","R^2");

  histname = "MRVsRsq_Jet120"; histtitle = "PF MR vs PF Rsq";
  bookME(ibooker,MRVsRsq_Jet120_ME_,histname,histtitle,mr_binning_.nbins, mr_binning_.xmin, mr_binning_.xmax, rsq_binning_.nbins, rsq_binning_.xmin, rsq_binning_.xmax);
  setMETitle(MRVsRsq_Jet120_ME_,"M_R [GeV]","R^2");

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

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken( metToken_, metHandle );
  reco::PFMET pfmet = metHandle->front();
  if ( ! metSelection_( pfmet ) ) return;

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
  
  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken( muoToken_, muoHandle );
  if ( int(muoHandle->size()) < nmuons_ ) return;
  std::vector<reco::Muon> muons;
  for ( auto const & m : *muoHandle ) {
    if ( muoSelection_( m ) ) muons.push_back(m);
  }
  if ( int(muons.size()) < nmuons_ ) return;

  edm::Handle< vector<math::XYZTLorentzVector> > hemispheres;
  iEvent.getByToken (theHemispheres_,hemispheres);

  float HT = 0.0;
  float nJets80 = 0;
  float nJets120 = 0;

  for (unsigned int i=0; i<jets.size(); i++) {
    if(std::abs(jets.at(i).eta()) < 3.0 && jets.at(i).pt() >= 40.0){
      HT += jets.at(i).pt();
      if (jets.at(i).pt() >= 80) nJets80++;
      if (jets.at(i).pt() >= 120) nJets120++;
    }
  }

  //float met = pfmet.pt();
  //float phi = pfmet.phi();

  if (not hemispheres.isValid()){
    return;
  }

  if(hemispheres->size() ==0){  // the Hemisphere Maker will produce an empty collection of hemispheres if # of jets is too high
    //edm::LogError("DQM_HLT_Razor") << "Cannot calculate M_R and R^2 because there are too many jets! (trigger passed automatically without forming the hemispheres)" << 
    //endl;
    return;
  }

  if(hemispheres->size() != 0 && hemispheres->size() != 2 && hemispheres->size() != 5 && hemispheres->size() != 10){
    edm::LogError("DQM_HLT_Razor") << "Invalid hemisphere collection!  hemispheres->size() = " << hemispheres->size() << endl;
    return;
  }

  TLorentzVector ja(hemispheres->at(0).x(),hemispheres->at(0).y(),hemispheres->at(0).z(),hemispheres->at(0).t());
  TLorentzVector jb(hemispheres->at(1).x(),hemispheres->at(1).y(),hemispheres->at(1).z(),hemispheres->at(1).t());

  //dummy vector (this trigger does not care about muons)
  std::vector<math::XYZTLorentzVector> muonVec;

  double MR = CalcMR(ja,jb);
  double R  = CalcR(MR,ja,jb,metHandle,muonVec);
  double Rsq = R*R;
  double dPhiR = deltaPhi(ja.Phi(), jb.Phi());

  std::cout << MR << ", "<< Rsq << ", " << dPhiR << std::endl;

  //offline selection
  if (Rsq<0.15 && MR<300) return;
  if (nJets80<2) return;

  // filling histograms (denominator)

  int ls = iEvent.id().luminosityBlock();

  // applying selection for denominator
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  if (Rsq>=0.15) {
    MR_ME_.denominator -> Fill(MR);
    MRVsLS_.denominator -> Fill(ls, MR);
  }

  if (MR>=300) {
    Rsq_ME_.denominator -> Fill(Rsq);
    RsqVsLS_.denominator -> Fill(ls, Rsq);
  }

  dPhiR_ME_.denominator -> Fill(dPhiR);
  dPhiRVsLS_.denominator -> Fill(ls, dPhiR);

  MRVsRsq_ME_.denominator -> Fill(MR, Rsq);

  if (nJets120>=2) {
    if (Rsq>=0.15) MR_Jet120_ME_.denominator -> Fill(MR);
    if (MR>=300)   Rsq_Jet120_ME_.denominator -> Fill(Rsq);
      MRVsRsq_Jet120_ME_.denominator -> Fill(MR, Rsq);
  }

  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  if (Rsq>0.15) {
    MR_ME_.numerator -> Fill(MR);
    MRVsLS_.numerator -> Fill(ls, MR);
  }

  if (MR>=300) {
    Rsq_ME_.numerator -> Fill(Rsq);
    RsqVsLS_.numerator -> Fill(ls, Rsq);
  }

  dPhiR_ME_.numerator -> Fill(dPhiR);
  dPhiRVsLS_.numerator -> Fill(ls, dPhiR);

  MRVsRsq_ME_.numerator -> Fill(MR, Rsq);

  if (nJets120>=2) {
    if (Rsq>=0.15) MR_Jet120_ME_.numerator -> Fill(MR);
    if (MR>=300)   Rsq_Jet120_ME_.numerator -> Fill(Rsq);
    MRVsRsq_Jet120_ME_.numerator -> Fill(MR, Rsq);
  }

}

//void RazorMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
//{
//  pset.add<int>   ( "nbins");
//  pset.add<double>( "xmin" );
//  pset.add<double>( "xmax" );
//}
//
//void RazorMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
//{
//  pset.add<int>   ( "nbins", 2500);
//}

void RazorMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/SUSY/Razor" );

  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<edm::InputTag>("hemispheres",edm::InputTag("hemispheres"))->setComment("hemisphere jets used to compute razor variables");
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

  //edm::ParameterSetDescription histoPSet;
  //edm::ParameterSetDescription metPSet;
  //fillHistoPSetDescription(metPSet);
  //histoPSet.add<edm::ParameterSetDescription>("metPSet", metPSet);
  //std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.};
  //histoPSet.add<std::vector<double> >("metBinning", bins);

  //edm::ParameterSetDescription lsPSet;
  //fillHistoLSPSetDescription(lsPSet);
  //histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  //desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

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

#include "DQMOffline/Trigger/plugins/BPHMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

BPHMonitor::BPHMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , muoToken_             ( mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )  
  , bsToken_              ( mayConsume<reco::BeamSpot>             (iConfig.getParameter<edm::InputTag>("beamSpot")))
  , trToken_              ( mayConsume<reco::TrackCollection>             (iConfig.getParameter<edm::InputTag>("tracks")))
  , phToken_              ( mayConsume<reco::PhotonCollection>             (iConfig.getParameter<edm::InputTag>("photons")))
  , phi_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("phiPSet")    ) )
  , pt_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("ptPSet")    ) )
  , eta_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("etaPSet")    ) )
  , d0_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("d0PSet")     ) )
  , z0_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("z0PSet")     ) )
  , dR_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("dRPSet")     ) )
  , mass_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("massPSet")     ) )
  , dca_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("dcaPSet")     ) )
  , ds_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("dsPSet")     ) )
  , cos_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("cosPSet")     ) )
  , prob_binning_           ( getHistoPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>   ("probPSet")     ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , muoSelection_ref ( iConfig.getParameter<std::string>("muoSelection_ref") )
  , muoSelection_tag ( iConfig.getParameter<std::string>("muoSelection_tag") )
  , muoSelection_probe ( iConfig.getParameter<std::string>("muoSelection_probe") )
  , nmuons_     ( iConfig.getParameter<int>("nmuons" )     )
  , tnp_     ( iConfig.getParameter<bool>("tnp" )     )
  , L3_     ( iConfig.getParameter<int>("L3" )     )
  , trOrMu_     ( iConfig.getParameter<int>("trOrMu" )     )
  , Jpsi_     ( iConfig.getParameter<int>("Jpsi" )     )
  , Upsilon_     ( iConfig.getParameter<int>("Upsilon" )     )//if ==1 path with Upsilon constraint
  , enum_     ( iConfig.getParameter<int>("enum" )     )
  , seagull_     ( iConfig.getParameter<int>("seagull" )     )
  , maxmass_     ( iConfig.getParameter<double>("maxmass" )     )
  , minmass_     ( iConfig.getParameter<double>("minmass" )     )
  , maxmassJpsi     ( iConfig.getParameter<double>("maxmassJpsi" )     )
  , minmassJpsi     ( iConfig.getParameter<double>("minmassJpsi" )     )
  , maxmassUpsilon     ( iConfig.getParameter<double>("maxmassUpsilon" )     )
  , minmassUpsilon     ( iConfig.getParameter<double>("minmassUpsilon" )     )
  , maxmassJpsiTk     ( iConfig.getParameter<double>("maxmassJpsiTk" )     )
  , minmassJpsiTk     ( iConfig.getParameter<double>("minmassJpsiTk" )     )
  , minprob     ( iConfig.getParameter<double>("minprob" )     )
  , mincos     ( iConfig.getParameter<double>("mincos" )     )
  , minDS     ( iConfig.getParameter<double>("minDS" )     )
  , hltTrigResTag_ (mayConsume<edm::TriggerResults>( iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet").getParameter<edm::InputTag>("hltInputTag")))
  , hltInputTag_ (mayConsume<trigger::TriggerEvent>( iConfig.getParameter<edm::InputTag>("hltTriggerSummaryAOD")))
  , hltpaths_num     ( iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet").getParameter<std::vector<std::string>>("hltPaths"))
  , hltpaths_den     ( iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet").getParameter<std::vector<std::string>>("hltPaths"))
  , trSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , trSelection_ref ( iConfig.getParameter<std::string>("trSelection_ref") )
  , DMSelection_ref ( iConfig.getParameter<std::string>("DMSelection_ref") )

{

  muPhi_.numerator   = nullptr;
  muPhi_.denominator = nullptr;
  muEta_.numerator   = nullptr;
  muEta_.denominator = nullptr;
  muPt_.numerator   = nullptr;
  muPt_.denominator = nullptr;
  mud0_.numerator   = nullptr;
  mud0_.denominator   = nullptr;
  muz0_.numerator   = nullptr;
  muz0_.denominator = nullptr;

  mu1Phi_.numerator   = nullptr;
  mu1Phi_.denominator = nullptr;
  mu1Eta_.numerator   = nullptr;
  mu1Eta_.denominator = nullptr;
  mu1Pt_.numerator   = nullptr;
  mu1Pt_.denominator = nullptr;

  mu2Phi_.numerator   = nullptr;
  mu2Phi_.denominator = nullptr;
  mu2Eta_.numerator   = nullptr;
  mu2Eta_.denominator = nullptr;
  mu2Pt_.numerator   = nullptr;
  mu2Pt_.denominator = nullptr;

  mu3Phi_.numerator   = nullptr;
  mu3Phi_.denominator = nullptr;
  mu3Eta_.numerator   = nullptr;
  mu3Eta_.denominator = nullptr;
  mu3Pt_.numerator   = nullptr;
  mu3Pt_.denominator = nullptr;

  phPhi_.numerator   = nullptr;
  phPhi_.denominator = nullptr;
  phEta_.numerator   = nullptr;
  phEta_.denominator = nullptr;
  phPt_.numerator   = nullptr;
  phPt_.denominator = nullptr;


  DiMuPhi_.numerator   = nullptr;
  DiMuPhi_.denominator = nullptr;
  DiMuEta_.numerator   = nullptr;
  DiMuEta_.denominator = nullptr;
  DiMuPt_.numerator   = nullptr;
  DiMuPt_.denominator = nullptr;
  DiMuPVcos_.numerator   = nullptr;
  DiMuPVcos_.denominator = nullptr;
  DiMuProb_.numerator   = nullptr;
  DiMuProb_.denominator = nullptr;
  DiMuDS_.numerator   = nullptr;
  DiMuDS_.denominator = nullptr;
  DiMuDCA_.numerator   = nullptr;
  DiMuDCA_.denominator = nullptr;
  DiMuMass_.numerator   = nullptr;
  DiMuMass_.denominator = nullptr;
  DiMudR_.numerator   = nullptr;
  DiMudR_.denominator = nullptr;


}

BPHMonitor::~BPHMonitor()
{
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

MEbinning BPHMonitor::getHistoPSet(edm::ParameterSet pset)
{
  // Due to the setup of the fillDescription only one of the
  // two cases is possible at this point.
  if (pset.existsAs<std::vector<double>>("edges")) {
    return MEbinning{pset.getParameter<std::vector<double>>("edges")};
  }

  return MEbinning{
    pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
      };
}

// MEbinning BPHMonitor::getHistoLSPSet(edm::ParameterSet pset)
// {
//   return MEbinning{
//     pset.getParameter<int32_t>("nbins"),
//       0.,
//       double(pset.getParameter<int32_t>("nbins"))
//       };
// }

void BPHMonitor::setMETitle(METME& me, std::string titleX, std::string titleY)
{
  me.numerator->setAxisTitle(titleX,1);
  me.numerator->setAxisTitle(titleY,2);
  me.denominator->setAxisTitle(titleX,1);
  me.denominator->setAxisTitle(titleY,2);

}

void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, std::string& histname, std::string& histtitle, int& nbins, double& min, double& max)
{
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, min, max);
}
void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, std::string& histname, std::string& histtitle, std::vector<double> binning)
{
  int nbins = binning.size()-1;
  std::vector<float> fbinning(binning.begin(),binning.end());
  float* arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname+"_numerator",   histtitle+" (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname+"_denominator", histtitle+" (denominator)", nbins, arr);
}
void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, double& ymin, double& ymax)
{
  me.numerator   = ibooker.bookProfile(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}
void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, int& nbinsY, double& ymin, double& ymax)
{
  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}
void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY)
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

void BPHMonitor::bookME(DQMStore::IBooker &ibooker, METME &me, std::string &histname, std::string &histtitle, /*const*/ MEbinning& binning)
{
  // If the vector in the binning is filled use the bins defined there
  // otherwise use a linear binning between min and max
  if (binning.edges.empty()) {
    this->bookME(ibooker, me, histname, histtitle, binning.nbins, binning.xmin, binning.xmax);
  } else {
    this->bookME(ibooker, me, histname, histtitle, binning.edges);
  }
}


void BPHMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				edm::Run const        & iRun,
				edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle, istnp, trMuPh;
  bool Ph_; if (enum_==7) Ph_ = true;
  if (tnp_) istnp = "Tag_and_Probe/"; else istnp = "";
  std::string currentFolder = folderName_ + istnp;
  ibooker.setCurrentFolder(currentFolder);
  if (trOrMu_) trMuPh = "tr";else if (Ph_) trMuPh = "ph";else trMuPh = "mu";

  if (enum_==7 || enum_==1 || enum_==9 || enum_==10){  
    histname = trMuPh+"Pt"; histtitle = trMuPh+"_P_{t}";
    bookME(ibooker,muPt_,histname,histtitle, pt_binning_);
    setMETitle(muPt_,trMuPh+"_Pt[GeV]","events/1GeV");

    histname =trMuPh+"Phi"; histtitle =trMuPh+"Phi";
    bookME(ibooker,muPhi_,histname,histtitle, phi_binning_);
    setMETitle(muPhi_,trMuPh+"_#phi","events / 0.1 rad");

    histname =trMuPh+"Eta"; histtitle = trMuPh+"_Eta";
    bookME(ibooker,muEta_,histname,histtitle, eta_binning_);
    setMETitle(muEta_,trMuPh+"_#eta","events/ ");
  }
  else if(enum_==11){
    trMuPh = "tr";
    histname = trMuPh+"1Pt"; histtitle = trMuPh+"1_P_{t}";
    bookME(ibooker,mu1Pt_,histname,histtitle, pt_binning_);
    setMETitle(mu1Pt_,trMuPh+"_Pt[GeV]","events/1GeV");

    histname =trMuPh+"1Phi"; histtitle =trMuPh+"1Phi";
    bookME(ibooker,mu1Phi_,histname,histtitle, phi_binning_);
    setMETitle(mu1Phi_,trMuPh+"_#phi","events / 0.1 rad");
  
    histname =trMuPh+"1Eta"; histtitle = trMuPh+"1_Eta";
    bookME(ibooker,mu1Eta_,histname,histtitle, eta_binning_);
    setMETitle(mu1Eta_,trMuPh+"_#eta","events/ ");

    histname = trMuPh+"2Pt"; histtitle = trMuPh+"2_P_{t}";
    bookME(ibooker,mu2Pt_,histname,histtitle, pt_binning_);
    setMETitle(mu2Pt_,trMuPh+"_Pt[GeV]","events/1GeV");

    histname =trMuPh+"2Phi"; histtitle =trMuPh+"2Phi";
    bookME(ibooker,mu2Phi_,histname,histtitle, phi_binning_);
    setMETitle(mu2Phi_,trMuPh+"_#phi","events / 0.1 rad");

    histname =trMuPh+"2Eta"; histtitle = trMuPh+"2_Eta";
    bookME(ibooker,mu2Eta_,histname,histtitle, eta_binning_);
    setMETitle(mu2Eta_,trMuPh+"_#eta","events/ ");

  }

  else{
    histname ="mu1Eta"; histtitle = "mu1Eta";
    bookME(ibooker,mu1Eta_,histname,histtitle, eta_binning_);
    setMETitle(mu1Eta_,"mu1#eta","events/ ");

    histname = "mu1Pt"; histtitle = "mu1_P_{t}";
    bookME(ibooker,mu1Pt_,histname,histtitle, pt_binning_);
    setMETitle(mu1Pt_,"mu1_Pt[GeV]","events/1GeV");

    histname ="mu1Phi"; histtitle ="mu1Phi";
    bookME(ibooker,mu1Phi_,histname,histtitle, phi_binning_);
    setMETitle(mu1Phi_,"mu1_#phi","events / 0.1 rad");

    histname ="mu2Eta"; histtitle = "mu2Eta";
    bookME(ibooker,mu2Eta_,histname,histtitle, eta_binning_);
    setMETitle(mu2Eta_,"mu2#eta","events/ ");

    histname = "mu2Pt"; histtitle = "mu2_P_{t}";
    bookME(ibooker,mu2Pt_,histname,histtitle, pt_binning_);
    setMETitle(mu2Pt_,"mu2_Pt[GeV]","events/1GeV");

    histname ="mu2Phi"; histtitle ="mu2Phi";
    bookME(ibooker,mu2Phi_,histname,histtitle, phi_binning_);
    setMETitle(mu2Phi_,"mu2_#phi","events / 0.1 rad");

    histname ="mu3Eta"; histtitle = "mu3Eta";
    bookME(ibooker,mu3Eta_,histname,histtitle, eta_binning_);
    setMETitle(mu3Eta_,"mu3#eta","events/ ");

    histname = "mu3Pt"; histtitle = "mu3_P_{t}";
    bookME(ibooker,mu3Pt_,histname,histtitle, pt_binning_);
    setMETitle(mu3Pt_,"mu3_Pt[GeV]","events/1GeV");

    histname ="mu3Phi"; histtitle ="mu3Phi";
    bookME(ibooker,mu3Phi_,histname,histtitle, phi_binning_);
    setMETitle(mu3Phi_,"mu3_#phi","events / 0.1 rad");

    histname ="DiMuEta"; histtitle = "DiMuEta";
    bookME(ibooker,DiMuEta_,histname,histtitle, eta_binning_);
    setMETitle(DiMuEta_,"DiMu#eta","events/ ");

    histname = "DiMuPt"; histtitle = "DiMu_P_{t}";
    bookME(ibooker,DiMuPt_,histname,histtitle, pt_binning_);
    setMETitle(DiMuPt_,"DiMu_Pt[GeV]","events/1GeV");

    histname ="DiMuPhi"; histtitle ="DiMuPhi";
    bookME(ibooker,DiMuPhi_,histname,histtitle, phi_binning_);
    setMETitle(DiMuPhi_,"DiMu_#phi","events / 0.1 rad");

    histname ="DiMuPVcos"; histtitle ="DiMuPVcos";
    bookME(ibooker,DiMuPVcos_,histname,histtitle, cos_binning_);
    setMETitle(DiMuPVcos_,"DiMu_#cosPV","events / ");

    histname ="DiMuProb"; histtitle ="DiMuProb";
    bookME(ibooker,DiMuProb_,histname,histtitle, prob_binning_);
    setMETitle(DiMuProb_,"DiMu_#prob","events / ");

    histname ="DiMuDS"; histtitle ="DiMuDS";
    bookME(ibooker,DiMuDS_,histname,histtitle, ds_binning_);
    setMETitle(DiMuDS_,"DiMu_#ds","events / ");


    histname ="DiMuDCA"; histtitle ="DiMuDCA";
    bookME(ibooker,DiMuDCA_,histname,histtitle, dca_binning_);
    setMETitle(DiMuDCA_,"DiMu_#dca","events / ");

    histname ="DiMuMass"; histtitle ="DiMuMass";
    bookME(ibooker,DiMuMass_,histname,histtitle, mass_binning_);
    setMETitle(DiMuMass_,"DiMu_#mass","events / ");

    histname ="DiMudR"; histtitle ="DiMudR";
    bookME(ibooker,DiMudR_,histname,histtitle, dR_binning_);
    setMETitle(DiMudR_,"DiMu_#dR","events / ");

  }

  if (trOrMu_) {
    histname =trMuPh+ "_d0"; histtitle =trMuPh+ "_d0";
    bookME(ibooker,mud0_,histname,histtitle, d0_binning_);
    setMETitle(mud0_,trMuPh+"_d0","events/bin ");

    histname = trMuPh+"_z0"; histtitle =trMuPh+"_z0";
    bookME(ibooker,muz0_,histname,histtitle, z0_binning_);
    setMETitle(muz0_,trMuPh+"_z0","events/bin ");
  }

  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TLorentzVector.h"
void BPHMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken( bsToken_,  beamSpot);

  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken( muoToken_, muoHandle );

  edm::Handle<reco::TrackCollection> trHandle;
  iEvent.getByToken( trToken_, trHandle );

  edm::Handle<reco::PhotonCollection> phHandle;
  iEvent.getByToken( phToken_, phHandle );

  edm::Handle<edm::TriggerResults> handleTriggerTrigRes; 
  edm::Handle<trigger::TriggerEvent> handleTriggerEvent; 
  edm::ESHandle<MagneticField> bFieldHandle;
  // Filter out events if Trigger Filtering is requested
  if (tnp_>0) {//TnP method 
    if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;
    iEvent.getByToken( hltInputTag_, handleTriggerEvent);
    if (handleTriggerEvent->sizeFilters()== 0)return;
    const std::string & hltpath = hltpaths_num[0]; 
    std::vector<reco::Muon> tagMuons;
    for ( auto const & m : *muoHandle ) {//applying tag selection 
      if(false && !matchToTrigger(hltpath,m, handleTriggerEvent)) continue;
      if ( muoSelection_ref( m ) ) tagMuons.push_back(m);
    }
    for (int i = 0; i<int(tagMuons.size());i++){
      for ( auto const & m : *muoHandle ) { 
        if(false && !matchToTrigger(hltpath,m, handleTriggerEvent)) continue;
        if ((tagMuons[i].pt() == m.pt()))continue;//not the same  
        if ((tagMuons[i].p4()+m.p4()).M() >minmass_&& (tagMuons[i].p4()+m.p4()).M() <maxmass_){//near to J/psi mass
          muPhi_.denominator->Fill(m.phi());
          muEta_.denominator->Fill(m.eta());
          muPt_.denominator ->Fill(m.pt());
          if (muoSelection_( m )){
            muPhi_.numerator->Fill(m.phi());
            muEta_.numerator->Fill(m.eta());
            muPt_.numerator ->Fill(m.pt());
          }
        }      
      }
        
    }
    

  }  
  else{//reference method
    if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;
    iEvent.getByToken( hltInputTag_, handleTriggerEvent);
    if (handleTriggerEvent->sizeFilters()== 0)return;
    const std::string & hltpath = hltpaths_den[0]; 
    for (auto const & m : *muoHandle ) {
      if(false && !matchToTrigger(hltpath,m, handleTriggerEvent)) continue;
      if(!muoSelection_ref(m))continue;   
      for (auto const & m1 : *muoHandle ) {
        if (m1.pt() == m.pt())continue;
        if(!muoSelection_ref(m1))continue;   
        if(false && !matchToTrigger(hltpath,m1, handleTriggerEvent)) continue;
        if (enum_<10){
          if (!DMSelection_ref(m1.p4() + m.p4()))continue;
          if (m.charge()*m1.charge()>0 )continue;
        }
        iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
        const reco::BeamSpot& vertexBeamSpot = *beamSpot;
        std::vector<reco::TransientTrack> j_tks;
        reco::TransientTrack mu1TT(m.track(), &(*bFieldHandle));
        reco::TransientTrack mu2TT(m1.track(), &(*bFieldHandle));
        j_tks.push_back(mu1TT);
        j_tks.push_back(mu2TT);
        KalmanVertexFitter jkvf;
        TransientVertex jtv = jkvf.vertex(j_tks);
        if (!jtv.isValid()) continue;
        reco::Vertex jpsivertex = jtv;
        float dimuonCL = 0;
        if( (jpsivertex.chi2()>=0) && (jpsivertex.ndof()>0) )//I think these values are "unphysical"(no one will need to change them ever)so the can be fixed
	  dimuonCL = TMath::Prob(jpsivertex.chi2(), jpsivertex.ndof() );
        math::XYZVector jpperp(m.px() + m1.px() ,
			       m.py() + m1.py() ,
			       0.);
        GlobalPoint jVertex = jtv.position();
        GlobalError jerr    = jtv.positionError();
        GlobalPoint displacementFromBeamspotJpsi( -1*((vertexBeamSpot.x0() - jVertex.x()) + (jVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()),
                                                  -1*((vertexBeamSpot.y0() - jVertex.y()) + (jVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()),
						  0);
        reco::Vertex::Point vperpj(displacementFromBeamspotJpsi.x(), displacementFromBeamspotJpsi.y(), 0.);
        float jpsi_cos = vperpj.Dot(jpperp)/(vperpj.R()*jpperp.R());
        TrajectoryStateClosestToPoint mu1TS = mu1TT.impactPointTSCP();
        TrajectoryStateClosestToPoint mu2TS = mu2TT.impactPointTSCP();
        ClosestApproachInRPhi cApp;
        if (mu1TS.isValid() && mu2TS.isValid()) {
          cApp.calculate(mu1TS.theState(), mu2TS.theState());
        }
        double DiMuMass = (m1.p4()+m.p4()).M();
        switch(enum_){//enum_ = 1...9, represents different sets of variables for different paths, we want to have different hists for different paths
        case 1: tnp_=1;//already filled hists for tnp method
        case 2:
          if ((Jpsi_) && (!Upsilon_)){
            if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
          }
          if ((!Jpsi_) && (Upsilon_)){
            if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
          }
          if (dimuonCL<minprob)continue;
          mu1Phi_.denominator->Fill(m.phi());
          mu1Eta_.denominator->Fill(m.eta());
          mu1Pt_.denominator ->Fill(m.pt());
          mu2Phi_.denominator->Fill(m1.phi());
          mu2Eta_.denominator->Fill(m1.eta());
          mu2Pt_.denominator ->Fill(m1.pt());
          DiMuPt_.denominator ->Fill((m1.p4()+m.p4()).Pt() );
          DiMuEta_.denominator ->Fill((m1.p4()+m.p4()).Eta() );
          DiMuPhi_.denominator ->Fill((m1.p4()+m.p4()).Phi());
          break;
        case 3:
          if ((Jpsi_) && (!Upsilon_)){
            if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
          }
          if ((!Jpsi_) && (Upsilon_)){
            if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
          }
          if (dimuonCL<minprob)continue;
          mu1Eta_.denominator->Fill(m.eta());
          mu1Pt_.denominator ->Fill(m.pt());
          mu2Eta_.denominator->Fill(m1.eta());
          mu2Pt_.denominator ->Fill(m1.pt());
          break; 
        case 4:
          if (dimuonCL<minprob)continue;
          DiMuMass_.denominator ->Fill(DiMuMass);
          if ((Jpsi_) && (!Upsilon_)){
            if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
          }
          if ((!Jpsi_) && (Upsilon_)){
            if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
          }
          mu1Phi_.denominator->Fill(m.phi());
          mu1Eta_.denominator->Fill(m.eta());
          mu1Pt_.denominator ->Fill(m.pt());
          mu2Phi_.denominator->Fill(m1.phi());
          mu2Eta_.denominator->Fill(m1.eta());
          mu2Pt_.denominator ->Fill(m1.pt());
          DiMuPt_.denominator ->Fill((m1.p4()+m.p4()).Pt() );
          DiMuEta_.denominator ->Fill((m1.p4()+m.p4()).Eta() );
          DiMuPhi_.denominator ->Fill((m1.p4()+m.p4()).Phi());
          DiMudR_.denominator ->Fill(reco::deltaR(m,m1));
          break;
        case 5:
          if (dimuonCL<minprob)continue;
          if ((Jpsi_) && (!Upsilon_)){
            if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
          }
  
          if ((!Jpsi_) && (Upsilon_)){
            if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
          }
          mu1Phi_.denominator->Fill(m.phi());
          mu1Eta_.denominator->Fill(m.eta());
          mu1Pt_.denominator ->Fill(m.pt());
          mu2Phi_.denominator->Fill(m1.phi());
          mu2Eta_.denominator->Fill(m1.eta());
          mu2Pt_.denominator ->Fill(m1.pt());
          DiMuPt_.denominator ->Fill((m1.p4()+m.p4()).Pt() );
          DiMuEta_.denominator ->Fill((m1.p4()+m.p4()).Eta() );
          DiMuPhi_.denominator ->Fill((m1.p4()+m.p4()).Phi());
          DiMudR_.denominator ->Fill(reco::deltaR(m,m1));
          break;
        case 6: 
          if (dimuonCL<minprob)continue;
          if ((Jpsi_) && (!Upsilon_)){
            if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
          }
          if ((!Jpsi_) && (Upsilon_)){
            if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
          }
          for (auto const & m2 : *muoHandle) {//triple muon paths
            if(false && !matchToTrigger(hltpath,m2, handleTriggerEvent)) continue;
            if (m2.pt() == m.pt())continue;
            mu1Phi_.denominator->Fill(m.phi());
            mu1Eta_.denominator->Fill(m.eta());
            mu1Pt_.denominator ->Fill(m.pt());
            mu2Phi_.denominator->Fill(m1.phi());
            mu2Eta_.denominator->Fill(m1.eta());
            mu2Pt_.denominator ->Fill(m1.pt());
            mu3Phi_.denominator->Fill(m2.phi());
            mu3Eta_.denominator->Fill(m2.eta());
            mu3Pt_.denominator ->Fill(m2.pt());
          }      
          break;    
  
        case 7:// the hists for photon monitoring will be filled on 515 line
          tnp_=0;
          break;
          
        case 8://vtx monitoring, filling probability, DS, DCA, cos of pointing angle to the PV, eta, pT of dimuon
          if ((Jpsi_) && (!Upsilon_)){
            if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
          }
  
          if ((!Jpsi_) && (Upsilon_)){
            if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
          }
  
          DiMuProb_.denominator ->Fill( dimuonCL);
          if (dimuonCL<minprob)continue;
          DiMuDS_.denominator ->Fill( displacementFromBeamspotJpsi.perp()/sqrt(jerr.rerr(displacementFromBeamspotJpsi)));
          DiMuPVcos_.denominator ->Fill(jpsi_cos );
          DiMuPt_.denominator ->Fill((m1.p4()+m.p4()).Pt() );
          DiMuEta_.denominator ->Fill((m1.p4()+m.p4()).Eta() );
          DiMuDCA_.denominator ->Fill( cApp.distance());
          break;
	case 9:
          if (dimuonCL<minprob)continue;
          if (fabs(jpsi_cos)<mincos)continue;
          if ((displacementFromBeamspotJpsi.perp()/sqrt(jerr.rerr(displacementFromBeamspotJpsi)))<minDS)continue;
  
	  if (trHandle.isValid()){
	    ////////////////////////
  
	    for (auto const & t : *trHandle) {
      
	      if(!trSelection_ref(t))continue;
	      if(false && !matchToTrigger(hltpath,t, handleTriggerEvent)) continue;
              reco::Track itrk1       = t ;                                                
              
              if((reco::deltaR(t,m1) <= 0.001))continue;//checking overlaping
              if((reco::deltaR(t,m) <= 0.001)) continue;
   
              if (! itrk1.quality(reco::TrackBase::highPurity))     continue;
  
              reco::Particle::LorentzVector pB, p1, p2, p3;
              double trackMass2 = 0.493677 *0.493677;
              double MuMass2 = 0.1056583745 *0.1056583745;
              double e1   = sqrt(m.momentum().Mag2()  + MuMass2          );
              double e2   = sqrt(m1.momentum().Mag2()  + MuMass2          );
              double e3   = sqrt(itrk1.momentum().Mag2() + trackMass2  );
              
              p1   = reco::Particle::LorentzVector(m.px() , m.py() , m.pz() , e1  );
              p2   = reco::Particle::LorentzVector(m1.px() , m1.py() , m1.pz() , e2  );
              p3   = reco::Particle::LorentzVector(itrk1.px(), itrk1.py(), itrk1.pz(), e3  );
              pB   = p1 + p2 + p3;
              if( pB.mass()> maxmassJpsiTk || pB.mass()< minmassJpsiTk)continue;
              reco::TransientTrack trTT(itrk1, &(*bFieldHandle));
  
              std::vector<reco::TransientTrack> t_tks;
              t_tks.push_back(mu1TT);
              t_tks.push_back(mu2TT);
              t_tks.push_back(trTT);
              
              KalmanVertexFitter kvf;
              TransientVertex tv  = kvf.vertex(t_tks);
              reco::Vertex vertex = tv;
              if (!tv.isValid()) continue;
              float JpsiTkCL = 0;
              if ((vertex.chi2()>=0.0) && (vertex.ndof()>0) )   
		JpsiTkCL = TMath::Prob(vertex.chi2(), vertex.ndof() );
              math::XYZVector pperp(m.px() + m1.px() + itrk1.px(),
                                    m.py() + m1.py() + itrk1.py(),
                                    0.);
              GlobalPoint secondaryVertex = tv.position();
              GlobalError err             = tv.positionError();
              GlobalPoint displacementFromBeamspot( -1*((vertexBeamSpot.x0() - secondaryVertex.x()) + 
                                                        (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()), 
                                                    -1*((vertexBeamSpot.y0() - secondaryVertex.y()) + 
                                                        (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()), 
                                                    0);
              reco::Vertex::Point vperp(displacementFromBeamspot.x(),displacementFromBeamspot.y(),0.);
              float jpsiKcos = vperp.Dot(pperp)/(vperp.R()*pperp.R());
	      if (JpsiTkCL<minprob)continue;
	      if (fabs(jpsiKcos)<mincos)continue;
	      if ((displacementFromBeamspot.perp()/sqrt(err.rerr(displacementFromBeamspot)))<minDS)continue;
	      muPhi_.denominator->Fill(t.phi());
	      muEta_.denominator->Fill(t.eta());
	      muPt_.denominator ->Fill(t.pt());
   
	      /////////////////////////
	    }
	  }
	  break;
	case 10:
	  if (trHandle.isValid()){
	    for (auto const & t : *trHandle) {
	      if(!trSelection_ref(t))continue;
	      if(false && !matchToTrigger(hltpath,t, handleTriggerEvent)) continue;
	      reco::Track itrk1       = t ;                                                
	      if((reco::deltaR(t,m1) <= 0.001))continue;//checking overlaping
	      if((reco::deltaR(t,m) <= 0.001)) continue;
	      if (! itrk1.quality(reco::TrackBase::highPurity))     continue;
	      reco::Particle::LorentzVector pB, p2, p3;
	      double trackMass2 = 0.493677 *0.493677;
	      double MuMass2 = 0.1056583745 *0.1056583745;
	      double e2   = sqrt(m1.momentum().Mag2()  + MuMass2          );
	      double e3   = sqrt(itrk1.momentum().Mag2() + trackMass2  );
	      p2   = reco::Particle::LorentzVector(m1.px() , m1.py() , m1.pz() , e2  );
	      p3   = reco::Particle::LorentzVector(itrk1.px(), itrk1.py(), itrk1.pz(), e3  );
	      pB   = p2 + p3;
	      if( pB.mass()> maxmassJpsiTk || pB.mass()< minmassJpsiTk)continue;
	      reco::TransientTrack trTT(itrk1, &(*bFieldHandle));
	      std::vector<reco::TransientTrack> t_tks;
	      t_tks.push_back(mu2TT);
	      t_tks.push_back(trTT);
	      KalmanVertexFitter kvf;
	      TransientVertex tv  = kvf.vertex(t_tks);
	      reco::Vertex vertex = tv;
	      if (!tv.isValid()) continue;
	      float JpsiTkCL = 0;
	      if ((vertex.chi2()>=0.0) && (vertex.ndof()>0) )   
		JpsiTkCL = TMath::Prob(vertex.chi2(), vertex.ndof() );
	      math::XYZVector pperp(m1.px() + itrk1.px(),
				    m1.py() + itrk1.py(),
				    0.);
	      GlobalPoint secondaryVertex = tv.position();
	      GlobalError err             = tv.positionError();
	      GlobalPoint displacementFromBeamspot( -1*((vertexBeamSpot.x0() - secondaryVertex.x()) + 
							(secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()), 
						    -1*((vertexBeamSpot.y0() - secondaryVertex.y()) + 
							(secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()), 
						    0);
	      reco::Vertex::Point vperp(displacementFromBeamspot.x(),displacementFromBeamspot.y(),0.);
	      if (JpsiTkCL<minprob)continue;
	      muPhi_.denominator->Fill(m1.phi());
	      muEta_.denominator->Fill(m1.eta());
	      muPt_.denominator ->Fill(m1.pt());
	    }
	  }
	  break;
	case 11:
	  if (dimuonCL<minprob)continue;
	  if (fabs(jpsi_cos)<mincos)continue;
	  if ((displacementFromBeamspotJpsi.perp()/sqrt(jerr.rerr(displacementFromBeamspotJpsi)))<minDS)continue;
	  if (trHandle.isValid()){
	    ////////////////////////
	    for (auto const & t : *trHandle) {
	      if(!trSelection_ref(t))continue;
	      if(false && !matchToTrigger(hltpath,t, handleTriggerEvent)) continue;
	      for (auto const & t1 : *trHandle) {
		if(!trSelection_ref(t1))continue;
		if(false && !matchToTrigger(hltpath,t1, handleTriggerEvent)) continue;
		reco::Track itrk1       = t ;                                                
		reco::Track itrk2       = t1 ;                                                
		if((reco::deltaR(t,m1) <= 0.001))continue;//checking overlaping
		if((reco::deltaR(t,t1) <= 0.001))continue;//checking overlaping
		if((reco::deltaR(t,m) <= 0.001)) continue;
		if (! itrk1.quality(reco::TrackBase::highPurity))     continue;
		if (! itrk2.quality(reco::TrackBase::highPurity))     continue;
		reco::Particle::LorentzVector pB, p1, p2, p3, p4;
		double trackMass2 = 0.493677 *0.493677;
		double MuMass2 = 0.1056583745 *0.1056583745;
		double e1   = sqrt(m.momentum().Mag2()  + MuMass2          );
		double e2   = sqrt(m1.momentum().Mag2()  + MuMass2          );
		double e3   = sqrt(itrk1.momentum().Mag2() + trackMass2  );
		double e4   = sqrt(itrk2.momentum().Mag2() + trackMass2  );
		p1   = reco::Particle::LorentzVector(m.px() , m.py() , m.pz() , e1  );
		p2   = reco::Particle::LorentzVector(m1.px() , m1.py() , m1.pz() , e2  );
		p3   = reco::Particle::LorentzVector(itrk1.px(), itrk1.py(), itrk1.pz(), e3  );
		p4   = reco::Particle::LorentzVector(itrk2.px(), itrk2.py(), itrk2.pz(), e4  );
		pB   = p1 + p2 + p3 + p4;
		if( pB.mass()> maxmassJpsiTk || pB.mass()< minmassJpsiTk)continue;
		reco::TransientTrack trTT(itrk1, &(*bFieldHandle));
		reco::TransientTrack tr1TT(itrk2, &(*bFieldHandle));
		std::vector<reco::TransientTrack> t_tks;
		t_tks.push_back(mu1TT);
		t_tks.push_back(mu2TT);
		t_tks.push_back(trTT);
		t_tks.push_back(tr1TT);
		KalmanVertexFitter kvf;
		TransientVertex tv  = kvf.vertex(t_tks);
		reco::Vertex vertex = tv;
		if (!tv.isValid()) continue;
		float JpsiTkCL = 0;
		if ((vertex.chi2()>=0.0) && (vertex.ndof()>0) )   
		  JpsiTkCL = TMath::Prob(vertex.chi2(), vertex.ndof() );
		math::XYZVector pperp(m.px() + m1.px() + itrk1.px() + itrk2.px(),
				      m.py() + m1.py() + itrk1.py() + itrk2.py(),
				      0.);
		GlobalPoint secondaryVertex = tv.position();
		GlobalError err             = tv.positionError();
		GlobalPoint displacementFromBeamspot( -1*((vertexBeamSpot.x0() - secondaryVertex.x()) + 
							  (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()), 
						      -1*((vertexBeamSpot.y0() - secondaryVertex.y()) + 
							  (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()), 
						      0);
		reco::Vertex::Point vperp(displacementFromBeamspot.x(),displacementFromBeamspot.y(),0.);
		float jpsiKcos = vperp.Dot(pperp)/(vperp.R()*pperp.R());
		if (JpsiTkCL<minprob)continue;
		if (fabs(jpsiKcos)<mincos)continue;
		if ((displacementFromBeamspot.perp()/sqrt(err.rerr(displacementFromBeamspot)))<minDS)continue;
		mu1Phi_.denominator->Fill(t.phi());
		mu1Eta_.denominator->Fill(t.eta());
		mu1Pt_.denominator ->Fill(t.pt());
		mu2Phi_.denominator->Fill(t1.phi());
		mu2Eta_.denominator->Fill(t1.eta());
		mu2Pt_.denominator ->Fill(t1.pt());
	      } 
	      /////////////////////////
	    }
	  }
	  break;
	} 
      }
    }

    if (enum_ == 7){//photons
      const std::string & hltpath = hltpaths_den[0];
      for (auto const & p : *phHandle) {
	if(false && !matchToTrigger(hltpath,p, handleTriggerEvent)) continue;
        phPhi_.denominator->Fill(p.phi());
        phEta_.denominator->Fill(p.eta());
        phPt_.denominator ->Fill(p.pt());
      }

    } 
    //
    /////////
    //filling numerator hists
    if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;
    iEvent.getByToken( hltInputTag_, handleTriggerEvent);
    if (handleTriggerEvent->sizeFilters()== 0)return;
    const std::string & hltpath1 = hltpaths_num[0]; 
    for (auto const & m : *muoHandle ) {
      if(false && !matchToTrigger(hltpath1,m, handleTriggerEvent)) continue;
      if(!muoSelection_ref(m))continue;   
      for (auto const & m1 : *muoHandle ) {
	if (seagull_ && ((m.charge()* deltaPhi(m.phi(), m1.phi())) > 0.) )continue;
	if (m.charge()*m1.charge()>0 )continue;
	if (m1.pt() == m.pt())continue;
	if(!muoSelection_ref(m1))continue;   
	if(false && !matchToTrigger(hltpath1,m1, handleTriggerEvent)) continue;
	if (!DMSelection_ref(m1.p4() + m.p4()))continue;
	iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
	const reco::BeamSpot& vertexBeamSpot = *beamSpot;
	std::vector<reco::TransientTrack> j_tks;
	reco::TransientTrack mu1TT(m.track(), &(*bFieldHandle));
	reco::TransientTrack mu2TT(m1.track(), &(*bFieldHandle));
	j_tks.push_back(mu1TT);
	j_tks.push_back(mu2TT);
	KalmanVertexFitter jkvf;
	TransientVertex jtv = jkvf.vertex(j_tks);
	if (!jtv.isValid()) continue;
	reco::Vertex jpsivertex = jtv;
	float dimuonCL = 0;
	if( (jpsivertex.chi2()>=0) && (jpsivertex.ndof()>0) )//I think these values are "unphysical"(no one will need to change them ever)so the can be fixed
	  dimuonCL = TMath::Prob(jpsivertex.chi2(), jpsivertex.ndof() );
	math::XYZVector jpperp(m.px() + m1.px() ,
			       m.py() + m1.py() ,
			       0.);
	GlobalPoint jVertex = jtv.position();
	GlobalError jerr    = jtv.positionError();
	GlobalPoint displacementFromBeamspotJpsi( -1*((vertexBeamSpot.x0() - jVertex.x()) + (jVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()),
						  -1*((vertexBeamSpot.y0() - jVertex.y()) + (jVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()),
						  0);
	reco::Vertex::Point vperpj(displacementFromBeamspotJpsi.x(), displacementFromBeamspotJpsi.y(), 0.);
	float jpsi_cos = vperpj.Dot(jpperp)/(vperpj.R()*jpperp.R());
	TrajectoryStateClosestToPoint mu1TS = mu1TT.impactPointTSCP();
	TrajectoryStateClosestToPoint mu2TS = mu2TT.impactPointTSCP();
	ClosestApproachInRPhi cApp;
	if (mu1TS.isValid() && mu2TS.isValid()) {
	  cApp.calculate(mu1TS.theState(), mu2TS.theState());
	}
	double DiMuMass = (m1.p4()+m.p4()).M();
	switch(enum_){//enum_ = 1...9, represents different sets of variables for different paths, we want to have different hists for different paths
	case 1: tnp_=1;//already filled hists for tnp method
	case 2:
	  if ((Jpsi_) && (!Upsilon_)){
	    if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
	  }
	  if ((!Jpsi_) && (Upsilon_)){
	    if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
	  }
	  if (dimuonCL<minprob)continue;
	  mu1Phi_.numerator->Fill(m.phi());
	  mu1Eta_.numerator->Fill(m.eta());
	  mu1Pt_.numerator ->Fill(m.pt());
	  mu2Phi_.numerator->Fill(m1.phi());
	  mu2Eta_.numerator->Fill(m1.eta());
	  mu2Pt_.numerator ->Fill(m1.pt());
	  DiMuPt_.numerator ->Fill((m1.p4()+m.p4()).Pt() );
	  DiMuEta_.numerator ->Fill((m1.p4()+m.p4()).Eta() );
	  DiMuPhi_.numerator ->Fill((m1.p4()+m.p4()).Phi());
	  break;
	case 3:
	  if ((Jpsi_) && (!Upsilon_)){
	    if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
	  }

	  if ((!Jpsi_) && (Upsilon_)){
	    if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
	  }
	  if (dimuonCL<minprob)continue;
	  mu1Eta_.numerator->Fill(m.eta());
	  mu1Pt_.numerator ->Fill(m.pt());
	  mu2Eta_.numerator->Fill(m1.eta());
	  mu2Pt_.numerator ->Fill(m1.pt());
	  break; 
	case 4:
	  if (dimuonCL<minprob)continue;
	  DiMuMass_.numerator ->Fill(DiMuMass);
	  if ((Jpsi_) && (!Upsilon_)){
	    if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
	  }
	  if ((!Jpsi_) && (Upsilon_)){
	    if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
	  }
	  mu1Phi_.numerator->Fill(m.phi());
	  mu1Eta_.numerator->Fill(m.eta());
	  mu1Pt_.numerator ->Fill(m.pt());
	  mu2Phi_.numerator->Fill(m1.phi());
	  mu2Eta_.numerator->Fill(m1.eta());
	  mu2Pt_.numerator ->Fill(m1.pt());
	  DiMuPt_.numerator ->Fill((m1.p4()+m.p4()).Pt() );
	  DiMuEta_.numerator ->Fill((m1.p4()+m.p4()).Eta() );
	  DiMuPhi_.numerator ->Fill((m1.p4()+m.p4()).Phi());
	  DiMudR_.numerator ->Fill(reco::deltaR(m,m1));
	  break;
	case 5:
	  if (dimuonCL<minprob)continue;
	  if ((Jpsi_) && (!Upsilon_)){
	    if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
	  }
	  if ((!Jpsi_) && (Upsilon_)){
	    if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
	  }
	  mu1Phi_.numerator->Fill(m.phi());
	  mu1Eta_.numerator->Fill(m.eta());
	  mu1Pt_.numerator ->Fill(m.pt());
	  mu2Phi_.numerator->Fill(m1.phi());
	  mu2Eta_.numerator->Fill(m1.eta());
	  mu2Pt_.numerator ->Fill(m1.pt());
	  DiMuPt_.numerator ->Fill((m1.p4()+m.p4()).Pt() );
	  DiMuEta_.numerator ->Fill((m1.p4()+m.p4()).Eta() );
	  DiMuPhi_.numerator ->Fill((m1.p4()+m.p4()).Phi());
	  DiMudR_.numerator ->Fill(reco::deltaR(m,m1));
	  break;
	case 6: 
	  if (dimuonCL<minprob)continue;
	  if ((Jpsi_) && (!Upsilon_)){
	    if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
	  }
	  if ((!Jpsi_) && (Upsilon_)){
	    if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
	  }
	  for (auto const & m2 : *muoHandle) {//triple muon paths
	    if(false && !matchToTrigger(hltpath1,m2, handleTriggerEvent)) continue;
	    if (m2.pt() == m.pt())continue;
	    mu1Phi_.numerator->Fill(m.phi());
	    mu1Eta_.numerator->Fill(m.eta());
	    mu1Pt_.numerator ->Fill(m.pt());
	    mu2Phi_.numerator->Fill(m1.phi());
	    mu2Eta_.numerator->Fill(m1.eta());
	    mu2Pt_.numerator ->Fill(m1.pt());
	    mu3Phi_.numerator->Fill(m2.phi());
	    mu3Eta_.numerator->Fill(m2.eta());
	    mu3Pt_.numerator ->Fill(m2.pt());
	  }      
	  break;    
	case 7:// the hists for photon monitoring will be filled on 515 line
	  tnp_=0;
	  break;
	case 8://vtx monitoring, filling probability, DS, DCA, cos of pointing angle to the PV, eta, pT of dimuon
	  if ((Jpsi_) && (!Upsilon_)){
	    if (DiMuMass> maxmassJpsi || DiMuMass< minmassJpsi)continue;
	  }
	  if ((!Jpsi_) && (Upsilon_)){
	    if (DiMuMass> maxmassUpsilon || DiMuMass< minmassUpsilon)continue;
	  }
	  DiMuProb_.numerator ->Fill( dimuonCL);
	  if (dimuonCL<minprob)continue;
	  DiMuDS_.numerator ->Fill( displacementFromBeamspotJpsi.perp()/sqrt(jerr.rerr(displacementFromBeamspotJpsi)));
	  DiMuPVcos_.numerator ->Fill(jpsi_cos );
	  DiMuPt_.numerator ->Fill((m1.p4()+m.p4()).Pt() );
	  DiMuEta_.numerator ->Fill((m1.p4()+m.p4()).Eta() );
	  DiMuDCA_.numerator ->Fill( cApp.distance());
	  break;
	case 9:
	  if (dimuonCL<minprob)continue;
	  if (fabs(jpsi_cos)<mincos)continue;
	  if ((displacementFromBeamspotJpsi.perp()/sqrt(jerr.rerr(displacementFromBeamspotJpsi)))<minDS)continue;
	  if (trHandle.isValid()){
	    for (auto const & t : *trHandle) {
	      if(!trSelection_ref(t))continue;
	      if(false && !matchToTrigger(hltpath1,t, handleTriggerEvent)) continue;
	      reco::Track itrk1       = t ;                                                
	      if((reco::deltaR(t,m1) <= 0.001))continue;//checking overlaping
	      if((reco::deltaR(t,m) <= 0.001)) continue;
	      if (! itrk1.quality(reco::TrackBase::highPurity))     continue;
	      reco::Particle::LorentzVector pB, p1, p2, p3;
	      double trackMass2 = 0.493677 *0.493677;
	      double MuMass2 = 0.1056583745 *0.1056583745;
	      double e1   = sqrt(m.momentum().Mag2()  + MuMass2          );
	      double e2   = sqrt(m1.momentum().Mag2()  + MuMass2          );
	      double e3   = sqrt(itrk1.momentum().Mag2() + trackMass2  );
	      p1   = reco::Particle::LorentzVector(m.px() , m.py() , m.pz() , e1  );
	      p2   = reco::Particle::LorentzVector(m1.px() , m1.py() , m1.pz() , e2  );
	      p3   = reco::Particle::LorentzVector(itrk1.px(), itrk1.py(), itrk1.pz(), e3  );
	      pB   = p1 + p2 + p3;
	      if( pB.mass()> maxmassJpsiTk || pB.mass()< minmassJpsiTk)continue;
	      reco::TransientTrack trTT(itrk1, &(*bFieldHandle));
	      std::vector<reco::TransientTrack> t_tks;
	      t_tks.push_back(mu1TT);
	      t_tks.push_back(mu2TT);
	      t_tks.push_back(trTT);
	      KalmanVertexFitter kvf;
	      TransientVertex tv  = kvf.vertex(t_tks);
	      reco::Vertex vertex = tv;
	      if (!tv.isValid()) continue;
	      float JpsiTkCL = 0;
	      if ((vertex.chi2()>=0.0) && (vertex.ndof()>0) )   
		JpsiTkCL = TMath::Prob(vertex.chi2(), vertex.ndof() );
	      math::XYZVector pperp(m.px() + m1.px() + itrk1.px(),
				    m.py() + m1.py() + itrk1.py(),
				    0.);
	      GlobalPoint secondaryVertex = tv.position();
	      GlobalError err             = tv.positionError();
	      GlobalPoint displacementFromBeamspot( -1*((vertexBeamSpot.x0() - secondaryVertex.x()) + 
							(secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()), 
						    -1*((vertexBeamSpot.y0() - secondaryVertex.y()) + 
							(secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()), 
						    0);
	      reco::Vertex::Point vperp(displacementFromBeamspot.x(),displacementFromBeamspot.y(),0.);
	      float jpsiKcos = vperp.Dot(pperp)/(vperp.R()*pperp.R());
	      if (JpsiTkCL<minprob)continue;
	      if (fabs(jpsiKcos)<mincos)continue;
	      if ((displacementFromBeamspot.perp()/sqrt(err.rerr(displacementFromBeamspot)))<minDS)continue;
	      muPhi_.numerator->Fill(t.phi());
	      muEta_.numerator->Fill(t.eta());
	      muPt_.numerator ->Fill(t.pt());
	    }
	  }
	  break;

	case 10:
	  if (trHandle.isValid()){
	    for (auto const & t : *trHandle) {
	      if(!trSelection_ref(t))continue;
	      if(false && !matchToTrigger(hltpath1,t, handleTriggerEvent)) continue;
	      reco::Track itrk1       = t ;                                                
	      if((reco::deltaR(t,m1) <= 0.001))continue;//checking overlaping
	      if((reco::deltaR(t,m) <= 0.001)) continue;
	      if (! itrk1.quality(reco::TrackBase::highPurity))     continue;
	      reco::Particle::LorentzVector pB, p2, p3;
	      double trackMass2 = 0.493677 *0.493677;
	      double MuMass2 = 0.1056583745 *0.1056583745;
	      double e2   = sqrt(m1.momentum().Mag2()  + MuMass2          );
	      double e3   = sqrt(itrk1.momentum().Mag2() + trackMass2  );
	      p2   = reco::Particle::LorentzVector(m1.px() , m1.py() , m1.pz() , e2  );
	      p3   = reco::Particle::LorentzVector(itrk1.px(), itrk1.py(), itrk1.pz(), e3  );
	      pB   = p2 + p3;
	      if( pB.mass()> maxmassJpsiTk || pB.mass()< minmassJpsiTk)continue;
	      reco::TransientTrack trTT(itrk1, &(*bFieldHandle));
	      std::vector<reco::TransientTrack> t_tks;
	      t_tks.push_back(mu2TT);
	      t_tks.push_back(trTT);
	      if (t_tks.size()!=2) continue;
	      KalmanVertexFitter kvf;
	      TransientVertex tv  = kvf.vertex(t_tks);
	      reco::Vertex vertex = tv;
	      if (!tv.isValid()) continue;
	      float JpsiTkCL = 0;
	      if ((vertex.chi2()>=0.0) && (vertex.ndof()>0) )   
		JpsiTkCL = TMath::Prob(vertex.chi2(), vertex.ndof() );
	      math::XYZVector pperp(m1.px() + itrk1.px(),
				    m1.py() + itrk1.py(),
				    0.);
	      GlobalPoint secondaryVertex = tv.position();
	      GlobalError err             = tv.positionError();
	      GlobalPoint displacementFromBeamspot( -1*((vertexBeamSpot.x0() - secondaryVertex.x()) + 
							(secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()), 
						    -1*((vertexBeamSpot.y0() - secondaryVertex.y()) + 
							(secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()), 
						    0);
	      reco::Vertex::Point vperp(displacementFromBeamspot.x(),displacementFromBeamspot.y(),0.);
	      if (JpsiTkCL<minprob)continue;
	      muPhi_.numerator->Fill(m1.phi());
	      muEta_.numerator->Fill(m1.eta());
	      muPt_.numerator ->Fill(m1.pt());
	    }
	  }
	  break;
	case 11:
	  if (dimuonCL<minprob)continue;
	  if (fabs(jpsi_cos)<mincos)continue;
	  if ((displacementFromBeamspotJpsi.perp()/sqrt(jerr.rerr(displacementFromBeamspotJpsi)))<minDS)continue;
	  if (trHandle.isValid()){
	    for (auto const & t : *trHandle) {
	      if(!trSelection_ref(t))continue;
	      if(false && !matchToTrigger(hltpath1,t, handleTriggerEvent)) continue;
	      for (auto const & t1 : *trHandle) {
		if(!trSelection_ref(t1))continue;
		if(false && !matchToTrigger(hltpath1,t1, handleTriggerEvent)) continue;
		reco::Track itrk1       = t ;
		reco::Track itrk2       = t1 ;
		if((reco::deltaR(t,m1) <= 0.001))continue;//checking overlaping
		if((reco::deltaR(t,t1) <= 0.001))continue;//checking overlaping
		if((reco::deltaR(t,m) <= 0.001)) continue;
		if (! itrk1.quality(reco::TrackBase::highPurity))     continue;
		if (! itrk2.quality(reco::TrackBase::highPurity))     continue;

		reco::Particle::LorentzVector pB, p1, p2, p3, p4;
		double trackMass2 = 0.493677 *0.493677;
		double MuMass2 = 0.1056583745 *0.1056583745;
		double e1   = sqrt(m.momentum().Mag2()  + MuMass2          );
		double e2   = sqrt(m1.momentum().Mag2()  + MuMass2          );
		double e3   = sqrt(itrk1.momentum().Mag2() + trackMass2  );
		double e4   = sqrt(itrk2.momentum().Mag2() + trackMass2  );
		p1   = reco::Particle::LorentzVector(m.px() , m.py() , m.pz() , e1  );
		p2   = reco::Particle::LorentzVector(m1.px() , m1.py() , m1.pz() , e2  );
		p3   = reco::Particle::LorentzVector(itrk1.px(), itrk1.py(), itrk1.pz(), e3  );
		p4   = reco::Particle::LorentzVector(itrk2.px(), itrk2.py(), itrk2.pz(), e4  );
		pB   = p1 + p2 + p3 + p4;
		if( pB.mass()> maxmassJpsiTk || pB.mass()< minmassJpsiTk)continue;
		reco::TransientTrack trTT(itrk1, &(*bFieldHandle));
		reco::TransientTrack tr1TT(itrk2, &(*bFieldHandle));
		std::vector<reco::TransientTrack> t_tks;
		t_tks.push_back(mu1TT);
		t_tks.push_back(mu2TT);
		t_tks.push_back(trTT);
		t_tks.push_back(tr1TT);
		KalmanVertexFitter kvf;
		TransientVertex tv  = kvf.vertex(t_tks);
		reco::Vertex vertex = tv;
		if (!tv.isValid()) continue;
		float JpsiTkCL = 0;
		if ((vertex.chi2()>=0.0) && (vertex.ndof()>0) )   
		  JpsiTkCL = TMath::Prob(vertex.chi2(), vertex.ndof() );
		math::XYZVector pperp(m.px() + m1.px() + itrk1.px() + itrk2.px(),
				      m.py() + m1.py() + itrk1.py() + itrk2.py(),
				      0.);
		GlobalPoint secondaryVertex = tv.position();
		GlobalError err             = tv.positionError();
		GlobalPoint displacementFromBeamspot( -1*((vertexBeamSpot.x0() - secondaryVertex.x()) + 
							  (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()), 
						      -1*((vertexBeamSpot.y0() - secondaryVertex.y()) + 
							  (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()), 
						      0);
		reco::Vertex::Point vperp(displacementFromBeamspot.x(),displacementFromBeamspot.y(),0.);
		float jpsiKcos = vperp.Dot(pperp)/(vperp.R()*pperp.R());
		if (JpsiTkCL<minprob)continue;
		if (fabs(jpsiKcos)<mincos)continue;
		if ((displacementFromBeamspot.perp()/sqrt(err.rerr(displacementFromBeamspot)))<minDS)continue;
		mu1Phi_.numerator->Fill(t.phi());
		mu1Eta_.numerator->Fill(t.eta());
		mu1Pt_.numerator ->Fill(t.pt());
		mu2Phi_.numerator->Fill(t1.phi());
		mu2Eta_.numerator->Fill(t1.eta());
		mu2Pt_.numerator ->Fill(t1.pt());
	      } 
	      /////////////////////////
	    }
	  }
	  break;
	} 
      }
    }
    if (enum_ == 7){//photons
      const std::string &hltpath = hltpaths_num[0];
      for (auto const & p : *phHandle) {
        if(false && !matchToTrigger(hltpath,p, handleTriggerEvent)) continue;
        phPhi_.numerator->Fill(p.phi());
        phEta_.numerator->Fill(p.eta());
        phPt_.numerator ->Fill(p.pt());
      }
    }
  }
}




void BPHMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.addNode((edm::ParameterDescription<int>("nbins", true) and
                edm::ParameterDescription<double>("xmin", true) and
                edm::ParameterDescription<double>("xmax", true)) xor
               edm::ParameterDescription<std::vector<double>>("edges", true));
}

void BPHMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<int>   ( "nbins", 2500);
}

void BPHMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/BPH/" );
  desc.add<edm::InputTag>( "tracks",  edm::InputTag("generalTracks") );
  desc.add<edm::InputTag>( "photons",  edm::InputTag("photons") );
  desc.add<edm::InputTag>( "offlinePVs",     edm::InputTag("offlinePrimaryVertices") );
  desc.add<edm::InputTag>( "beamSpot",edm::InputTag("offlineBeamSpot") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<edm::InputTag>( "hltTriggerSummaryAOD",    edm::InputTag("hltTriggerSummaryAOD","","HLT") );
  desc.add<std::string>("muoSelection", "abs(eta)<1.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0");
  desc.add<std::string>("muoSelection_ref", "isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0");
  desc.add<std::string>("muoSelection_tag",  "isGlobalMuon && isPFMuon && isTrackerMuon && abs(eta) < 2.4 && innerTrack.hitPattern.numberOfValidPixelHits > 0 && innerTrack.hitPattern.trackerLayersWithMeasurement > 5 && globalTrack.hitPattern.numberOfValidMuonHits > 0 && globalTrack.normalizedChi2 < 10");//tight selection for tag muon
  desc.add<std::string>("muoSelection_probe", "isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0");
  desc.add<std::string>("trSelection_ref", "");
  desc.add<std::string>("DMSelection_ref", "Pt>4 & abs(eta)");

  desc.add<int>("nmuons",     1);
  desc.add<bool>( "tnp", 0 );
  desc.add<int>( "L3", 0 );
  desc.add<int>( "trOrMu", 0 );//if =0, track param monitoring
  desc.add<int>( "Jpsi", 0 );
  desc.add<int>( "Upsilon", 0 );
  desc.add<int>( "enum", 1 );//1...9, 9 sets of variables to be filled, depends on the hlt path
  desc.add<int>( "seagull", 1 );//1...9, 9 sets of variables to be filled, depends on the hlt path
  desc.add<double>( "maxmass", 3.596 );
  desc.add<double>( "minmass", 2.596 );
  desc.add<double>( "maxmassJpsi", 3.2 );
  desc.add<double>( "minmassJpsi", 3. );
  desc.add<double>( "maxmassUpsilon", 8.1 );
  desc.add<double>( "minmassUpsilon", 8. );
  desc.add<double>( "maxmassJpsiTk", 5.46 );
  desc.add<double>( "minmassJpsiTk", 5.1 );
  desc.add<double>( "minprob", 0.005 );
  desc.add<double>( "mincos", 0.95 );
  desc.add<double>( "minDS", 3. );

  edm::ParameterSetDescription genericTriggerEventPSet;
  genericTriggerEventPSet.add<bool>("andOr");
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi") );
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT") );
  genericTriggerEventPSet.add<std::vector<int> >("dcsPartitions",{});
  genericTriggerEventPSet.add<bool>("andOrDcs", false);
  genericTriggerEventPSet.add<bool>("errorReplyDcs", true);
  genericTriggerEventPSet.add<std::string>("dbLabel","");
  genericTriggerEventPSet.add<bool>("andOrHlt", true);
  genericTriggerEventPSet.add<bool>("andOrL1", true);
  genericTriggerEventPSet.add<std::vector<std::string> >("hltPaths",{});
  genericTriggerEventPSet.add<std::vector<std::string> >("l1Algorithms",{});
  genericTriggerEventPSet.add<std::string>("hltDBKey","");
  genericTriggerEventPSet.add<bool>("errorReplyHlt",false);
  genericTriggerEventPSet.add<bool>("errorReplyL1",true);
  genericTriggerEventPSet.add<bool>("l1BeforeMask",true);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel",0);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription phiPSet;
  edm::ParameterSetDescription etaPSet;
  edm::ParameterSetDescription ptPSet;
  edm::ParameterSetDescription d0PSet;
  edm::ParameterSetDescription z0PSet;
  edm::ParameterSetDescription dRPSet;
  edm::ParameterSetDescription massPSet;
  edm::ParameterSetDescription dcaPSet;
  edm::ParameterSetDescription dsPSet;
  edm::ParameterSetDescription cosPSet;
  edm::ParameterSetDescription probPSet;
  fillHistoPSetDescription(phiPSet);
  fillHistoPSetDescription(ptPSet);
  fillHistoPSetDescription(etaPSet);
  fillHistoPSetDescription(z0PSet);
  fillHistoPSetDescription(d0PSet);
  fillHistoPSetDescription(dRPSet);
  fillHistoPSetDescription(massPSet);
  fillHistoPSetDescription(dcaPSet);
  fillHistoPSetDescription(dsPSet);
  fillHistoPSetDescription(cosPSet);
  fillHistoPSetDescription(probPSet);
  histoPSet.add<edm::ParameterSetDescription>("d0PSet", d0PSet);
  histoPSet.add<edm::ParameterSetDescription>("etaPSet", etaPSet);
  histoPSet.add<edm::ParameterSetDescription>("phiPSet", phiPSet);
  histoPSet.add<edm::ParameterSetDescription>("ptPSet", ptPSet);
  histoPSet.add<edm::ParameterSetDescription>("z0PSet", z0PSet);
  histoPSet.add<edm::ParameterSetDescription>("dRPSet", dRPSet);
  histoPSet.add<edm::ParameterSetDescription>("massPSet", massPSet);
  histoPSet.add<edm::ParameterSetDescription>("dcaPSet", dcaPSet);
  histoPSet.add<edm::ParameterSetDescription>("dsPSet", dsPSet);
  histoPSet.add<edm::ParameterSetDescription>("cosPSet", cosPSet);
  histoPSet.add<edm::ParameterSetDescription>("probPSet", probPSet);
  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("bphMonitoring", desc);
}
template <typename T>
bool BPHMonitor::matchToTrigger(const std::string  &theTriggerName , T t, edm::Handle<trigger::TriggerEvent> handleTriggerEvent){
  //bool BPHMonitor::matchToTrigger(std::string theTriggerName,T t, edm::Handle<trigger::TriggerEventWithRefs> handleTriggerEvent){
   
  bool matchedToTrigger = false;
  if (handleTriggerEvent->sizeFilters() >0){
    const trigger::TriggerObjectCollection & toc(handleTriggerEvent->getObjects());//Handle< trigger::TriggerEvent > handleTriggerEvent;
    for ( size_t ia = 0; ia < handleTriggerEvent->sizeFilters(); ++ ia) {
      std::string fullname = handleTriggerEvent->filterTag(ia).encode();
      std::string name;
      size_t p = fullname.find_first_of(':');
      if ( p != std::string::npos) {name = fullname.substr(0, p);}
      else {name = fullname;}
      const trigger::Keys & k = handleTriggerEvent->filterKeys(ia);
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	reco::Particle theTriggerParticle = toc[*ki].particle();
	if(name.find(theTriggerName) != string::npos){
	  if((reco::deltaR(t.eta(), t.phi(),theTriggerParticle.eta(),theTriggerParticle.phi()) <= 0.2)){
	    matchedToTrigger = true;
	  }
	}    
      }
    }

    return matchedToTrigger;
  }
  else {cout<<theTriggerName<<"\t\tNo HLT filters"<<endl; return false;}
}




// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BPHMonitor);

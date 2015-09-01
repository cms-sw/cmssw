
// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TTree.h"

//
// class declaration
//

class HiEvtAnalyzer : public edm::EDAnalyzer {
public:
  explicit HiEvtAnalyzer(const edm::ParameterSet&);
  ~HiEvtAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  // ----------member data ---------------------------
  edm::InputTag CentralityTag_;
  edm::InputTag CentralityBinTag_;

  edm::InputTag EvtPlaneTag_;
  edm::InputTag EvtPlaneFlatTag_;

  edm::InputTag HiMCTag_;
  edm::InputTag VertexTag_;

  bool doEvtPlane_;
  bool doEvtPlaneFlat_;
  bool doCentrality_;

  bool doMC_;
  bool doVertex_;

  edm::Service<TFileService> fs_;

  TTree * thi_;

  float *hiEvtPlane;
  int nEvtPlanes;
  int HltEvtCnt;
  int hiBin;
  int hiNpix, hiNpixelTracks, hiNtracks, hiNtracksPtCut, hiNtracksEtaCut, hiNtracksEtaPtCut;
  float hiHF, hiHFplus, hiHFminus, hiHFplusEta4, hiHFminusEta4, hiHFhit, hiHFhitPlus, hiHFhitMinus, hiEB, hiET, hiEE, hiEEplus, hiEEminus, hiZDC, hiZDCplus, hiZDCminus;

  float fNpart;
  float fNcoll;
  float fNhard;
  float fPhi0;
  float fb;

  int fNcharged;
  int fNchargedMR;
  float fMeanPt;
  float fMeanPtMR;
  float fEtMR;
  int fNchargedPtCut;
  int fNchargedPtCutMR;

  int proc_id;

  float vx,vy,vz;

  int event;
  int run;
  int lumi;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HiEvtAnalyzer::HiEvtAnalyzer(const edm::ParameterSet& iConfig) :
  CentralityTag_(iConfig.getParameter<edm::InputTag> ("CentralitySrc")),
  CentralityBinTag_(iConfig.getParameter<edm::InputTag> ("CentralityBinSrc")),
  EvtPlaneTag_(iConfig.getParameter<edm::InputTag> ("EvtPlane")),
  EvtPlaneFlatTag_(iConfig.getParameter<edm::InputTag> ("EvtPlaneFlat")),
  HiMCTag_(iConfig.getParameter<edm::InputTag> ("HiMC")),
  VertexTag_(iConfig.getParameter<edm::InputTag> ("Vertex")),
  doEvtPlane_(iConfig.getParameter<bool> ("doEvtPlane")),
  doEvtPlaneFlat_(iConfig.getParameter<bool> ("doEvtPlaneFlat")),
  doCentrality_(iConfig.getParameter<bool> ("doCentrality")),
  doMC_(iConfig.getParameter<bool> ("doMC")),
  doVertex_(iConfig.getParameter<bool>("doVertex"))
{

}

HiEvtAnalyzer::~HiEvtAnalyzer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
HiEvtAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // Run info
  event = iEvent.id().event();
  run = iEvent.id().run();
  lumi = iEvent.id().luminosityBlock();

  edm::Handle<reco::EvtPlaneCollection> evtPlanes;

  if(doMC_){
    edm::Handle<edm::GenHIEvent> mchievt;
    iEvent.getByLabel(edm::InputTag(HiMCTag_),mchievt);
    fb = mchievt->b();
    fNpart = mchievt->Npart();
    fNcoll = mchievt->Ncoll();
    fNhard = mchievt->Nhard();
    fPhi0 = mchievt->evtPlane();
    fNcharged = mchievt->Ncharged();
    fNchargedMR = mchievt->NchargedMR();
    fMeanPt = mchievt->MeanPt();
    fMeanPtMR = mchievt->MeanPtMR();
    fEtMR = mchievt->EtMR();
    fNchargedPtCut = mchievt->NchargedPtCut();
    fNchargedPtCutMR = mchievt->NchargedPtCutMR();

    edm::Handle<edm::HepMCProduct> hepmcevt;
    iEvent.getByLabel("generator", hepmcevt);
    proc_id =  hepmcevt->GetEvent()->signal_process_id();
  }

  if (doCentrality_) {

    edm::Handle<int> cbin_;
    iEvent.getByLabel(CentralityBinTag_,cbin_);
    hiBin = *cbin_;

    edm::Handle<reco::Centrality> centrality;
    iEvent.getByLabel(CentralityTag_, centrality);

    hiNpix = centrality->multiplicityPixel();
    hiNpixelTracks = centrality->NpixelTracks();
    hiNtracks = centrality->Ntracks();
    hiNtracksPtCut = centrality->NtracksPtCut();
    hiNtracksEtaCut = centrality->NtracksEtaCut();
    hiNtracksEtaPtCut = centrality->NtracksEtaPtCut();

    hiHF = centrality->EtHFtowerSum();
    hiHFplus = centrality->EtHFtowerSumPlus();
    hiHFminus = centrality->EtHFtowerSumMinus();
    hiHFplusEta4 = centrality->EtHFtruncatedPlus();
    hiHFminusEta4 = centrality->EtHFtruncatedMinus();
    hiHFhit = centrality->EtHFhitSum();
    hiHFhitPlus = centrality->EtHFhitSumPlus();
    hiHFhitMinus = centrality->EtHFhitSumMinus();

    hiZDC = centrality->zdcSum();
    hiZDCplus = centrality->zdcSumPlus();
    hiZDCminus = centrality->zdcSumMinus();

    hiEEplus = centrality->EtEESumPlus();
    hiEEminus = centrality->EtEESumMinus();
    hiEE = centrality->EtEESum();
    hiEB = centrality->EtEBSum();
    hiET = centrality->EtMidRapiditySum();
  }

  nEvtPlanes = 0;
  if (doEvtPlane_) {
    iEvent.getByLabel(EvtPlaneTag_,evtPlanes);
    if(evtPlanes.isValid()){
      nEvtPlanes += evtPlanes->size();
      for(unsigned int i = 0; i < evtPlanes->size(); ++i){
	hiEvtPlane[i] = (*evtPlanes)[i].angle();
      }
    }
  }

  if (doEvtPlaneFlat_) {
    iEvent.getByLabel(EvtPlaneFlatTag_,evtPlanes);
    if(evtPlanes.isValid()){
      for(unsigned int i = 0; i < evtPlanes->size(); ++i){
	hiEvtPlane[nEvtPlanes+i] = (*evtPlanes)[i].angle();
      }
      nEvtPlanes += evtPlanes->size();
    }
  }


  if (doVertex_) {
    edm::Handle<std::vector<reco::Vertex> > vertex;
    iEvent.getByLabel(VertexTag_, vertex);
    vx=vertex->begin()->x();
    vy=vertex->begin()->y();
    vz=vertex->begin()->z();
  }

  // Done w/ all vars
  thi_->Fill();
}


// ------------ method called once each job just before starting event loop  ------------
void
HiEvtAnalyzer::beginJob()
{
  thi_ = fs_->make<TTree>("HiTree", "");

  //centProvider = 0;
  HltEvtCnt = 0;
  const int kMaxEvtPlanes = 1000;

  fNpart = -1;
  fNcoll = -1;
  fNhard = -1;
  fPhi0 = -1;
  fb = -1;
  fNcharged = -1;
  fNchargedMR = -1;
  fMeanPt = -1;
  fMeanPtMR = -1;

  fEtMR = -1;
  fNchargedPtCut = -1;
  fNchargedPtCutMR = -1;

  nEvtPlanes = 0;
  hiBin = -1;
  hiEvtPlane = new float[kMaxEvtPlanes];

  vx = -100;
  vy = -100;
  vz = -100;

  // Run info
  thi_->Branch("run",&run,"run/I");
  thi_->Branch("evt",&event,"evt/I");
  thi_->Branch("lumi",&lumi,"lumi/I");

  // Vertex
  thi_->Branch("vx",&vx,"vx/F");
  thi_->Branch("vy",&vy,"vy/F");
  thi_->Branch("vz",&vz,"vz/F");

  // Centrality
  if (doMC_) {
    thi_->Branch("Npart",&fNpart,"Npart/F");
    thi_->Branch("Ncoll",&fNcoll,"Ncoll/F");
    thi_->Branch("Nhard",&fNhard,"Nhard/F");
    thi_->Branch("phi0",&fPhi0,"NPhi0/F");
    thi_->Branch("b",&fb,"b/F");
    thi_->Branch("Ncharged",&fNcharged,"Ncharged/I");
    thi_->Branch("NchargedMR",&fNchargedMR,"NchargedMR/I");
    thi_->Branch("MeanPt",&fMeanPt,"MeanPt/F");
    thi_->Branch("MeanPtMR",&fMeanPtMR,"MeanPtMR/F");
    thi_->Branch("EtMR",&fEtMR,"EtMR/F");
    thi_->Branch("NchargedPtCut",&fNchargedPtCut,"NchargedPtCut/I");
    thi_->Branch("NchargedPtCutMR",&fNchargedPtCutMR,"NchargedPtCutMR/I");

    thi_->Branch("ProcessID",&proc_id,"ProcessID/I");
  }

  thi_->Branch("hiBin",&hiBin,"hiBin/I");
  thi_->Branch("hiHF",&hiHF,"hiHF/F");
  thi_->Branch("hiHFplus",&hiHFplus,"hiHFplus/F");
  thi_->Branch("hiHFminus",&hiHFminus,"hiHFminus/F");
  thi_->Branch("hiHFplusEta4",&hiHFplusEta4,"hiHFplusEta4/F");
  thi_->Branch("hiHFminusEta4",&hiHFminusEta4,"hiHFminusEta4/F");

  thi_->Branch("hiZDC",&hiZDC,"hiZDC/F");
  thi_->Branch("hiZDCplus",&hiZDCplus,"hiZDCplus/F");
  thi_->Branch("hiZDCminus",&hiZDCminus,"hiZDCminus/F");

  thi_->Branch("hiHFhit",&hiHFhit,"hiHFhit/F");
  thi_->Branch("hiHFhitPlus",&hiHFhitPlus,"hiHFhitPlus/F");
  thi_->Branch("hiHFhitMinus",&hiHFhitMinus,"hiHFhitMinus/F");

  thi_->Branch("hiET",&hiET,"hiET/F");
  thi_->Branch("hiEE",&hiEE,"hiEE/F");
  thi_->Branch("hiEB",&hiEB,"hiEB/F");
  thi_->Branch("hiEEplus",&hiEEplus,"hiEEplus/F");
  thi_->Branch("hiEEminus",&hiEEminus,"hiEEminus/F");
  thi_->Branch("hiNpix",&hiNpix,"hiNpix/I");
  thi_->Branch("hiNpixelTracks",&hiNpixelTracks,"hiNpixelTracks/I");
  thi_->Branch("hiNtracks",&hiNtracks,"hiNtracks/I");
  thi_->Branch("hiNtracksPtCut",&hiNtracksPtCut,"hiNtracksPtCut/I");
  thi_->Branch("hiNtracksEtaCut",&hiNtracksEtaCut,"hiNtracksEtaCut/I");
  thi_->Branch("hiNtracksEtaPtCut",&hiNtracksEtaPtCut,"hiNtracksEtaPtCut/I");

  // Event plane
  if (doEvtPlane_) {
    thi_->Branch("hiNevtPlane",&nEvtPlanes,"hiNevtPlane/I");
    thi_->Branch("hiEvtPlanes",hiEvtPlane,"hiEvtPlanes[hiNevtPlane]/F");
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void
HiEvtAnalyzer::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HiEvtAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiEvtAnalyzer);

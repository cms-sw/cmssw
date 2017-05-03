
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
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include <HepMC/PdfInfo.h>

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include <HepMC/PdfInfo.h>

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
  edm::EDGetTokenT<reco::Centrality> CentralityTag_;
  edm::EDGetTokenT<int> CentralityBinTag_;

  edm::EDGetTokenT<reco::EvtPlaneCollection> EvtPlaneTag_;
  edm::EDGetTokenT<reco::EvtPlaneCollection> EvtPlaneFlatTag_;

  edm::EDGetTokenT<edm::GenHIEvent> HiMCTag_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> VertexTag_;

  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> puInfoToken_;
  edm::EDGetTokenT<GenEventInfoProduct> genInfoToken_;

  bool doEvtPlane_;
  bool doEvtPlaneFlat_;
  bool doCentrality_;

  bool doMC_;
  bool doHiMC_;
  bool useHepMC_;
  bool doVertex_;

  int evtPlaneLevel_;

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
  float pthat;
  float weight;
  float alphaQCD;
  float alphaQED;
  float qScale;
  int   nMEPartons;
  int   nMEPartonsFiltered;
  std::pair<int, int> pdfID;
  std::pair<float, float> pdfX;
  std::pair<float, float> pdfXpdf;
  std::vector<int> npus;    //number of pileup interactions
  std::vector<float> tnpus; //true number of interactions

  float vx,vy,vz;

  unsigned long long event;
  unsigned int run;
  unsigned int lumi;

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
  CentralityTag_(consumes<reco::Centrality>(iConfig.getParameter<edm::InputTag>("CentralitySrc"))),
  CentralityBinTag_(consumes<int>(iConfig.getParameter<edm::InputTag>("CentralityBinSrc"))),
  EvtPlaneTag_(consumes<reco::EvtPlaneCollection>(iConfig.getParameter<edm::InputTag>("EvtPlane"))),
  EvtPlaneFlatTag_(consumes<reco::EvtPlaneCollection>(iConfig.getParameter<edm::InputTag>("EvtPlaneFlat"))),
  HiMCTag_(consumes<edm::GenHIEvent>(iConfig.getParameter<edm::InputTag>("HiMC"))),
  VertexTag_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("Vertex"))),
  puInfoToken_(consumes<std::vector<PileupSummaryInfo>>(edm::InputTag("addPileupInfo"))),
  genInfoToken_(consumes<GenEventInfoProduct>(edm::InputTag("generator"))),
  doEvtPlane_(iConfig.getParameter<bool> ("doEvtPlane")),
  doEvtPlaneFlat_(iConfig.getParameter<bool> ("doEvtPlaneFlat")),
  doCentrality_(iConfig.getParameter<bool> ("doCentrality")),
  doMC_(iConfig.getParameter<bool> ("doMC")),
  doHiMC_(iConfig.getParameter<bool> ("doHiMC")),
  useHepMC_(iConfig.getParameter<bool> ("useHepMC")),
  doVertex_(iConfig.getParameter<bool>("doVertex")),
  evtPlaneLevel_(iConfig.getParameter<int>("evtPlaneLevel"))
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

  if(doHiMC_){
    edm::Handle<edm::GenHIEvent> mchievt;
    if(iEvent.getByToken(HiMCTag_, mchievt)) {
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
    }
  }
    
  if(doMC_){
    if(useHepMC_) {
      edm::Handle<edm::HepMCProduct> hepmcevt;
      iEvent.getByLabel("generator", hepmcevt);
      proc_id  = hepmcevt->GetEvent()->signal_process_id();
      weight   = hepmcevt->GetEvent()->weights()[0];
      alphaQCD = hepmcevt->GetEvent()->alphaQCD();
      alphaQED = hepmcevt->GetEvent()->alphaQED();
      qScale   = hepmcevt->GetEvent()->event_scale();
      const HepMC::PdfInfo *hepPDF = hepmcevt->GetEvent()->pdf_info();
      if (hepPDF) {
        pdfID = std::make_pair(hepPDF->id1(), hepPDF->id2());
        pdfX = std::make_pair(hepPDF->x1(), hepPDF->x2());
        pdfXpdf = std::make_pair(hepPDF->pdf1(), hepPDF->pdf2());
      }
    }
    else {
      edm::Handle<GenEventInfoProduct> genInfo;
      if(iEvent.getByToken(genInfoToken_, genInfo)) {
        proc_id = genInfo->signalProcessID();
        if (genInfo->hasBinningValues())
          pthat = genInfo->binningValues()[0];
        weight = genInfo->weight();
        nMEPartons = genInfo->nMEPartons();
        nMEPartonsFiltered = genInfo->nMEPartonsFiltered();
        alphaQCD = genInfo->alphaQCD();
        alphaQED = genInfo->alphaQED();
        qScale = genInfo->qScale();

        if (genInfo->hasPDF()) {
          pdfID = genInfo->pdf()->id;
          pdfX.first = genInfo->pdf()->x.first;
          pdfX.second = genInfo->pdf()->x.second;
          pdfXpdf.first = genInfo->pdf()->xPDF.first;
          pdfXpdf.second = genInfo->pdf()->xPDF.second;
        }
      }
    }

    // MC PILEUP INFORMATION
    edm::Handle<std::vector<PileupSummaryInfo>>    puInfos;
    if (iEvent.getByToken(puInfoToken_, puInfos)) {
      for (const auto& pu: *puInfos) {
        npus.push_back(pu.getPU_NumInteractions());
        tnpus.push_back(pu.getTrueNumInteractions());
      }
    }
  }

  

  if (doCentrality_) {

    edm::Handle<int> cbin_;
    iEvent.getByToken(CentralityBinTag_,cbin_);
    hiBin = *cbin_;

    edm::Handle<reco::Centrality> centrality;
    iEvent.getByToken(CentralityTag_, centrality);

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
  edm::Handle<reco::EvtPlaneCollection> evtPlanes;

  if (doEvtPlane_) {
    iEvent.getByToken(EvtPlaneTag_,evtPlanes);
    if(evtPlanes.isValid()){
      nEvtPlanes += evtPlanes->size();
      for(unsigned int i = 0; i < evtPlanes->size(); ++i){
        hiEvtPlane[i] = (*evtPlanes)[i].angle(evtPlaneLevel_);
      }
    }
  }

  if (doEvtPlaneFlat_) {
    iEvent.getByToken(EvtPlaneFlatTag_,evtPlanes);
    if(evtPlanes.isValid()){
      for(unsigned int i = 0; i < evtPlanes->size(); ++i){
	hiEvtPlane[nEvtPlanes+i] = (*evtPlanes)[i].angle();
      }
      nEvtPlanes += evtPlanes->size();
    }
  }


  if (doVertex_) {
    edm::Handle<std::vector<reco::Vertex>> vertex;
    iEvent.getByToken(VertexTag_, vertex);
    vx=vertex->begin()->x();
    vy=vertex->begin()->y();
    vz=vertex->begin()->z();
  }

  thi_->Fill();
}


// ------------ method called once each job just before starting event loop  ------------
void
HiEvtAnalyzer::beginJob()
{
  thi_ = fs_->make<TTree>("HiTree", "");

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

  proc_id =   -1;
  pthat   =   -1.;
  weight  =   -1.;
  alphaQCD =  -1.;
  alphaQED =  -1.;
  qScale   =  -1.;
  //  npu      =   1;

  nEvtPlanes = 0;
  hiBin = -1;
  hiEvtPlane = new float[kMaxEvtPlanes];

  vx = -100;
  vy = -100;
  vz = -100;

  // Run info
  thi_->Branch("run",&run,"run/i");
  thi_->Branch("evt",&event,"evt/l");
  thi_->Branch("lumi",&lumi,"lumi/i");

  // Vertex
  thi_->Branch("vx",&vx,"vx/F");
  thi_->Branch("vy",&vy,"vy/F");
  thi_->Branch("vz",&vz,"vz/F");

  //Event observables
  if (doHiMC_) {
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
  }
  if (doMC_) {
    thi_->Branch("ProcessID",&proc_id,"ProcessID/I");
    thi_->Branch("pthat",&pthat,"pthat/F");
    thi_->Branch("weight",&weight,"weight/F");
    thi_->Branch("alphaQCD",&alphaQCD,"alphaQCD/F");
    thi_->Branch("alphaQED",&alphaQED,"alphaQED/F");
    thi_->Branch("qScale",&qScale,"qScale/F");
    thi_->Branch("nMEPartons",&nMEPartons,"nMEPartons/I");
    thi_->Branch("nMEPartonsFiltered",&nMEPartonsFiltered,"nMEPartonsFiltered/I");
    thi_->Branch("pdfID",&pdfID);
    thi_->Branch("pdfX",&pdfX);
    thi_->Branch("pdfXpdf",&pdfXpdf);
    thi_->Branch("npus",&npus);
    thi_->Branch("tnpus",&tnpus);
  }

  // Centrality
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

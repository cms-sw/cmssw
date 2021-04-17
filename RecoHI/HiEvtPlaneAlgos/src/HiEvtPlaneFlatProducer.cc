#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Math/Vector3D.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/HeavyIonRPRcd.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/HIObjects/interface/RPFlatParams.h"

#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneFlatten.h"
#include <ctime>
#include <cstdlib>

#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"
#include "RecoHI/HiEvtPlaneAlgos/interface/LoadEPDB.h"

using namespace std;
using namespace hi;

#include <vector>
using std::vector;

//
// class declaration
//

class HiEvtPlaneFlatProducer : public edm::stream::EDProducer<> {
public:
  explicit HiEvtPlaneFlatProducer(const edm::ParameterSet&);
  ~HiEvtPlaneFlatProducer() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  std::string centralityVariable_;
  std::string centralityLabel_;
  std::string centralityMC_;

  edm::InputTag centralityBinTag_;
  edm::EDGetTokenT<int> centralityBinToken_;

  edm::InputTag centralityTag_;
  edm::EDGetTokenT<reco::Centrality> centralityToken_;

  edm::InputTag vertexTag_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> vertexToken_;

  edm::InputTag inputPlanesTag_;
  edm::EDGetTokenT<reco::EvtPlaneCollection> inputPlanesToken_;

  edm::InputTag trackTag_;
  edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  edm::Handle<reco::TrackCollection> trackCollection_;

  edm::ESWatcher<HeavyIonRcd> hiWatcher;
  edm::ESWatcher<HeavyIonRPRcd> hirpWatcher;

  const int FlatOrder_;
  int NumFlatBins_;
  int flatnvtxbins_;
  double flatminvtx_;
  double flatdelvtx_;
  double caloCentRef_;
  double caloCentRefWidth_;
  int CentBinCompression_;
  HiEvtPlaneFlatten* flat[NumEPNames];
  bool useOffsetPsi_;
  double nCentBins_;
};
//
// constructors and destructor
//
HiEvtPlaneFlatProducer::HiEvtPlaneFlatProducer(const edm::ParameterSet& iConfig)
    : centralityVariable_(iConfig.getParameter<std::string>("centralityVariable")),
      centralityBinTag_(iConfig.getParameter<edm::InputTag>("centralityBinTag")),
      centralityTag_(iConfig.getParameter<edm::InputTag>("centralityTag")),
      vertexTag_(iConfig.getParameter<edm::InputTag>("vertexTag")),
      inputPlanesTag_(iConfig.getParameter<edm::InputTag>("inputPlanesTag")),
      trackTag_(iConfig.getParameter<edm::InputTag>("trackTag")),
      FlatOrder_(iConfig.getParameter<int>("FlatOrder")),
      NumFlatBins_(iConfig.getParameter<int>("NumFlatBins")),
      flatnvtxbins_(iConfig.getParameter<int>("flatnvtxbins")),
      flatminvtx_(iConfig.getParameter<double>("flatminvtx")),
      flatdelvtx_(iConfig.getParameter<double>("flatdelvtx")),
      caloCentRef_(iConfig.getParameter<double>("caloCentRef")),
      caloCentRefWidth_(iConfig.getParameter<double>("caloCentRefWidth")),
      CentBinCompression_(iConfig.getParameter<int>("CentBinCompression")),
      useOffsetPsi_(iConfig.getParameter<bool>("useOffsetPsi")) {
  nCentBins_ = 200.;

  if (iConfig.exists("nonDefaultGlauberModel")) {
    centralityMC_ = iConfig.getParameter<std::string>("nonDefaultGlauberModel");
  }
  centralityLabel_ = centralityVariable_ + centralityMC_;

  centralityBinToken_ = consumes<int>(centralityBinTag_);

  centralityToken_ = consumes<reco::Centrality>(centralityTag_);

  vertexToken_ = consumes<std::vector<reco::Vertex>>(vertexTag_);

  trackToken_ = consumes<reco::TrackCollection>(trackTag_);

  inputPlanesToken_ = consumes<reco::EvtPlaneCollection>(inputPlanesTag_);

  //register your products
  produces<reco::EvtPlaneCollection>();
  //now do what ever other initialization is needed
  for (int i = 0; i < NumEPNames; i++) {
    flat[i] = new HiEvtPlaneFlatten();
    flat[i]->init(FlatOrder_, NumFlatBins_, flatnvtxbins_, flatminvtx_, flatdelvtx_, EPNames[i], EPOrder[i]);
  }
}

HiEvtPlaneFlatProducer::~HiEvtPlaneFlatProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  for (int i = 0; i < NumEPNames; i++) {
    delete flat[i];
  }
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HiEvtPlaneFlatProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace reco;

  if (hiWatcher.check(iSetup)) {
    //
    //Get Size of Centrality Table
    //
    edm::ESHandle<CentralityTable> centDB_;
    iSetup.get<HeavyIonRcd>().get(centralityLabel_, centDB_);
    nCentBins_ = centDB_->m_table.size();
    for (int i = 0; i < NumEPNames; i++) {
      if (caloCentRef_ > 0) {
        int minbin = (caloCentRef_ - caloCentRefWidth_ / 2.) * nCentBins_ / 100.;
        int maxbin = (caloCentRef_ + caloCentRefWidth_ / 2.) * nCentBins_ / 100.;
        minbin /= CentBinCompression_;
        maxbin /= CentBinCompression_;
        if (minbin > 0 && maxbin >= minbin) {
          if (EPDet[i] == HF || EPDet[i] == Castor)
            flat[i]->setCaloCentRefBins(minbin, maxbin);
        }
      }
    }
  }
  //
  //Get flattening parameter file.
  //
  if (hirpWatcher.check(iSetup)) {
    edm::ESHandle<RPFlatParams> flatparmsDB_;
    iSetup.get<HeavyIonRPRcd>().get(flatparmsDB_);
    LoadEPDB db(flatparmsDB_, flat);
  }  //rp record change

  //
  //Get Centrality
  //
  int bin = 0;
  int cbin = 0;
  cbin = iEvent.get(centralityBinToken_);
  bin = cbin / CentBinCompression_;
  //
  //Get Vertex
  //

  //best vertex
  double bestvz = -999.9;
  const reco::Vertex& vtx = iEvent.get(vertexToken_)[0];
  bestvz = vtx.z();

  //
  //Get Event Planes
  //

  auto const& evtPlanes = iEvent.get(inputPlanesToken_);

  auto evtplaneOutput = std::make_unique<EvtPlaneCollection>();
  EvtPlane* ep[NumEPNames];
  for (int i = 0; i < NumEPNames; i++) {
    ep[i] = nullptr;
  }
  int indx = 0;
  for (auto&& rp : (evtPlanes)) {
    double s = rp.sumSin(0);
    double c = rp.sumCos(0);
    uint m = rp.mult();
    double soff = s;
    double coff = c;
    double psiOffset = -10;
    double psiFlat = -10;
    if (rp.angle(0) > -5) {
      if (useOffsetPsi_) {
        soff = flat[indx]->soffset(s, bestvz, bin);
        coff = flat[indx]->coffset(c, bestvz, bin);
        psiOffset = flat[indx]->offsetPsi(soff, coff);
      }
      psiFlat = flat[indx]->getFlatPsi(psiOffset, bestvz, bin);
    }
    ep[indx] = new EvtPlane(indx, 2, psiFlat, soff, coff, rp.sumw(), rp.sumw2(), rp.sumPtOrEt(), rp.sumPtOrEt2(), m);
    ep[indx]->addLevel(0, rp.angle(0), s, c);
    ep[indx]->addLevel(3, 0., rp.sumSin(3), rp.sumCos(3));
    if (useOffsetPsi_)
      ep[indx]->addLevel(1, psiOffset, soff, coff);
    ++indx;
  }

  for (int i = 0; i < NumEPNames; i++) {
    if (ep[i] != nullptr)
      evtplaneOutput->push_back(*ep[i]);
  }
  iEvent.put(std::move(evtplaneOutput));
  for (int i = 0; i < indx; i++)
    delete ep[i];
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiEvtPlaneFlatProducer);

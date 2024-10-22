// ----------------------------------------------------------------------
// PCCNTupler
// ---------

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <bitset>

#include "PCCNTupler.h"

#include "CondFormats/Alignment/interface/Definitions.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
DEFINE_FWK_MODULE(PCCNTupler);

#include <TROOT.h>
#include <TSystem.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>

using namespace std;
using namespace edm;
using namespace reco;

// ----------------------------------------------------------------------
PCCNTupler::PCCNTupler(edm::ParameterSet const& iConfig)
    : fPrimaryVertexCollectionLabel(
          iConfig.getUntrackedParameter<InputTag>("vertexCollLabel", edm::InputTag("offlinePrimaryVertices"))),
      fPixelClusterLabel(
          iConfig.getUntrackedParameter<InputTag>("pixelClusterLabel", edm::InputTag("siPixelClusters"))),
      fPileUpInfoLabel(edm::InputTag("addPileupInfo")),
      saveType(iConfig.getUntrackedParameter<string>("saveType")),
      sampleType(iConfig.getUntrackedParameter<string>("sampleType")),
      includeVertexInformation(iConfig.getUntrackedParameter<bool>("includeVertexInformation", 1)),
      includePixels(iConfig.getUntrackedParameter<bool>("includePixels", 1)),
      includeJets(iConfig.getUntrackedParameter<bool>("includeJets", 0)),
      splitByBX(iConfig.getUntrackedParameter<bool>("splitByBX", 1)),
      pixelPhase2Geometry(iConfig.getUntrackedParameter<bool>("pixelPhase2Geometry", 0)) {
  cout << "----------------------------------------------------------------------" << endl;
  cout << "--- PCCNTupler constructor" << endl;

  edm::Service<TFileService> fs;

  tree = fs->make<TTree>("tree", "Pixel Cluster Counters");
  tree->Branch("run", &run, "run/I");
  tree->Branch("LS", &LS, "LS/I");
  tree->Branch("LN", &LN, "LN/I");
  tree->Branch("timeStamp_begin", &timeStamp_begin, "timeStamp_begin/i");
  tree->Branch("timeStamp_end", &timeStamp_end, "timeStamp_end/i");
  tree->Branch("eventCounter", &eventCounter, "eventCounter/I");
  tree->Branch("BXNo", "map<int,int>", &BXNo);
  if (saveType == "Event") {
    tree->Branch("event", &event, "event/i");
    tree->Branch("orbit", &orbit, "orbit/I");
    tree->Branch("bunchCrossing", &bunchCrossing, "bunchCrossing/I");
  }

  pileup = fs->make<TH1F>("pileup", "pileup", 100, 0, 100);
  if (includeVertexInformation) {
    tree->Branch("nGoodVtx", "map<int,int>", &nGoodVtx);
    tree->Branch("nValidVtx", "map<int,int>", &nValidVtx);
    recoVtxToken = consumes<reco::VertexCollection>(fPrimaryVertexCollectionLabel);
    if (saveType == "Event") {
      tree->Branch("nVtx", &nVtx, "nVtx/I");
      tree->Branch("vtx_nTrk", &vtx_nTrk, "vtx_nTrk[nVtx]/I");
      tree->Branch("vtx_ndof", &vtx_ndof, "vtx_ndof[nVtx]/I");
      tree->Branch("vtx_x", &vtx_x, "vtx_x[nVtx]/F");
      tree->Branch("vtx_y", &vtx_y, "vtx_y[nVtx]/F");
      tree->Branch("vtx_z", &vtx_z, "vtx_z[nVtx]/F");
      tree->Branch("vtx_xError", &vtx_xError, "vtx_xError[nVtx]/F");
      tree->Branch("vtx_yError", &vtx_yError, "vtx_yError[nVtx]/F");
      tree->Branch("vtx_zError", &vtx_zError, "vtx_zError[nVtx]/F");
      tree->Branch("vtx_chi2", &vtx_chi2, "vtx_chi2[nVtx]/F");
      tree->Branch("vtx_normchi2", &vtx_normchi2, "vtx_normchi2[nVtx]/F");
      tree->Branch("vtx_isValid", &vtx_isValid, "vtx_isValid[nVtx]/O");
      tree->Branch("vtx_isFake", &vtx_isFake, "vtx_isFake[nVtx]/O");
      tree->Branch("vtx_isGood", &vtx_isGood, "vtx_isGood[nVtx]/O");
    }
  }

  if (includePixels) {
    tree->Branch("nPixelClusters", "map<std::pair<int,int>,int>", &nPixelClusters);
    tree->Branch("nClusters", "map<std::pair<int,int>,int>", &nClusters);
    //tree->Branch("nPixelClusters","map<int,int>",&nPixelClusters);
    //tree->Branch("nClusters","map<int,int>",&nClusters);
    tree->Branch("layers", "map<int,int>", &layers);
    pixelToken = consumes<edmNew::DetSetVector<SiPixelCluster> >(fPixelClusterLabel);
  }

  if (sampleType == "MC") {
    pileUpToken = consumes<std::vector<PileupSummaryInfo> >(fPileUpInfoLabel);
    tree->Branch("nPU", &nPU, "nPU/I");
  }

  if (includeJets) {
    hltjetsToken_ = consumes<reco::CaloJetCollection>(edm::InputTag("ak4CaloJets"));
    const int kMaxJetCal = 100;
    jhcalpt = new float[kMaxJetCal];
    jhcalphi = new float[kMaxJetCal];
    jhcaleta = new float[kMaxJetCal];
    jhcale = new float[kMaxJetCal];
    jhcalemf = new float[kMaxJetCal];
    jhcaln90 = new float[kMaxJetCal];
    jhcaln90hits = new float[kMaxJetCal];

    //ccla HLTJETS
    tree->Branch("NohJetCal", &nhjetcal, "NohJetCal/I");
    tree->Branch("ohJetCalPt", jhcalpt, "ohJetCalPt[NohJetCal]/F");
    tree->Branch("ohJetCalPhi", jhcalphi, "ohJetCalPhi[NohJetCal]/F");
    tree->Branch("ohJetCalEta", jhcaleta, "ohJetCalEta[NohJetCal]/F");
    tree->Branch("ohJetCalE", jhcale, "ohJetCalE[NohJetCal]/F");
    tree->Branch("ohJetCalEMF", jhcalemf, "ohJetCalEMF[NohJetCal]/F");
    tree->Branch("ohJetCalN90", jhcaln90, "ohJetCalN90[NohJetCal]/F");
    tree->Branch("ohJetCalN90hits", jhcaln90hits, "ohJetCalN90hits[NohJetCal]/F");
  }
}

// ----------------------------------------------------------------------
PCCNTupler::~PCCNTupler() {}

// ----------------------------------------------------------------------
void PCCNTupler::endJob() { cout << "==>PCCNTupler> Succesfully gracefully ended job" << endl; }

// ----------------------------------------------------------------------
void PCCNTupler::beginJob() {}

void PCCNTupler::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& isetup) {
  firstEvent = true;
  Reset();
}

void PCCNTupler::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& isetup) { tree->Fill(); }

// ----------------------------------------------------------------------
void PCCNTupler::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using reco::VertexCollection;

  eventCounter++;

  saveAndReset = false;
  sameEvent = (event == (int)iEvent.id().event());
  sameLumiNib = true;  // FIXME where is this info?
  sameLumiSect = (LS == (int)iEvent.getLuminosityBlock().luminosityBlock());

  // When arriving at the new LS, LN or event the tree
  // must be filled and branches must be reset.
  // The final entry is saved in the deconstructor.
  saveAndReset = (saveType == "LumiSect" && !sameLumiSect) || (saveType == "LumiNib" && !sameLumiNib) ||
                 (saveType == "Event" && !sameEvent);

  if (!saveAndReset && !sameLumiSect && !sameLumiNib && !sameEvent) {
    std::cout << "Diff LS, LN and Event, but not saving/resetting..." << std::endl;
  }

  if (saveAndReset) {
    SaveAndReset();
  }

  if (sampleType == "MC") {
    edm::Handle<std::vector<PileupSummaryInfo> > pileUpInfo;
    iEvent.getByToken(pileUpToken, pileUpInfo);
    std::vector<PileupSummaryInfo>::const_iterator PVI;
    for (PVI = pileUpInfo->begin(); PVI != pileUpInfo->end(); ++PVI) {
      int pu_bunchcrossing = PVI->getBunchCrossing();
      //std::cout<<"pu_bunchcrossing getPU_NumInteractions getTrueNumInteractions "<<pu_bunchcrossing<<" "<<PVI->getPU_NumInteractions()<<" "<<PVI->getTrueNumInteractions()<<std::endl;
      if (pu_bunchcrossing == 0) {
        nPU = PVI->getPU_NumInteractions();
        pileup->Fill(nPU);
      }
    }
  }

  // Get the Run, Lumi Section, and Event numbers, etc.
  run = iEvent.id().run();
  LS = iEvent.getLuminosityBlock().luminosityBlock();
  //LN    = -99; // FIXME need the luminibble
  event = iEvent.id().event();
  bunchCrossing = iEvent.bunchCrossing();
  if (!splitByBX) {  //if no splitting by BX then we can remove info.
    bunchCrossing = -10;
  }
  timeStamp_local = iEvent.time().unixTime();
  if (timeStamp_end < timeStamp_local)
    timeStamp_end = timeStamp_local;
  if (timeStamp_begin > timeStamp_local)
    timeStamp_begin = timeStamp_local;
  orbit = iEvent.orbitNumber();
  //LN    = ((int) (orbit/pow(2,12)) % 64);
  //int LN2    = iEvent.nibble;
  LN = ((int)(orbit >> 12) % 64);  // FIXME need the luminibble

  bxModKey.first = bunchCrossing;
  bxModKey.second = -1;

  if ((BXNo.count(bunchCrossing) == 0 || nGoodVtx.count(bunchCrossing) == 0) &&
      !(BXNo.count(bunchCrossing) == 0 && nGoodVtx.count(bunchCrossing) == 0)) {
    std::cout << "BXNo and nGoodVtx should have the same keys but DO NOT!!!" << std::endl;
  }

  if (BXNo.count(bunchCrossing) == 0) {
    BXNo[bunchCrossing] = 0;
  }

  if (nGoodVtx.count(bunchCrossing) == 0) {
    nGoodVtx[bunchCrossing] = 0;
    nValidVtx[bunchCrossing] = 0;
  }

  BXNo[bunchCrossing] = BXNo[bunchCrossing] + 1;
  // add the vertex information

  if (includeVertexInformation) {
    edm::Handle<reco::VertexCollection> recVtxs;
    iEvent.getByToken(recoVtxToken, recVtxs);

    if (recVtxs.isValid()) {
      //nVtx=recVtxs->size();
      int ivtx = 0;
      for (reco::VertexCollection::const_iterator v = recVtxs->begin(); v != recVtxs->end(); ++v) {
        if (v->isFake())
          continue;
        vtx_isGood[ivtx] = false;
        vtx_nTrk[ivtx] = v->tracksSize();
        vtx_ndof[ivtx] = (int)v->ndof();
        vtx_x[ivtx] = v->x();
        vtx_y[ivtx] = v->y();
        vtx_z[ivtx] = v->z();
        vtx_xError[ivtx] = v->xError();
        vtx_yError[ivtx] = v->yError();
        vtx_zError[ivtx] = v->zError();
        vtx_chi2[ivtx] = v->chi2();
        vtx_normchi2[ivtx] = v->normalizedChi2();
        vtx_isValid[ivtx] = v->isValid();
        vtx_isFake[ivtx] = v->isFake();
        if (vtx_isValid[ivtx] && (vtx_isFake[ivtx] == 0)) {
          nValidVtx[bunchCrossing] = nValidVtx[bunchCrossing] + 1;
        }
        if (vtx_ndof[ivtx] > 4 && vtx_isValid[ivtx] && (vtx_isFake[ivtx] == 0)) {
          if (vtx_nTrk[ivtx] > 0) {
            nGoodVtx[bunchCrossing] = nGoodVtx[bunchCrossing] + 1;
            vtx_isGood[ivtx] = true;
          }
        }
        ivtx++;
      }
      nVtx = ivtx;
    }
  }

  if (includeJets) {
    edm::Handle<reco::CaloJetCollection> hltjets;
    iEvent.getByToken(hltjetsToken_, hltjets);
    bool valid = hltjets.isValid();
    if (not valid) {
      std::cout << "hltjets not valid " << std::endl;
      nhjetcal = -1;
    } else {
      reco::CaloJetCollection mycalojets;
      mycalojets = *hltjets;
      //std::sort(mycalojets.begin(),mycalojets.end(),PtGreater());
      typedef reco::CaloJetCollection::const_iterator cjiter;
      int jhcal = 0;
      for (cjiter i = mycalojets.begin(); i != mycalojets.end(); i++) {
        if (i->pt() > 5 && i->energy() > 0.) {
          jhcalpt[jhcal] = i->pt();
          jhcalphi[jhcal] = i->phi();
          jhcaleta[jhcal] = i->eta();
          jhcale[jhcal] = i->energy();
          jhcalemf[jhcal] = i->emEnergyFraction();
          jhcaln90[jhcal] = i->n90();
          //jetID->calculate( iEvent, *i );
          //jhcaln90hits[jhcal] = jetID->n90Hits();
          jhcal++;
        }
      }
      nhjetcal = jhcal;
    }
  }

  int NumPixelBarrelLayers = 3;
  if (pixelPhase2Geometry) {
    NumPixelBarrelLayers = 4;
  }
  // -- Pixel cluster
  if (includePixels) {
    edm::Handle<edmNew::DetSetVector<SiPixelCluster> > hClusterColl;
    iEvent.getByToken(pixelToken, hClusterColl);
    if (!hClusterColl.failedToGet()) {
      const edmNew::DetSetVector<SiPixelCluster>& clustColl = *hClusterColl;
      // ----------------------------------------------------------------------
      // -- Clusters without tracks

      for (edmNew::DetSetVector<SiPixelCluster>::const_iterator isearch = clustColl.begin(); isearch != clustColl.end();
           ++isearch) {
        // these are sorted by modules so we pick the current one
        edmNew::DetSet<SiPixelCluster> mod = *isearch;
        if (mod.empty()) {
          continue;
        }  // skip empty modules
        DetId detId = mod.id();

        bxModKey.second = detId();
        for (edmNew::DetSet<SiPixelCluster>::const_iterator di = mod.begin(); di != mod.end(); ++di) {
          if (nPixelClusters.count(bxModKey) == 0) {
            nPixelClusters[bxModKey] = 0;
          }
          nPixelClusters[bxModKey] = nPixelClusters[bxModKey] + 1;

          int nCluster = isearch->size();
          if (nClusters.count(bxModKey) == 0) {
            nClusters[bxModKey] = 0;
          }
          nClusters[bxModKey] += nCluster;

          if (detId.subdetId() == PixelSubdetector::PixelBarrel) {
            PixelBarrelName detName = PixelBarrelName(detId);
            int layer = detName.layerName();
            if (layers.count(detId()) == 0) {
              layers[detId()] = layer;
            }
          } else {
            assert(detId.subdetId() == PixelSubdetector::PixelEndcap);
            PixelEndcapName detName = PixelEndcapName(detId);
            int disk = detName.diskName();
            if (layers.count(detId()) == 0) {
              layers[detId()] = disk + NumPixelBarrelLayers;
            }
          }
          //}
        }
      }
    }
  }
}

void PCCNTupler::Reset() {
  nVtx = 0;
  nPixelClusters.clear();
  nClusters.clear();
  layers.clear();
  BXNo.clear();
  nValidVtx.clear();
  nGoodVtx.clear();
  eventCounter = 1;
  timeStamp_end = 0;
  timeStamp_begin = -1;
}

void PCCNTupler::SaveAndReset() {
  if (!firstEvent)
    tree->Fill();
  Reset();
  firstEvent = false;
}

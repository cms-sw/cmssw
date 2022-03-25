/* Software to emulate the hardware 2-layer jet-finding algorithm (fixed-point). *Layers 1 and 2*
 *
 * 2021
 *
 * Created based on Rishi Patel's L1TrackJetProducer.cc file.
 * Authors: Samuel Edwin Leigh, Tyler Wu
 * Rutgers, the State University of New Jersey
 *  Revolutionary for 250 years
 */

//Holds data from tracks, converted from their integer versions.

// system include files

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1Trigger/interface/TkJetWord.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/L1Trigger/interface/Vertex.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstdlib>
#include "TH1D.h"
#include "TH2D.h"
#include <TMath.h>
#include <ap_int.h>
#include "L1Trigger/L1TTrackMatch/interface/L1TrackJetEmulationProducer.h"

using namespace std;
using namespace edm;
class L1TrackJetEmulationProducer : public stream::EDProducer<> {
public:
  explicit L1TrackJetEmulationProducer(const ParameterSet &);
  ~L1TrackJetEmulationProducer() override;
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef vector<L1TTTrackType> L1TTTrackCollectionType;

  static void fillDescriptions(ConfigurationDescriptions &descriptions);
  bool trackQualityCuts(int trk_nstub, float trk_chi2, float trk_bendchi2);
  void L2_cluster(vector<Ptr<L1TTTrackType>> L1TrkPtrs_, TrackJetEmulationMaxZBin &mzb);
  virtual TrackJetEmulationEtaPhiBin *L1_cluster(TrackJetEmulationEtaPhiBin *phislice);

private:
  void beginStream(StreamID) override;
  void produce(Event &, const EventSetup &) override;
  void endStream() override;

  // ----------member data ---------------------------

  vector<Ptr<L1TTTrackType>> L1TrkPtrs_;
  vector<int> zBinCount_;
  vector<int> tdtrk_;
  const float MaxDzTrackPV;
  const float trkZMax_;
  const float trkPtMax_;
  const float trkPtMin_;
  const float trkEtaMax_;
  const float trkChi2dofMax_;
  const float trkBendChi2Max_;
  const int trkNPSStubMin_;
  const double minTrkJetpT_;
  int etaBins_;
  int phiBins_;
  int zBins_;
  float d0CutNStubs4_;
  float d0CutNStubs5_;
  int lowpTJetMinTrackMultiplicity_;
  float lowpTJetMinpT_;
  int highpTJetMinTrackMultiplicity_;
  float highpTJetMinpT_;
  bool displaced_;
  float nStubs4DisplacedChi2_;
  float nStubs5DisplacedChi2_;
  float nStubs4DisplacedBend_;
  float nStubs5DisplacedBend_;
  int nDisplacedTracks_;

  float PVz;
  z0_intern zStep_;
  glbeta_intern etaStep_;
  glbphi_intern phiStep_;

  const EDGetTokenT<vector<TTTrack<Ref_Phase2TrackerDigi_>>> trackToken_;
  const EDGetTokenT<l1t::VertexWordCollection> PVtxToken_;
  ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
};

L1TrackJetEmulationProducer::L1TrackJetEmulationProducer(const ParameterSet &iConfig)
    : MaxDzTrackPV((float)iConfig.getParameter<double>("MaxDzTrackPV")),
      trkZMax_((float)iConfig.getParameter<double>("trk_zMax")),
      trkPtMax_((float)iConfig.getParameter<double>("trk_ptMax")),
      trkPtMin_((float)iConfig.getParameter<double>("trk_ptMin")),
      trkEtaMax_((float)iConfig.getParameter<double>("trk_etaMax")),
      trkChi2dofMax_((float)iConfig.getParameter<double>("trk_chi2dofMax")),
      trkBendChi2Max_((float)iConfig.getParameter<double>("trk_bendChi2Max")),
      trkNPSStubMin_((int)iConfig.getParameter<int>("trk_nPSStubMin")),
      minTrkJetpT_(iConfig.getParameter<double>("minTrkJetpT")),
      etaBins_((int)iConfig.getParameter<int>("etaBins")),
      phiBins_((int)iConfig.getParameter<int>("phiBins")),
      zBins_((int)iConfig.getParameter<int>("zBins")),
      d0CutNStubs4_((float)iConfig.getParameter<double>("d0_cutNStubs4")),
      d0CutNStubs5_((float)iConfig.getParameter<double>("d0_cutNStubs5")),
      lowpTJetMinTrackMultiplicity_((int)iConfig.getParameter<int>("lowpTJetMinTrackMultiplicity")),
      lowpTJetMinpT_((float)iConfig.getParameter<double>("lowpTJetMinpT")),
      highpTJetMinTrackMultiplicity_((int)iConfig.getParameter<int>("highpTJetMinTrackMultiplicity")),
      highpTJetMinpT_((float)iConfig.getParameter<double>("highpTJetMinpT")),
      displaced_(iConfig.getParameter<bool>("displaced")),
      nStubs4DisplacedChi2_((float)iConfig.getParameter<double>("nStubs4DisplacedChi2")),
      nStubs5DisplacedChi2_((float)iConfig.getParameter<double>("nStubs5DisplacedChi2")),
      nStubs4DisplacedBend_((float)iConfig.getParameter<double>("nStubs4Displacedbend")),
      nStubs5DisplacedBend_((float)iConfig.getParameter<double>("nStubs5Displacedbend")),
      nDisplacedTracks_((int)iConfig.getParameter<int>("nDisplacedTracks")),
      trackToken_(consumes<vector<TTTrack<Ref_Phase2TrackerDigi_>>>(iConfig.getParameter<InputTag>("L1TrackInputTag"))),
      PVtxToken_(consumes<l1t::VertexWordCollection>(iConfig.getParameter<InputTag>("VertexInputTag"))),
      tTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""))) {
  zStep_ = convert::makeZ0(2.0 * trkZMax_ / (zBins_ + 1));
  etaStep_ = convert::makeGlbEta(2.0 * trkEtaMax_ / etaBins_);  //etaStep is the width of an etabin
  phiStep_ = convert::makeGlbPhi(2.0 * M_PI / phiBins_);        ////phiStep is the width of a phibin

  if (displaced_)
    produces<l1t::TkJetWordCollection>("L1TrackJetsExtended");
  else
    produces<l1t::TkJetWordCollection>("L1TrackJets");
}

L1TrackJetEmulationProducer::~L1TrackJetEmulationProducer() {}

void L1TrackJetEmulationProducer::produce(Event &iEvent, const EventSetup &iSetup) {
  unique_ptr<l1t::TkJetWordCollection> L1L1TrackJetProducer(new l1t::TkJetWordCollection);

  // For TTStubs
  const TrackerTopology &tTopo = iSetup.getData(tTopoToken_);

  edm::Handle<vector<TTTrack<Ref_Phase2TrackerDigi_>>> TTTrackHandle;
  iEvent.getByToken(trackToken_, TTTrackHandle);
  vector<TTTrack<Ref_Phase2TrackerDigi_>>::const_iterator iterL1Track;

  edm::Handle<l1t::VertexWordCollection> PVtx;
  iEvent.getByToken(PVtxToken_, PVtx);
  float PVz = (PVtx->at(0)).z0();

  L1TrkPtrs_.clear();
  zBinCount_.clear();
  tdtrk_.clear();
  unsigned int this_l1track = 0;
  for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
    edm::Ptr<L1TTTrackType> trkPtr(TTTrackHandle, this_l1track);
    this_l1track++;
    float trk_pt = trkPtr->momentum().perp();
    int trk_nstubs = (int)trkPtr->getStubRefs().size();
    float trk_chi2dof = trkPtr->chi2Red();
    float trk_d0 = trkPtr->d0();
    float trk_bendchi2 = trkPtr->stubPtConsistency();
    float trk_z0 = trkPtr->z0();

    int trk_nPS = 0;
    for (int istub = 0; istub < trk_nstubs; istub++) {  // loop over the stubs
      DetId detId(trkPtr->getStubRefs().at(istub)->getDetId());
      if (detId.det() == DetId::Detector::Tracker) {
        if ((detId.subdetId() == StripSubdetector::TOB && tTopo.tobLayer(detId) <= 3) ||
            (detId.subdetId() == StripSubdetector::TID && tTopo.tidRing(detId) <= 9))
          trk_nPS++;
      }
    }

    if (trk_nPS < trkNPSStubMin_)
      continue;
    if (!trackQualityCuts(trk_nstubs, trk_chi2dof, trk_bendchi2))
      continue;
    if (std::abs(trk_z0 - PVz) > MaxDzTrackPV)
      continue;
    if (std::abs(trk_z0) > trkZMax_)
      continue;
    if (std::abs(trkPtr->momentum().eta()) > trkEtaMax_)
      continue;
    if (trk_pt < trkPtMin_)
      continue;
    L1TrkPtrs_.push_back(trkPtr);
    zBinCount_.push_back(0);

    if ((std::abs(trk_d0) > d0CutNStubs5_ && trk_nstubs >= 5 && d0CutNStubs5_ >= 0) ||
        (trk_nstubs == 4 && std::abs(trk_d0) > d0CutNStubs4_ && d0CutNStubs4_ >= 0))
      tdtrk_.push_back(1);  //displaced track
    else
      tdtrk_.push_back(0);  // not displaced track
  }

  if (!L1TrkPtrs_.empty()) {
    TrackJetEmulationMaxZBin mzb;

    L2_cluster(L1TrkPtrs_, mzb);
    if (mzb.clusters != nullptr) {
      for (int j = 0; j < mzb.nclust; ++j) {
        //FILL Two Layer Jets for Jet Collection
        if (mzb.clusters[j].pTtot < convert::makePtFromFloat(trkPtMin_))
          continue;  //protects against reading bad memory
        if (mzb.clusters[j].ntracks < 1)
          continue;
        if (mzb.clusters[j].ntracks > 5000)
          continue;
        l1t::TkJetWord::glbeta_t jetEta = mzb.clusters[j].eta * convert::ETA_LSB_POW;
        l1t::TkJetWord::glbphi_t jetPhi = mzb.clusters[j].phi * convert::PHI_LSB_POW;
        l1t::TkJetWord::z0_t jetZ0 = mzb.zbincenter * convert::Z0_LSB_POW;
        l1t::TkJetWord::pt_t jetPt = mzb.clusters[j].pTtot;
        l1t::TkJetWord::nt_t totalntracks_ = mzb.clusters[j].ntracks;
        l1t::TkJetWord::nx_t totalxtracks_ = mzb.clusters[j].nxtracks;
        l1t::TkJetWord::tkjetunassigned_t unassigned_ = 0;

        l1t::TkJetWord trkJet(jetPt, jetEta, jetPhi, jetZ0, totalntracks_, totalxtracks_, unassigned_);
        //trkJet.setDispCounters(DispCounters);
        L1L1TrackJetProducer->push_back(trkJet);
      }
    } else if (mzb.clusters == nullptr) {
      edm::LogWarning("L1TrackJetEmulationProducer") << "mzb.clusters Not Assigned!\n";
    }

    if (displaced_)
      iEvent.put(std::move(L1L1TrackJetProducer), "L1TrackJetsExtended");
    else
      iEvent.put(std::move(L1L1TrackJetProducer), "L1TrackJets");
    delete[] mzb.clusters;
  } else if (L1TrkPtrs_.empty()) {
    edm::LogWarning("L1TrackJetEmulationProducer") << "L1TrkPtrs Not Assigned!\n";
  }
}

void L1TrackJetEmulationProducer::L2_cluster(vector<Ptr<L1TTTrackType>> L1TrkPtrs_, TrackJetEmulationMaxZBin &mzb) {
  enum TrackBitWidths {
    kEtaSize = 16,    // Width of z-position (40cm / 0.1)
    kEtaMagSize = 3,  // Width of z-position magnitude (signed)
    kPtSize = 14,     // Width of pt
    kPtMagSize = 9,   // Width of pt magnitude (unsigned)
    kPhiSize = 12,
    kPhiMagSize = 1,
  };

  const int nz = zBins_ + 1;
  TrackJetEmulationMaxZBin all_zBins[nz];
  static TrackJetEmulationMaxZBin mzbtemp;
  for (int z = 0; z < nz; ++z)
    all_zBins[z] = mzbtemp;

  z0_intern zmin = convert::makeZ0(-1.0 * trkZMax_);
  z0_intern zmax = zmin + 2 * zStep_;

  TrackJetEmulationEtaPhiBin epbins[phiBins_][etaBins_];  // create grid of phiBins
  glbphi_intern phi = convert::makeGlbPhi(-1.0 * M_PI);
  glbeta_intern eta;
  glbeta_intern etamin, etamax, phimin, phimax;
  for (int i = 0; i < phiBins_; ++i) {
    eta = convert::makeGlbEta(-1.0 * trkEtaMax_);
    for (int j = 0; j < etaBins_; ++j) {
      phimin = phi;
      phimax = phi + phiStep_;
      etamin = eta;
      eta = eta + etaStep_;
      etamax = eta;
      epbins[i][j].phi = (phimin + phimax) / 2;
      epbins[i][j].eta = (etamin + etamax) / 2;
      epbins[i][j].pTtot = 0;
      epbins[i][j].ntracks = 0;
      epbins[i][j].nxtracks = 0;
    }  // for each etabin
    phi = phi + phiStep_;
  }  // for each phibin (finished creating epbins)

  mzb = all_zBins[0];
  mzb.ht = 0;
  int ntracks = L1TrkPtrs_.size();
  // uninitalized arrays
  TrackJetEmulationEtaPhiBin *L1clusters[phiBins_];
  TrackJetEmulationEtaPhiBin L2cluster[ntracks];

  for (int zbin = 0; zbin < zBins_; ++zbin) {
    for (int i = 0; i < phiBins_; ++i) {  //First initialize pT, numtracks, used to 0 (or false)
      for (int j = 0; j < etaBins_; ++j) {
        epbins[i][j].pTtot = 0;
        epbins[i][j].used = false;
        epbins[i][j].ntracks = 0;
        epbins[i][j].nxtracks = 0;
      }  //for each etabin
      L1clusters[i] = epbins[i];
    }  //for each phibin

    for (unsigned int k = 0; k < L1TrkPtrs_.size(); ++k) {
      ap_ufixed<TrackBitWidths::kPtSize, TrackBitWidths::kPtMagSize, AP_RND_CONV, AP_SAT> inputTrkPt = 0;
      inputTrkPt.V = L1TrkPtrs_[k]->getTrackWord()(TTTrack_TrackWord::TrackBitLocations::kRinvMSB - 1,
                                                   TTTrack_TrackWord::TrackBitLocations::kRinvLSB);
      pt_intern trkpt = inputTrkPt;

      ap_fixed<TrackBitWidths::kEtaSize, TrackBitWidths::kEtaMagSize, AP_RND_CONV, AP_SAT> trketainput = 0;
      trketainput.V = L1TrkPtrs_[k]->getTrackWord()(TTTrack_TrackWord::TrackBitLocations::kTanlMSB,
                                                    TTTrack_TrackWord::TrackBitLocations::kTanlLSB);
      ap_ufixed<64 + ETA_EXTRABITS, 32 + ETA_EXTRABITS> eta_conv =
          1.0 / convert::ETA_LSB;  //conversion factor from input eta format to internal format
      glbeta_intern trketa = eta_conv * trketainput;

      int Sector = L1TrkPtrs_[k]->phiSector();
      glbphi_intern trkphiSector = 0;
      if (Sector < 5) {
        trkphiSector = Sector * convert::makeGlbPhi(2.0 * M_PI / 9.0);
      } else {
        trkphiSector =
            convert::makeGlbPhi(-1.0 * M_PI + M_PI / 9.0) + (Sector - 5) * convert::makeGlbPhi(2.0 * M_PI / 9.0);
      }

      ap_fixed<TrackBitWidths::kPhiSize, TrackBitWidths::kPhiMagSize, AP_RND_CONV, AP_SAT> trkphiinput = 0;
      trkphiinput.V = L1TrkPtrs_[k]->getTrackWord()(TTTrack_TrackWord::TrackBitLocations::kPhiMSB,
                                                    TTTrack_TrackWord::TrackBitLocations::kPhiLSB);
      ap_ufixed<64 + PHI_EXTRABITS, 32 + PHI_EXTRABITS> phi_conv =
          TTTrack_TrackWord::stepPhi0 / convert::PHI_LSB *
          (1 << (TrackBitWidths::kPhiSize - TrackBitWidths::kPhiMagSize));
      //phi_conv is a conversion factor from input phi format to the internal format
      glbeta_intern localphi = phi_conv * trkphiinput;
      glbeta_intern trkphi = localphi + trkphiSector;

      ap_int<TTTrack_TrackWord::TrackBitWidths::kZ0Size> inputTrkZ0 = L1TrkPtrs_[k]->getTrackWord()(
          TTTrack_TrackWord::TrackBitLocations::kZ0MSB, TTTrack_TrackWord::TrackBitLocations::kZ0LSB);
      ap_ufixed<32 + Z0_EXTRABITS, 1 + Z0_EXTRABITS> z0_conv =
          TTTrack_TrackWord::stepZ0 / convert::Z0_LSB;  //conversion factor from input z format to internal format
      z0_intern trkZ = z0_conv * inputTrkZ0;

      for (int i = 0; i < phiBins_; ++i) {
        for (int j = 0; j < etaBins_; ++j) {
          L2cluster[k] = epbins[i][j];
          if ((zmin <= trkZ && zmax >= trkZ) &&
              ((epbins[i][j].eta - etaStep_ / 2 < trketa && epbins[i][j].eta + etaStep_ / 2 >= trketa) &&
               epbins[i][j].phi - phiStep_ / 2 < trkphi && epbins[i][j].phi + phiStep_ / 2 >= trkphi &&
               (zBinCount_[k] != 2))) {
            zBinCount_.at(k) = zBinCount_.at(k) + 1;

            if (trkpt < convert::makePtFromFloat(trkPtMax_))
              epbins[i][j].pTtot += trkpt;
            else
              epbins[i][j].pTtot += convert::makePtFromFloat(trkPtMax_);
            ++epbins[i][j].ntracks;
            //x-bit is currently not used in firmware, so we leave nxtracks = 0 for now
          }  // if right bin
        }    // for each phibin: j loop
      }      // for each phibin: i loop
    }        // end loop over tracks

    for (int phislice = 0; phislice < phiBins_; ++phislice) {
      L1clusters[phislice] = L1_cluster(epbins[phislice]);
      for (int ind = 0; L1clusters[phislice][ind].pTtot != 0; ++ind) {
        L1clusters[phislice][ind].used = false;
      }
    }

    //Create clusters array to hold output cluster data for Layer2; can't have more clusters than tracks.
    //Find eta-phibin with maxpT, make center of cluster, add neighbors if not already used.
    pt_intern hipT = 0;
    int nclust = 0;
    int phibin = 0;
    int imax = -1;
    int index1;  //index of clusters array for each phislice
    pt_intern E1 = 0;
    pt_intern E0 = 0;
    pt_intern E2 = 0;
    l1t::TkJetWord::nt_t ntrk1, ntrk2;
    l1t::TkJetWord::nx_t nxtrk1, nxtrk2;
    int used1, used2, used3, used4;

    for (phibin = 0; phibin < phiBins_; ++phibin) {  //Find eta-phibin with highest pT
      while (true) {
        hipT = 0;
        for (index1 = 0; L1clusters[phibin][index1].pTtot > 0; ++index1) {
          if (!L1clusters[phibin][index1].used && L1clusters[phibin][index1].pTtot >= hipT) {
            hipT = L1clusters[phibin][index1].pTtot;
            imax = index1;
          }
        }  // for each index within the phibin

        if (hipT == 0)
          break;    // If highest pT is 0, all bins are used
        E0 = hipT;  // E0 is pT of first phibin of the cluster
        E1 = 0;
        E2 = 0;
        ntrk1 = 0;
        ntrk2 = 0;
        nxtrk1 = 0;
        nxtrk2 = 0;
        L2cluster[nclust] = L1clusters[phibin][imax];

        L1clusters[phibin][imax].used = true;
        // Add pT of upper neighbor
        // E1 is pT of the middle phibin (should be highest pT)
        if (phibin != phiBins_ - 1) {
          used1 = -1;
          used2 = -1;
          for (index1 = 0; L1clusters[phibin + 1][index1].pTtot != 0; ++index1) {
            if (L1clusters[phibin + 1][index1].used)
              continue;
            if (L1clusters[phibin + 1][index1].eta - L1clusters[phibin][imax].eta <= 3 * etaStep_ / 2 &&
                L1clusters[phibin][imax].eta - L1clusters[phibin + 1][index1].eta <= 3 * etaStep_ / 2) {
              E1 += L1clusters[phibin + 1][index1].pTtot;
              ntrk1 += L1clusters[phibin + 1][index1].ntracks;
              nxtrk1 += L1clusters[phibin + 1][index1].nxtracks;
              if (used1 < 0)
                used1 = index1;
              else
                used2 = index1;
            }  // if cluster is within one phibin
          }    // for each cluster in above phibin

          if (E1 < E0) {  // if E1 isn't higher, E0 and E1 are their own cluster
            L2cluster[nclust].pTtot += E1;
            L2cluster[nclust].ntracks += ntrk1;
            L2cluster[nclust].nxtracks += nxtrk1;
            if (used1 >= 0)
              L1clusters[phibin + 1][used1].used = true;
            if (used2 >= 0)
              L1clusters[phibin + 1][used2].used = true;
            nclust++;
            continue;
          }

          if (phibin != phiBins_ - 2) {  // E2 will be the pT of the third phibin (should be lower than E1)
            used3 = -1;
            used4 = -1;
            for (index1 = 0; L1clusters[phibin + 2][index1].pTtot != 0; ++index1) {
              if (L1clusters[phibin + 2][index1].used)
                continue;
              if (L1clusters[phibin + 2][index1].eta - L1clusters[phibin][imax].eta <= 3 * etaStep_ / 2 &&
                  L1clusters[phibin][imax].eta - L1clusters[phibin + 2][index1].eta <= 3 * etaStep_ / 2) {
                E2 += L1clusters[phibin + 2][index1].pTtot;
                ntrk2 += L1clusters[phibin + 2][index1].ntracks;
                nxtrk2 += L1clusters[phibin + 2][index1].nxtracks;
                if (used3 < 0)
                  used3 = index1;
                else
                  used4 = index1;
              }
            }
            // if indeed E2 < E1, add E1 and E2 to E0, they're all a cluster together
            // otherwise, E0 is its own cluster
            if (E2 < E1) {
              L2cluster[nclust].pTtot += E1 + E2;
              L2cluster[nclust].ntracks += ntrk1 + ntrk2;
              L2cluster[nclust].nxtracks += nxtrk1 + nxtrk2;
              L2cluster[nclust].phi = L1clusters[phibin + 1][used1].phi;
              if (used1 >= 0)
                L1clusters[phibin + 1][used1].used = true;
              if (used2 >= 0)
                L1clusters[phibin + 1][used2].used = true;
              if (used3 >= 0)
                L1clusters[phibin + 2][used3].used = true;
              if (used4 >= 0)
                L1clusters[phibin + 2][used4].used = true;
            }
            nclust++;
            continue;
          }  // end Not phiBins-2
          else {
            L2cluster[nclust].pTtot += E1;
            L2cluster[nclust].ntracks += ntrk1;
            L2cluster[nclust].nxtracks += nxtrk1;
            L2cluster[nclust].phi = L1clusters[phibin + 1][used1].phi;
            if (used1 >= 0)
              L1clusters[phibin + 1][used1].used = true;
            if (used2 >= 0)
              L1clusters[phibin + 1][used2].used = true;
            nclust++;
            continue;
          }
        }       //End not last phibin(23)
        else {  //if it is phibin 23
          L1clusters[phibin][imax].used = true;
          nclust++;
        }
      }  // while hipT not 0
    }    // for each phibin

    for (phibin = 0; phibin < phiBins_; ++phibin)
      delete[] L1clusters[phibin];

    // Now merge clusters, if necessary
    for (int m = 0; m < nclust - 1; ++m) {
      for (int n = m + 1; n < nclust; ++n) {
        if (L2cluster[n].eta == L2cluster[m].eta &&
            ((L2cluster[n].phi - L2cluster[m].phi < 3 * phiStep_ / 2 &&
              L2cluster[m].phi - L2cluster[n].phi < 3 * phiStep_ / 2) ||
             (L2cluster[n].phi - L2cluster[m].phi > 2 * convert::makeGlbPhi(M_PI) - phiStep_ ||
              L2cluster[m].phi - L2cluster[n].phi > 2 * convert::makeGlbPhi(M_PI) - phiStep_))) {
          if (L2cluster[n].pTtot > L2cluster[m].pTtot) {
            L2cluster[m].phi = L2cluster[n].phi;
          }
          L2cluster[m].pTtot += L2cluster[n].pTtot;
          L2cluster[m].ntracks += L2cluster[n].ntracks;
          L2cluster[m].nxtracks += L2cluster[n].nxtracks;
          for (int m1 = n; m1 < nclust - 1; ++m1) {
            L2cluster[m1] = L2cluster[m1 + 1];
          }
          nclust--;
          m = -1;
          break;  //?????
        }         // end if clusters neighbor in eta
      }
    }  // end for (m) loop
    // sum up all pTs in this zbin to find ht

    pt_intern ht = 0;
    for (int k = 0; k < nclust; ++k) {
      if (L2cluster[k].pTtot > convert::makePtFromFloat(lowpTJetMinpT_) &&
          L2cluster[k].ntracks < lowpTJetMinTrackMultiplicity_)
        continue;
      if (L2cluster[k].pTtot > convert::makePtFromFloat(highpTJetMinpT_) &&
          L2cluster[k].ntracks < highpTJetMinTrackMultiplicity_)
        continue;
      if (L2cluster[k].pTtot > convert::makePtFromFloat(minTrkJetpT_))
        ht += L2cluster[k].pTtot;
    }
    // if ht is larger than previous max, this is the new vertex zbin
    all_zBins[zbin].znum = zbin;
    all_zBins[zbin].clusters = new TrackJetEmulationEtaPhiBin[nclust];
    all_zBins[zbin].nclust = nclust;
    all_zBins[zbin].zbincenter = (zmin + zmax) / 2;
    for (int k = 0; k < nclust; ++k) {
      all_zBins[zbin].clusters[k].phi = L2cluster[k].phi;
      all_zBins[zbin].clusters[k].eta = L2cluster[k].eta;
      all_zBins[zbin].clusters[k].pTtot = L2cluster[k].pTtot;
      all_zBins[zbin].clusters[k].ntracks = L2cluster[k].ntracks;
      all_zBins[zbin].clusters[k].nxtracks = L2cluster[k].nxtracks;
    }
    all_zBins[zbin].ht = ht;
    if (ht >= mzb.ht) {
      mzb = all_zBins[zbin];
      mzb.zbincenter = (zmin + zmax) / 2;
    }
    // Prepare for next zbin!
    zmin = zmin + zStep_;
    zmax = zmax + zStep_;
  }  // for each zbin

  for (int zbin = 0; zbin < zBins_; ++zbin) {
    if (zbin == mzb.znum) {
      continue;
    }
    delete[] all_zBins[zbin].clusters;
  }
}

TrackJetEmulationEtaPhiBin *L1TrackJetEmulationProducer::L1_cluster(TrackJetEmulationEtaPhiBin *phislice) {
  TrackJetEmulationEtaPhiBin *clusters = new TrackJetEmulationEtaPhiBin[etaBins_ / 2];
  for (int etabin = 0; etabin < etaBins_ / 2; ++etabin) {
    clusters[etabin].pTtot = 0;
    clusters[etabin].ntracks = 0;
    clusters[etabin].nxtracks = 0;
    clusters[etabin].phi = 0;
    clusters[etabin].eta = 0;
    clusters[etabin].used = false;
  }

  if (clusters == nullptr)
    edm::LogWarning("L1TrackJetEmulationProducer") << "Clusters memory not assigned!\n";

  // Find eta-phibin with maxpT, make center of cluster, add neighbors if not already used
  pt_intern my_pt, left_pt, right_pt, right2pt;
  int nclust = 0;
  right2pt = 0;
  for (int etabin = 0; etabin < etaBins_; ++etabin) {
    // assign values for my pT and neighbors' pT
    if (phislice[etabin].used)
      continue;
    my_pt = phislice[etabin].pTtot;
    if (etabin > 0 && !phislice[etabin - 1].used) {
      left_pt = phislice[etabin - 1].pTtot;
    } else
      left_pt = 0;
    if (etabin < etaBins_ - 1 && !phislice[etabin + 1].used) {
      right_pt = phislice[etabin + 1].pTtot;
      if (etabin < etaBins_ - 2 && !phislice[etabin + 2].used) {
        right2pt = phislice[etabin + 2].pTtot;
      } else
        right2pt = 0;
    } else
      right_pt = 0;

    // if I'm not a cluster, move on
    if (my_pt < left_pt || my_pt <= right_pt) {
      // if unused pT in the left neighbor, spit it out as a cluster
      if (left_pt > 0) {
        clusters[nclust] = phislice[etabin - 1];
        phislice[etabin - 1].used = true;
        nclust++;
      }
      continue;
    }

    // I guess I'm a cluster-- should I use my right neighbor?
    // Note: left neighbor will definitely be used because if it
    //       didn't belong to me it would have been used already
    clusters[nclust] = phislice[etabin];
    phislice[etabin].used = true;
    if (left_pt > 0) {
      clusters[nclust].pTtot += left_pt;
      clusters[nclust].ntracks += phislice[etabin - 1].ntracks;
      clusters[nclust].nxtracks += phislice[etabin - 1].nxtracks;
    }
    if (my_pt >= right2pt && right_pt > 0) {
      clusters[nclust].pTtot += right_pt;
      clusters[nclust].ntracks += phislice[etabin + 1].ntracks;
      clusters[nclust].nxtracks += phislice[etabin + 1].nxtracks;
      phislice[etabin + 1].used = true;
    }
    nclust++;
  }  // for each etabin

  // Now merge clusters, if necessary
  for (int m = 0; m < nclust - 1; ++m) {
    if (clusters[m + 1].eta - clusters[m].eta < 3 * etaStep_ / 2 &&
        clusters[m].eta - clusters[m + 1].eta < 3 * etaStep_ / 2) {
      if (clusters[m + 1].pTtot > clusters[m].pTtot) {
        clusters[m].eta = clusters[m + 1].eta;
      }
      clusters[m].pTtot += clusters[m + 1].pTtot;
      clusters[m].ntracks += clusters[m + 1].ntracks;  // Previous version didn't add tracks when merging
      clusters[m].nxtracks += clusters[m + 1].nxtracks;
      for (int m1 = m + 1; m1 < nclust - 1; ++m1)
        clusters[m1] = clusters[m1 + 1];
      nclust--;
      m = -1;
    }  // end if clusters neighbor in eta
  }    // end for (m) loop

  for (int i = nclust; i < etaBins_ / 2; ++i)  // zero out remaining unused clusters
    clusters[i].pTtot = 0;
  return clusters;
}

void L1TrackJetEmulationProducer::beginStream(StreamID) {}

void L1TrackJetEmulationProducer::endStream() {}

bool L1TrackJetEmulationProducer::trackQualityCuts(int trk_nstub, float trk_chi2, float trk_bendchi2) {
  bool PassQuality = false;
  if (trk_bendchi2 < trkBendChi2Max_ && trk_chi2 < trkChi2dofMax_ && trk_nstub >= 4 && !displaced_)
    PassQuality = true;
  if (displaced_ && trk_bendchi2 < nStubs4DisplacedBend_ && trk_chi2 < nStubs4DisplacedChi2_ && trk_nstub == 4)
    PassQuality = true;
  if (displaced_ && trk_bendchi2 < nStubs5DisplacedBend_ && trk_chi2 < nStubs5DisplacedChi2_ && trk_nstub > 4)
    PassQuality = true;
  return PassQuality;
}

void L1TrackJetEmulationProducer::fillDescriptions(ConfigurationDescriptions &descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackJetEmulationProducer);

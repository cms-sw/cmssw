// Original Author: Samuel Edwin Leigh, Tyler Wu,
//                  Rutgers, the State University of New Jersey
//
// Rewritting/Improvements:      George Karathanasis,
//                          georgios.karathanasis@cern.ch, CU Boulder
//                          Claire Savard (claire.savard@colorado.edu)
//
//         Created:  Wed, 01 Aug 2018 14:01:41 GMT
//         Latest update: Nov 2023 (by CS)
//
// Track jets are clustered in a two-layer process, first by clustering in phi,
// then by clustering in eta. The code proceeds as following: putting all tracks// in a grid of eta vs phi space, and then cluster them. Finally we merge the cl
// usters when needed. The code is improved to use the same module between emula
// tion and simulation was also improved, with bug fixes and being faster.
// Introduction to object (p10-13):
// https://indico.cern.ch/event/791517/contributions/3341650/attachments/1818736/2973771/TrackBasedAlgos_L1TMadrid_MacDonald.pdf
// New and improved version: https://indico.cern.ch/event/1203796/contributions/5073056/attachments/2519806/4333006/trackjet_emu.pdf

// L1T include files
#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1Trigger/interface/TkJetWord.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"

// system include files
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

//own headers
#include "L1TrackJetClustering.h"

//general
#include <ap_int.h>

using namespace std;
using namespace edm;
using namespace l1t;
using namespace l1ttrackjet;

class L1TrackJetEmulatorProducer : public stream::EDProducer<> {
public:
  explicit L1TrackJetEmulatorProducer(const ParameterSet &);
  ~L1TrackJetEmulatorProducer() override = default;
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef vector<L1TTTrackType> L1TTTrackCollectionType;
  typedef edm::RefVector<L1TTTrackCollectionType> L1TTTrackRefCollectionType;
  static void fillDescriptions(ConfigurationDescriptions &descriptions);

private:
  void produce(Event &, const EventSetup &) override;

  // ----------member data ---------------------------

  std::vector<edm::Ptr<L1TTTrackType>> L1TrkPtrs_;
  const float trkZMax_;
  const float trkPtMax_;
  const float trkEtaMax_;
  const int lowpTJetMinTrackMultiplicity_;
  const float lowpTJetThreshold_;
  const int highpTJetMinTrackMultiplicity_;
  const float highpTJetThreshold_;
  const int zBins_;
  const int etaBins_;
  const int phiBins_;
  const double minTrkJetpT_;
  const bool displaced_;
  const float d0CutNStubs4_;
  const float d0CutNStubs5_;
  const int nDisplacedTracks_;

  float zStep_;
  glbeta_intern etaStep_;
  glbphi_intern phiStep_;

  TTTrack_TrackWord trackword;

  const EDGetTokenT<L1TTTrackRefCollectionType> trackToken_;
};

//constructor
L1TrackJetEmulatorProducer::L1TrackJetEmulatorProducer(const ParameterSet &iConfig)
    : trkZMax_(iConfig.getParameter<double>("trk_zMax")),
      trkPtMax_(iConfig.getParameter<double>("trk_ptMax")),
      trkEtaMax_(iConfig.getParameter<double>("trk_etaMax")),
      lowpTJetMinTrackMultiplicity_(iConfig.getParameter<int>("lowpTJetMinTrackMultiplicity")),
      lowpTJetThreshold_(iConfig.getParameter<double>("lowpTJetThreshold")),
      highpTJetMinTrackMultiplicity_(iConfig.getParameter<int>("highpTJetMinTrackMultiplicity")),
      highpTJetThreshold_(iConfig.getParameter<double>("highpTJetThreshold")),
      zBins_(iConfig.getParameter<int>("zBins")),
      etaBins_(iConfig.getParameter<int>("etaBins")),
      phiBins_(iConfig.getParameter<int>("phiBins")),
      minTrkJetpT_(iConfig.getParameter<double>("minTrkJetpT")),
      displaced_(iConfig.getParameter<bool>("displaced")),
      d0CutNStubs4_(iConfig.getParameter<double>("d0_cutNStubs4")),
      d0CutNStubs5_(iConfig.getParameter<double>("d0_cutNStubs5")),
      nDisplacedTracks_(iConfig.getParameter<int>("nDisplacedTracks")),
      trackToken_(consumes<L1TTTrackRefCollectionType>(iConfig.getParameter<InputTag>("L1TrackInputTag"))) {
  zStep_ = 2.0 * trkZMax_ / (zBins_ + 1);                 // added +1 in denom
  etaStep_ = glbeta_intern(2.0 * trkEtaMax_ / etaBins_);  //etaStep is the width of an etabin
  phiStep_ = DoubleToBit(2.0 * (M_PI) / phiBins_,
                         TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit,
                         TTTrack_TrackWord::stepPhi0);  ///phiStep is the width of a phibin

  if (displaced_)
    produces<l1t::TkJetWordCollection>("L1TrackJetsExtended");
  else
    produces<l1t::TkJetWordCollection>("L1TrackJets");
}

void L1TrackJetEmulatorProducer::produce(Event &iEvent, const EventSetup &iSetup) {
  unique_ptr<l1t::TkJetWordCollection> L1TrackJetContainer(new l1t::TkJetWordCollection);

  // L1 tracks
  edm::Handle<L1TTTrackRefCollectionType> TTTrackHandle;
  iEvent.getByToken(trackToken_, TTTrackHandle);

  L1TrkPtrs_.clear();
  // track selection
  for (unsigned int this_l1track = 0; this_l1track < TTTrackHandle->size(); this_l1track++) {
    edm::Ptr<L1TTTrackType> trkPtr(TTTrackHandle, this_l1track);
    L1TrkPtrs_.push_back(trkPtr);
  }

  // if no tracks pass selection return empty containers
  if (L1TrkPtrs_.empty()) {
    if (displaced_)
      iEvent.put(std::move(L1TrackJetContainer), "L1TrackJetsExtended");
    else
      iEvent.put(std::move(L1TrackJetContainer), "L1TrackJets");
    return;
  }

  TrackJetEmulationMaxZBin mzb;
  mzb.znum = 0;
  mzb.nclust = 0;
  mzb.ht = 0;

  TrackJetEmulationEtaPhiBin epbins_default[phiBins_][etaBins_];  // create grid of phiBins
  glbphi_intern phi = DoubleToBit(
      -1.0 * M_PI, TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit, TTTrack_TrackWord::stepPhi0);
  for (int i = 0; i < phiBins_; ++i) {
    glbeta_intern eta = -1 * trkEtaMax_;
    for (int j = 0; j < etaBins_; ++j) {
      epbins_default[i][j].phi = (phi + (phi + phiStep_)) / 2;  // phi coord of bin center
      epbins_default[i][j].eta = (eta + (eta + etaStep_)) / 2;  // eta coord of bin center
      eta += etaStep_;
    }  // for each etabin
    phi += phiStep_;
  }  // for each phibin (finished creating epbins)

  // bins for z if multibin option is selected
  std::vector<z0_intern> zmins, zmaxs;
  for (int zbin = 0; zbin < zBins_; zbin++) {
    zmins.push_back(DoubleToBit(
        -1.0 * trkZMax_ + zStep_ * zbin, TTTrack_TrackWord::TrackBitWidths::kZ0Size, TTTrack_TrackWord::stepZ0));
    zmaxs.push_back(DoubleToBit(-1.0 * trkZMax_ + zStep_ * zbin + 2 * zStep_,
                                TTTrack_TrackWord::TrackBitWidths::kZ0Size,
                                TTTrack_TrackWord::stepZ0));
  }

  // create vectors that hold clusters
  std::vector<std::vector<TrackJetEmulationEtaPhiBin>> L1clusters;
  L1clusters.reserve(phiBins_);
  std::vector<TrackJetEmulationEtaPhiBin> L2clusters;

  // default (empty) grid
  for (int i = 0; i < phiBins_; ++i) {
    for (int j = 0; j < etaBins_; ++j) {
      epbins_default[i][j].pTtot = 0;
      epbins_default[i][j].used = false;
      epbins_default[i][j].ntracks = 0;
      epbins_default[i][j].nxtracks = 0;
      epbins_default[i][j].trackidx.clear();
    }
  }

  //Begin Firmware-style clustering
  // logic: loop over z bins find tracks in this bin and arrange them in a 2D eta-phi matrix
  for (unsigned int zbin = 0; zbin < zmins.size(); ++zbin) {
    // initialize matrices for every z bin
    z0_intern zmin = zmins[zbin];
    z0_intern zmax = zmaxs[zbin];

    TrackJetEmulationEtaPhiBin epbins[phiBins_][etaBins_];

    std::copy(&epbins_default[0][0], &epbins_default[0][0] + phiBins_ * etaBins_, &epbins[0][0]);

    L1clusters.clear();
    L2clusters.clear();
    for (unsigned int k = 0; k < L1TrkPtrs_.size(); ++k) {
      z0_intern trkZ = L1TrkPtrs_[k]->getZ0Word();

      if (zmax < trkZ)
        continue;
      if (zmin > trkZ)
        continue;
      if (zbin == 0 && zmin == trkZ)
        continue;

      // Pt
      ap_uint<TTTrack_TrackWord::TrackBitWidths::kRinvSize - 1> ptEmulationBits = L1TrkPtrs_[k]->getRinvWord();
      pt_intern trkpt;
      trkpt.V = ptEmulationBits.range();

      // d0
      d0_intern abs_trkD0 = L1TrkPtrs_[k]->getD0Word();

      // nstubs
      int trk_nstubs = (int)L1TrkPtrs_[k]->getStubRefs().size();

      // Phi bin
      int i = phi_bin_firmwareStyle(L1TrkPtrs_[k]->phiSector(),
                                    L1TrkPtrs_[k]->getPhiWord());  //Function defined in L1TrackJetClustering.h

      // Eta bin
      int j = eta_bin_firmwareStyle(L1TrkPtrs_[k]->getTanlWord());  //Function defined in L1TrackJetClustering.h

      //This is a quick fix to eta going outside of scope - also including protection against phi going outside
      //of scope as well. The eta index, j, cannot be less than zero or greater than 23 (the number of eta bins
      //minus one). The phi index, i, cannot be less than zero or greater than 26 (the number of phi bins
      //minus one).
      if ((j < 0) || (j > (etaBins_ - 1)) || (i < 0) || (i > (phiBins_ - 1)))
        continue;

      if (trkpt < pt_intern(trkPtMax_))
        epbins[i][j].pTtot += trkpt;
      else
        epbins[i][j].pTtot += pt_intern(trkPtMax_);
      if ((abs_trkD0 >
               DoubleToBit(d0CutNStubs5_, TTTrack_TrackWord::TrackBitWidths::kD0Size, TTTrack_TrackWord::stepD0) &&
           trk_nstubs >= 5 && d0CutNStubs5_ >= 0) ||
          (abs_trkD0 >
               DoubleToBit(d0CutNStubs4_, TTTrack_TrackWord::TrackBitWidths::kD0Size, TTTrack_TrackWord::stepD0) &&
           trk_nstubs == 4 && d0CutNStubs4_ >= 0))
        epbins[i][j].nxtracks += 1;

      epbins[i][j].trackidx.push_back(k);
      ++epbins[i][j].ntracks;
    }
    //End Firmware style clustering

    // first layer clustering - in eta using grid
    for (int phibin = 0; phibin < phiBins_; ++phibin) {
      L1clusters.push_back(L1_clustering<TrackJetEmulationEtaPhiBin, pt_intern, glbeta_intern, glbphi_intern>(
          epbins[phibin], etaBins_, etaStep_));
    }

    //second layer clustering in phi for all eta clusters
    L2clusters = L2_clustering<TrackJetEmulationEtaPhiBin, pt_intern, glbeta_intern, glbphi_intern>(
        L1clusters, phiBins_, phiStep_, etaStep_);

    // sum all cluster pTs in this zbin to find max
    pt_intern sum_pt = 0;
    for (unsigned int k = 0; k < L2clusters.size(); ++k) {
      if (L2clusters[k].pTtot > pt_intern(highpTJetThreshold_) && L2clusters[k].ntracks < lowpTJetMinTrackMultiplicity_)
        continue;
      if (L2clusters[k].pTtot > pt_intern(highpTJetThreshold_) &&
          L2clusters[k].ntracks < highpTJetMinTrackMultiplicity_)
        continue;

      if (L2clusters[k].pTtot > pt_intern(minTrkJetpT_))
        sum_pt += L2clusters[k].pTtot;
    }
    if (sum_pt < mzb.ht)
      continue;
    // if ht is larger than previous max, this is the new vertex zbin
    mzb.ht = sum_pt;
    mzb.znum = zbin;
    mzb.clusters = L2clusters;
    mzb.nclust = L2clusters.size();
    mzb.zbincenter = (zmin + zmax) / 2.0;
  }  //zbin loop

  vector<edm::Ptr<L1TTTrackType>> L1TrackAssocJet;
  for (unsigned int j = 0; j < mzb.clusters.size(); ++j) {
    l1t::TkJetWord::glbeta_t jetEta = DoubleToBit(double(mzb.clusters[j].eta),
                                                  TkJetWord::TkJetBitWidths::kGlbEtaSize,
                                                  TkJetWord::MAX_ETA / (1 << TkJetWord::TkJetBitWidths::kGlbEtaSize));
    l1t::TkJetWord::glbphi_t jetPhi = DoubleToBit(
        BitToDouble(mzb.clusters[j].phi, TTTrack_TrackWord::TrackBitWidths::kPhiSize + 4, TTTrack_TrackWord::stepPhi0),
        TkJetWord::TkJetBitWidths::kGlbPhiSize,
        (2. * std::abs(M_PI)) / (1 << TkJetWord::TkJetBitWidths::kGlbPhiSize));
    l1t::TkJetWord::z0_t jetZ0 = 0;
    l1t::TkJetWord::pt_t jetPt = mzb.clusters[j].pTtot;
    l1t::TkJetWord::nt_t total_ntracks = mzb.clusters[j].ntracks;
    l1t::TkJetWord::nx_t total_disptracks = mzb.clusters[j].nxtracks;
    l1t::TkJetWord::dispflag_t dispflag = 0;
    l1t::TkJetWord::tkjetunassigned_t unassigned = 0;

    if (total_disptracks >= nDisplacedTracks_)
      dispflag = 1;
    L1TrackAssocJet.clear();
    for (unsigned int itrk = 0; itrk < mzb.clusters[j].trackidx.size(); itrk++)
      L1TrackAssocJet.push_back(L1TrkPtrs_[mzb.clusters[j].trackidx[itrk]]);

    l1t::TkJetWord trkJet(jetPt, jetEta, jetPhi, jetZ0, total_ntracks, total_disptracks, dispflag, unassigned);

    L1TrackJetContainer->push_back(trkJet);
  }

  std::sort(L1TrackJetContainer->begin(), L1TrackJetContainer->end(), [](auto &a, auto &b) { return a.pt() > b.pt(); });
  if (displaced_)
    iEvent.put(std::move(L1TrackJetContainer), "L1TrackJetsExtended");
  else
    iEvent.put(std::move(L1TrackJetContainer), "L1TrackJets");
}

void L1TrackJetEmulatorProducer::fillDescriptions(ConfigurationDescriptions &descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"));
  desc.add<double>("trk_zMax", 15.0);
  desc.add<double>("trk_ptMax", 200.0);
  desc.add<double>("trk_etaMax", 2.4);
  desc.add<double>("minTrkJetpT", -1.0);
  desc.add<int>("etaBins", 24);
  desc.add<int>("phiBins", 27);
  desc.add<int>("zBins", 1);
  desc.add<double>("d0_cutNStubs4", -1);
  desc.add<double>("d0_cutNStubs5", -1);
  desc.add<int>("lowpTJetMinTrackMultiplicity", 2);
  desc.add<double>("lowpTJetThreshold", 50.0);
  desc.add<int>("highpTJetMinTrackMultiplicity", 3);
  desc.add<double>("highpTJetThreshold", 100.0);
  desc.add<bool>("displaced", false);
  desc.add<int>("nDisplacedTracks", 2);
  descriptions.add("l1tTrackJetsEmulator", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackJetEmulatorProducer);

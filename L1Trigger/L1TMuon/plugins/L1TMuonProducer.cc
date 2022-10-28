// -*- C++ -*-
//
// Package:    L1TMuonProducer
// Class:      L1TMuonProducer
//
/**\class L1TMuonProducer L1TMuonProducer.cc L1Trigger/L1TMuon/src/L1TMuonProducer.cc

 Description: Takes txt-file input and produces barrel- / overlap- / forward TF muons

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Joschka Philip Lingemann,40 3-B01,+41227671598,
//         Created:  Thu Oct  3 16:31:34 CEST 2013
// $Id$
//
//

// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTRankPtQualLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTIsolationUnit.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTCancelOutUnit.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"
#include "L1Trigger/L1TMuon/interface/GMTInternalMuon.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsO2ORcd.h"
#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParamsHelper.h"
#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParams_PUBLIC.h"

#include "TMath.h"
//
// class declaration
//
using namespace l1t;

class L1TMuonProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TMuonProducer(const edm::ParameterSet&);
  ~L1TMuonProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  static bool compareMuons(const std::shared_ptr<MicroGMTConfiguration::InterMuon>& mu1,
                           const std::shared_ptr<MicroGMTConfiguration::InterMuon>& mu2);

  void sortMuons(MicroGMTConfiguration::InterMuonList&, unsigned) const;

  void calculateRank(MicroGMTConfiguration::InterMuonList& muons) const;

  void splitAndConvertMuons(edm::Handle<MicroGMTConfiguration::InputCollection> const& in,
                            MicroGMTConfiguration::InterMuonList& out_pos,
                            MicroGMTConfiguration::InterMuonList& out_neg,
                            GMTInternalWedges& wedges_pos,
                            GMTInternalWedges& wedges_neg,
                            int bx) const;

  void convertMuons(edm::Handle<MicroGMTConfiguration::InputCollection> const& in,
                    MicroGMTConfiguration::InterMuonList& out,
                    GMTInternalWedges& wedges,
                    int bx) const;

  int computeMuonIdx(const RegionalMuonCand& mu, int currentLink, int muIdxAuto) const;

  void addMuonsToCollections(MicroGMTConfiguration::InterMuonList& coll,
                             MicroGMTConfiguration::InterMuonList& interout,
                             std::unique_ptr<MuonBxCollection>& out,
                             int bx) const;

  // ----------member data ---------------------------
  bool m_autoBxRange;
  int m_bxMin;
  int m_bxMax;
  bool m_autoCancelMode;
  std::bitset<72> m_inputsToDisable;
  std::bitset<28> m_caloInputsToDisable;
  std::bitset<12> m_bmtfInputsToDisable;
  std::bitset<12> m_omtfInputsToDisable;
  std::bitset<12> m_emtfInputsToDisable;
  std::bitset<72> m_maskedInputs;
  std::bitset<28> m_maskedCaloInputs;
  std::bitset<12> m_maskedBmtfInputs;
  std::bitset<12> m_maskedOmtfInputs;
  std::bitset<12> m_maskedEmtfInputs;
  std::unique_ptr<L1TMuonGlobalParamsHelper> microGMTParamsHelper;
  edm::InputTag m_barrelTfInputTag;
  edm::InputTag m_overlapTfInputTag;
  edm::InputTag m_endcapTfInputTag;
  edm::InputTag m_trigTowerTag;
  std::shared_ptr<MicroGMTRankPtQualLUT> m_rankPtQualityLUT;
  MicroGMTIsolationUnit m_isolationUnit;
  MicroGMTCancelOutUnit m_cancelOutUnit;
  std::ofstream m_debugOut;
  l1t::cancelmode m_bmtfCancelMode;
  l1t::cancelmode m_emtfCancelMode;

  edm::EDGetTokenT<MicroGMTConfiguration::InputCollection> m_barrelTfInputToken;
  edm::EDGetTokenT<MicroGMTConfiguration::InputCollection> m_overlapTfInputToken;
  edm::EDGetTokenT<MicroGMTConfiguration::InputCollection> m_endcapTfInputToken;
  edm::EDGetTokenT<MicroGMTConfiguration::CaloInputCollection> m_caloTowerInputToken;
  edm::ESGetToken<L1TMuonGlobalParams, L1TMuonGlobalParamsRcd> m_microGMTParamsToken;
  edm::ESGetToken<L1TMuonGlobalParams, L1TMuonGlobalParamsO2ORcd> m_o2oProtoToken;
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
L1TMuonProducer::L1TMuonProducer(const edm::ParameterSet& iConfig)
    : m_debugOut("test/debug/iso_debug.dat"),
      m_bmtfCancelMode(cancelmode::tracks),
      m_emtfCancelMode(cancelmode::coordinate) {
  // edm::InputTag barrelTfInputTag = iConfig.getParameter<edm::InputTag>("barrelTFInput");
  // edm::InputTag overlapTfInputTag = iConfig.getParameter<edm::InputTag>("overlapTFInput");
  // edm::InputTag forwardTfInputTag = iConfig.getParameter<edm::InputTag>("forwardTFInput");

  m_barrelTfInputTag = iConfig.getParameter<edm::InputTag>("barrelTFInput");
  m_overlapTfInputTag = iConfig.getParameter<edm::InputTag>("overlapTFInput");
  m_endcapTfInputTag = iConfig.getParameter<edm::InputTag>("forwardTFInput");
  m_trigTowerTag = iConfig.getParameter<edm::InputTag>("triggerTowerInput");

  m_autoBxRange = iConfig.getParameter<bool>("autoBxRange");
  m_bxMin = iConfig.getParameter<int>("bxMin");
  m_bxMax = iConfig.getParameter<int>("bxMax");

  m_autoCancelMode = iConfig.getParameter<bool>("autoCancelMode");
  if (!m_autoCancelMode) {
    if (iConfig.getParameter<std::string>("bmtfCancelMode").find("kftracks") == 0) {
      m_bmtfCancelMode = cancelmode::kftracks;
    }
    if (iConfig.getParameter<std::string>("emtfCancelMode").find("tracks") == 0) {
      m_emtfCancelMode = cancelmode::tracks;
    }
  }

  m_barrelTfInputToken = consumes<MicroGMTConfiguration::InputCollection>(m_barrelTfInputTag);
  m_overlapTfInputToken = consumes<MicroGMTConfiguration::InputCollection>(m_overlapTfInputTag);
  m_endcapTfInputToken = consumes<MicroGMTConfiguration::InputCollection>(m_endcapTfInputTag);
  m_caloTowerInputToken = consumes<MicroGMTConfiguration::CaloInputCollection>(m_trigTowerTag);
  m_microGMTParamsToken = esConsumes<L1TMuonGlobalParams, L1TMuonGlobalParamsRcd, edm::Transition::BeginRun>();
  m_o2oProtoToken = esConsumes<L1TMuonGlobalParams, L1TMuonGlobalParamsO2ORcd, edm::Transition::BeginRun>();

  //register your products
  produces<MuonBxCollection>();
  produces<MuonBxCollection>("imdMuonsBMTF");
  produces<MuonBxCollection>("imdMuonsEMTFPos");
  produces<MuonBxCollection>("imdMuonsEMTFNeg");
  produces<MuonBxCollection>("imdMuonsOMTFPos");
  produces<MuonBxCollection>("imdMuonsOMTFNeg");
}

L1TMuonProducer::~L1TMuonProducer() { m_debugOut.close(); }

//
// member functions
//

// ------------ method called to produce the data  ------------
void L1TMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::unique_ptr<MuonBxCollection> outMuons(new MuonBxCollection());
  std::unique_ptr<MuonBxCollection> imdMuonsBMTF(new MuonBxCollection());
  std::unique_ptr<MuonBxCollection> imdMuonsEMTFPos(new MuonBxCollection());
  std::unique_ptr<MuonBxCollection> imdMuonsEMTFNeg(new MuonBxCollection());
  std::unique_ptr<MuonBxCollection> imdMuonsOMTFPos(new MuonBxCollection());
  std::unique_ptr<MuonBxCollection> imdMuonsOMTFNeg(new MuonBxCollection());

  Handle<MicroGMTConfiguration::InputCollection> bmtfMuons;
  Handle<MicroGMTConfiguration::InputCollection> emtfMuons;
  Handle<MicroGMTConfiguration::InputCollection> omtfMuons;
  Handle<MicroGMTConfiguration::CaloInputCollection> trigTowers;

  iEvent.getByToken(m_barrelTfInputToken, bmtfMuons);
  iEvent.getByToken(m_endcapTfInputToken, emtfMuons);
  iEvent.getByToken(m_overlapTfInputToken, omtfMuons);
  iEvent.getByToken(m_caloTowerInputToken, trigTowers);

  // find out the BX range from the inputs
  // the smallest BX window defines the output BX window
  if (m_autoBxRange) {
    int bxMin = -1000;
    int bxMax = 1000;
    if (!(m_caloInputsToDisable.all() || m_maskedCaloInputs.all())) {
      bxMin = std::max(bxMin, trigTowers->getFirstBX());
      bxMax = std::min(bxMax, trigTowers->getLastBX());
    }
    if (!(m_bmtfInputsToDisable.all() || m_maskedBmtfInputs.all())) {
      bxMin = std::max(bxMin, bmtfMuons->getFirstBX());
      bxMax = std::min(bxMax, bmtfMuons->getLastBX());
    }
    if (!(m_omtfInputsToDisable.all() || m_maskedOmtfInputs.all())) {
      bxMin = std::max(bxMin, omtfMuons->getFirstBX());
      bxMax = std::min(bxMax, omtfMuons->getLastBX());
    }
    if (!(m_emtfInputsToDisable.all() || m_maskedEmtfInputs.all())) {
      bxMin = std::max(bxMin, emtfMuons->getFirstBX());
      bxMax = std::min(bxMax, emtfMuons->getLastBX());
    }
    if (bxMin > 0) {
      bxMin = 0;
    }
    if (bxMax < 0) {
      bxMax = 0;
    }
    if (bxMin > -1000) {
      m_bxMin = bxMin;
    } else {
      m_bxMin = 0;
    }
    if (bxMax < 1000) {
      m_bxMax = bxMax;
    } else {
      m_bxMax = 0;
    }
  }

  // set BX range for outputs
  outMuons->setBXRange(m_bxMin, m_bxMax);
  imdMuonsBMTF->setBXRange(m_bxMin, m_bxMax);
  imdMuonsEMTFPos->setBXRange(m_bxMin, m_bxMax);
  imdMuonsEMTFNeg->setBXRange(m_bxMin, m_bxMax);
  imdMuonsOMTFPos->setBXRange(m_bxMin, m_bxMax);
  imdMuonsOMTFNeg->setBXRange(m_bxMin, m_bxMax);

  for (int bx = m_bxMin; bx <= m_bxMax; ++bx) {
    m_isolationUnit.setTowerSums(*trigTowers, bx);
    MicroGMTConfiguration::InterMuonList internMuonsBmtf;
    MicroGMTConfiguration::InterMuonList internMuonsEmtfPos;
    MicroGMTConfiguration::InterMuonList internMuonsEmtfNeg;
    MicroGMTConfiguration::InterMuonList internMuonsOmtfPos;
    MicroGMTConfiguration::InterMuonList internMuonsOmtfNeg;

    // These wedges contain shared pointers to the ones in the InterMuonList
    GMTInternalWedges omtfNegWedges;
    GMTInternalWedges bmtfWedges;
    GMTInternalWedges emtfPosWedges;
    GMTInternalWedges emtfNegWedges;
    GMTInternalWedges omtfPosWedges;

    // this converts the InputMuon type to the InternalMuon type and splits them into
    // positive / negative eta collections necessary as LUTs may differ for pos / neg.
    convertMuons(bmtfMuons, internMuonsBmtf, bmtfWedges, bx);
    splitAndConvertMuons(emtfMuons, internMuonsEmtfPos, internMuonsEmtfNeg, emtfPosWedges, emtfNegWedges, bx);
    splitAndConvertMuons(omtfMuons, internMuonsOmtfPos, internMuonsOmtfNeg, omtfPosWedges, omtfNegWedges, bx);

    // cancel out within the track finders:
    m_cancelOutUnit.setCancelOutBits(bmtfWedges, tftype::bmtf, m_bmtfCancelMode);
    m_cancelOutUnit.setCancelOutBits(omtfPosWedges, tftype::omtf_pos, cancelmode::coordinate);
    m_cancelOutUnit.setCancelOutBits(omtfNegWedges, tftype::omtf_neg, cancelmode::coordinate);
    m_cancelOutUnit.setCancelOutBits(emtfPosWedges, tftype::emtf_pos, m_emtfCancelMode);
    m_cancelOutUnit.setCancelOutBits(emtfNegWedges, tftype::emtf_neg, m_emtfCancelMode);

    // cancel out between track finder acceptance overlaps:
    m_cancelOutUnit.setCancelOutBitsOverlapBarrel(omtfPosWedges, bmtfWedges, cancelmode::coordinate);
    m_cancelOutUnit.setCancelOutBitsOverlapBarrel(omtfNegWedges, bmtfWedges, cancelmode::coordinate);
    m_cancelOutUnit.setCancelOutBitsOverlapEndcap(omtfPosWedges, emtfPosWedges, cancelmode::coordinate);
    m_cancelOutUnit.setCancelOutBitsOverlapEndcap(omtfNegWedges, emtfNegWedges, cancelmode::coordinate);

    m_isolationUnit.extrapolateMuons(internMuonsBmtf);
    m_isolationUnit.extrapolateMuons(internMuonsEmtfNeg);
    m_isolationUnit.extrapolateMuons(internMuonsEmtfPos);
    m_isolationUnit.extrapolateMuons(internMuonsOmtfNeg);
    m_isolationUnit.extrapolateMuons(internMuonsOmtfPos);

    // the rank calculated here is used in the sort below
    calculateRank(internMuonsBmtf);
    calculateRank(internMuonsEmtfNeg);
    calculateRank(internMuonsEmtfPos);
    calculateRank(internMuonsOmtfNeg);
    calculateRank(internMuonsOmtfPos);

    // The sort function both sorts and removes all but best "nSurvivors"
    sortMuons(internMuonsBmtf, 8);
    sortMuons(internMuonsOmtfPos, 4);
    sortMuons(internMuonsOmtfNeg, 4);
    sortMuons(internMuonsEmtfPos, 4);
    sortMuons(internMuonsEmtfNeg, 4);

    // This combines the 5 streams into one InternalMuon collection for
    // the final global sort.
    MicroGMTConfiguration::InterMuonList internalMuons;
    addMuonsToCollections(internMuonsEmtfPos, internalMuons, imdMuonsEMTFPos, bx);
    addMuonsToCollections(internMuonsOmtfPos, internalMuons, imdMuonsOMTFPos, bx);
    addMuonsToCollections(internMuonsBmtf, internalMuons, imdMuonsBMTF, bx);
    addMuonsToCollections(internMuonsOmtfNeg, internalMuons, imdMuonsOMTFNeg, bx);
    addMuonsToCollections(internMuonsEmtfNeg, internalMuons, imdMuonsEMTFNeg, bx);

    // sort internal muons and delete all but best 8
    sortMuons(internalMuons, 8);

    m_isolationUnit.isolatePreSummed(internalMuons);
    // copy muons to output collection...
    for (const auto& mu : internalMuons) {
      if (mu->hwPt() > 0) {
        math::PtEtaPhiMLorentzVector vec{
            (mu->hwPt() - 1) * 0.5, mu->hwEta() * 0.010875, mu->hwGlobalPhi() * 0.010908, 0.0};
        int iso = mu->hwAbsIso() + (mu->hwRelIso() << 1);
        int outMuQual = MicroGMTConfiguration::setOutputMuonQuality(mu->hwQual(), mu->trackFinderType(), mu->hwHF());
        Muon outMu{vec,
                   mu->hwPt(),
                   mu->hwEta(),
                   mu->hwGlobalPhi(),
                   outMuQual,
                   mu->hwSign(),
                   mu->hwSignValid(),
                   iso,
                   mu->tfMuonIndex(),
                   0,
                   true,
                   mu->hwIsoSum(),
                   mu->hwDPhi(),
                   mu->hwDEta(),
                   mu->hwRank()};

        // Set coordinates at the vertex
        outMu.setHwEtaAtVtx(MicroGMTConfiguration::calcMuonHwEtaExtra(outMu));
        outMu.setHwPhiAtVtx(MicroGMTConfiguration::calcMuonHwPhiExtra(outMu));
        outMu.setEtaAtVtx(MicroGMTConfiguration::calcMuonEtaExtra(outMu));
        outMu.setPhiAtVtx(MicroGMTConfiguration::calcMuonPhiExtra(outMu));

        // Set displacement information
        int hwPtUnconstrained{mu->hwPtUnconstrained()};
        outMu.setPtUnconstrained(
            hwPtUnconstrained == 0
                ? 0
                : (hwPtUnconstrained - 1));  // Don't want negative pT, unconstr. pT has LSB of 1 GeV.
        outMu.setHwPtUnconstrained(hwPtUnconstrained);
        outMu.setHwDXY(mu->hwDXY());

        if (mu->hwSignValid()) {
          outMu.setCharge(1 - 2 * mu->hwSign());
        } else {
          outMu.setCharge(0);
        }
        m_debugOut << mu->hwCaloPhi() << " " << mu->hwCaloEta() << std::endl;
        outMuons->push_back(bx, outMu);
      }
    }
  }

  iEvent.put(std::move(outMuons));
  iEvent.put(std::move(imdMuonsBMTF), "imdMuonsBMTF");
  iEvent.put(std::move(imdMuonsEMTFPos), "imdMuonsEMTFPos");
  iEvent.put(std::move(imdMuonsEMTFNeg), "imdMuonsEMTFNeg");
  iEvent.put(std::move(imdMuonsOMTFPos), "imdMuonsOMTFPos");
  iEvent.put(std::move(imdMuonsOMTFNeg), "imdMuonsOMTFNeg");
}

bool L1TMuonProducer::compareMuons(const std::shared_ptr<MicroGMTConfiguration::InterMuon>& mu1,
                                   const std::shared_ptr<MicroGMTConfiguration::InterMuon>& mu2) {
  return (mu1->hwWins() >= mu2->hwWins());
}

void L1TMuonProducer::sortMuons(MicroGMTConfiguration::InterMuonList& muons, unsigned nSurvivors) const {
  MicroGMTConfiguration::InterMuonList::iterator mu1;
  // reset from previous sort stage
  for (mu1 = muons.begin(); mu1 != muons.end(); ++mu1) {
    (*mu1)->setHwWins(0);
  }

  int nCancelled = 0;
  for (mu1 = muons.begin(); mu1 != muons.end(); ++mu1) {
    int mu1CancelBit = (*mu1)->hwCancelBit();
    nCancelled += mu1CancelBit;
    auto mu2 = mu1;
    mu2++;
    for (; mu2 != muons.end(); ++mu2) {
      if (mu1CancelBit != 1 && (*mu2)->hwCancelBit() != 1) {
        if ((*mu1)->hwRank() >= (*mu2)->hwRank()) {
          (*mu1)->increaseWins();
        } else {
          (*mu2)->increaseWins();
        }
      } else if (mu1CancelBit != 1) {
        (*mu1)->increaseWins();
      } else if ((*mu2)->hwCancelBit() != 1) {
        (*mu2)->increaseWins();
      }
    }
  }

  size_t nMuonsBefore = muons.size();
  int minWins = nMuonsBefore - nSurvivors;

  // remove all muons that were cancelled or that do not have sufficient rank
  // (reduces the container size to nSurvivors)
  muons.remove_if([&minWins](auto muon) { return ((muon->hwWins() < minWins) || (muon->hwCancelBit() == 1)); });
  muons.sort(L1TMuonProducer::compareMuons);
}

void L1TMuonProducer::calculateRank(MicroGMTConfiguration::InterMuonList& muons) const {
  for (auto& mu1 : muons) {
    int rank = m_rankPtQualityLUT->lookup(mu1->hwPt(), mu1->hwQual());
    mu1->setHwRank(rank);
  }
}

void L1TMuonProducer::addMuonsToCollections(MicroGMTConfiguration::InterMuonList& coll,
                                            MicroGMTConfiguration::InterMuonList& interout,
                                            std::unique_ptr<MuonBxCollection>& out,
                                            int bx) const {
  for (auto& mu : coll) {
    interout.push_back(mu);
    math::PtEtaPhiMLorentzVector vec{(mu->hwPt() - 1) * 0.5, mu->hwEta() * 0.010875, mu->hwGlobalPhi() * 0.010908, 0.0};
    int outMuQual = MicroGMTConfiguration::setOutputMuonQuality(mu->hwQual(), mu->trackFinderType(), mu->hwHF());
    // set tfMuonIndex and iso to 0 like in the FW
    Muon outMu{vec,
               mu->hwPt(),
               mu->hwEta(),
               mu->hwGlobalPhi(),
               outMuQual,
               mu->hwSign(),
               mu->hwSignValid(),
               0,
               0,
               0,
               true,
               0,
               mu->hwDPhi(),
               mu->hwDEta(),
               mu->hwRank()};

    int hwPtUnconstrained{mu->hwPtUnconstrained()};
    outMu.setPtUnconstrained(hwPtUnconstrained == 0
                                 ? 0
                                 : (hwPtUnconstrained - 1));  // Don't want negative pT, unconstr. pT has LSB of 1 GeV.
    outMu.setHwPtUnconstrained(hwPtUnconstrained);
    outMu.setHwDXY(mu->hwDXY());

    if (mu->hwSignValid()) {
      outMu.setCharge(1 - 2 * mu->hwSign());
    } else {
      outMu.setCharge(0);
    }

    out->push_back(bx, outMu);
  }
}

void L1TMuonProducer::splitAndConvertMuons(const edm::Handle<MicroGMTConfiguration::InputCollection>& in,
                                           MicroGMTConfiguration::InterMuonList& out_pos,
                                           MicroGMTConfiguration::InterMuonList& out_neg,
                                           GMTInternalWedges& wedges_pos,
                                           GMTInternalWedges& wedges_neg,
                                           int bx) const {
  // initialize the wedge collections:
  for (int i = 0; i < 6; ++i) {
    wedges_pos[i] = std::vector<std::shared_ptr<GMTInternalMuon>>();
    wedges_pos[i].reserve(3);
    wedges_neg[i] = std::vector<std::shared_ptr<GMTInternalMuon>>();
    wedges_neg[i].reserve(3);
  }
  if (bx < in->getFirstBX() || bx > in->getLastBX())
    return;
  int muIdxAuto = 0;
  int currentLink = 0;
  for (size_t i = 0; i < in->size(bx); ++i, ++muIdxAuto) {
    if (in->at(bx, i).hwPt() > 0) {
      int link = in->at(bx, i).link();
      if (m_inputsToDisable.test(link) || m_maskedInputs.test(link)) {
        continue;  // only process if input link is enabled and not masked
      }
      if (currentLink != link) {
        muIdxAuto = 0;
        currentLink = link;
      }
      int gPhi = MicroGMTConfiguration::calcGlobalPhi(
          in->at(bx, i).hwPhi(), in->at(bx, i).trackFinderType(), in->at(bx, i).processor());
      int tfMuonIdx{computeMuonIdx(in->at(bx, i), currentLink, muIdxAuto)};
      std::shared_ptr<GMTInternalMuon> out = std::make_shared<GMTInternalMuon>(in->at(bx, i), gPhi, tfMuonIdx);
      if (in->at(bx, i).hwEta() > 0) {
        out_pos.push_back(out);
        wedges_pos[in->at(bx, i).processor()].push_back(out);
      } else {
        out_neg.emplace_back(out);
        wedges_neg[in->at(bx, i).processor()].push_back(out);
      }
    }
  }
  for (int i = 0; i < 6; ++i) {
    if (wedges_pos[i].size() > 3)
      edm::LogWarning("Input Mismatch") << " too many inputs per processor for emtf+ / omtf+. Wedge " << i << ": Size "
                                        << wedges_pos[i].size() << std::endl;
    if (wedges_neg[i].size() > 3)
      edm::LogWarning("Input Mismatch") << " too many inputs per processor for emtf- / omtf-. Wedge " << i << ": Size "
                                        << wedges_neg[i].size() << std::endl;
  }
}

void L1TMuonProducer::convertMuons(const edm::Handle<MicroGMTConfiguration::InputCollection>& in,
                                   MicroGMTConfiguration::InterMuonList& out,
                                   GMTInternalWedges& wedges,
                                   int bx) const {
  // initialize the wedge collection:
  for (int i = 0; i < 12; ++i) {
    wedges[i] = std::vector<std::shared_ptr<GMTInternalMuon>>();
    wedges[i].reserve(3);
  }
  if (bx < in->getFirstBX() || bx > in->getLastBX()) {
    return;
  }
  int muIdxAuto = 0;
  int currentLink = 0;
  for (size_t i = 0; i < in->size(bx); ++i, ++muIdxAuto) {
    if (in->at(bx, i).hwPt() > 0) {
      int link = in->at(bx, i).link();
      if (m_inputsToDisable.test(link) || m_maskedInputs.test(link)) {
        continue;  // only process if input link is enabled and not masked
      }
      if (currentLink != link) {
        muIdxAuto = 0;
        currentLink = link;
      }
      int gPhi = MicroGMTConfiguration::calcGlobalPhi(
          in->at(bx, i).hwPhi(), in->at(bx, i).trackFinderType(), in->at(bx, i).processor());
      int tfMuonIdx{computeMuonIdx(in->at(bx, i), currentLink, muIdxAuto)};
      std::shared_ptr<GMTInternalMuon> outMu = std::make_shared<GMTInternalMuon>(in->at(bx, i), gPhi, tfMuonIdx);
      out.emplace_back(outMu);
      wedges[in->at(bx, i).processor()].push_back(outMu);
    }
  }
  for (int i = 0; i < 12; ++i) {
    if (wedges[i].size() > 3) {
      edm::LogWarning("Input Mismatch") << " too many inputs per processor for barrel. Wedge " << i << ": Size "
                                        << wedges[i].size() << std::endl;
    }
  }
}

int L1TMuonProducer::computeMuonIdx(const RegionalMuonCand& mu, int currentLink, int muIdxAuto) const {
  // If the muon index was set in the data format we should use that. Otherwise we use the value computed from the position in the vector.
  if (mu.muIdx() != -1) {
    return 3 * (currentLink - 36) + mu.muIdx();
  } else {
    return 3 * (currentLink - 36) + muIdxAuto;
  }
}

// ------------ method called when starting to processes a run  ------------
void L1TMuonProducer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  edm::ESHandle<L1TMuonGlobalParams> microGMTParamsHandle = iSetup.getHandle(m_microGMTParamsToken);

  std::unique_ptr<L1TMuonGlobalParams_PUBLIC> microGMTParams(
      new L1TMuonGlobalParams_PUBLIC(cast_to_L1TMuonGlobalParams_PUBLIC(*microGMTParamsHandle.product())));
  if (microGMTParams->pnodes_.empty()) {
    edm::ESHandle<L1TMuonGlobalParams> o2oProtoHandle = iSetup.getHandle(m_o2oProtoToken);
    microGMTParamsHelper = std::make_unique<L1TMuonGlobalParamsHelper>(*o2oProtoHandle.product());
  } else
    microGMTParamsHelper =
        std::make_unique<L1TMuonGlobalParamsHelper>(cast_to_L1TMuonGlobalParams(*microGMTParams.get()));

  //microGMTParamsHelper->print(std::cout);
  m_inputsToDisable = microGMTParamsHelper->inputsToDisable();
  edm::LogVerbatim("L1TMuonProducer")
      << "uGMT inputsToDisable: " << m_inputsToDisable
      << "\n                      EMTF-|OMTF-|   BMTF    |OMTF+|EMTF+|            CALO           |  res  0";
  m_caloInputsToDisable = microGMTParamsHelper->caloInputsToDisable();
  m_bmtfInputsToDisable = microGMTParamsHelper->bmtfInputsToDisable();
  m_omtfInputsToDisable = microGMTParamsHelper->omtfInputsToDisable();
  m_emtfInputsToDisable = microGMTParamsHelper->emtfInputsToDisable();
  m_maskedInputs = microGMTParamsHelper->maskedInputs();
  edm::LogVerbatim("L1TMuonProducer")
      << "uGMT maskedInputs:    " << m_maskedInputs
      << "\n                      EMTF-|OMTF-|   BMTF    |OMTF+|EMTF+|            CALO           |  res  0";
  m_maskedCaloInputs = microGMTParamsHelper->maskedCaloInputs();
  m_maskedBmtfInputs = microGMTParamsHelper->maskedBmtfInputs();
  m_maskedOmtfInputs = microGMTParamsHelper->maskedOmtfInputs();
  m_maskedEmtfInputs = microGMTParamsHelper->maskedEmtfInputs();
  m_rankPtQualityLUT =
      l1t::MicroGMTRankPtQualLUTFactory::create(microGMTParamsHelper->sortRankLUT(), microGMTParamsHelper->fwVersion());
  m_isolationUnit.initialise(microGMTParamsHelper.get());
  m_cancelOutUnit.initialise(microGMTParamsHelper.get());

  if (m_autoCancelMode) {
    if (microGMTParamsHelper->fwVersion() >= 0x6000000) {
      m_bmtfCancelMode = cancelmode::kftracks;
    }
    // TODO: No decision yet on when to use EMTF track addresses for cancel-out.
    // if (microGMTParamsHelper->fwVersion() > 0x5000000) {
    //   m_emtfCancelMode = cancelmode::tracks;
    // }
  }
}

// ------------ method called when ending the processing of a run  ------------
void L1TMuonProducer::endRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method called when starting to processes a luminosity block  ------------
void L1TMuonProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

// ------------ method called when ending the processing of a luminosity block  ------------
void L1TMuonProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonProducer);

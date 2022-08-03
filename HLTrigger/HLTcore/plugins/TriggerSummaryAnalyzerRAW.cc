/** \class TriggerSummaryAnalyzerRAW
 *
 *  This class is an EDAnalyzer analyzing the HLT summary object for RAW
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

//
// class declaration
//
class TriggerSummaryAnalyzerRAW : public edm::global::EDAnalyzer<> {
public:
  explicit TriggerSummaryAnalyzerRAW(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

private:
  /// InputTag of TriggerEventWithRefs to analyze
  const edm::InputTag inputTag_;
  const edm::EDGetTokenT<trigger::TriggerEventWithRefs> inputToken_;
};

//
// constructors and destructor
//
TriggerSummaryAnalyzerRAW::TriggerSummaryAnalyzerRAW(const edm::ParameterSet& ps)
    : inputTag_(ps.getParameter<edm::InputTag>("inputTag")),
      inputToken_(consumes<trigger::TriggerEventWithRefs>(inputTag_)) {}

//
// member functions
//

void TriggerSummaryAnalyzerRAW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag", edm::InputTag("hltTriggerSummaryRAW"));
  descriptions.add("triggerSummaryAnalyzerRAW", desc);
}

// ------------ method called to produce the data  ------------
void TriggerSummaryAnalyzerRAW::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;
  using namespace l1t;

  LogVerbatim("TriggerSummaryAnalyzerRAW") << endl;
  LogVerbatim("TriggerSummaryAnalyzerRAW")
      << "TriggerSummaryAnalyzerRAW: content of TriggerEventWithRefs: " << inputTag_.encode();

  Handle<TriggerEventWithRefs> handle;
  iEvent.getByToken(inputToken_, handle);
  if (handle.isValid()) {
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "Used Processname: " << handle->usedProcessName() << endl;
    const size_type nFO(handle->size());
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "Number of TriggerFilterObjects: " << nFO << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "The TriggerFilterObjects: #, tag" << endl;
    for (size_type iFO = 0; iFO != nFO; ++iFO) {
      LogVerbatim("TriggerSummaryAnalyzerRAW") << iFO << " " << handle->filterTag(iFO).encode() << endl;
      LogVerbatim("TriggerSummaryAnalyzerRAW") << "  # of objects:";

      const unsigned int nPhotons(handle->photonSlice(iFO).second - handle->photonSlice(iFO).first);
      if (nPhotons > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " Photons: " << nPhotons;

      const unsigned int nElectrons(handle->electronSlice(iFO).second - handle->electronSlice(iFO).first);
      if (nElectrons > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " Electrons: " << nElectrons;

      const unsigned int nMuons(handle->muonSlice(iFO).second - handle->muonSlice(iFO).first);
      if (nMuons > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " Muons: " << nMuons;

      const unsigned int nJets(handle->jetSlice(iFO).second - handle->jetSlice(iFO).first);
      if (nJets > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " Jets: " << nJets;

      const unsigned int nComposites(handle->compositeSlice(iFO).second - handle->compositeSlice(iFO).first);
      if (nComposites > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " Composites: " << nComposites;

      const unsigned int nBaseMETs(handle->basemetSlice(iFO).second - handle->basemetSlice(iFO).first);
      if (nBaseMETs > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " BaseMETs: " << nBaseMETs;

      const unsigned int nCaloMETs(handle->calometSlice(iFO).second - handle->calometSlice(iFO).first);
      if (nCaloMETs > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " CaloMETs: " << nCaloMETs;

      const unsigned int nPixTracks(handle->pixtrackSlice(iFO).second - handle->pixtrackSlice(iFO).first);
      if (nPixTracks > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " PixTracks: " << nPixTracks;

      const unsigned int nL1EM(handle->l1emSlice(iFO).second - handle->l1emSlice(iFO).first);
      if (nL1EM > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1EM: " << nL1EM;

      const unsigned int nL1Muon(handle->l1muonSlice(iFO).second - handle->l1muonSlice(iFO).first);
      if (nL1Muon > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1Muon: " << nL1Muon;

      const unsigned int nL1Jet(handle->l1jetSlice(iFO).second - handle->l1jetSlice(iFO).first);
      if (nL1Jet > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1Jet: " << nL1Jet;

      const unsigned int nL1EtMiss(handle->l1etmissSlice(iFO).second - handle->l1etmissSlice(iFO).first);
      if (nL1EtMiss > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1EtMiss: " << nL1EtMiss;

      const unsigned int nL1HfRings(handle->l1hfringsSlice(iFO).second - handle->l1hfringsSlice(iFO).first);
      if (nL1HfRings > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1HfRings: " << nL1HfRings;

      const unsigned int nPFJets(handle->pfjetSlice(iFO).second - handle->pfjetSlice(iFO).first);
      if (nPFJets > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " PFJets: " << nPFJets;

      const unsigned int nPFTaus(handle->pftauSlice(iFO).second - handle->pftauSlice(iFO).first);
      if (nPFTaus > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " PFTaus: " << nPFTaus;

      const unsigned int nPFMETs(handle->pfmetSlice(iFO).second - handle->pfmetSlice(iFO).first);
      if (nPFMETs > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " PFMETs: " << nPFMETs;

      const unsigned int nL1TMuon(handle->l1tmuonSlice(iFO).second - handle->l1tmuonSlice(iFO).first);
      if (nL1TMuon > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TMuon: " << nL1TMuon;

      const unsigned int nL1TMuonShower(handle->l1tmuonShowerSlice(iFO).second - handle->l1tmuonShowerSlice(iFO).first);
      if (nL1TMuonShower > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TMuonShower: " << nL1TMuonShower;

      const unsigned int nL1TEGamma(handle->l1tegammaSlice(iFO).second - handle->l1tegammaSlice(iFO).first);
      if (nL1TEGamma > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TEGamma: " << nL1TEGamma;

      const unsigned int nL1TJet(handle->l1tjetSlice(iFO).second - handle->l1tjetSlice(iFO).first);
      if (nL1TJet > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TJet: " << nL1TJet;

      const unsigned int nL1TTau(handle->l1ttauSlice(iFO).second - handle->l1ttauSlice(iFO).first);
      if (nL1TTau > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TTau: " << nL1TTau;

      const unsigned int nL1TEtSum(handle->l1tetsumSlice(iFO).second - handle->l1tetsumSlice(iFO).first);
      if (nL1TEtSum > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TEtSum: " << nL1TEtSum;

      /* Phase-2 */
      const unsigned int nL1TTkMuon(handle->l1ttkmuonSlice(iFO).second - handle->l1ttkmuonSlice(iFO).first);
      if (nL1TTkMuon > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TTrackerMuon: " << nL1TTkMuon;

      const unsigned int nL1TTkEle(handle->l1ttkeleSlice(iFO).second - handle->l1ttkeleSlice(iFO).first);
      if (nL1TTkEle > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TTkEle: " << nL1TTkEle;

      const unsigned int nL1TTkEm(handle->l1ttkemSlice(iFO).second - handle->l1ttkemSlice(iFO).first);
      if (nL1TTkEm > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TTkEm: " << nL1TTkEm;

      const unsigned int nL1TPFJet(handle->l1tpfjetSlice(iFO).second - handle->l1tpfjetSlice(iFO).first);
      if (nL1TPFJet > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TPFJet: " << nL1TPFJet;

      const unsigned int nL1TPFTau(handle->l1tpftauSlice(iFO).second - handle->l1tpftauSlice(iFO).first);
      if (nL1TPFTau > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TPFTau: " << nL1TPFTau;

      const unsigned int nL1THPSPFTau(handle->l1thpspftauSlice(iFO).second - handle->l1thpspftauSlice(iFO).first);
      if (nL1THPSPFTau > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1THPSPFTau: " << nL1THPSPFTau;

      const unsigned int nL1TPFTrack(handle->l1tpftrackSlice(iFO).second - handle->l1tpftrackSlice(iFO).first);
      if (nL1TPFTrack > 0)
        LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1TPFTrack: " << nL1TPFTrack;

      LogVerbatim("TriggerSummaryAnalyzerRAW") << endl;
    }
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "Elements in linearised collections of Refs: " << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Photons:       " << handle->photonSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Electrons:     " << handle->electronSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Muons:         " << handle->muonSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Jets:          " << handle->jetSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Composites:    " << handle->compositeSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  BaseMETs:      " << handle->basemetSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  CaloMETs:      " << handle->calometSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Pixtracks:     " << handle->pixtrackSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1EM:          " << handle->l1emSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1Muon:        " << handle->l1muonSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1Jet:         " << handle->l1jetSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1EtMiss:      " << handle->l1etmissSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1HfRings:     " << handle->l1hfringsSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  PFJets:        " << handle->pfjetSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  PFTaus:        " << handle->pftauSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  PFMETs:        " << handle->pfmetSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TMuon:       " << handle->l1tmuonSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TMuonShower: " << handle->l1tmuonShowerSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TEGamma:     " << handle->l1tegammaSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TJet:        " << handle->l1tjetSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TTau:        " << handle->l1ttauSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TEtSum:      " << handle->l1tetsumSize() << endl;
    /* Phase-2 */
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TTrackerMuon:" << handle->l1ttkmuonSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TTkEle:     " << handle->l1ttkeleSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TTkEm:      " << handle->l1ttkemSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TPFJet:     " << handle->l1tpfjetSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TPFTau:     " << handle->l1tpftauSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1THPSPFTau:  " << handle->l1thpspftauSize() << endl;
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1TPFTrack:   " << handle->l1tpftrackSize() << endl;

  } else {
    LogVerbatim("TriggerSummaryAnalyzerRAW") << "Handle invalid! Check InputTag provided." << endl;
  }
  LogVerbatim("TriggerSummaryAnalyzerRAW") << endl;

  return;
}

DEFINE_FWK_MODULE(TriggerSummaryAnalyzerRAW);

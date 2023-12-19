#include "L1TriggerScouting/Utilities/interface/printScObjects.h"

namespace l1ScoutingRun3 {

  void printMuon(const Muon& muon, std::ostream& outs) {
    outs << "  Pt  [GeV/Hw]: " << ugmt::fPt(muon.hwPt()) << "/" << muon.hwPt() << "\n"
         << "  Eta [rad/Hw]: " << ugmt::fEta(muon.hwEta()) << "/" << muon.hwEta() << "\n"
         << "  Phi [rad/Hw]: " << ugmt::fPhi(muon.hwPhi()) << "/" << muon.hwPhi() << "\n"
         << "  Charge/valid: " << muon.hwCharge() << "/" << muon.hwChargeValid() << "\n"
         << "  PhiVtx  [rad/Hw]: " << ugmt::fPhiAtVtx(muon.hwPhiAtVtx()) << "/" << muon.hwPhiAtVtx() << "\n"
         << "  EtaVtx  [rad/Hw]: " << ugmt::fEtaAtVtx(muon.hwEtaAtVtx()) << "/" << muon.hwEtaAtVtx() << "\n"
         << "  Pt uncon[GeV/Hw]: " << ugmt::fPtUnconstrained(muon.hwPtUnconstrained()) << "/"
         << muon.hwPtUnconstrained() << "\n"
         << "  Dxy: " << muon.hwDXY() << "\n"
         << "  Qual: " << muon.hwQual() << "\n"
         << "  TF index: " << muon.tfMuonIndex() << "\n";
  }

  template <typename T>
  void printCaloObject(const T& obj, std::ostream& outs) {
    outs << "  Et  [GeV/Hw]: " << demux::fEt(obj.hwEt()) << "/" << obj.hwEt() << "\n"
         << "  Eta [rad/Hw]: " << demux::fEta(obj.hwEta()) << "/" << obj.hwEta() << "\n"
         << "  Phi [rad/Hw]: " << demux::fPhi(obj.hwPhi()) << "/" << obj.hwPhi() << "\n"
         << "  Iso [Hw]: " << obj.hwIso() << "\n";
  }

  void printJet(const Jet& jet, std::ostream& outs) { printCaloObject<Jet>(jet, outs); }
  void printEGamma(const EGamma& eGamma, std::ostream& outs) { printCaloObject<EGamma>(eGamma, outs); }
  void printTau(const Tau& tau, std::ostream& outs) { printCaloObject<Tau>(tau, outs); }

  void printBxSums(const BxSums& sums, std::ostream& outs) {
    outs << "Total ET\n"
         << "  Et [GeV/Hw]: " << demux::fEt(sums.hwTotalEt()) << "/" << sums.hwTotalEt() << "\n"
         << "Total ETEm\n"
         << "  Et [GeV/Hw]: " << demux::fEt(sums.hwTotalEtEm()) << "/" << sums.hwTotalEtEm() << "\n"
         << "Total HT\n"
         << "  Et [GeV/Hw]: " << demux::fEt(sums.hwTotalHt()) << "/" << sums.hwTotalHt() << "\n"
         << "Missing ET\n"
         << "  Et [GeV/Hw] : " << demux::fEt(sums.hwMissEt()) << "/" << sums.hwMissEt() << "\n"
         << "  Phi [Rad/Hw]: " << demux::fPhi(sums.hwMissEtPhi()) << "/" << sums.hwMissEtPhi() << "\n"
         << "Missing HT\n"
         << "  Et [GeV/Hw] : " << demux::fEt(sums.hwMissHt()) << "/" << sums.hwMissHt() << "\n"
         << "  Phi [Rad/Hw]: " << demux::fPhi(sums.hwMissHtPhi()) << "/" << sums.hwMissHtPhi() << "\n"
         << "Missing ETHF\n"
         << "  Et [GeV/Hw] : " << demux::fEt(sums.hwMissEtHF()) << "/" << sums.hwMissEtHF() << "\n"
         << "  Phi [Rad/Hw]: " << demux::fPhi(sums.hwMissEtHFPhi()) << "/" << sums.hwMissEtHFPhi() << "\n"
         << "Missing HTHF\n"
         << "  Et [GeV/Hw] : " << demux::fEt(sums.hwMissHtHF()) << "/" << sums.hwMissHtHF() << "\n"
         << "  Phi [Rad/Hw]: " << demux::fPhi(sums.hwMissHtHFPhi()) << "/" << sums.hwMissHtHFPhi() << "\n"
         << "AsymEt\n"
         << "  Et [GeV/Hw] : " << demux::fEt(sums.hwAsymEt()) << "/" << sums.hwAsymEt() << "\n"
         << "AsymHt\n"
         << "  Et [GeV/Hw] : " << demux::fEt(sums.hwAsymHt()) << "/" << sums.hwAsymHt() << "\n"
         << "AsymEtHF\n"
         << "  Et [GeV/Hw] : " << demux::fEt(sums.hwAsymEtHF()) << "/" << sums.hwAsymEtHF() << "\n"
         << "AsymHtHF\n"
         << "  Et [GeV/Hw] : " << demux::fEt(sums.hwAsymHtHF()) << "/" << sums.hwAsymHtHF() << "\n"
         << "MinBiasHFP0\n"
         << "  Hw: " << sums.minBiasHFP0() << "\n"
         << "MinBiasHFM0\n"
         << "  Hw: " << sums.minBiasHFM0() << "\n"
         << "MinBiasHFP1\n"
         << "  Hw: " << sums.minBiasHFP1() << "\n"
         << "MinBiasHFM1\n"
         << "  Hw: " << sums.minBiasHFM1() << "\n"
         << "Centrality\n"
         << "  Hw: " << sums.centrality() << "\n"
         << "Tower Count\n"
         << "  Hw: " << sums.towerCount() << "\n";
  }
}  // namespace l1ScoutingRun3
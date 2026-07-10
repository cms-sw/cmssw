#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "CommonTools/Utils/interface/FormulaEvaluator.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCaloTower.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCaloJet.h"
#include "DataFormats/Math/interface/libminifloat.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"

class L1ScoutingCaloJetProducer : public edm::global::EDProducer<> {
public:
  explicit L1ScoutingCaloJetProducer(const edm::ParameterSet&);
  ~L1ScoutingCaloJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  class JetCorrector {
  public:
    explicit JetCorrector() = default;
    explicit JetCorrector(std::string const& filePath) {
      std::ifstream infile(filePath);
      if (!infile) {
        throw cms::Exception("InputError") << "failed to open JetCorrector input file: " << filePath;
      }

      std::string line{};
      while (std::getline(infile, line)) {
        std::istringstream iss(line);

        float ptMin{0.f};
        float ptMax{0.f};
        float etaMin{0.f};
        float etaMax{0.f};
        int puProxyMin{0};
        int puProxyMax{0};
        int formNParams{0};
        std::string formEvalStr{""};

        if (!(iss >> ptMin >> ptMax >> etaMin >> etaMax >> puProxyMin >> puProxyMax >> formEvalStr >> formNParams)) {
          throw cms::Exception("InvalidInput")
              << "failed to read line from input file (invalid format): \"" << line << "\"";
        }

        if (ptMin <= 0) {
          throw cms::Exception("InvalidInput")
              << "invalid value for parameter \"ptMin\" (must be greater than zero): " << ptMin;
        }

        if (ptMax <= 0) {
          throw cms::Exception("InvalidInput")
              << "invalid value for parameter \"ptMax\" (must be greater than zero): " << ptMax;
        }

        if (ptMin >= ptMax) {
          throw cms::Exception("InvalidInput") << "inconsistent values for parameters \"ptMin\" and \"ptMax\" (the "
                                                  "latter must be greater than the former): ptMin="
                                               << ptMin << " ptMax=" << ptMax;
        }

        if (etaMin >= etaMax) {
          throw cms::Exception("InvalidInput") << "inconsistent values for parameters \"etaMin\" and \"etaMax\" (the "
                                                  "latter must be greater than the former): etaMin="
                                               << etaMin << " etaMax=" << etaMax;
        }

        if (puProxyMin > 0 and puProxyMax > 0 and puProxyMin >= puProxyMax) {
          throw cms::Exception("InvalidInput")
              << "inconsistent values for parameters \"puProxyMin\" and \"puProxyMax\" (if both are greater than zero, "
                 "the latter must be greater than the former): puProxyMin="
              << puProxyMin << " puProxyMax=" << puProxyMax;
        }

        if (formNParams <= 0) {
          throw cms::Exception("InvalidInput")
              << "invalid value for parameter \"formNParams\" (must be greater than zero): " << formNParams;
        }

        reco::FormulaEvaluator formEval{formEvalStr};

        std::vector<double> formParams(formNParams);
        for (auto idx = 0; idx < formNParams; ++idx) {
          if (!(iss >> formParams[idx])) {
            throw cms::Exception("InvalidInput")
                << "failed to read line from input file (invalid format, formula parameter #" << idx << "): \"" << line
                << "\"";
          }
        }

        data_.emplace_back(
            ptMin, ptMax, etaMin, etaMax, puProxyMin, puProxyMax, std::move(formEval), std::move(formParams));
      }
    }

    double correction(float const pt, float const eta, int const puProxy) const {
      for (auto const& entry : data_) {
        if (eta >= entry.etaMin and eta < entry.etaMax and (entry.puProxyMin < 0 or puProxy >= entry.puProxyMin) and
            (entry.puProxyMax < 0 or puProxy < entry.puProxyMax)) {
          std::vector<double> vars{std::clamp(pt, entry.ptMin, entry.ptMax)};
          return (entry.formulaEvaluator.evaluate(vars, entry.formulaParameters) / vars[0]);
        }
      }
      return 0;
    }

  private:
    struct Entry {
      float ptMin;
      float ptMax;
      float etaMin;
      float etaMax;
      int puProxyMin;
      int puProxyMax;
      reco::FormulaEvaluator formulaEvaluator;
      std::vector<double> formulaParameters;
    };

    std::vector<Entry> data_;
  };

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // Number of BXs per orbit
  static constexpr unsigned int kNBXPlus1 = 3565;

  edm::EDGetTokenT<l1ScoutingRun3::CaloTowerOrbitCollection> const src_;
  double const akR_;
  double const ptMin_;
  int const towerMinHwEt_;
  int const towerMaxHwEt_;

  bool const applyJECs_;
  JetCorrector const jetCorrector_;
  int const jecPUProxyTowerMinHwEt_;
  int const jecPUProxyTowerMaxHwEt_;
  int const jecPUProxyTowerMinAbsHwEta_;
  int const jecPUProxyTowerMaxAbsHwEta_;

  int const mantissaPrecision_;
};

L1ScoutingCaloJetProducer::L1ScoutingCaloJetProducer(const edm::ParameterSet& iPSet)
    : src_(consumes(iPSet.getParameter<edm::InputTag>("src"))),
      akR_(iPSet.getParameter<double>("akR")),
      ptMin_(iPSet.getParameter<double>("ptMin")),
      towerMinHwEt_(iPSet.getParameter<int>("towerMinHwEt")),
      towerMaxHwEt_(iPSet.getParameter<int>("towerMaxHwEt")),
      applyJECs_(iPSet.getParameter<bool>("applyJECs")),
      jetCorrector_(applyJECs_ ? JetCorrector(iPSet.getParameter<edm::FileInPath>("jecFile").fullPath())
                               : JetCorrector{}),
      jecPUProxyTowerMinHwEt_(iPSet.getParameter<int>("jecPUProxyTowerMinHwEt")),
      jecPUProxyTowerMaxHwEt_(iPSet.getParameter<int>("jecPUProxyTowerMaxHwEt")),
      jecPUProxyTowerMinAbsHwEta_(iPSet.getParameter<int>("jecPUProxyTowerMinAbsHwEta")),
      jecPUProxyTowerMaxAbsHwEta_(iPSet.getParameter<int>("jecPUProxyTowerMaxAbsHwEta")),
      mantissaPrecision_(iPSet.getParameter<int>("mantissaPrecision")) {
  produces<l1ScoutingRun3::CaloJetOrbitCollection>("CaloJet").setBranchAlias("CaloJetOrbitCollection");
}

// ------------ method called for each ORBIT  ------------
void L1ScoutingCaloJetProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  auto const& caloTowerCollection = iEvent.get(src_);

  auto caloJetCollection = std::make_unique<l1ScoutingRun3::CaloJetOrbitCollection>();
  std::vector<std::vector<l1ScoutingRun3::CaloJet>> caloJetBuffer(kNBXPlus1);
  unsigned int nCaloJet = 0;

  // define fastjet algorithm
  fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, akR_);

  // create pseudojet vector to be filled
  std::vector<fastjet::PseudoJet> pjCTs;

  // loop over valid bunch crossings
  for (auto const bx : caloTowerCollection.getFilledBxs()) {
    LogTrace("L1ScoutingCaloJetProducer")
        << "[L1ScoutingCaloJetProducer:" << moduleDescription().moduleLabel() << "] BX = " << bx;

    LogTrace("L1ScoutingCaloJetProducer") << "[L1ScoutingCaloJetProducer:" << moduleDescription().moduleLabel()
                                          << "]   Inputs (l1ScoutingRun3::CaloTower and fastjet::PseudoJet)";

    auto const& cts = caloTowerCollection.bxIterator(bx);

    // prepare PseudoJets to give in input to fastjet
    pjCTs.clear();
    pjCTs.reserve(cts.size());
    for (auto const& ct : cts) {
      if (not((towerMinHwEt_ < 0 or ct.hwEt() >= towerMinHwEt_) and
              (towerMaxHwEt_ < 0 or ct.hwEt() <= towerMaxHwEt_))) {
        continue;
      }

      if (not l1ScoutingRun3::calol1::validHwEta(ct.hwEta())) {
        edm::LogWarning("L1ScoutingCaloJetProducer") << "CaloTower in BX=" << bx << " with invalid hwEta value ("
                                                     << ct.hwEta() << ") will not be used for jet clustering !";
        continue;
      }

      if (not l1ScoutingRun3::calol1::validHwPhi(ct.hwPhi())) {
        edm::LogWarning("L1ScoutingCaloJetProducer") << "CaloTower in BX=" << bx << " with invalid hwPhi value ("
                                                     << ct.hwPhi() << ") will not be used for jet clustering !";
        continue;
      }

      float const ctEt = l1ScoutingRun3::calol1::fEt(ct.hwEt());
      float const ctEta = l1ScoutingRun3::calol1::fEta(ct.hwEta());
      float const ctPhi = l1ScoutingRun3::calol1::fPhi(ct.hwPhi());

      pjCTs.emplace_back(fastjet::PtYPhiM(ctEt, ctEta, ctPhi, 0));

      LogTrace("L1ScoutingCaloJetProducer")
          << "[L1ScoutingCaloJetProducer:" << moduleDescription().moduleLabel() << "]     [" << (pjCTs.size() - 1)
          << "] hwEt=" << ct.hwEt() << " hwEta=" << ct.hwEta() << " hwPhi=" << ct.hwPhi() << " (PseudoJet: pt=" << ctEt
          << " eta=" << ctEta << " phi=" << ctPhi << " px=" << pjCTs.back().px() << " py=" << pjCTs.back().py()
          << " pz=" << pjCTs.back().pz() << " E=" << pjCTs.back().E() << ")";
    }

    // if JECs are applied, compute a per-BX PU proxy used as input to the evaluation of the JECs
    // (the PU proxy corresponds to the number of CaloTowers passing predefined cuts on hwEt and |hwEta|)
    int puProxy{0};
    if (applyJECs_) {
      for (auto const& ct : cts) {
        if (not((jecPUProxyTowerMinHwEt_ < 0 or ct.hwEt() >= jecPUProxyTowerMinHwEt_) and
                (jecPUProxyTowerMaxHwEt_ < 0 or ct.hwEt() <= jecPUProxyTowerMaxHwEt_))) {
          continue;
        }

        auto const absHwEta{std::abs(ct.hwEta())};
        if (not((jecPUProxyTowerMinAbsHwEta_ < 0 or absHwEta >= jecPUProxyTowerMinAbsHwEta_) and
                (jecPUProxyTowerMaxAbsHwEta_ < 0 or absHwEta <= jecPUProxyTowerMaxAbsHwEta_))) {
          continue;
        }

        ++puProxy;
      }

      LogTrace("L1ScoutingCaloJetProducer") << "[L1ScoutingCaloJetProducer:" << moduleDescription().moduleLabel()
                                            << "]   PU proxy for JECs (CaloTower multiplicity): " << puProxy;
    }

    LogTrace("L1ScoutingCaloJetProducer") << "[L1ScoutingCaloJetProducer:" << moduleDescription().moduleLabel()
                                          << "]   Running jet clustering and applying JECs";

    // run the jet clustering with the given jet definition
    fastjet::ClusterSequence clustSeq(pjCTs, jetDef);

    // get the resulting jets ordered in pt
    std::vector<fastjet::PseudoJet> incJets = clustSeq.inclusive_jets();

    // fill l1ScoutingRun3::CaloJet objects buffer
    auto& bufferThisBX = caloJetBuffer[bx];
    bufferThisBX.reserve(incJets.size());

    for (auto idx = 0u; idx < incJets.size(); ++idx) {
      auto const& incJet = incJets[idx];
      int const nConst = incJet.has_constituents() ? incJet.constituents().size() : 0;
      double const energyCorr{applyJECs_ ? jetCorrector_.correction(incJet.pt(), incJet.eta(), puProxy) : 1};

      LogTrace("L1ScoutingCaloJetProducer")
          << "[L1ScoutingCaloJetProducer:" << moduleDescription().moduleLabel() << "]     [" << idx
          << "] pt=" << incJet.pt() << " eta=" << incJet.eta() << " phi=" << incJet.phi_std() << " mass=" << incJet.m()
          << " energyCorr=" << energyCorr << " nConst=" << nConst << " (before JECs and pT cut)";

      if (energyCorr <= 0) {
        continue;
      }

      float const jet_pt = incJet.pt() * energyCorr;

      if (jet_pt < ptMin_) {
        continue;
      }

      float const jet_mass = incJet.m() * energyCorr;

      LogTrace("L1ScoutingCaloJetProducer")
          << "[L1ScoutingCaloJetProducer:" << moduleDescription().moduleLabel() << "]     [" << idx << "] pt=" << jet_pt
          << " eta=" << incJet.eta() << " phi=" << incJet.phi_std() << " mass=" << jet_mass
          << " energyCorr=" << energyCorr << " nConst=" << nConst << " (after JECs and pT cut)";

      bufferThisBX.emplace_back(MiniFloatConverter::reduceMantissaToNbitsRounding(jet_pt, mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(incJet.eta(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(incJet.phi_std(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(jet_mass, mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(energyCorr, mantissaPrecision_),
                                nConst);
      ++nCaloJet;
    }

    std::sort(bufferThisBX.begin(), bufferThisBX.end(), [](auto const& a, auto const& b) { return a.pt() > b.pt(); });

    LogTrace("L1ScoutingCaloJetProducer") << "[L1ScoutingCaloJetProducer:" << moduleDescription().moduleLabel()
                                          << "]   Final outputs (l1ScoutingRun3::CaloJet)";

#ifdef EDM_ML_DEBUG
    for (auto idx = 0u; idx < bufferThisBX.size(); ++idx) {
      auto const& obj = bufferThisBX[idx];
      LogTrace("L1ScoutingCaloJetProducer")
          << "[L1ScoutingCaloJetProducer:" << moduleDescription().moduleLabel() << "]     [" << idx
          << "] pt=" << obj.pt() << " eta=" << obj.eta() << " phi=" << obj.phi() << " mass=" << obj.mass()
          << " energyCorr=" << obj.energyCorr() << " nConst=" << obj.nConst();
    }
#endif
  }

  // fill orbit collection with reconstructed jets
  caloJetCollection->fillAndClear(caloJetBuffer, nCaloJet);

  iEvent.put(std::move(caloJetCollection), "CaloJet");
}

void L1ScoutingCaloJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src")->setComment(
      "Input collection of CaloTowers (type: l1ScoutingRun3::CaloTowerOrbitCollection)");
  desc.add<double>("akR")->setComment("Value of R parameter for anti-kt clustering");
  desc.add<double>("ptMin")->setComment(
      "Minimum pT of output jets (applies to the corrected jet-pT if applyJECs==True)");
  desc.add<int>("towerMinHwEt", 1)
      ->setComment("Min hwEt (inclusive) of CaloTowers used for jet clustering (ignored if negative)");
  desc.add<int>("towerMaxHwEt", -1)
      ->setComment("Max hwEt (inclusive) of CaloTowers used for jet clustering (ignored if negative)");

  desc.add<bool>("applyJECs", false)
      ->setComment("Apply jet-energy-scale corrections (and output corrected jets, ordered by their corrected pT)");
  desc.add<edm::FileInPath>("jecFile")->setComment(
      "Path to text file containing jet-energy-scale corrections (used only if applyJECs==True)");
  desc.add<int>("jecPUProxyTowerMinHwEt", 1)
      ->setComment(
          "Min CaloTower hwEt (inclusive) used when computing the CaloTower multiplicity taken as PU proxy to evaluate "
          "JECs (used only if applyJECs==True, and ignored if negative)");
  desc.add<int>("jecPUProxyTowerMaxHwEt", -1)
      ->setComment(
          "Max CaloTower hwEt (inclusive) used when computing the CaloTower multiplicity taken as PU proxy to evaluate "
          "JECs (used only if applyJECs==True, and ignored if negative)");
  desc.add<int>("jecPUProxyTowerMinAbsHwEta", 0)
      ->setComment(
          "Min CaloTower |hwEta| (inclusive) used when computing the CaloTower multiplicity taken as PU proxy to "
          "evaluate JECs (used only if applyJECs==True, and ignored if negative)");
  desc.add<int>("jecPUProxyTowerMaxAbsHwEta", 4)
      ->setComment(
          "Max CaloTower |hwEta| (inclusive) used when computing the CaloTower multiplicity taken as PU proxy to "
          "evaluate JECs (used only if applyJECs==True, and ignored if negative)");

  desc.add<int>("mantissaPrecision", 10)->setComment("default float16, change to 23 for float32");

  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1ScoutingCaloJetProducer);

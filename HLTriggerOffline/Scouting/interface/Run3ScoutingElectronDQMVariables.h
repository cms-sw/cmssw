#ifndef HLTriggerOffline_Scouting_interface_Run3ScoutingElectronDQMVariables_h
#define HLTriggerOffline_Scouting_interface_Run3ScoutingElectronDQMVariables_h

#include "DQMServices/Components/interface/GenericObjectDQMSource.h"
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"

template <>
struct DQMVariableTraits<Run3ScoutingElectron> {
  static std::vector<DQMVariable<Run3ScoutingElectron>> variables() {
    return {
        {"pt", "p_{T} [GeV]", 100, 0., 200., [](Run3ScoutingElectron const& e) { return e.pt(); }},
        {"eta", "#eta", 100, -4., 4., [](Run3ScoutingElectron const& e) { return e.eta(); }},
        {"phi", "#phi", 100, -3.15, 3.15, [](Run3ScoutingElectron const& e) { return e.phi(); }},
        {"m", "mass [GeV]", 100, 0., 1., [](Run3ScoutingElectron const& e) { return e.m(); }},
        {"rawEnergy", "raw energy [GeV]", 100, 0., 300., [](Run3ScoutingElectron const& e) { return e.rawEnergy(); }},
        {"preshowerEnergy",
         "preshower energy [GeV]",
         100,
         0.,
         20.,
         [](Run3ScoutingElectron const& e) { return e.preshowerEnergy(); }},
        {"corrEcalEnergyError",
         "#sigma(E_{ECAL}) [GeV]",
         100,
         0.,
         10.,
         [](Run3ScoutingElectron const& e) { return e.corrEcalEnergyError(); }},
        {"dEtaIn", "#Delta#eta_{in}", 100, -0.05, 0.05, [](Run3ScoutingElectron const& e) { return e.dEtaIn(); }},
        {"dPhiIn", "#Delta#phi_{in}", 100, -0.1, 0.1, [](Run3ScoutingElectron const& e) { return e.dPhiIn(); }},
        {"sigmaIetaIeta",
         "#sigma_{i#etai#eta}",
         100,
         0.,
         0.05,
         [](Run3ScoutingElectron const& e) { return e.sigmaIetaIeta(); }},
        {"hOverE", "H/E", 100, 0., 0.5, [](Run3ScoutingElectron const& e) { return e.hOverE(); }},
        {"ooEMOop", "1/E - 1/p [1/GeV]", 100, -0.2, 0.2, [](Run3ScoutingElectron const& e) { return e.ooEMOop(); }},
        {"missingHits", "N_{missing hits}", 10, 0., 10., [](Run3ScoutingElectron const& e) { return e.missingHits(); }},
        {"trackfbrem", "f_{brem}", 100, 0., 1., [](Run3ScoutingElectron const& e) { return e.trackfbrem(); }},
        {"ecalIso", "ECAL iso [GeV]", 100, 0., 20., [](Run3ScoutingElectron const& e) { return e.ecalIso(); }},
        {"hcalIso", "HCAL iso [GeV]", 100, 0., 20., [](Run3ScoutingElectron const& e) { return e.hcalIso(); }},
        {"trackIso", "track iso [GeV]", 100, 0., 20., [](Run3ScoutingElectron const& e) { return e.trackIso(); }},
        {"r9", "R9", 100, 0., 1.2, [](Run3ScoutingElectron const& e) { return e.r9(); }},
        {"sMin", "s_{min}", 100, 0., 1., [](Run3ScoutingElectron const& e) { return e.sMin(); }},
        {"sMaj", "s_{maj}", 100, 0., 5., [](Run3ScoutingElectron const& e) { return e.sMaj(); }},
        {"seedId", "seed detId", 100, 0., 1.e9, [](Run3ScoutingElectron const& e) { return e.seedId(); }},
        {"nClusters", "N_{clusters}", 10, 0., 10., [](Run3ScoutingElectron const& e) { return e.nClusters(); }},
        {"nCrystals", "N_{crystals}", 30, 0., 30., [](Run3ScoutingElectron const& e) { return e.nCrystals(); }},
        {"rechitZeroSuppression",
         "rechitZeroSuppression",
         2,
         -0.5,
         1.5,
         [](Run3ScoutingElectron const& e) { return e.rechitZeroSuppression(); }},
    };
  }

  static std::vector<DQMVectorVariable<Run3ScoutingElectron>> vectorVariables() {
    return {
        {"trkd0",
         "per-track d_{0} [cm]",
         100,
         -1.,
         1.,
         [](Run3ScoutingElectron const& e) { return std::vector<double>(e.trkd0().begin(), e.trkd0().end()); }},
        {"trkdz",
         "per-track d_{z} [cm]",
         100,
         -20.,
         20.,
         [](Run3ScoutingElectron const& e) { return std::vector<double>(e.trkdz().begin(), e.trkdz().end()); }},
        {"trkpt",
         "per-track p_{T} [GeV]",
         100,
         0.,
         200.,
         [](Run3ScoutingElectron const& e) { return std::vector<double>(e.trkpt().begin(), e.trkpt().end()); }},
        {"trketa",
         "per-track #eta",
         100,
         -3.,
         3.,
         [](Run3ScoutingElectron const& e) { return std::vector<double>(e.trketa().begin(), e.trketa().end()); }},
        {"trkphi",
         "per-track #phi",
         100,
         -3.15,
         3.15,
         [](Run3ScoutingElectron const& e) { return std::vector<double>(e.trkphi().begin(), e.trkphi().end()); }},
        {"trkpMode",
         "per-track p_{mode} [GeV]",
         100,
         0.,
         200.,
         [](Run3ScoutingElectron const& e) { return std::vector<double>(e.trkpMode().begin(), e.trkpMode().end()); }},
        {"trketaMode",
         "per-track #eta_{mode}",
         100,
         -3.,
         3.,
         [](Run3ScoutingElectron const& e) {
           return std::vector<double>(e.trketaMode().begin(), e.trketaMode().end());
         }},
        {"trkphiMode",
         "per-track #phi_{mode}",
         100,
         -3.15,
         3.15,
         [](Run3ScoutingElectron const& e) {
           return std::vector<double>(e.trkphiMode().begin(), e.trkphiMode().end());
         }},
        {"trkqoverpModeError",
         "per-track #sigma(q/p_{mode}) [1/GeV]",
         100,
         0.,
         0.5,
         [](Run3ScoutingElectron const& e) {
           return std::vector<double>(e.trkqoverpModeError().begin(), e.trkqoverpModeError().end());
         }},
        {"trkchi2overndf",
         "per-track #chi^{2}/ndof",
         100,
         0.,
         10.,
         [](Run3ScoutingElectron const& e) {
           return std::vector<double>(e.trkchi2overndf().begin(), e.trkchi2overndf().end());
         }},
        {"trkcharge",
         "per-track charge",
         5,
         -2.5,
         2.5,
         [](Run3ScoutingElectron const& e) { return std::vector<double>(e.trkcharge().begin(), e.trkcharge().end()); }},
        {"energyMatrix",
         "per-crystal energy [GeV]",
         100,
         0.,
         50.,
         [](Run3ScoutingElectron const& e) {
           return std::vector<double>(e.energyMatrix().begin(), e.energyMatrix().end());
         }},
        {"detIds",
         "per-crystal detId",
         100,
         0.,
         1.e9,
         [](Run3ScoutingElectron const& e) { return std::vector<double>(e.detIds().begin(), e.detIds().end()); }},
        {"timingMatrix",
         "per-crystal timing [ns]",
         100,
         -25.,
         25.,
         [](Run3ScoutingElectron const& e) {
           return std::vector<double>(e.timingMatrix().begin(), e.timingMatrix().end());
         }},
    };
  }
};

#endif

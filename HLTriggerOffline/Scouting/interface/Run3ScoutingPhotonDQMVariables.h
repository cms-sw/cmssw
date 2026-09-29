#ifndef HLTriggerOffline_Scouting_interface_Run3ScoutingPhotonDQMVariables_h
#define HLTriggerOffline_Scouting_interface_Run3ScoutingPhotonDQMVariables_h

#include "DQMServices/Components/interface/GenericObjectDQMSource.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"

template <>
struct DQMVariableTraits<Run3ScoutingPhoton> {
  static std::vector<DQMVariable<Run3ScoutingPhoton>> variables() {
    return {
        {"pt", "p_{T} [GeV]", 100, 0., 200., [](Run3ScoutingPhoton const& p) { return p.pt(); }},
        {"eta", "#eta", 100, -4., 4., [](Run3ScoutingPhoton const& p) { return p.eta(); }},
        {"phi", "#phi", 100, -3.15, 3.15, [](Run3ScoutingPhoton const& p) { return p.phi(); }},
        {"m", "mass [GeV]", 100, 0., 1., [](Run3ScoutingPhoton const& p) { return p.m(); }},
        {"rawEnergy", "raw energy [GeV]", 100, 0., 300., [](Run3ScoutingPhoton const& p) { return p.rawEnergy(); }},
        {"preshowerEnergy",
         "preshower energy [GeV]",
         100,
         0.,
         20.,
         [](Run3ScoutingPhoton const& p) { return p.preshowerEnergy(); }},
        {"corrEcalEnergyError",
         "#sigma(E_{ECAL}) [GeV]",
         100,
         0.,
         10.,
         [](Run3ScoutingPhoton const& p) { return p.corrEcalEnergyError(); }},
        {"sigmaIetaIeta",
         "#sigma_{i#etai#eta}",
         100,
         0.,
         0.05,
         [](Run3ScoutingPhoton const& p) { return p.sigmaIetaIeta(); }},
        {"hOverE", "H/E", 100, 0., 0.5, [](Run3ScoutingPhoton const& p) { return p.hOverE(); }},
        {"ecalIso", "ECAL iso [GeV]", 100, 0., 20., [](Run3ScoutingPhoton const& p) { return p.ecalIso(); }},
        {"hcalIso", "HCAL iso [GeV]", 100, 0., 20., [](Run3ScoutingPhoton const& p) { return p.hcalIso(); }},
        {"trkIso", "track iso [GeV]", 100, 0., 20., [](Run3ScoutingPhoton const& p) { return p.trkIso(); }},
        {"r9", "R9", 100, 0., 1.2, [](Run3ScoutingPhoton const& p) { return p.r9(); }},
        {"sMin", "s_{min}", 100, 0., 1., [](Run3ScoutingPhoton const& p) { return p.sMin(); }},
        {"sMaj", "s_{maj}", 100, 0., 5., [](Run3ScoutingPhoton const& p) { return p.sMaj(); }},
        {"seedId", "seed detId", 100, 0., 1.e9, [](Run3ScoutingPhoton const& p) { return p.seedId(); }},
        {"nClusters", "N_{clusters}", 10, 0., 10., [](Run3ScoutingPhoton const& p) { return p.nClusters(); }},
        {"nCrystals", "N_{crystals}", 30, 0., 30., [](Run3ScoutingPhoton const& p) { return p.nCrystals(); }},
        {"rechitZeroSuppression",
         "rechitZeroSuppression",
         2,
         -0.5,
         1.5,
         [](Run3ScoutingPhoton const& p) { return p.rechitZeroSuppression(); }},
    };
  }

  static std::vector<DQMVectorVariable<Run3ScoutingPhoton>> vectorVariables() {
    return {
        {"energyMatrix",
         "per-crystal energy [GeV]",
         100,
         0.,
         50.,
         [](Run3ScoutingPhoton const& p) {
           return std::vector<double>(p.energyMatrix().begin(), p.energyMatrix().end());
         }},
        {"timingMatrix",
         "per-crystal timing [ns]",
         100,
         -25.,
         25.,
         [](Run3ScoutingPhoton const& p) {
           return std::vector<double>(p.timingMatrix().begin(), p.timingMatrix().end());
         }},
        {"detIds",
         "per-crystal detId",
         100,
         0.,
         1.e9,
         [](Run3ScoutingPhoton const& p) { return std::vector<double>(p.detIds().begin(), p.detIds().end()); }},
    };
  }
};

#endif

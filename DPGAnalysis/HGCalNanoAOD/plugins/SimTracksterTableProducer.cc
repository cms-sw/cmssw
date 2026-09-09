#include <algorithm>
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SimTracksterTableProducer : public edm::global::EDProducer<> {
public:
  SimTracksterTableProducer(const edm::ParameterSet& cfg)
      : tableName_(cfg.getParameter<std::string>("tableName")),
        skipNonExistingSrc_(cfg.getParameter<bool>("skipNonExistingSrc")),
        simTrackstersToken_(mayConsume<std::vector<ticl::Trackster>>(cfg.getParameter<edm::InputTag>("simTracksters"))),
        caloParticlesToken_(mayConsume<std::vector<CaloParticle>>(cfg.getParameter<edm::InputTag>("caloParticles"))),
        simClustersToken_(mayConsume<std::vector<SimCluster>>(cfg.getParameter<edm::InputTag>("simClusters"))),
        caloParticleToSimClustersMap_token_(mayConsume<std::map<uint, std::vector<uint>>>(
            cfg.getParameter<edm::InputTag>("caloParticleToSimClustersMap"))),
        precision_(cfg.getParameter<int>("precision")) {
    produces<nanoaod::FlatTable>(tableName_);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("tableName", "hltSimTrackstersTable")
        ->setComment("Table name, needs to be the same as the main Tau table");
    desc.add<bool>("skipNonExistingSrc", false)
        ->setComment("whether or not to skip producing the table on absent input product");
    desc.add<edm::InputTag>("simTracksters", edm::InputTag("hltTiclSimTracksters"));
    desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
    desc.add<edm::InputTag>("simClusters", edm::InputTag("mix", "MergedCaloTruth"));
    desc.add<edm::InputTag>("caloParticleToSimClustersMap", edm::InputTag("hltTiclSimTracksters"));
    desc.add<int>("precision", 7);
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& setup) const override {
    const auto simTrackstersHandle = event.getHandle(simTrackstersToken_);
    const auto caloParticlesHandle = event.getHandle(caloParticlesToken_);
    const auto simClustersHandle = event.getHandle(simClustersToken_);
    const auto cpToSCMapHandle = event.getHandle(caloParticleToSimClustersMap_token_);

    const size_t nSimTracksters = simTrackstersHandle.isValid() ? simTrackstersHandle->size() : 0;

    static constexpr float default_value = std::numeric_limits<float>::quiet_NaN();
    static constexpr int default_int_value = -1;

    std::vector<float> boundaryX(nSimTracksters, default_value);
    std::vector<float> boundaryY(nSimTracksters, default_value);
    std::vector<float> boundaryZ(nSimTracksters, default_value);
    std::vector<float> boundaryPx(nSimTracksters, default_value);
    std::vector<float> boundaryPy(nSimTracksters, default_value);
    std::vector<float> boundaryPz(nSimTracksters, default_value);
    std::vector<float> boundaryEta(nSimTracksters, default_value);
    std::vector<float> boundaryPhi(nSimTracksters, default_value);
    std::vector<float> simEnergy(nSimTracksters, default_value);
    std::vector<float> simTime(nSimTracksters, default_value);
    std::vector<float> genPt(nSimTracksters, default_value);
    std::vector<float> mass(nSimTracksters, default_value);
    std::vector<int> caloParticleIdx(nSimTracksters, default_int_value);
    std::vector<int8_t> isPU(nSimTracksters, default_int_value);

    if ((simTrackstersHandle.isValid() && caloParticlesHandle.isValid() && simClustersHandle.isValid() &&
         cpToSCMapHandle.isValid()) ||
        !(this->skipNonExistingSrc_)) {
      const auto& simTracksters = *simTrackstersHandle;
      const auto& caloParticles = *caloParticlesHandle;
      const auto& simClusters = *simClustersHandle;
      const auto& cpToSCMap = *cpToSCMapHandle;

      //utility lambda for filling vectors
      auto fillVectors = [&](const auto& obj, size_t iSim, float time, int cpIdx) {
        const auto& simTrack = obj.g4Tracks()[0];
        const auto caloPt = obj.pt();
        const auto simHitSumEnergy = obj.simEnergy();
        const auto caloMass = obj.mass();

        boundaryX[iSim] = simTrack.getPositionAtBoundary().x();
        boundaryY[iSim] = simTrack.getPositionAtBoundary().y();
        boundaryZ[iSim] = simTrack.getPositionAtBoundary().z();
        boundaryEta[iSim] = simTrack.getPositionAtBoundary().eta();
        boundaryPhi[iSim] = simTrack.getPositionAtBoundary().phi();
        boundaryPx[iSim] = simTrack.getMomentumAtBoundary().x();
        boundaryPy[iSim] = simTrack.getMomentumAtBoundary().y();
        boundaryPz[iSim] = simTrack.getMomentumAtBoundary().z();

        simTime[iSim] = time;
        simEnergy[iSim] = simHitSumEnergy;
        genPt[iSim] = caloPt;
        mass[iSim] = caloMass;

        // PU flag: a non-zero event/bunch-crossing on the
        // seed g4Track means the CaloParticle/SimCluster came from a pileup interaction.
        isPU[iSim] = (simTrack.eventId().event() != 0 || simTrack.eventId().bunchCrossing() != 0) ? 1 : 0;

        caloParticleIdx[iSim] = cpIdx;
      };

      for (size_t iSim = 0; iSim < simTracksters.size(); ++iSim) {
        const auto& simT = simTracksters[iSim];
        float time = default_value;

        if (simT.seedID() == caloParticlesHandle.id()) {
          const int cpIdx = static_cast<int>(simT.seedIndex());
          const auto& cp = caloParticles[simT.seedIndex()];
          time = cp.simTime();
          fillVectors(cp, iSim, time, cpIdx);
        } else {
          const auto& sc = simClusters[simT.seedIndex()];
          int cpIdx = default_int_value;
          //SCtoCP map not availalbe, use CPtoSC map instead
          for (const auto& [cpIdxCandidate, scVec] : cpToSCMap) {
            if (std::ranges::find(scVec, simT.seedIndex()) != scVec.end()) {
              cpIdx = static_cast<int>(cpIdxCandidate);
              time = caloParticles[cpIdxCandidate].simTime();
              break;  //dont need to check further
            }
          }
          fillVectors(sc, iSim, time, cpIdx);
        }
      }
    }

    auto simTrackstersTable =
        std::make_unique<nanoaod::FlatTable>(nSimTracksters, tableName_, /*singleton*/ false, /*extension*/ true);
    simTrackstersTable->addColumn<float>(
        "boundaryX", boundaryX, "CaloVolume boundary Position X [cm] of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>(
        "boundaryY", boundaryY, "CaloVolume boundary Position Y [cm] of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>(
        "boundaryZ", boundaryZ, "CaloVolume boundary Position Z [cm] of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>(
        "boundaryEta", boundaryEta, "CaloVolume boundary pseudorapidity of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>(
        "boundaryPhi", boundaryPhi, "CaloVolume boundary phi of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>(
        "boundaryPx", boundaryPx, "X component of momentum at CaloVolume boundary of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>(
        "boundaryPy", boundaryPy, "Y component of momentum at CaloVolume boundary of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>(
        "boundaryPz", boundaryPz, "Z component of momentum at CaloVolume boundary of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>("simTime", simTime, "Sim-Time of simulated object [ns]", precision_);
    simTrackstersTable->addColumn<float>("genPt", genPt, "Gen-pT associated with SimObject", precision_);
    simTrackstersTable->addColumn<float>("mass", mass, "mass associated with SimObject", precision_);
    simTrackstersTable->addColumn<int>(
        "caloParticleIdx", caloParticleIdx, "Index of the parent CaloParticle in the CaloParticle collection");
    simTrackstersTable->addColumn<int>(
        "isPU", isPU, "PU flag of the associated Simobject: 1 = pileup (non-zero event/bx), 0 = otherwise");

    event.put(std::move(simTrackstersTable), tableName_);
  }

private:
  const std::string tableName_;
  const bool skipNonExistingSrc_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> simTrackstersToken_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticlesToken_;
  const edm::EDGetTokenT<std::vector<SimCluster>> simClustersToken_;
  const edm::EDGetTokenT<std::map<uint, std::vector<uint>>> caloParticleToSimClustersMap_token_;
  const unsigned int precision_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimTracksterTableProducer);

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
    const auto& simTracksters = *simTrackstersHandle;
    const auto caloParticlesHandle = event.getHandle(caloParticlesToken_);
    const auto& caloParticles = *caloParticlesHandle;
    const auto simClustersHandle = event.getHandle(simClustersToken_);
    const auto& simClusters = *simClustersHandle;
    const auto cpToSCMap = event.get(caloParticleToSimClustersMap_token_);
    const size_t nSimTracksters = simTrackstersHandle.isValid() ? simTrackstersHandle->size() : 0;

    static constexpr float default_value = std::numeric_limits<float>::quiet_NaN();

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

    //utility lambda for filling vectors
    auto fillVectors = [&](const auto& obj, size_t iSim, float time) {
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
    };

    if (simTrackstersHandle.isValid() || !(this->skipNonExistingSrc_)) {
      for (size_t iSim = 0; iSim < simTracksters.size(); ++iSim) {
        const auto& simT = simTracksters[iSim];
        float time = default_value;

        if (simT.seedID() == caloParticlesHandle.id()) {
          const auto& cp = caloParticles[simT.seedIndex()];
          time = cp.simTime();
          fillVectors(cp, iSim, time);
        } else {
          const auto& sc = simClusters[simT.seedIndex()];
          //SCtoCP map not availalbe, use CPtoSC map instead
          for (const auto& [cpIdx, scVec] : cpToSCMap) {
            if (std::ranges::find(scVec, simT.seedIndex()) != scVec.end()) {
              time = caloParticles[cpIdx].simTime();
              break;  //dont need to check further
            }
          }
          fillVectors(sc, iSim, time);
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
        "boundaryPhi", boundaryEta, "CaloVolume boundary phi of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>(
        "boundaryPx", boundaryPx, "X component of momentum at CaloVolume boundary of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>(
        "boundaryPy", boundaryPy, "Y component of momentum at CaloVolume boundary of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>(
        "boundaryPz", boundaryPz, "Z component of momentum at CaloVolume boundary of associated Simobject", precision_);
    simTrackstersTable->addColumn<float>("simTime", simTime, "Sim-Time of simulated object [ns]", precision_);
    simTrackstersTable->addColumn<float>("genPt", genPt, "Gen-pT associated with SimObject", precision_);
    simTrackstersTable->addColumn<float>("mass", mass, "mass associated with SimObject", precision_);

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

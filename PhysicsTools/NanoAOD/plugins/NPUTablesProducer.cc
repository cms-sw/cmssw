#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <vector>
#include <iostream>

class NPUTablesProducer : public edm::global::EDProducer<> {
public:
  NPUTablesProducer(edm::ParameterSet const& params)
      : npuTag_(consumes<std::vector<PileupSummaryInfo>>(params.getParameter<edm::InputTag>("src"))),
        pvTag_(consumes<std::vector<reco::Vertex>>(params.getParameter<edm::InputTag>("pvsrc"))),
        vz_(params.getParameter<std::vector<double>>("zbins")) {
    produces<nanoaod::FlatTable>();
  }

  ~NPUTablesProducer() override {}

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
    auto npuTab = std::make_unique<nanoaod::FlatTable>(1, "Pileup", true);

    edm::Handle<std::vector<reco::Vertex>> pvsIn;
    iEvent.getByToken(pvTag_, pvsIn);
    const double refpvz = (*pvsIn)[0].position().z();

    edm::Handle<std::vector<PileupSummaryInfo>> npuInfo;
    if (iEvent.getByToken(npuTag_, npuInfo)) {
      fillNPUObjectTable(*npuInfo, *npuTab, refpvz);
    }

    iEvent.put(std::move(npuTab));
  }

  void fillNPUObjectTable(const std::vector<PileupSummaryInfo>& npuProd, nanoaod::FlatTable& out, double refpvz) const {
    // Get BX 0
    unsigned int bx0 = 0;
    unsigned int nt = 0;
    unsigned int npu = 0;

    auto zbin = std::lower_bound(vz_.begin(), vz_.end() - 1, std::abs(refpvz));
    float pudensity = 0;
    float gpudensity = 0;

    for (unsigned int ibx = 0; ibx < npuProd.size(); ibx++) {
      if (npuProd[ibx].getBunchCrossing() == 0) {
        bx0 = ibx;
        nt = npuProd[ibx].getTrueNumInteractions();
        npu = npuProd[ibx].getPU_NumInteractions();

        std::vector<float> zpositions;
        unsigned int nzpositions = npuProd[ibx].getPU_zpositions().size();
        for (unsigned int j = 0; j < nzpositions; ++j) {
          zpositions.push_back(npuProd[ibx].getPU_zpositions()[j]);
          if (std::abs(zpositions.back() - refpvz) < 0.1)
            pudensity++;  //N_PU/mm
          auto bin = std::lower_bound(vz_.begin(), vz_.end() - 1, std::abs(zpositions.back()));
          if (bin != vz_.end() && bin == zbin)
            gpudensity++;
        }
        gpudensity /= (20.0 * (*(zbin) - *(zbin - 1)));
      }
    }
    unsigned int eoot = 0;
    for (unsigned int ipu = 0; ipu < bx0; ipu++) {
      eoot += npuProd[ipu].getPU_NumInteractions();
    }
    unsigned int loot = 0;
    for (unsigned int ipu = npuProd.size() - 1; ipu > bx0; ipu--) {
      loot += npuProd[ipu].getPU_NumInteractions();
    }
    out.addColumnValue<float>("nTrueInt",
                              nt,
                              "the true mean number of the poisson distribution for this event from which the number "
                              "of interactions each bunch crossing has been sampled",
                              nanoaod::FlatTable::FloatColumn);
    out.addColumnValue<int>(
        "nPU",
        npu,
        "the number of pileup interactions that have been added to the event in the current bunch crossing",
        nanoaod::FlatTable::IntColumn);
    out.addColumnValue<int>("sumEOOT", eoot, "number of early out of time pileup", nanoaod::FlatTable::IntColumn);
    out.addColumnValue<int>("sumLOOT", loot, "number of late out of time pileup", nanoaod::FlatTable::IntColumn);
    out.addColumnValue<float>("pudensity", pudensity, "PU vertices / mm", nanoaod::FlatTable::FloatColumn);
    out.addColumnValue<float>(
        "gpudensity", gpudensity, "Generator-level PU vertices / mm", nanoaod::FlatTable::FloatColumn);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("slimmedAddPileupInfo"))
        ->setComment("tag for the PU information (vector<PileupSummaryInfo>)");
    desc.add<edm::InputTag>("pvsrc", edm::InputTag("offlineSlimmedPrimaryVertices"))->setComment("tag for the PVs");
    desc.add<std::vector<double>>("zbins", {})
        ->setComment("Z bins to compute the generator-level number of PU vertices per mm");
    descriptions.add("puTable", desc);
  }

protected:
  const edm::EDGetTokenT<std::vector<PileupSummaryInfo>> npuTag_;
  const edm::EDGetTokenT<std::vector<reco::Vertex>> pvTag_;

  const std::vector<double> vz_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(NPUTablesProducer);

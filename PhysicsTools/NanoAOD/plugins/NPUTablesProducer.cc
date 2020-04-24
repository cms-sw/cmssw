#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include <vector>
#include <iostream>


class NPUTablesProducer : public edm::global::EDProducer<> {
    public:
        NPUTablesProducer( edm::ParameterSet const & params ) :
  	   npuTag_(consumes<std::vector<PileupSummaryInfo>>(params.getParameter<edm::InputTag>("src")))
        {
            produces<nanoaod::FlatTable>();
        }

        ~NPUTablesProducer() override {}

        void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
            auto npuTab  = std::make_unique<nanoaod::FlatTable>(1, "Pileup", true);

            edm::Handle<std::vector<PileupSummaryInfo> > npuInfo;
            if (iEvent.getByToken(npuTag_, npuInfo)) {
                fillNPUObjectTable(*npuInfo, *npuTab);
            }

            iEvent.put(std::move(npuTab));
        }

        void fillNPUObjectTable(const std::vector<PileupSummaryInfo> & npuProd, nanoaod::FlatTable & out) const {
	  // Get BX 0
	  unsigned int bx0 = 0;
	  unsigned int nt = 0;
	  unsigned int npu = 0;
	  for(unsigned int ibx=0; ibx<npuProd.size(); ibx++) {
	    if(npuProd[ibx].getBunchCrossing()==0) {
	      bx0 = ibx;
	      nt = npuProd[ibx].getTrueNumInteractions();
	      npu = npuProd[ibx].getPU_NumInteractions();
	    }
	  }
	  unsigned int eoot = 0;
	  for(unsigned int ipu=0; ipu<bx0; ipu++) {
	    eoot+=npuProd[ipu].getPU_NumInteractions();
	  }
	  unsigned int loot = 0;
	  for(unsigned int ipu=npuProd.size()-1; ipu>bx0; ipu--) {
	    loot+=npuProd[ipu].getPU_NumInteractions();
	  }
	  out.addColumnValue<float>("nTrueInt", nt, "the true mean number of the poisson distribution for this event from which the number of interactions each bunch crossing has been sampled", nanoaod::FlatTable::FloatColumn);
	  out.addColumnValue<int>("nPU",    npu, "the number of pileup interactions that have been added to the event in the current bunch crossing", nanoaod::FlatTable::IntColumn);
	  out.addColumnValue<int>("sumEOOT", eoot, "number of early out of time pileup" , nanoaod::FlatTable::IntColumn);
	  out.addColumnValue<int>("sumLOOT", loot, "number of late out of time pileup" , nanoaod::FlatTable::IntColumn);
        }

        static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
            edm::ParameterSetDescription desc;
            desc.add<edm::InputTag>("src", edm::InputTag("slimmedAddPileupInfo"))->setComment("tag for the PU information (vector<PileupSummaryInfo>)");
            descriptions.add("puTable", desc);
        }

    protected:
       const edm::EDGetTokenT<std::vector<PileupSummaryInfo>> npuTag_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(NPUTablesProducer);


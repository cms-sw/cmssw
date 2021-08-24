#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/ValidHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class DummyHepMCAnalyzer : public edm::one::EDAnalyzer<> {
private:
  bool dumpHepMC_;
  bool dumpPDF_;
  bool checkPDG_;
  edm::EDGetTokenT<edm::HepMCProduct> srcToken_;

public:
  explicit DummyHepMCAnalyzer(const edm::ParameterSet& cfg)
      : dumpHepMC_(cfg.getUntrackedParameter<bool>("dumpHepMC")),
        dumpPDF_(cfg.getUntrackedParameter<bool>("dumpPDF")),
        checkPDG_(cfg.getUntrackedParameter<bool>("checkPDG")),
        srcToken_(consumes<edm::HepMCProduct>(cfg.getParameter<edm::InputTag>("src"))) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("dumpHepMC", true);
    desc.addUntracked<bool>("dumpPDF", false);
    desc.addUntracked<bool>("checkPDG", false);
    desc.add<edm::InputTag>("src", edm::InputTag("generatorSmeared"))
        ->setComment("Input generated HepMC event after vtx smearing");
    descriptions.add("dummyHepMCAnalyzer", desc);
  }

  void analyze(const edm::Event& evt, const edm::EventSetup& es) override {
    auto hepMC = makeValid(evt.getHandle(srcToken_));

    const HepMC::GenEvent* mc = hepMC->GetEvent();
    edm::LogPrint("HepMCAnalyzer") << "\n particles #: " << mc->particles_size();

    if (dumpPDF_) {
      edm::LogPrint("HepMCAnalyzer") << "\n PDF info: " << mc->pdf_info();
    }

    if (dumpHepMC_) {
      mc->print(std::cout);
    }

    if (checkPDG_) {
      edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable;
      es.getData(fPDGTable);
      for (HepMC::GenEvent::particle_const_iterator part = mc->particles_begin(); part != mc->particles_end(); ++part) {
        const HepPDT::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID((*part)->pdg_id()));
        if (!PData) {
          edm::LogWarning("HepMCAnalyzer") << "Missing entry in particle table for PDG code = " << (*part)->pdg_id();
        }
      }
    }
  }
};

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DummyHepMCAnalyzer);

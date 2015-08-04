#ifndef GeneratorInterface_Core_EventVertexGenerator_h
#define GeneratorInterface_Core_EventVertexGenerator_h

namespace HepMC {
   class FourVector;
}

namespace edm {
  class ConsumesCollector;
  class Event;
  class EventSetup;
  class HepMCProduct;
  class LuminosityBlock;
  class ParameterSet;
  class Run;
}

namespace CLHEP {
  class HepRandomEngine;
}

class BaseEvtVtxGenerator {
public:

  explicit BaseEvtVtxGenerator();
  virtual ~BaseEvtVtxGenerator();

  BaseEvtVtxGenerator(BaseEvtVtxGenerator const&) = delete;
  BaseEvtVtxGenerator const& operator=(BaseEvtVtxGenerator const&) = delete;

  void generateNewVertex(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) {generateNewVertex_(product, engine);}
  void beginRun(edm::Run const& run, edm::EventSetup const& setup) {beginRun_(run, setup);}
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {beginLuminosityBlock_(lumi, setup);}

private:
  virtual void generateNewVertex_(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) = 0;
  virtual void beginRun_(edm::Run const&, edm::EventSetup const&) {}
  virtual void beginLuminosityBlock_(edm::LuminosityBlock const&, edm::EventSetup const&) {}
};

#endif

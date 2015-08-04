#ifndef GeneratorInterface_Core_EventVertexHelper_h
#define GeneratorInterface_Core_EventVertexHelper_h

#include <memory>

class BaseEvtVtxGenerator;

namespace edm {
  class ConsumesCollector;
  class Event;
  class EventSetup;
  class HepMCProduct;
  class LuminosityBlock;
  class ParameterSet;
  class Run;
}

class EventVertexHelper {
public:
   // ctor & dtor
   explicit EventVertexHelper(edm::ParameterSet const&, edm::ConsumesCollector&&);
   virtual ~EventVertexHelper();
      
  EventVertexHelper(EventVertexHelper const&) = delete;
  EventVertexHelper& operator=(EventVertexHelper const&) = delete;
  void smearVertex(edm::Event const&, edm::HepMCProduct&);
  void beginRun(edm::Run const& , edm::EventSetup const&);
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

private :

  std::unique_ptr<BaseEvtVtxGenerator> vertexGenerator_;
};

#endif

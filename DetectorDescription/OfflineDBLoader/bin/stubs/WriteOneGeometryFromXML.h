#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class WriteOneGeometryFromXML : public edm::EDAnalyzer {

 public:
  explicit WriteOneGeometryFromXML( const edm::ParameterSet& iConfig );
  ~WriteOneGeometryFromXML();
  virtual void beginRun( const edm::Run&, edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};

 private:
  std::string label_;
  int rotNumSeed_;
};

#ifndef CSCGeometryBuilder_DTRecoIdealDBLoader_h
#define CSCGeometryBuilder_DTRecoIdealDBLoader_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class CSCGeometry;

class DTRecoIdealDBLoader : public edm::EDAnalyzer {

 public:
  explicit DTRecoIdealDBLoader( const edm::ParameterSet& iConfig );
  ~DTRecoIdealDBLoader();
  virtual void beginJob() {};
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};

 private:
  std::string label_;
  int rotNumSeed_;
};

#endif

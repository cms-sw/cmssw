#ifndef CondTools_CSCRecoIdealDBLoader_h
#define CondTools_CSCRecoIdealDBLoader_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class CSCGeometry;

class CSCRecoIdealDBLoader : public edm::EDAnalyzer {

 public:
  explicit CSCRecoIdealDBLoader( const edm::ParameterSet& iConfig );
  ~CSCRecoIdealDBLoader();
  virtual void beginJob( edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};

 private:
  std::string label_;
  int rotNumSeed_;
};

#endif

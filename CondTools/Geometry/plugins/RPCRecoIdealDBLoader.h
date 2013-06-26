#ifndef CondTools_RPCRecoIdealDBLoader_h
#define CondTools_RPCRecoIdealDBLoader_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class RPCRecoIdealDBLoader : public edm::EDAnalyzer {

 public:
  explicit RPCRecoIdealDBLoader( const edm::ParameterSet& iConfig );
  ~RPCRecoIdealDBLoader();
  virtual void beginRun( const edm::Run&, edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};

 private:
  std::string label_;
  int rotNumSeed_;
};

#endif

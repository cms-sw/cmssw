#ifndef L1RCTProducer_h
#define L1RCTProducer_h 

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

class L1RCT;
class L1RCTLookupTables;

class L1RCTProducer : public edm::EDProducer
{
 public:
  explicit L1RCTProducer(const edm::ParameterSet& ps);
  virtual ~L1RCTProducer();
  virtual void beginJob(const edm::EventSetup& c);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
 private:
  L1RCT* rct;
  L1RCTLookupTables* rctLookupTables;
  edm::FileInPath src;
  bool orcaFileInput;
  edm::FileInPath lutFile;
  std::string rctTestInputFile;
  std::string rctTestOutputFile;
  bool patternTest;
  edm::FileInPath lutFile2;
  bool useEcal;
  bool useHcal;
  edm::InputTag ecalDigisLabel;
  edm::InputTag hcalDigisLabel;
};
#endif

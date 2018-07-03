#ifndef ElectronIDAnalyzer_h
#define ElectronIDAnalyzer_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ElectronIDAnalyzer : public edm::EDAnalyzer
{
 public:

  explicit ElectronIDAnalyzer(const edm::ParameterSet& conf);
  ~ElectronIDAnalyzer() override{};

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

 private:

  edm::ParameterSet conf_;

  std::string electronProducer_;

  std::string electronLabelRobustLoose_;
  std::string electronLabelRobustTight_;
  std::string electronLabelLoose_;
  std::string electronLabelTight_;

};

#endif

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
  virtual ~ElectronIDAnalyzer(){};

  virtual void beginJob(edm::EventSetup const& iSetup){};
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
   
 private:

  edm::ParameterSet conf_;

  std::string electronProducer_;
  std::string electronLabel_;
  std::string electronIDAssocProducer_;

};

#endif

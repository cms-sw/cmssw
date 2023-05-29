#ifndef RecoTBCalo_EcalTBHodoscopeReconstructor_EcalTBHodoscopeRawInfoDumper_HH
#define RecoTBCalo_EcalTBHodoscopeReconstructor_EcalTBHodoscopeRawInfoDumper_HH

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>
//#include "TTree.h"
#include "TH1.h"
#include "TGraph.h"
#include "TH2.h"
class EcalTBHodoscopeRawInfoDumper : public edm::one::EDAnalyzer<> {
public:
  explicit EcalTBHodoscopeRawInfoDumper(const edm::ParameterSet& ps);
  ~EcalTBHodoscopeRawInfoDumper() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  std::string rawInfoProducer_;    // name of module/plugin/producer making digis
  std::string rawInfoCollection_;  // secondary name given to collection of digis
  std::string rootfile_;
  TH1F* h_numberOfFiredHits_[4];
  TH1F* h_firedHits_[4];
};
#endif

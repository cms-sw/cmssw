#ifndef RecoExamples_SimpleJetDump_h
#define RecoExamples_SimpleJetDump_h
#include <TH1.h>
/* \class SimpleJetDump
 *
 * \author Robert Harris
 *
 * \version 1
 *
 */
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

class SimpleJetDump : public edm::one::EDAnalyzer<> {
public:
  SimpleJetDump(const edm::ParameterSet&);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  std::string CaloJetAlg, GenJetAlg;
  //Internal parameters
  int evtCount;
};

#endif

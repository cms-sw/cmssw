#ifndef DataFormats_ReadMath_h
#define DataFormats_ReadMath_h
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ReadMath : public edm::one::EDAnalyzer<> {
public:
  ReadMath(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&);
  edm::InputTag src;
};

#endif

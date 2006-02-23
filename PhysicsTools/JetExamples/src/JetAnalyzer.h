#ifndef JetExamples_JetAnalyzer_h
#define JetExamples_JetAnalyzer_h
// $Id: JetAnalyzer.h,v 1.1 2006/02/07 15:43:43 llista Exp $
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "TH1.h"
#include "TFile.h"

namespace edm {
  class ParameterSet;
}

class JetAnalyzer : public edm::EDAnalyzer {
public:
  JetAnalyzer( const edm::ParameterSet & );
  ~JetAnalyzer();

private:
  void analyze( const edm::Event& , const edm::EventSetup& );
  std::string src;
  TFile file;
  TH1F  jetPt;
};

#endif

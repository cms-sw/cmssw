#ifndef JetExamples_JetAnalyzer_h
#define JetExamples_JetAnalyzer_h
// $Id: JetAnalyzer.h,v 1.4 2005/12/12 03:52:23 llista Exp $
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
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

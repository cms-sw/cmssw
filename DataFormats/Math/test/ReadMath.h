#ifndef DataFormats_ReadMath_h
#define DataFormats_ReadMath_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>

class ReadMath : public edm::EDAnalyzer {
public:
  ReadMath( const edm::ParameterSet& );
private:
  void analyze( const edm::Event &, const edm::EventSetup & );
  std::string src;
};

#endif

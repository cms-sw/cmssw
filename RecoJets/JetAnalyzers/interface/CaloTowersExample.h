#ifndef RecoExamples_CaloTowersExample_h
#define RecoExamples_CaloTowersExample_h
#include <TH1.h>
/* \class CaloTowersExample
 *
 * \author Robert Harris
 *
 * \version 1
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"

class TFile;

class CaloTowersExample : public edm::EDAnalyzer {
public:
  CaloTowersExample( const edm::ParameterSet & );

private:
  void beginJob( );
  void analyze( const edm::Event& , const edm::EventSetup& );
  void endJob();
  std::string CaloTowersAlgorithm;
  TH1F h_et;
  TFile* m_file;
};

#endif

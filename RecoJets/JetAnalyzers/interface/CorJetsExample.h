#ifndef RecoExamples_CorJetsExample_h
#define RecoExamples_CorJetsExample_h
#include <TH1.h>
/* \class CorJetsExample
 *
 * \author Robert Harris
 *
 * \version 1
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"

class TFile;

class CorJetsExample : public edm::EDAnalyzer {
public:
  CorJetsExample( const edm::ParameterSet & );

private:
  void beginJob( const edm::EventSetup & );
  void analyze( const edm::Event& , const edm::EventSetup& );
  void endJob();
  std::string CaloJetAlgorithm, CorJetAlgorithm, JetCorrectionService, GenJetAlgorithm;
  TH1F h_ptCal, h_ptGen, h_ptCor, h_ptCorOnFly;
  TFile* m_file;
};

#endif

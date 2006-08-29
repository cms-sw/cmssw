#ifndef RecoExamples_JetPlotsExample_h
#define RecoExamples_JetPlotsExample_h
#include <TH1.h>
/* \class JetPlotsExample
 *
 * \author Robert Harris
 *
 * \version 1
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"

class JetPlotsExample : public edm::EDAnalyzer {
public:
  JetPlotsExample( const edm::ParameterSet & );

private:
  void beginJob( const edm::EventSetup & );
  void analyze( const edm::Event& , const edm::EventSetup& );
  void endJob();
  std::string CaloJetAlgorithm, GenJetAlgorithm;
  TH1F h_ptCal, h_etaCal, h_phiCal;
  TH1F h_ptGen, h_etaGen, h_phiGen;
  TFile* m_file;
};

#endif

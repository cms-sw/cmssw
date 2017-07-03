
#ifndef PyquenAnalyzer_H
#define PyquenAnalyzer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

// forward declarations
class TFile;
class TH1D;


class PyquenAnalyzer : public edm::EDAnalyzer
{ //analyzer module to analyze pythia events
 public:
  explicit PyquenAnalyzer(const edm::ParameterSet& );
  ~PyquenAnalyzer() override {} 

  void analyze(const edm::Event&, const edm::EventSetup& ) override;
  void beginJob() override;
  void endJob() override;

 private:
 
  TH1D*        phdNdEta;           // histogram for dN/deta
  TH1D*        phdNdY;             // histogram for dN/dy
  TH1D*        phdNdPt;            // histogram for dN/dpt
  TH1D*        phdNdPhi;           // histogram for dN/dphi
};

#endif


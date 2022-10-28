
#ifndef PyquenAnalyzer_H
#define PyquenAnalyzer_H

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

// forward declarations
class TFile;
class TH1D;

class PyquenAnalyzer : public edm::one::EDAnalyzer<> {  //analyzer module to analyze pythia events
public:
  explicit PyquenAnalyzer(const edm::ParameterSet&);
  virtual ~PyquenAnalyzer() {}

  void analyze(const edm::Event&, const edm::EventSetup&) final;
  void beginJob() final;
  void endJob() final;

  edm::EDGetTokenT<edm::HepMCProduct> srcT_;

private:
  TH1D* phdNdEta;  // histogram for dN/deta
  TH1D* phdNdY;    // histogram for dN/dy
  TH1D* phdNdPt;   // histogram for dN/dpt
  TH1D* phdNdPhi;  // histogram for dN/dphi
};

#endif

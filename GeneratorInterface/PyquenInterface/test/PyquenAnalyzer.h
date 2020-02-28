
#ifndef PyquenAnalyzer_H
#define PyquenAnalyzer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

// forward declarations
class TFile;
class TH1D;

class PyquenAnalyzer : public edm::EDAnalyzer {  //analyzer module to analyze pythia events
public:
  explicit PyquenAnalyzer(const edm::ParameterSet&);
  virtual ~PyquenAnalyzer() {}

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void endJob();

  edm::EDGetTokenT<edm::HepMCProduct> srcT_;

private:
  TH1D* phdNdEta;  // histogram for dN/deta
  TH1D* phdNdY;    // histogram for dN/dy
  TH1D* phdNdPt;   // histogram for dN/dpt
  TH1D* phdNdPhi;  // histogram for dN/dphi
};

#endif

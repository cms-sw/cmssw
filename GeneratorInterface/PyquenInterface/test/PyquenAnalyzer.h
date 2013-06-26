// $Id: PyquenAnalyzer.h,v 1.4 2009/12/15 12:25:55 fabiocos Exp $

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
  virtual ~PyquenAnalyzer() {} 

  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  virtual void beginJob();
  virtual void endJob();

 private:
 
  TH1D*        phdNdEta;           // histogram for dN/deta
  TH1D*        phdNdY;             // histogram for dN/dy
  TH1D*        phdNdPt;            // histogram for dN/dpt
  TH1D*        phdNdPhi;           // histogram for dN/dphi
};

#endif


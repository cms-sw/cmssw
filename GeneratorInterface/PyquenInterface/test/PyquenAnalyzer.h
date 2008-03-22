#ifndef PyquenAnalyzer_H
#define PyquenAnalyzer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

// forward declarations
class TFile;
class TH1D;


class PyquenAnalyzer : public edm::EDAnalyzer
{
 public:
  explicit PyquenAnalyzer(const edm::ParameterSet& );
  virtual ~PyquenAnalyzer() {} 

  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  virtual void beginJob(const edm::EventSetup& );
  virtual void endJob();

 private:
 
  std::string  sOutFileName;       // name of the output file

  TFile*       pfOutFile;          // output file
  TH1D*        phdNdEta;           // histogram for dN/deta
  TH1D*        phdNdY;             // histogram for dN/dy
  TH1D*        phdNdPt;            // histogram for dN/dpt
  TH1D*        phdNdPhi;           // histogram for dN/dphi

};

#endif
//module to analyze pythia events

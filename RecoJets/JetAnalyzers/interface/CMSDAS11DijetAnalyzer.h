// CMSDAS11DijetAnalyzer.cc
// Description: A basic dijet analyzer for the CMSDAS 2011
// Author: John Paul Chou
// Date: January 12, 2011

#ifndef __CMSDAS11_DIJET_ANALYZER_H__
#define __CMSDAS11_DIJET_ANALYZER_H__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>

class TH1D;

class CMSDAS11DijetAnalyzer : public edm::EDAnalyzer {
 public:
  CMSDAS11DijetAnalyzer(const edm::ParameterSet &);
  void analyze( const edm::Event& , const edm::EventSetup& );
  virtual ~CMSDAS11DijetAnalyzer() {}
  virtual void beginJob() {}
  virtual void endJob() {}

 private:

  // Parameters
  edm::InputTag jetSrc;
  edm::InputTag vertexSrc;
  std::string jetCorrections;

  // Histograms to be filled
  TH1D* hVertexZ;
  TH1D* hJetCorrPt;
};

#endif

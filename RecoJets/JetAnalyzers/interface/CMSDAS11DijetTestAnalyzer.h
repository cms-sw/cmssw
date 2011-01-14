// CMSDAS11DijetTestAnalyzer.h
// Description: A basic dijet analyzer for the CMSDAS 2011
// Author: John Paul Chou
// Date: January 12, 2011

#ifndef __CMSDAS11_DIJET_TEST_ANALYZER_H__
#define __CMSDAS11_DIJET_TEST_ANALYZER_H__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include <string>

class TH1D;

class CMSDAS11DijetTestAnalyzer : public edm::EDAnalyzer {
 public:
  CMSDAS11DijetTestAnalyzer(const edm::ParameterSet &);
  void analyze( const edm::Event& , const edm::EventSetup& );
  virtual ~CMSDAS11DijetTestAnalyzer() {}
  virtual void beginJob() {}
  virtual void endJob() {}
  
  static bool compare_JetPt(const reco::CaloJet& jet1, const reco::CaloJet& jet2) {
    return (jet1.pt() > jet2.pt() );
  }
 private:

  // Parameters
  edm::InputTag jetSrc;
  edm::InputTag vertexSrc;
  std::string jetCorrections;
  double innerDeltaEta;
  double outerDeltaEta;
  double JESbias;

  // Histograms to be filled
  TH1D* hVertexZ;
  TH1D* hJetCorrPt;
  TH1D* hJetRawPt;
  TH1D* hJetEta;
  TH1D* hJetPhi;
  TH1D* hJetEMF;

  TH1D* hRawDijetMass;
  TH1D* hCorDijetMass;
  TH1D* hJet1Pt;
  TH1D* hJet1Eta;
  TH1D* hJet1Phi;
  TH1D* hJet1EMF;
  TH1D* hJet2Pt;
  TH1D* hJet2Eta;
  TH1D* hJet2Phi;
  TH1D* hJet2EMF;
  TH1D* hDijetDeltaPhi;
  TH1D* hDijetDeltaEta;

  TH1D* hInnerDijetMass;
  TH1D* hOuterDijetMass;

};


#endif

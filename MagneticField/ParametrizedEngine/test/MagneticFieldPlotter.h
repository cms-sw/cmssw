// -*- C++ -*-
//
// Package:    MagneticFieldPlotter
// Class:      MagneticFieldPlotter
// 
/**\class MagneticFieldPlotter MagneticFieldPlotter.cc MyAnalyzers/MagneticFieldPlotter/src/MagneticFieldPlotter.cc

 Description: Plots Magnetic Field Components in the Tracker Volume

 Implementation:
     This Analyzer fills some histograms with the Magnetic Field components in the tracker volume. It's mainly aimed to look
     at differences between the Veikko parametrized field and the VolumeBased one.
*/
//
// Original Author:  Massimiliano Chiorboli
//         Created:  Mon Jun 11 17:20:15 CEST 2007
// $Id: MagneticFieldPlotter.h,v 1.2 2009/12/14 22:23:22 wmtan Exp $
//
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TGraph2D.h"
#include "TGraph.h"
#include "TH1.h"
#include "TFile.h"


class MagneticFieldPlotter : public edm::EDAnalyzer {
public:
  explicit MagneticFieldPlotter(const edm::ParameterSet&);
  ~MagneticFieldPlotter();
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  
  TH2D* gBz[5];
  TH2D* gBphi[5];
  TH2D* gBr[5];
  
  int nZstep;
  int nPhistep;
  float zHalfLength;
  
  const MagneticField* theMGField;
  std::string HistoFileName;
  TFile* theHistoFile;
  
};




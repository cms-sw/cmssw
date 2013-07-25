#ifndef TreeSplitter_HH
#define TreeSplitter_HH

// -*- C++ -*-
//
// Package:    TreeSplitter
// Class:      TreeSplitter
// 
/**\class TreeSplitter TreeSplitter.cc MuonAnalysis/MomentumScaleCalibration/plugins/TreeSplitter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Marco De Mattia
//         Created:  Thu Sep 11 12:16:00 CEST 2008
// $Id: TreeSplitter.h,v 1.1 2010/07/13 10:50:38 demattia Exp $
//

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <MuonAnalysis/MomentumScaleCalibration/interface/RootTreeHandler.h>

class TreeSplitter : public edm::EDAnalyzer
{
public:
  explicit TreeSplitter(const edm::ParameterSet&);
  ~TreeSplitter();

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) {};
  virtual void endJob();

  TString treeFileName_;
  TString outputFileName_;
  int32_t maxEvents_;
  uint32_t subSampleFirstEvent_;
  uint32_t subSampleMaxEvents_;
};

#endif // TREESPLITTER_HH

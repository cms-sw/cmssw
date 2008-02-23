#ifndef SiStripPerformanceSummaryBuilder_H
#define SiStripPerformanceSummaryBuilder_H
// -*- C++ -*-
// Package:    CondTools/SiStrip
// Class:      SiStripPerformanceSummaryBuilder
/**\class SiStripPerformanceSummaryBuilder SiStripPerformanceSummaryBuilder.cc CondTools/SiStrip/test/SiStripPerformanceSummaryBuilder.cc
 Description: <one line class summary>
 Implementation:
     <Notes on implementation>
*/
// Original Author:  Dorian Kcira
//         Created:  Mon Apr 30 17:46:00 CEST 2007
// $Id: SiStripPerformanceSummaryBuilder.h,v 1.1 2007/09/05 15:22:27 dkcira Exp $
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/SiStripObjects/interface/SiStripPerformanceSummary.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

class SiStripPerformanceSummaryBuilder : public edm::EDAnalyzer {
 public:
  explicit SiStripPerformanceSummaryBuilder(const edm::ParameterSet&);
  ~SiStripPerformanceSummaryBuilder(){};

 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  edm::FileInPath fp_;
  uint32_t printdebug_;
};
#endif

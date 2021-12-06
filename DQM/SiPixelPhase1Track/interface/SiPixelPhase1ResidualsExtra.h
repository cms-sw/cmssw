#ifndef SiPixelPhase1ResidualsExtra_SiPixelPhase1ResidualsExtra_h
#define SiPixelPhase1ResidualsExtra_SiPixelPhase1ResidualsExtra_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1ResidualsExtra
// Class  :     SiPixelPhase1ResidualsExtra
//
/*

 Description: Introduce some computation over the PixelPhase1 residuals distributions

 Usage:
    <usage>

*/
//
// Original Author:  Alessandro Rossi
//         Created:  21st May 2021
//

//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiPixelPhase1ResidualsExtra : public DQMEDHarvester {
public:
  explicit SiPixelPhase1ResidualsExtra(const edm::ParameterSet& conf);
  ~SiPixelPhase1ResidualsExtra() override;

protected:
  // BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  // EndJob
  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) override;

private:
  std::string topFolderName_;
  int minHits_;
  edm::ParameterSet conf_;

  std::map<std::string, MonitorElement*> residuals_;
  std::map<std::string, MonitorElement*> DRnR_;

  //Book Monitoring Elements
  void bookMEs(DQMStore::IBooker& iBooker);

  //Fill Monitoring Elements
  void fillMEs(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter);
};

#endif

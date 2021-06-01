#ifndef SiPixelPhase1Barycenter_SiPixelPhase1Barycenter_h
#define SiPixelPhase1Barycenter_SiPixelPhase1Barycenter_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1Barycenter
// Class  :     SiPixelPhase1Barycenter
//
/**

 Description: Barycenter plot generation for the Phase 1 pixel

 Usage:
    <usage>

*/
//
// Original Author:  Danilo Meuser
//         Created:  26th May 2021
//

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Alignment/interface/Alignments.h"

class SiPixelPhase1Barycenter : public DQMEDHarvester {
public:
  explicit SiPixelPhase1Barycenter(const edm::ParameterSet& conf);
  ~SiPixelPhase1Barycenter() override;

protected:
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  void dqmEndLuminosityBlock(DQMStore::IBooker& iBooker,
                             DQMStore::IGetter& iGetter,
                             edm::LuminosityBlock const& lumiSeg,
                             edm::EventSetup const& c) override;
  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) override;

private:  
  std::map<std::string, MonitorElement*> barycenters_;

  //book the barycenter histograms
  void bookBarycenterHistograms(DQMStore::IBooker& iBooker);
  
  //fill the barycenter histograms
  void fillBarycenterHistograms(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, const std::vector<AlignTransform>& input, const std::vector<AlignTransform>& GPR);

};

#endif

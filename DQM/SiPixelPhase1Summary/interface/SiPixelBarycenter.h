#ifndef DQM_SiPixelPhase1Summary_SiPixelBarycenter_h
#define DQM_SiPixelPhase1Summary_SiPixelBarycenter_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1Summary
// Class  :     SiPixelBarycenter
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

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

class SiPixelBarycenter : public DQMEDHarvester {
public:
  explicit SiPixelBarycenter(const edm::ParameterSet& conf);
  ~SiPixelBarycenter() override = default;

protected:
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  void dqmEndRun(DQMStore::IBooker& iBooker,
                 DQMStore::IGetter& iGetter,
                 edm::Run const& iRun,
                 edm::EventSetup const& c) override;
  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) override;

private:
  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> alignmentToken_;
  const edm::ESGetToken<Alignments, GlobalPositionRcd> gprToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;

  std::map<std::string, MonitorElement*> barycenters_;

  const std::array<std::string, 9> subdetectors_ = {
      {"BPIX", "FPIX_zm", "FPIX_zp", "BPIX_xp", "BPIX_xm", "FPIX_zp_xp", "FPIX_zm_xp", "FPIX_zp_xm", "FPIX_zm_xm"}};

  //book the barycenter histograms
  void bookBarycenterHistograms(DQMStore::IBooker& iBooker);

  //fill the barycenter histograms
  void fillBarycenterHistograms(DQMStore::IBooker& iBooker,
                                DQMStore::IGetter& iGetter,
                                const std::vector<AlignTransform>& input,
                                const std::vector<AlignTransform>& GPR,
                                const TrackerTopology& tTopo);
};

#endif

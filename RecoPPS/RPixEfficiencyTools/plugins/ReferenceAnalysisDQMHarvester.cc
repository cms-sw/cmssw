// -*- C++ -*-
//
// Package:    RecoPPS/RPixEfficiencyTools
// Class:      ReferenceAnalysisDQMHarvester
//
/**\class ReferenceAnalysisDQMHarvester ReferenceAnalysisDQMHarvester.cc
 RecoPPS/RPixEfficiencyTools/plugins/ReferenceAnalysisDQMHarvester.cc

 Description: [one line class summary]

 Implementation:
                 [Notes on implementation]
*/
//
// Original Author:  Andrea Bellora
//         Created:  Wed, 22 Aug 2017 09:55:05 GMT
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <exception>
#include <fstream>
#include <memory>
#include <set>

class ReferenceAnalysisDQMHarvester : public DQMEDHarvester {
public:
  explicit ReferenceAnalysisDQMHarvester(const edm::ParameterSet &);
  ~ReferenceAnalysisDQMHarvester();
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
  void beginRun(edm::Run const &run, edm::EventSetup const &eventSetup) override;

private:
  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geomEsToken_;
  std::vector<CTPPSPixelDetId> detids_;
};

ReferenceAnalysisDQMHarvester::ReferenceAnalysisDQMHarvester(const edm::ParameterSet &iConfig)
    : geomEsToken_(esConsumes<edm::Transition::BeginRun>()) {}

void ReferenceAnalysisDQMHarvester::beginRun(edm::Run const &run, edm::EventSetup const &eventSetup) {
  if (detids_.size() == 0) {  //first run
    const auto &geom = eventSetup.getData(geomEsToken_);
    for (auto it = geom.beginSensor(); it != geom.endSensor(); ++it) {
      if (!CTPPSPixelDetId::check(it->first))
        continue;
      const CTPPSPixelDetId detid(it->first);

      detids_.emplace_back(detid);
    }
  }
}
ReferenceAnalysisDQMHarvester::~ReferenceAnalysisDQMHarvester() {}

void ReferenceAnalysisDQMHarvester::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  igetter.cd();
  for (auto &rpId : detids_) {
    uint32_t arm = rpId.arm();
    uint32_t station = rpId.station();
    uint32_t rp = rpId.rp();
    std::string rpDirName = Form("Arm%i_st%i_rp%i", arm, station, rp);
    ibooker.cd(rpDirName.data());

    std::string numMonitorName_ = Form("h2RefinedTrackEfficiencyBuffer_arm%i_st%i_rp%i", arm, station, rp);
    std::string denMonitorName_ = Form("h2TrackHitDistribution_arm%i_st%i_rp%i", arm, station, rp);
    std::string resultName_ = Form("h2RefinedTrackEfficiency_arm%i_st%i_rp%i", arm, station, rp);

    MonitorElement *numerator = igetter.get(numMonitorName_);
    MonitorElement *denominator = igetter.get(denMonitorName_);
    MonitorElement *result = igetter.get(resultName_);

    result->divide(numerator, denominator, 1., 1., "B");
    result->getTH2D()->SetMaximum(1.);
    if (station == 0) {
      std::string numMonitorName_ = Form("h2RefinedTrackEfficiencyBuffer_rotated_arm%i_st%i_rp%i", arm, station, rp);
      std::string denMonitorName_ = Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp%i", arm, station, rp);
      std::string resultName_ = Form("h2RefinedTrackEfficiency_rotated_arm%i_st%i_rp%i", arm, station, rp);

      MonitorElement *numerator = igetter.get(numMonitorName_);
      MonitorElement *denominator = igetter.get(denMonitorName_);
      MonitorElement *result = igetter.get(resultName_);
      result->divide(numerator, denominator, 1., 1., "B");
      result->getTH2D()->SetMaximum(1.);
    }

    //EFFICIENCY VS XI
    //1
    numMonitorName_ = Form("h1EfficiencyVsXi_arm%i_st%i_rp%i", arm, station, rp);
    denMonitorName_ = Form("h1Xi_arm%i_st%i_rp%i", arm, station, rp);
    resultName_ = Form("h1EfficiencyVsXiFinal_arm%i_st%i_rp%i", arm, station, rp);

    numerator = igetter.get(numMonitorName_);
    denominator = igetter.get(denMonitorName_);
    result = igetter.get(resultName_);

    result->divide(numerator, denominator, 1., 1., "B");
    result->getTH1D()->SetMaximum(1.);

    //2
    numMonitorName_ = Form("h1EfficiencyVsTx_arm%i_st%i_rp%i", arm, station, rp);
    denMonitorName_ = Form("h1Tx_arm%i_st%i_rp%i", arm, station, rp);
    resultName_ = Form("h1EfficiencyVsTxFinal_arm%i_st%i_rp%i", arm, station, rp);

    numerator = igetter.get(numMonitorName_);
    denominator = igetter.get(denMonitorName_);
    result = igetter.get(resultName_);

    result->divide(numerator, denominator, 1., 1., "B");
    result->getTH1D()->SetMaximum(1.);

    //3
    numMonitorName_ = Form("h1EfficiencyVsTy_arm%i_st%i_rp%i", arm, station, rp);
    denMonitorName_ = Form("h1Ty_arm%i_st%i_rp%i", arm, station, rp);
    resultName_ = Form("h1EfficiencyVsTyFinal_arm%i_st%i_rp%i", arm, station, rp);

    numerator = igetter.get(numMonitorName_);
    denominator = igetter.get(denMonitorName_);
    result = igetter.get(resultName_);

    result->divide(numerator, denominator, 1., 1., "B");
    result->getTH1D()->SetMaximum(1.);
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(ReferenceAnalysisDQMHarvester);
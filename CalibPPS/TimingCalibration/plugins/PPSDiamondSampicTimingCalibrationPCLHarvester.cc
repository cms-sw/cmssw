// -*- C++ -*-
//
// Package:    CalibPPS/TimingCalibration/PPSDiamondSampicTimingCalibrationPCLHarvester
// Class:      PPSDiamondSampicTimingCalibrationPCLHarvester
//
/**\class PPSDiamondSampicTimingCalibrationPCLHarvester PPSDiamondSampicTimingCalibrationPCLHarvester.cc CalibPPS/TimingCalibration/PPSDiamondSampicTimingCalibrationPCLHarvester/plugins/PPSDiamondSampicTimingCalibrationPCLHarvester.cc

 Description: Harvester of the DiamondSampic calibration which produces sqlite file with a new channel alignment

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Misan
//         Created:  Mon, 26 Jul 2021 16:36:13 GMT
//
//
#include "TAxis.h"
#include "TH1.h"
#include "TArrayD.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "CalibPPS/TimingCalibration/interface/TimingCalibrationStruct.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"

namespace pt = boost::property_tree;

class PPSDiamondSampicTimingCalibrationPCLHarvester : public DQMEDHarvester {
public:
  PPSDiamondSampicTimingCalibrationPCLHarvester(const edm::ParameterSet&);
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  void calibJson(DQMStore::IGetter& iGetter);
  void calibDb(DQMStore::IGetter& iGetter);
  bool getDbSampicChannel(
      DQMStore::IGetter& iGetter, int& db, int& sampic, int& channel, std::string ch_name, CTPPSDiamondDetId detid);
  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geomEsToken_;
  edm::ESGetToken<PPSTimingCalibration, PPSTimingCalibrationRcd> timingCalibrationToken_;
  edm::ESHandle<PPSTimingCalibration> hTimingCalib_;
  std::vector<CTPPSDiamondDetId> detids_;
  const std::string dqmDir_;
  const unsigned int min_entries_;
  const std::string jsonCalibFile_, jsonOutputPath_;
};

//------------------------------------------------------------------------------

PPSDiamondSampicTimingCalibrationPCLHarvester::PPSDiamondSampicTimingCalibrationPCLHarvester(
    const edm::ParameterSet& iConfig)
    : geomEsToken_(esConsumes<edm::Transition::BeginRun>()),
      dqmDir_(iConfig.getParameter<std::string>("dqmDir")),
      min_entries_(iConfig.getParameter<unsigned int>("minEntries")),
      jsonCalibFile_(iConfig.getParameter<std::string>("jsonCalibFile")),
      jsonOutputPath_(iConfig.getParameter<std::string>("jsonOutputPath")) {
  if (jsonCalibFile_.empty())
    timingCalibrationToken_ = esConsumes<edm::Transition::BeginRun>(
        edm::ESInputTag(iConfig.getParameter<std::string>("timingCalibrationTag")));
}

//------------------------------------------------------------------------------

void PPSDiamondSampicTimingCalibrationPCLHarvester::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (jsonCalibFile_.empty())
    hTimingCalib_ = iSetup.getHandle(timingCalibrationToken_);
  const auto& geom = iSetup.getData(geomEsToken_);
  for (auto it = geom.beginSensor(); it != geom.endSensor(); ++it) {
    if (!CTPPSDiamondDetId::check(it->first))
      continue;
    const CTPPSDiamondDetId detid(it->first);
    detids_.emplace_back(detid);
  }
}

//------------------------------------------------------------------------------

bool PPSDiamondSampicTimingCalibrationPCLHarvester::getDbSampicChannel(
    DQMStore::IGetter& iGetter, int& db, int& sampic, int& channel, std::string path, CTPPSDiamondDetId detid) {
  auto histDb = iGetter.get(path + "db");
  auto histSampic = iGetter.get(path + "sampic");
  auto histChannel = iGetter.get(path + "channel");

  if (histDb == nullptr) {
    edm::LogWarning("PPSDiamondSampicTimingCalibrationPCLHarvester:dqmEndJob")
        << "Failed to retrieve db for detid: " << detid;
    return false;
  }

  if (histSampic == nullptr) {
    edm::LogWarning("PPSDiamondSampicTimingCalibrationPCLHarvester:dqmEndJob")
        << "Failed to retrieve sampic for detid: " << detid;
    return false;
  }

  if (histChannel == nullptr) {
    edm::LogWarning("PPSDiamondSampicTimingCalibrationPCLHarvester:dqmEndJob")
        << "Failed to retrieve channel hwId for detid: " << detid;
    return false;
  }

  db = histDb->getIntValue();
  sampic = histSampic->getIntValue();
  channel = histChannel->getIntValue();

  return true;
}
//------------------------------------------------------------------------------

void PPSDiamondSampicTimingCalibrationPCLHarvester::calibJson(DQMStore::IGetter& iGetter) {
  std::unordered_map<uint32_t, dqm::reco::MonitorElement*> timeHisto;
  std::string ch_name, path;

  pt::ptree node;
  pt::read_json(jsonCalibFile_, node);
  const std::string formula = node.get<std::string>("formula");

  for (const auto& detid : detids_) {
    detid.channelName(path, CTPPSDiamondDetId::nPath);
    detid.channelName(ch_name);
    path = dqmDir_ + "/" + path + "/" + ch_name;
    const auto chid = detid.rawId();

    int db, sampic, channel;
    if (!getDbSampicChannel(iGetter, db, sampic, channel, path, detid))
      continue;

    int ct = 0;
    for (pt::ptree::value_type& par : node.get_child("parameters." + std::to_string(db))) {
      double new_time_offset;
      if (ct == 16 * (1 - sampic) + channel) {  //flip the calibration - sampic 1 is first in json
        double old_time_offset = par.second.get<double>("time_offset");

        timeHisto[chid] = iGetter.get(path);

        if (timeHisto[chid] == nullptr) {
          edm::LogWarning("PPSDiamondSampicTimingCalibrationPCLHarvester:dqmEndJob")
              << "Failed to retrieve time monitor for detid" << detid;
          par.second.put<double>("time_offset", old_time_offset);
          continue;
        }

        if (min_entries_ > 0 && timeHisto[chid]->getEntries() < min_entries_) {
          edm::LogWarning("PPSDiamondSampicTimingCalibrationPCLHarvester:dqmEndJob")
              << "Not enough entries for channel (" << detid << "): " << timeHisto[chid]->getEntries() << " < "
              << min_entries_ << ". Skipping calibration.";
          par.second.put<double>("time_offset", old_time_offset);
          continue;
        }
        new_time_offset = old_time_offset - timeHisto[chid]->getMean();
        //scale x axis of the plots by calculated offset
        timeHisto[chid]->getTH1F()->GetXaxis()->Set(
            timeHisto[chid]->getTH1F()->GetXaxis()->GetNbins(),
            timeHisto[chid]->getTH1F()->GetXaxis()->GetXmin() + new_time_offset,   // new Xmin
            timeHisto[chid]->getTH1F()->GetXaxis()->GetXmax() + new_time_offset);  // new Xmax
        timeHisto[chid]->getTH1F()->ResetStats();

        par.second.put<double>("time_offset", new_time_offset);
        break;
      }
      ct++;
    }
  }
  pt::write_json(jsonOutputPath_, node);
}

//------------------------------------------------------------------------------

void PPSDiamondSampicTimingCalibrationPCLHarvester::calibDb(DQMStore::IGetter& iGetter) {
  PPSTimingCalibration calib = *hTimingCalib_;

  // book the parameters containers
  PPSTimingCalibration::ParametersMap params;
  PPSTimingCalibration::TimingMap time_info;

  std::unordered_map<uint32_t, dqm::reco::MonitorElement*> timeHisto;
  std::string rp_name, plane_name, ch_name, path;
  const std::string& formula = calib.formula();

  for (const auto& detid : detids_) {
    detid.channelName(path, CTPPSDiamondDetId::nPath);
    detid.channelName(ch_name);
    path = dqmDir_ + "/" + path + "/" + ch_name;
    const auto chid = detid.rawId();

    int db, sampic, channel;
    if (!getDbSampicChannel(iGetter, db, sampic, channel, path, detid))
      continue;

    PPSTimingCalibration::Key key;
    key.db = db;
    key.sampic = sampic;
    key.channel = channel;

    double timeOffset = calib.timeOffset(db, sampic, channel);
    double timePrecision = calib.timePrecision(db, sampic, channel);
    if (timeOffset == 0 && timePrecision == 0) {
      edm::LogWarning("PPSDiamondSampicTimingCalibrationPCLHarvester:dqmEndJob")
          << "No calibration found for db: " << db << " sampic: " << sampic << " channel: " << channel;
      continue;
    }

    int cell_ct = 0;
    while (!calib.parameters(db, sampic, channel, cell_ct).empty()) {
      auto parameters = calib.parameters(db, sampic, channel, cell_ct);
      key.cell = cell_ct;
      params[key] = parameters;
      cell_ct++;
    }

    key.cell = -1;

    time_info[key] = {timeOffset, timePrecision};
    timeHisto[chid] = iGetter.get(path);
    if (timeHisto[chid] == nullptr) {
      edm::LogWarning("PPSDiamondSampicTimingCalibrationPCLHarvester:dqmEndJob")
          << "Failed to retrieve time monitor for detid: " << detid;
      continue;
    }

    if (min_entries_ > 0 && timeHisto[chid]->getEntries() < min_entries_) {
      edm::LogInfo("PPSDiamondSampicTimingCalibrationPCLHarvester:dqmEndJob")
          << "Not enough entries (" << detid << "): " << timeHisto[chid]->getEntries() << " < " << min_entries_
          << ". Skipping calibration.";
      continue;
    }

    double new_time_offset = timeOffset - timeHisto[chid]->getMean();
    //scale x axis of the plots by calculated offset
    timeHisto[chid]->getTH1F()->GetXaxis()->Set(
        timeHisto[chid]->getTH1F()->GetXaxis()->GetNbins(),
        timeHisto[chid]->getTH1F()->GetXaxis()->GetXmin() + new_time_offset,   // new Xmin
        timeHisto[chid]->getTH1F()->GetXaxis()->GetXmax() + new_time_offset);  // new Xmax
    timeHisto[chid]->getTH1F()->ResetStats();

    time_info[key] = {new_time_offset, timePrecision};
  }

  auto calibPPS = PPSTimingCalibration(formula, params, time_info);
  // write the object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  poolDbService->writeOneIOV(calibPPS, poolDbService->currentTime(), "PPSTimingCalibrationRcd_SAMPIC");
}

//------------------------------------------------------------------------------

void PPSDiamondSampicTimingCalibrationPCLHarvester::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  iBooker.cd();
  iBooker.setCurrentFolder(dqmDir_);
  if (jsonCalibFile_.empty())
    calibDb(iGetter);
  else
    calibJson(iGetter);
}

//------------------------------------------------------------------------------

void PPSDiamondSampicTimingCalibrationPCLHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("timingCalibrationTag", "GlobalTag:PPSDiamondSampicCalibration")
      ->setComment("input tag for timing calibration retrieval");
  desc.add<std::string>("dqmDir", "AlCaReco/PPSDiamondSampicTimingCalibrationPCL")
      ->setComment("input path for the various DQM plots");
  desc.add<unsigned int>("minEntries", 1)->setComment("minimal number of hits to extract calibration");
  desc.add<std::string>("jsonCalibFile", "")
      ->setComment(
          "input path for json file containing calibration, if none, calibration will be obtained from db instead");
  desc.add<std::string>("jsonOutputPath", "offset_cal.json")->setComment("output path for the new json calibration");
  descriptions.add("PPSDiamondSampicTimingCalibrationPCLHarvester", desc);
}

DEFINE_FWK_MODULE(PPSDiamondSampicTimingCalibrationPCLHarvester);

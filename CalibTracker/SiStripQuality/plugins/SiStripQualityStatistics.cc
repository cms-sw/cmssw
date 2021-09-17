// -*- C++ -*-
//
// Package:    SiStripQualityStatistics
// Class:      SiStripQualityStatistics
//
/**\class SiStripQualityStatistics SiStripQualityStatistics.h CalibTracker/SiStripQuality/plugins/SiStripQualityStatistics.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 12:11:10 CEST 2007
//
//

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibTracker/SiStripQuality/plugins/SiStripQualityStatistics.h"

SiStripQualityStatistics::SiStripQualityStatistics(const edm::ParameterSet& iConfig)
    : TkMapFileName_(iConfig.getUntrackedParameter<std::string>("TkMapFileName", "")),
      saveTkHistoMap_(iConfig.getUntrackedParameter<bool>("SaveTkHistoMap", true)),
      tkMap(nullptr),
      tkMapFullIOVs(nullptr),
      tTopoToken_(esConsumes<edm::Transition::EndRun>()),
      tkDetMapToken_(esConsumes<edm::Transition::EndRun>()),
      withFedErrHelper_{iConfig, consumesCollector(), true} {
  detInfo_ = SiStripDetInfoFileReader::read(
      iConfig.getUntrackedParameter<edm::FileInPath>("file", edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))
          .fullPath());

  tkMapFullIOVs = new TrackerMap("BadComponents");
  tkhisto = nullptr;
}

void SiStripQualityStatistics::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("TkMapFileName", "");
  desc.addUntracked<bool>("SaveTkHistoMap", true);
  desc.addUntracked<edm::FileInPath>("file", edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile));
  SiStripQualityWithFromFedErrorsHelper::fillDescription(desc);
  descriptions.add("siStripQualityStatistics", desc);
}

void SiStripQualityStatistics::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  if (withFedErrHelper_.addBadCompFromFedErr()) {
    updateAndSave(&withFedErrHelper_.getMergedQuality(getter));
  }
  std::string filename = TkMapFileName_;
  if (!filename.empty()) {
    tkMapFullIOVs->save(false, 0, 0, filename);
    filename.erase(filename.begin() + filename.find('.'), filename.end());
    tkMapFullIOVs->print(false, 0, 0, filename);

    if (saveTkHistoMap_) {
      tkhisto->save(filename + ".root");
      tkhisto->saveAsCanvas(filename + "_Canvas.root", "E");
    }
  }
}

void SiStripQualityStatistics::endRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  tTopo_ = std::make_unique<TrackerTopology>(iSetup.getData(tTopoToken_));
  if ((!tkhisto) && (!TkMapFileName_.empty())) {
    //here the baseline (the value of the empty,not assigned bins) is put to -1 (default is zero)
    tkhisto = std::make_unique<TkHistoMap>(&iSetup.getData(tkDetMapToken_), "BadComp", "BadComp", -1.);
  }

  if (withFedErrHelper_.endRun(iSetup) && !withFedErrHelper_.addBadCompFromFedErr()) {
    run_ = run.id();
    updateAndSave(&iSetup.getData(withFedErrHelper_.qualityToken()));
  }
}

void SiStripQualityStatistics::updateAndSave(const SiStripQuality* siStripQuality) {
  for (int i = 0; i < 4; ++i) {
    NTkBadComponent[i] = 0;
    for (int j = 0; j < 19; ++j) {
      ssV[i][j].str("");
      for (int k = 0; k < 4; ++k)
        NBadComponent[i][j][k] = 0;
    }
  }

  if (tkMap)
    delete tkMap;
  tkMap = new TrackerMap("BadComponents");

  std::stringstream ss;
  std::vector<uint32_t> detids = detInfo_.getAllDetIds();
  std::vector<uint32_t>::const_iterator idet = detids.begin();
  for (; idet != detids.end(); ++idet) {
    ss << "detid " << (*idet) << " IsModuleUsable " << siStripQuality->IsModuleUsable((*idet)) << "\n";
    if (siStripQuality->IsModuleUsable((*idet)))
      tkMap->fillc(*idet, 0x00ff00);
  }
  LogDebug("SiStripQualityStatistics") << ss.str() << std::endl;

  std::vector<SiStripQuality::BadComponent> BC = siStripQuality->getBadComponentList();

  for (size_t i = 0; i < BC.size(); ++i) {
    //&&&&&&&&&&&&&
    //Full Tk
    //&&&&&&&&&&&&&

    if (BC[i].BadModule)
      NTkBadComponent[0]++;
    if (BC[i].BadFibers)
      NTkBadComponent[1] += ((BC[i].BadFibers >> 2) & 0x1) + ((BC[i].BadFibers >> 1) & 0x1) + ((BC[i].BadFibers) & 0x1);
    if (BC[i].BadApvs)
      NTkBadComponent[2] += ((BC[i].BadApvs >> 5) & 0x1) + ((BC[i].BadApvs >> 4) & 0x1) + ((BC[i].BadApvs >> 3) & 0x1) +
                            ((BC[i].BadApvs >> 2) & 0x1) + ((BC[i].BadApvs >> 1) & 0x1) + ((BC[i].BadApvs) & 0x1);

    //&&&&&&&&&&&&&&&&&
    //Single SubSyste
    //&&&&&&&&&&&&&&&&&
    int component;
    DetId detectorId = DetId(BC[i].detid);
    int subDet = detectorId.subdetId();
    if (subDet == StripSubdetector::TIB) {
      //&&&&&&&&&&&&&&&&&
      //TIB
      //&&&&&&&&&&&&&&&&&

      component = tTopo_->tibLayer(BC[i].detid);
      SetBadComponents(0, component, BC[i]);

    } else if (subDet == StripSubdetector::TID) {
      //&&&&&&&&&&&&&&&&&
      //TID
      //&&&&&&&&&&&&&&&&&

      component = tTopo_->tidSide(BC[i].detid) == 2 ? tTopo_->tidWheel(BC[i].detid) : tTopo_->tidWheel(BC[i].detid) + 3;
      SetBadComponents(1, component, BC[i]);

    } else if (subDet == StripSubdetector::TOB) {
      //&&&&&&&&&&&&&&&&&
      //TOB
      //&&&&&&&&&&&&&&&&&

      component = tTopo_->tobLayer(BC[i].detid);
      SetBadComponents(2, component, BC[i]);

    } else if (subDet == StripSubdetector::TEC) {
      //&&&&&&&&&&&&&&&&&
      //TEC
      //&&&&&&&&&&&&&&&&&

      component = tTopo_->tecSide(BC[i].detid) == 2 ? tTopo_->tecWheel(BC[i].detid) : tTopo_->tecWheel(BC[i].detid) + 9;
      SetBadComponents(3, component, BC[i]);
    }
  }

  //&&&&&&&&&&&&&&&&&&
  // Single Strip Info
  //&&&&&&&&&&&&&&&&&&
  float percentage = 0;

  SiStripQuality::RegistryIterator rbegin = siStripQuality->getRegistryVectorBegin();
  SiStripQuality::RegistryIterator rend = siStripQuality->getRegistryVectorEnd();

  for (SiStripBadStrip::RegistryIterator rp = rbegin; rp != rend; ++rp) {
    uint32_t detid = rp->detid;

    int subdet = -999;
    int component = -999;
    DetId detectorId = DetId(detid);
    int subDet = detectorId.subdetId();
    if (subDet == StripSubdetector::TIB) {
      subdet = 0;
      component = tTopo_->tibLayer(detid);
    } else if (subDet == StripSubdetector::TID) {
      subdet = 1;
      component = tTopo_->tidSide(detid) == 2 ? tTopo_->tidWheel(detid) : tTopo_->tidWheel(detid) + 3;
    } else if (subDet == StripSubdetector::TOB) {
      subdet = 2;
      component = tTopo_->tobLayer(detid);
    } else if (subDet == StripSubdetector::TEC) {
      subdet = 3;
      component = tTopo_->tecSide(detid) == 2 ? tTopo_->tecWheel(detid) : tTopo_->tecWheel(detid) + 9;
    }

    SiStripQuality::Range sqrange = SiStripQuality::Range(siStripQuality->getDataVectorBegin() + rp->ibegin,
                                                          siStripQuality->getDataVectorBegin() + rp->iend);

    percentage = 0;
    for (int it = 0; it < sqrange.second - sqrange.first; it++) {
      unsigned int range = siStripQuality->decode(*(sqrange.first + it)).range;
      NTkBadComponent[3] += range;
      NBadComponent[subdet][0][3] += range;
      NBadComponent[subdet][component][3] += range;
      percentage += range;
    }
    if (percentage != 0)
      percentage /= 128. * detInfo_.getNumberOfApvsAndStripLength(detid).first;
    if (percentage > 1)
      edm::LogError("SiStripQualityStatistics") << "PROBLEM detid " << detid << " value " << percentage << std::endl;

    //------- Global Statistics on percentage of bad components along the IOVs ------//
    tkMapFullIOVs->fill(detid, percentage);
    if (tkhisto != nullptr)
      tkhisto->fill(detid, percentage);
  }

  //&&&&&&&&&&&&&&&&&&
  // printout
  //&&&&&&&&&&&&&&&&&&

  ss.str("");
  ss << "\n-----------------\nNew IOV starting from run " << run_.run() << "\n-----------------\n";
  ss << "\n-----------------\nGlobal Info\n-----------------";
  ss << "\nBadComponent \t   Modules \tFibers "
        "\tApvs\tStrips\n----------------------------------------------------------------";
  ss << "\nTracker:\t\t" << NTkBadComponent[0] << "\t" << NTkBadComponent[1] << "\t" << NTkBadComponent[2] << "\t"
     << NTkBadComponent[3];
  ss << "\n";
  ss << "\nTIB:\t\t\t" << NBadComponent[0][0][0] << "\t" << NBadComponent[0][0][1] << "\t" << NBadComponent[0][0][2]
     << "\t" << NBadComponent[0][0][3];
  ss << "\nTID:\t\t\t" << NBadComponent[1][0][0] << "\t" << NBadComponent[1][0][1] << "\t" << NBadComponent[1][0][2]
     << "\t" << NBadComponent[1][0][3];
  ss << "\nTOB:\t\t\t" << NBadComponent[2][0][0] << "\t" << NBadComponent[2][0][1] << "\t" << NBadComponent[2][0][2]
     << "\t" << NBadComponent[2][0][3];
  ss << "\nTEC:\t\t\t" << NBadComponent[3][0][0] << "\t" << NBadComponent[3][0][1] << "\t" << NBadComponent[3][0][2]
     << "\t" << NBadComponent[3][0][3];
  ss << "\n";

  for (int i = 1; i < 5; ++i)
    ss << "\nTIB Layer " << i << " :\t\t" << NBadComponent[0][i][0] << "\t" << NBadComponent[0][i][1] << "\t"
       << NBadComponent[0][i][2] << "\t" << NBadComponent[0][i][3];
  ss << "\n";
  for (int i = 1; i < 4; ++i)
    ss << "\nTID+ Disk " << i << " :\t\t" << NBadComponent[1][i][0] << "\t" << NBadComponent[1][i][1] << "\t"
       << NBadComponent[1][i][2] << "\t" << NBadComponent[1][i][3];
  for (int i = 4; i < 7; ++i)
    ss << "\nTID- Disk " << i - 3 << " :\t\t" << NBadComponent[1][i][0] << "\t" << NBadComponent[1][i][1] << "\t"
       << NBadComponent[1][i][2] << "\t" << NBadComponent[1][i][3];
  ss << "\n";
  for (int i = 1; i < 7; ++i)
    ss << "\nTOB Layer " << i << " :\t\t" << NBadComponent[2][i][0] << "\t" << NBadComponent[2][i][1] << "\t"
       << NBadComponent[2][i][2] << "\t" << NBadComponent[2][i][3];
  ss << "\n";
  for (int i = 1; i < 10; ++i)
    ss << "\nTEC+ Disk " << i << " :\t\t" << NBadComponent[3][i][0] << "\t" << NBadComponent[3][i][1] << "\t"
       << NBadComponent[3][i][2] << "\t" << NBadComponent[3][i][3];
  for (int i = 10; i < 19; ++i)
    ss << "\nTEC- Disk " << i - 9 << " :\t\t" << NBadComponent[3][i][0] << "\t" << NBadComponent[3][i][1] << "\t"
       << NBadComponent[3][i][2] << "\t" << NBadComponent[3][i][3];
  ss << "\n";

  ss << "\n----------------------------------------------------------------\n\t\t   Detid  \tModules Fibers "
        "Apvs\n----------------------------------------------------------------";
  for (int i = 1; i < 5; ++i)
    ss << "\nTIB Layer " << i << " :" << ssV[0][i].str();
  ss << "\n";
  for (int i = 1; i < 4; ++i)
    ss << "\nTID+ Disk " << i << " :" << ssV[1][i].str();
  for (int i = 4; i < 7; ++i)
    ss << "\nTID- Disk " << i - 3 << " :" << ssV[1][i].str();
  ss << "\n";
  for (int i = 1; i < 7; ++i)
    ss << "\nTOB Layer " << i << " :" << ssV[2][i].str();
  ss << "\n";
  for (int i = 1; i < 10; ++i)
    ss << "\nTEC+ Disk " << i << " :" << ssV[3][i].str();
  for (int i = 10; i < 19; ++i)
    ss << "\nTEC- Disk " << i - 9 << " :" << ssV[3][i].str();

  edm::LogInfo("SiStripQualityStatistics") << ss.str() << std::endl;

  std::string filename = TkMapFileName_;
  std::stringstream sRun;
  sRun.str("");
  sRun << "_Run_" << std::setw(6) << std::setfill('0') << run_.run() << std::setw(0);

  if (!filename.empty()) {
    filename.insert(filename.find('.'), sRun.str());
    tkMap->save(true, 0, 0, filename);
    filename.erase(filename.begin() + filename.find('.'), filename.end());
    tkMap->print(true, 0, 0, filename);
  }
}

void SiStripQualityStatistics::SetBadComponents(int i, int component, SiStripQuality::BadComponent& BC) {
  int napv = detInfo_.getNumberOfApvsAndStripLength(BC.detid).first;

  ssV[i][component] << "\n\t\t " << BC.detid << " \t " << BC.BadModule << " \t " << ((BC.BadFibers) & 0x1) << " ";
  if (napv == 4)
    ssV[i][component] << "x " << ((BC.BadFibers >> 1) & 0x1);

  if (napv == 6)
    ssV[i][component] << ((BC.BadFibers >> 1) & 0x1) << " " << ((BC.BadFibers >> 2) & 0x1);
  ssV[i][component] << " \t " << ((BC.BadApvs) & 0x1) << " " << ((BC.BadApvs >> 1) & 0x1) << " ";
  if (napv == 4)
    ssV[i][component] << "x x " << ((BC.BadApvs >> 2) & 0x1) << " " << ((BC.BadApvs >> 3) & 0x1);
  if (napv == 6)
    ssV[i][component] << ((BC.BadApvs >> 2) & 0x1) << " " << ((BC.BadApvs >> 3) & 0x1) << " "
                      << ((BC.BadApvs >> 4) & 0x1) << " " << ((BC.BadApvs >> 5) & 0x1) << " ";

  if (BC.BadApvs) {
    NBadComponent[i][0][2] += ((BC.BadApvs >> 5) & 0x1) + ((BC.BadApvs >> 4) & 0x1) + ((BC.BadApvs >> 3) & 0x1) +
                              ((BC.BadApvs >> 2) & 0x1) + ((BC.BadApvs >> 1) & 0x1) + ((BC.BadApvs) & 0x1);
    NBadComponent[i][component][2] += ((BC.BadApvs >> 5) & 0x1) + ((BC.BadApvs >> 4) & 0x1) +
                                      ((BC.BadApvs >> 3) & 0x1) + ((BC.BadApvs >> 2) & 0x1) +
                                      ((BC.BadApvs >> 1) & 0x1) + ((BC.BadApvs) & 0x1);
    tkMap->fillc(BC.detid, 0xff0000);
  }
  if (BC.BadFibers) {
    NBadComponent[i][0][1] += ((BC.BadFibers >> 2) & 0x1) + ((BC.BadFibers >> 1) & 0x1) + ((BC.BadFibers) & 0x1);
    NBadComponent[i][component][1] +=
        ((BC.BadFibers >> 2) & 0x1) + ((BC.BadFibers >> 1) & 0x1) + ((BC.BadFibers) & 0x1);
    tkMap->fillc(BC.detid, 0x0000ff);
  }
  if (BC.BadModule) {
    NBadComponent[i][0][0]++;
    NBadComponent[i][component][0]++;
    tkMap->fillc(BC.detid, 0x0);
  }
}

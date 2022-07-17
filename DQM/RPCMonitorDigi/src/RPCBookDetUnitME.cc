#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>
#include <DQM/RPCMonitorClient/interface/RPCBookFolderStructure.h>
#include <DQM/RPCMonitorClient/interface/RPCRollMapHisto.h>
#include <DQM/RPCMonitorClient/interface/RPCNameHelper.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>

void RPCMonitorDigi::bookRollME(DQMStore::IBooker& ibooker,
                                const RPCDetId& detId,
                                const RPCGeometry* rpcGeo,
                                const std::string& recHitType,
                                std::map<std::string, MonitorElement*>& meMap) {
  ibooker.setCurrentFolder(
      fmt::format("{}/{}/{}", subsystemFolder_, recHitType, RPCBookFolderStructure::folderStructure(detId)));

  //get number of strips in current roll
  int nstrips = this->stripsInRoll(detId, rpcGeo);
  if (nstrips == 0) {
    nstrips = 1;
  }

  /// Name components common to current RPCDetId
  const std::string nameRoll = RPCNameHelper::name(detId, useRollInfo_);

  if (!RPCMonitorDigi::useRollInfo_) {
    if (detId.region() != 0 ||                                                     //Endcaps
        (abs(detId.ring()) == 2 && detId.station() == 2 && detId.layer() != 1) ||  //Wheel -/+2 RB2out
        (abs(detId.ring()) != 2 && detId.station() == 2 && detId.layer() == 1)) {
      nstrips *= 3;
    }  //Wheel -1,0,+1 RB2in
    else {
      nstrips *= 2;
    }
  }

  std::string tmpStr;

  tmpStr = "Occupancy_" + nameRoll;
  meMap[tmpStr] = ibooker.book1D(tmpStr, tmpStr, nstrips, 0.5, nstrips + 0.5);

  tmpStr = "BXDistribution_" + nameRoll;
  meMap[tmpStr] = ibooker.book1D(tmpStr, tmpStr, 7, -3.5, 3.5);

  if (detId.region() == 0) {
    tmpStr = "Multiplicity_" + nameRoll;
    meMap[tmpStr] = ibooker.book1D(tmpStr, tmpStr, 30, 0.5, 30.5);

  } else {
    tmpStr = "Multiplicity_" + nameRoll;
    meMap[tmpStr] = ibooker.book1D(tmpStr, tmpStr, 15, 0.5, 15.5);
  }

  tmpStr = "NumberOfClusters_" + nameRoll;
  meMap[tmpStr] = ibooker.book1D(tmpStr, tmpStr, 10, 0.5, 10.5);
}

void RPCMonitorDigi::bookSectorRingME(DQMStore::IBooker& ibooker,
                                      const std::string& recHitType,
                                      std::map<std::string, MonitorElement*>& meMap) {
  for (int wheel = -2; wheel <= 2; wheel++) {
    ibooker.setCurrentFolder(
        fmt::format("{}/{}/Barrel/Wheel_{}/SummaryBySectors", subsystemFolder_, recHitType, wheel));

    for (int sector = 1; sector <= 12; sector++) {
      const std::string meName = fmt::format("Occupancy_Wheel_{}_Sector_{}", wheel, sector);
      const std::string meClus = fmt::format("ClusterSize_Wheel_{}_Sector_{}", wheel, sector);

      if (sector == 9 || sector == 11) {
        meMap[meName] = ibooker.book2D(meName, meName, 91, 0.5, 91.5, 15, 0.5, 15.5);
        meMap[meClus] = ibooker.book2D(meClus, meClus, 16, 1, 17, 15, 0.5, 15.5);
      } else if (sector == 4) {
        meMap[meName] = ibooker.book2D(meName, meName, 91, 0.5, 91.5, 21, 0.5, 21.5);
        meMap[meClus] = ibooker.book2D(meClus, meClus, 16, 1, 17, 21, 0.5, 21.5);
      } else {
        meMap[meName] = ibooker.book2D(meName, meName, 91, 0.5, 91.5, 17, 0.5, 17.5);
        meMap[meClus] = ibooker.book2D(meClus, meClus, 16, 1, 17, 17, 0.5, 17.5);
      }

      meMap[meName]->setAxisTitle("strip", 1);
      RPCRollMapHisto::setBarrelRollAxis(meMap[meName], wheel, 2, true);

      meMap[meClus]->setAxisTitle("Cluster size", 1);
      meMap[meClus]->setBinLabel(1, "1", 1);
      meMap[meClus]->setBinLabel(5, "5", 1);
      meMap[meClus]->setBinLabel(10, "10", 1);
      meMap[meClus]->setBinLabel(15, "15", 1);
      meMap[meClus]->setBinLabel(meMap[meClus]->getNbinsX(), "Overflow", 1);
      RPCRollMapHisto::setBarrelRollAxis(meMap[meClus], wheel, 2, true);
    }
  }

  for (int region = -1; region <= 1; region++) {
    if (region == 0)
      continue;

    std::string regionName = "Endcap-";
    if (region == 1)
      regionName = "Endcap+";

    for (int disk = 1; disk <= RPCMonitorDigi::numberOfDisks_; disk++) {
      ibooker.setCurrentFolder(
          fmt::format("{}/{}/{}/Disk_{}/SummaryByRings/", subsystemFolder_, recHitType, regionName, region * disk));

      for (int ring = RPCMonitorDigi::numberOfInnerRings_; ring <= 3; ring++) {
        //Occupancy
        const std::string meName1 = fmt::format("Occupancy_Disk_{}_Ring_{}_CH01-CH18", (region * disk), ring);

        auto me1 = ibooker.book2D(meName1, meName1, 96, 0.5, 96.5, 18, 0.5, 18.5);
        me1->setAxisTitle("strip", 1);

        for (int i = 1; i <= 18; i++) {
          const std::string ylabel = fmt::format("R{}_CH{:02d}", ring, i);
          me1->setBinLabel(i, ylabel, 2);
        }

        me1->setBinLabel(1, "1", 1);
        me1->setBinLabel(16, "RollA", 1);
        me1->setBinLabel(32, "32", 1);
        me1->setBinLabel(33, "1", 1);
        me1->setBinLabel(48, "RollB", 1);
        me1->setBinLabel(64, "32", 1);
        me1->setBinLabel(65, "1", 1);
        me1->setBinLabel(80, "RollC", 1);
        me1->setBinLabel(96, "32", 1);

        const std::string meName2 = fmt::format("Occupancy_Disk_{}_Ring_{}_CH19-CH36", (region * disk), ring);

        auto me2 = ibooker.book2D(meName2, meName2, 96, 0.5, 96.5, 18, 18.5, 36.5);
        me2->setAxisTitle("strip", 1);

        for (int i = 1; i <= 18; i++) {
          const std::string ylabel = fmt::format("R{}_CH{:02d}", ring, i + 18);
          me2->setBinLabel(i, ylabel, 2);
        }

        me2->setBinLabel(1, "1", 1);
        me2->setBinLabel(16, "RollA", 1);
        me2->setBinLabel(32, "32", 1);
        me2->setBinLabel(33, "1", 1);
        me2->setBinLabel(48, "RollB", 1);
        me2->setBinLabel(64, "32", 1);
        me2->setBinLabel(65, "1", 1);
        me2->setBinLabel(80, "RollC", 1);
        me2->setBinLabel(96, "32", 1);

        meMap[meName1] = me1;
        meMap[meName2] = me2;

        //Cluster size
        const std::string meClus1 = fmt::format("ClusterSize_Disk_{}_Ring_{}_CH01-CH09", (region * disk), ring);
        const std::string meClus2 = fmt::format("ClusterSize_Disk_{}_Ring_{}_CH10-CH18", (region * disk), ring);
        const std::string meClus3 = fmt::format("ClusterSize_Disk_{}_Ring_{}_CH19-CH27", (region * disk), ring);
        const std::string meClus4 = fmt::format("ClusterSize_Disk_{}_Ring_{}_CH28-CH36", (region * disk), ring);

        auto mecl1 = ibooker.book2D(meClus1, meClus1, 11, 1, 12, 27, 0.5, 27.5);
        auto mecl2 = ibooker.book2D(meClus2, meClus2, 11, 1, 12, 27, 27.5, 54.5);
        auto mecl3 = ibooker.book2D(meClus3, meClus3, 11, 1, 12, 27, 54.5, 81.5);
        auto mecl4 = ibooker.book2D(meClus4, meClus4, 11, 1, 12, 27, 81.5, 108.5);

        std::array<MonitorElement*, 4> meCls = {{mecl1, mecl2, mecl3, mecl4}};

        for (unsigned int icl = 0; icl < meCls.size(); icl++) {
          meCls[icl]->setAxisTitle("Cluster size", 1);

          for (int i = 1; i <= 9; i++) {
            const std::string ylabel1 = fmt::format("R{}_CH{:02d}_C", ring, (icl * 9) + i);
            const std::string ylabel2 = fmt::format("R{}_CH{:02d}_B", ring, (icl * 9) + i);
            const std::string ylabel3 = fmt::format("R{}_CH{:02d}_A", ring, (icl * 9) + i);
            meCls[icl]->setBinLabel(1 + (i - 1) * 3, ylabel1, 2);
            meCls[icl]->setBinLabel(2 + (i - 1) * 3, ylabel2, 2);
            meCls[icl]->setBinLabel(3 + (i - 1) * 3, ylabel3, 2);
          }
          meCls[icl]->setBinLabel(1, "1", 1);
          meCls[icl]->setBinLabel(5, "5", 1);
          meCls[icl]->setBinLabel(10, "10", 1);
          meCls[icl]->setBinLabel(mecl1->getNbinsX(), "Overflow", 1);
        }

        meMap[meClus1] = mecl1;
        meMap[meClus2] = mecl2;
        meMap[meClus3] = mecl3;
        meMap[meClus4] = mecl4;
      }  //loop ring
    }    //loop disk
  }      //loop region
}

void RPCMonitorDigi::bookWheelDiskME(DQMStore::IBooker& ibooker,
                                     const std::string& recHitType,
                                     std::map<std::string, MonitorElement*>& meMap) {
  ibooker.setCurrentFolder(subsystemFolder_ + "/" + recHitType + "/" + globalFolder_);

  std::string tmpStr;

  for (int wheel = -2; wheel <= 2; wheel++) {  //Loop on wheel
    tmpStr = fmt::format("1DOccupancy_Wheel_{}", wheel);
    meMap[tmpStr] = ibooker.book1D(tmpStr, tmpStr, 12, 0.5, 12.5);
    for (int i = 1; i <= 12; ++i) {
      meMap[tmpStr]->setBinLabel(i, fmt::format("Sec{}", i), 1);
    }

    tmpStr = fmt::format("Occupancy_Roll_vs_Sector_Wheel_{}", wheel);
    meMap[tmpStr] = RPCRollMapHisto::bookBarrel(ibooker, wheel, tmpStr, tmpStr, true);

    tmpStr = fmt::format("BxDistribution_Wheel_{}", wheel);
    meMap[tmpStr] = ibooker.book1D(tmpStr, tmpStr, 9, -4.5, 4.5);

  }  //end loop on wheel

  for (int disk = -RPCMonitorDigi::numberOfDisks_; disk <= RPCMonitorDigi::numberOfDisks_; disk++) {
    if (disk == 0)
      continue;

    tmpStr = fmt::format("Occupancy_Ring_vs_Segment_Disk_{}", disk);
    meMap[tmpStr] = RPCRollMapHisto::bookEndcap(ibooker, disk, tmpStr, tmpStr, true);

    tmpStr = fmt::format("BxDistribution_Disk_{}", disk);
    meMap[tmpStr] = ibooker.book1D(tmpStr, tmpStr, 9, -4.5, 4.5);
  }

  for (int ring = RPCMonitorDigi::numberOfInnerRings_; ring <= 3; ring++) {
    const std::string meName = fmt::format("1DOccupancy_Ring_{}", ring);
    meMap[meName] = ibooker.book1D(
        meName, meName, RPCMonitorDigi::numberOfDisks_ * 2, 0.5, (RPCMonitorDigi::numberOfDisks_ * 2.0) + 0.5);
    for (int xbin = 1; xbin <= RPCMonitorDigi::numberOfDisks_ * 2; xbin++) {
      std::string label;
      if (xbin < RPCMonitorDigi::numberOfDisks_ + 1)
        label = fmt::format("Disk {}", (xbin - (RPCMonitorDigi::numberOfDisks_ + 1)));
      else
        label = fmt::format("Disk {}", (xbin - RPCMonitorDigi::numberOfDisks_));
      meMap[meName]->setBinLabel(xbin, label, 1);
    }
  }
}

//returns the number of strips in each roll
int RPCMonitorDigi::stripsInRoll(const RPCDetId& id, const RPCGeometry* rpcGeo) const {
  const RPCRoll* rpcRoll = rpcGeo->roll(id);
  if (!rpcRoll)
    return 1;

  return rpcRoll->nstrips();
}

void RPCMonitorDigi::bookRegionME(DQMStore::IBooker& ibooker,
                                  const std::string& recHitType,
                                  std::map<std::string, MonitorElement*>& meMap) {
  std::string currentFolder = subsystemFolder_ + "/" + recHitType + "/" + globalFolder_;
  ibooker.setCurrentFolder(currentFolder);

  std::stringstream name;
  std::stringstream title;

  //Number of Digis
  name.str("");
  title.str("");
  name << "Multiplicity_Barrel";
  title << "Multiplicity per Event per Roll - Barrel";
  meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 50, 0.5, 50.5);

  name.str("");
  title.str("");
  name << "Multiplicity_Endcap+";
  title << "Multiplicity per Event per Roll - Endcap+";
  meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 32, 0.5, 32.5);

  name.str("");
  title.str("");
  name << "Multiplicity_Endcap-";
  title << "Multiplicity per Event per Roll - Endcap-";
  meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 32, 0.5, 32.5);

  meMap["Occupancy_for_Endcap"] = ibooker.book2D("Occupancy_for_Endcap",
                                                 "Occupancy Endcap",
                                                 (int)RPCMonitorDigi::numberOfDisks_ * 2.0,
                                                 0.5,
                                                 ((float)RPCMonitorDigi::numberOfDisks_ * 2.0) + 0.5,
                                                 2,
                                                 1.5,
                                                 3.5);
  meMap["Occupancy_for_Endcap"]->setAxisTitle("Disk", 1);
  meMap["Occupancy_for_Endcap"]->setAxisTitle("Ring", 2);

  std::stringstream binlabel;
  for (int bin = 1; bin <= RPCMonitorDigi::numberOfDisks_ * 2; bin++) {
    binlabel.str("");
    if (bin < (RPCMonitorDigi::numberOfDisks_ + 1)) {  //negative endcap
      binlabel << (bin - (RPCMonitorDigi::numberOfDisks_ + 1));
    } else {  //positive endcaps
      binlabel << (bin - RPCMonitorDigi::numberOfDisks_);
    }
    meMap["Occupancy_for_Endcap"]->setBinLabel(bin, binlabel.str(), 1);
  }

  meMap["Occupancy_for_Endcap"]->setBinLabel(1, "2", 2);
  meMap["Occupancy_for_Endcap"]->setBinLabel(2, "3", 2);

  meMap["Occupancy_for_Barrel"] =
      ibooker.book2D("Occupancy_for_Barrel", "Occupancy Barrel", 12, 0.5, 12.5, 5, -2.5, 2.5);
  meMap["Occupancy_for_Barrel"]->setAxisTitle("Sec", 1);
  meMap["Occupancy_for_Barrel"]->setAxisTitle("Wheel", 2);

  for (int bin = 1; bin <= 12; bin++) {
    binlabel.str("");
    binlabel << bin;
    meMap["Occupancy_for_Barrel"]->setBinLabel(bin, binlabel.str(), 1);
    if (bin <= 5) {
      binlabel.str("");
      binlabel << (bin - 3);
      meMap["Occupancy_for_Barrel"]->setBinLabel(bin, binlabel.str(), 2);
    }
  }
}

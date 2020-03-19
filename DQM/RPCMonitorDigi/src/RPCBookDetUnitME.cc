#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>
#include <DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <DQM/RPCMonitorDigi/interface/utils.h>
#include <iomanip>

void RPCMonitorDigi::bookRollME(DQMStore::IBooker& ibooker,
                                const RPCDetId& detId,
                                const RPCGeometry* rpcGeo,
                                const std::string& recHitType,
                                std::map<std::string, MonitorElement*>& meMap) {
  RPCBookFolderStructure folderStr;
  std::string folder = subsystemFolder_ + "/" + recHitType + "/" + folderStr.folderStructure(detId);

  ibooker.setCurrentFolder(folder);

  //get number of strips in current roll
  int nstrips = this->stripsInRoll(detId, rpcGeo);
  if (nstrips == 0) {
    nstrips = 1;
  }

  /// Name components common to current RPCDetId
  RPCGeomServ RPCname(detId);
  std::string nameRoll = "";

  if (RPCMonitorDigi::useRollInfo_) {
    nameRoll = RPCname.name();
  } else {
    nameRoll = RPCname.chambername();

    if (detId.region() != 0 ||                                                     //Endcaps
        (abs(detId.ring()) == 2 && detId.station() == 2 && detId.layer() != 1) ||  //Wheel -/+2 RB2out
        (abs(detId.ring()) != 2 && detId.station() == 2 && detId.layer() == 1)) {
      nstrips *= 3;
    }  //Wheel -1,0,+1 RB2in
    else {
      nstrips *= 2;
    }
  }

  std::stringstream os;
  os.str("");
  os << "Occupancy_" << nameRoll;
  meMap[os.str()] = ibooker.book1D(os.str(), os.str(), nstrips, 0.5, nstrips + 0.5);

  os.str("");
  os << "BXDistribution_" << nameRoll;
  meMap[os.str()] = ibooker.book1D(os.str(), os.str(), 7, -3.5, 3.5);

  if (detId.region() == 0) {
    os.str("");
    os << "ClusterSize_" << nameRoll;
    meMap[os.str()] = ibooker.book1D(os.str(), os.str(), 15, 0.5, 15.5);

    os.str("");
    os << "Multiplicity_" << nameRoll;
    meMap[os.str()] = ibooker.book1D(os.str(), os.str(), 30, 0.5, 30.5);

  } else {
    os.str("");
    os << "ClusterSize_" << nameRoll;
    meMap[os.str()] = ibooker.book1D(os.str(), os.str(), 10, 0.5, 10.5);

    os.str("");
    os << "Multiplicity_" << nameRoll;
    meMap[os.str()] = ibooker.book1D(os.str(), os.str(), 15, 0.5, 15.5);
  }

  os.str("");
  os << "NumberOfClusters_" << nameRoll;
  meMap[os.str()] = ibooker.book1D(os.str(), os.str(), 10, 0.5, 10.5);
}

void RPCMonitorDigi::bookSectorRingME(DQMStore::IBooker& ibooker,
                                      const std::string& recHitType,
                                      std::map<std::string, MonitorElement*>& meMap) {
  std::stringstream os;

  for (int wheel = -2; wheel <= 2; wheel++) {
    os.str("");
    os << subsystemFolder_ << "/" << recHitType << "/Barrel/Wheel_" << wheel << "/SummaryBySectors";
    ibooker.setCurrentFolder(os.str());

    for (int sector = 1; sector <= 12; sector++) {
      os.str("");
      os << "Occupancy_Wheel_" << wheel << "_Sector_" << sector;

      if (sector == 9 || sector == 11)
        meMap[os.str()] = ibooker.book2D(os.str(), os.str(), 91, 0.5, 91.5, 15, 0.5, 15.5);
      else if (sector == 4)
        meMap[os.str()] = ibooker.book2D(os.str(), os.str(), 91, 0.5, 91.5, 21, 0.5, 21.5);
      else
        meMap[os.str()] = ibooker.book2D(os.str(), os.str(), 91, 0.5, 91.5, 17, 0.5, 17.5);

      meMap[os.str()]->setAxisTitle("strip", 1);
      rpcdqm::utils rpcUtils;
      rpcUtils.labelYAxisRoll(meMap[os.str()], 0, wheel, true);
    }
  }

  for (int region = -1; region <= 1; region++) {
    if (region == 0)
      continue;

    std::string regionName = "Endcap-";
    if (region == 1)
      regionName = "Endcap+";

    for (int disk = 1; disk <= RPCMonitorDigi::numberOfDisks_; disk++) {
      os.str("");
      os << subsystemFolder_ << "/" << recHitType << "/" << regionName << "/Disk_" << (region * disk)
         << "/SummaryByRings/";

      ibooker.setCurrentFolder(os.str());

      for (int ring = RPCMonitorDigi::numberOfInnerRings_; ring <= 3; ring++) {
        os.str("");
        os << "Occupancy_Disk_" << (region * disk) << "_Ring_" << ring << "_CH01-CH18";

        meMap[os.str()] = ibooker.book2D(os.str(), os.str(), 96, 0.5, 96.5, 18, 0.5, 18.5);
        meMap[os.str()]->setAxisTitle("strip", 1);
        rpcdqm::RPCMEHelper::setNoAlphanumeric(meMap[os.str()]);

        std::stringstream yLabel;
        for (int i = 1; i <= 18; i++) {
          yLabel.str("");
          yLabel << "R" << ring << "_CH" << std::setw(2) << std::setfill('0') << i;
          meMap[os.str()]->setBinLabel(i, yLabel.str(), 2);
        }

        for (int i = 1; i <= 96; i++) {
          if (i == 1)
            meMap[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 16)
            meMap[os.str()]->setBinLabel(i, "RollA", 1);
          else if (i == 32)
            meMap[os.str()]->setBinLabel(i, "32", 1);
          else if (i == 33)
            meMap[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 48)
            meMap[os.str()]->setBinLabel(i, "RollB", 1);
          else if (i == 64)
            meMap[os.str()]->setBinLabel(i, "32", 1);
          else if (i == 65)
            meMap[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 80)
            meMap[os.str()]->setBinLabel(i, "RollC", 1);
          else if (i == 96)
            meMap[os.str()]->setBinLabel(i, "32", 1);
          else
            meMap[os.str()]->setBinLabel(i, "", 1);
        }

        os.str("");
        os << "Occupancy_Disk_" << (region * disk) << "_Ring_" << ring << "_CH19-CH36";

        meMap[os.str()] = ibooker.book2D(os.str(), os.str(), 96, 0.5, 96.5, 18, 18.5, 36.5);
        meMap[os.str()]->setAxisTitle("strip", 1);
        rpcdqm::RPCMEHelper::setNoAlphanumeric(meMap[os.str()]);

        for (int i = 1; i <= 18; i++) {
          yLabel.str("");
          yLabel << "R" << ring << "_CH" << i + 18;
          meMap[os.str()]->setBinLabel(i, yLabel.str(), 2);
        }

        for (int i = 1; i <= 96; i++) {
          if (i == 1)
            meMap[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 16)
            meMap[os.str()]->setBinLabel(i, "RollA", 1);
          else if (i == 32)
            meMap[os.str()]->setBinLabel(i, "32", 1);
          else if (i == 33)
            meMap[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 48)
            meMap[os.str()]->setBinLabel(i, "RollB", 1);
          else if (i == 64)
            meMap[os.str()]->setBinLabel(i, "32", 1);
          else if (i == 65)
            meMap[os.str()]->setBinLabel(i, "1", 1);
          else if (i == 80)
            meMap[os.str()]->setBinLabel(i, "RollC", 1);
          else if (i == 96)
            meMap[os.str()]->setBinLabel(i, "32", 1);
          else
            meMap[os.str()]->setBinLabel(i, "", 1);
        }

      }  //loop ring
    }    //loop disk
  }      //loop region
}

void RPCMonitorDigi::bookWheelDiskME(DQMStore::IBooker& ibooker,
                                     const std::string& recHitType,
                                     std::map<std::string, MonitorElement*>& meMap) {
  ibooker.setCurrentFolder(subsystemFolder_ + "/" + recHitType + "/" + globalFolder_);

  std::stringstream os, label, name, title;
  rpcdqm::utils rpcUtils;

  for (int wheel = -2; wheel <= 2; wheel++) {  //Loop on wheel
    os.str("");
    os << "1DOccupancy_Wheel_" << wheel;
    meMap[os.str()] = ibooker.book1D(os.str(), os.str(), 12, 0.5, 12.5);
    for (int i = 1; i < 12; i++) {
      label.str("");
      label << "Sec" << i;
      meMap[os.str()]->setBinLabel(i, label.str(), 1);
    }

    os.str("");
    os << "Occupancy_Roll_vs_Sector_Wheel_" << wheel;
    meMap[os.str()] = ibooker.book2D(os.str(), os.str(), 12, 0.5, 12.5, 21, 0.5, 21.5);
    rpcUtils.labelXAxisSector(meMap[os.str()]);
    rpcUtils.labelYAxisRoll(meMap[os.str()], 0, wheel, true);

    os.str("");
    os << "BxDistribution_Wheel_" << wheel;
    meMap[os.str()] = ibooker.book1D(os.str(), os.str(), 9, -4.5, 4.5);

    for (int layer = 1; layer <= 6; layer++) {
      name.str("");
      title.str("");
      name << "ClusterSize_Wheel_" << wheel << "_Layer" << layer;
      title << "ClusterSize - Wheel " << wheel << " Layer" << layer;
      meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 16, 0.5, 16.5);
    }

  }  //end loop on wheel

  for (int disk = -RPCMonitorDigi::numberOfDisks_; disk <= RPCMonitorDigi::numberOfDisks_; disk++) {
    if (disk == 0)
      continue;

    os.str("");
    os << "Occupancy_Ring_vs_Segment_Disk_" << disk;
    meMap[os.str()] = ibooker.book2D(os.str(), os.str(), 36, 0.5, 36.5, 6, 0.5, 6.5);

    rpcUtils.labelXAxisSegment(meMap[os.str()]);
    rpcUtils.labelYAxisRing(meMap[os.str()], 2, true);

    os.str("");
    os << "BxDistribution_Disk_" << disk;
    meMap[os.str()] = ibooker.book1D(os.str(), os.str(), 9, -4.5, 4.5);

    for (int ring = RPCMonitorDigi::numberOfInnerRings_; ring <= 3; ring++) {
      name.str("");
      title.str("");
      name << "ClusterSize_Disk_" << disk << "_Ring" << ring;
      title << "ClusterSize - Disk" << disk << " Ring" << ring;
      meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 16, 0.5, 16.5);
    }
  }

  for (int ring = RPCMonitorDigi::numberOfInnerRings_; ring <= 3; ring++) {
    os.str("");
    os << "1DOccupancy_Ring_" << ring;
    meMap[os.str()] = ibooker.book1D(os.str(),
                                     os.str(),
                                     RPCMonitorDigi::numberOfDisks_ * 2,
                                     0.5,
                                     ((double)RPCMonitorDigi::numberOfDisks_ * 2.0) + 0.5);
    for (int xbin = 1; xbin <= RPCMonitorDigi::numberOfDisks_ * 2; xbin++) {
      label.str("");
      if (xbin < RPCMonitorDigi::numberOfDisks_ + 1)
        label << "Disk " << (xbin - (RPCMonitorDigi::numberOfDisks_ + 1));
      else
        label << "Disk " << (xbin - RPCMonitorDigi::numberOfDisks_);
      meMap[os.str()]->setBinLabel(xbin, label.str(), 1);
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
  for (int r = 0; r < 3; r++) {  //RPC regions are E-, B, and E+

    std::string regionName = RPCMonitorDigi::regionNames_[r];
    //Cluster size
    name.str("");
    title.str("");
    name << "ClusterSize_" << regionName;
    title << "ClusterSize - " << regionName;
    meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 16, 0.5, 16.5);
  }

  //Number of Cluster
  name.str("");
  title.str("");
  name << "NumberOfClusters_Barrel";
  title << "Number of Clusters per Event - Barrel";
  meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 30, 0.5, 30.5);

  name.str("");
  title.str("");
  name << "NumberOfClusters_Endcap+";
  title << "Number of Clusters per Event - Endcap+";
  meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 15, 0.5, 15.5);

  name.str("");
  title.str("");
  name << "NumberOfClusters_Endcap-";
  title << "Number of Clusters per Event - Endcap-";
  meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 15, 0.5, 15.5);

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

  for (int layer = 1; layer <= 6; layer++) {
    name.str("");
    title.str("");
    name << "ClusterSize_Layer" << layer;
    title << "ClusterSize - Layer" << layer;
    meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 16, 0.5, 16.5);
  }

  for (int ring = RPCMonitorDigi::numberOfInnerRings_; ring <= 3; ring++) {
    name.str("");
    title.str("");
    name << "ClusterSize_Ring" << ring;
    title << "ClusterSize - Ring" << ring;
    meMap[name.str()] = ibooker.book1D(name.str(), title.str(), 16, 0.5, 16.5);
  }

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

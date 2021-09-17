#include "CalibTracker/SiStripCommon/interface/SiStripDCSStatus.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

// FED cabling and numbering
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

//
// -- Constructor
//
SiStripDCSStatus::SiStripDCSStatus(edm::ConsumesCollector& iC)
    : TIBTIDinDAQ(false),
      TOBinDAQ(false),
      TECFinDAQ(false),
      TECBinDAQ(false),
      trackerAbsent(false),
      rawdataAbsent(true),
      initialised(false) {
  dcsStatusToken_ = iC.consumes<DcsStatusCollection>(edm::InputTag("scalersRawToDigi"));
  dcsRecordToken_ = iC.consumes<DCSRecord>(edm::InputTag("onlineMetaDataDigis"));
  rawDataToken_ = iC.consumes<FEDRawDataCollection>(edm::InputTag("rawDataCollector"));
  tTopoToken_ = iC.esConsumes<TrackerTopology, TrackerTopologyRcd>();
  fedCablingToken_ = iC.esConsumes<SiStripFedCabling, SiStripFedCablingRcd>();
}
//
// -- Get State
//
bool SiStripDCSStatus::getStatus(edm::Event const& e, edm::EventSetup const& eSetup) {
  bool retVal = true;
  if (!initialised)
    initialise(e, eSetup);

  edm::Handle<DcsStatusCollection> dcsStatus;
  e.getByToken(dcsStatusToken_, dcsStatus);

  edm::Handle<DCSRecord> dcsRecord;
  e.getByToken(dcsRecordToken_, dcsRecord);

  bool statusTIBTID(true), statusTOB(true), statusTECF(true), statusTECB(true);
  bool dcsTIBTID(true), dcsTOB(true), dcsTECF(true), dcsTECB(true);

  if (trackerAbsent || (!dcsStatus.isValid() && !dcsRecord.isValid())) {
    return retVal;
  }

  if ((*dcsStatus).empty()) {
    if (e.eventAuxiliary().isRealData()) {
      dcsTIBTID = (*dcsRecord).highVoltageReady(DCSRecord::Partition::TIBTID);
      dcsTOB = (*dcsRecord).highVoltageReady(DCSRecord::Partition::TOB);
      dcsTECF = (*dcsRecord).highVoltageReady(DCSRecord::Partition::TECp);
      dcsTECB = (*dcsRecord).highVoltageReady(DCSRecord::Partition::TECm);
    } else {
      return retVal;
    }
  } else {
    dcsTIBTID = ((*dcsStatus)[0].ready(DcsStatus::TIBTID));
    dcsTOB = ((*dcsStatus)[0].ready(DcsStatus::TOB));
    dcsTECF = ((*dcsStatus)[0].ready(DcsStatus::TECp));
    dcsTECB = ((*dcsStatus)[0].ready(DcsStatus::TECm));
  }

  if (rawdataAbsent) {
    statusTIBTID = dcsTIBTID;
    statusTOB = dcsTOB;
    statusTECF = dcsTECF;
    statusTECB = dcsTECB;
  } else {
    if (TIBTIDinDAQ && !dcsTIBTID)
      statusTIBTID = false;
    if (TOBinDAQ && !dcsTOB)
      statusTOB = false;
    if (TECFinDAQ && !dcsTECF)
      statusTECF = false;
    if (TECBinDAQ && !dcsTECB)
      statusTECB = false;
  }

  LogDebug("SiStripDCSStatus") << " SiStripDCSStatus :: Detectors in DAQ (TIBTID, TOB, TEC+ TEC-)" << TIBTIDinDAQ << " "
                               << TOBinDAQ << " " << TECFinDAQ << " " << TECBinDAQ << std::endl;
  LogDebug("SiStripDCSStatus") << " SiStripDCSStatus :: Detectors in ON (TIBTID, TOB, TEC+ TEC-)" << dcsTIBTID << " "
                               << dcsTOB << " " << dcsTECF << " " << dcsTECB << std::endl;

  LogDebug("SiStripDCSStatus") << " SiStripDCSStatus :: Final Flags     (TIBTID, TOB, TEC+ TEC-)" << statusTIBTID << " "
                               << statusTOB << " " << statusTECF << " " << statusTECB << std::endl;
  if (statusTIBTID && statusTOB && statusTECF && statusTECB)
    retVal = true;
  else
    retVal = false;
  LogDebug("SiStripDCSStatus") << " Return Value " << retVal;
  return retVal;
}
//
// -- initialise
//
void SiStripDCSStatus::initialise(edm::Event const& e, edm::EventSetup const& eSetup) {
  //Retrieve tracker topology from geometry
  const auto& tTopo = eSetup.getData(tTopoToken_);
  const auto& fedCabling_ = eSetup.getData(fedCablingToken_);

  auto connectedFEDs = fedCabling_.fedIds();

  edm::Handle<FEDRawDataCollection> rawDataHandle;
  //  e.getByLabel("source", rawDataHandle);
  e.getByToken(rawDataToken_, rawDataHandle);

  if (!rawDataHandle.isValid()) {
    rawdataAbsent = true;
    return;
  }

  rawdataAbsent = false;
  const FEDRawDataCollection& rawDataCollection = *rawDataHandle;
  for (std::vector<unsigned short>::const_iterator ifed = connectedFEDs.begin(); ifed != connectedFEDs.end(); ifed++) {
    auto fedChannels = fedCabling_.fedConnections(*ifed);
    if (!(rawDataCollection.FEDData(*ifed).size()) || !(rawDataCollection.FEDData(*ifed).data()))
      continue;
    // Check Modules Connected
    for (std::vector<FedChannelConnection>::const_iterator iconn = fedChannels.begin(); iconn < fedChannels.end();
         iconn++) {
      if (!iconn->isConnected())
        continue;
      uint32_t detId = iconn->detId();
      StripSubdetector subdet(detId);
      if ((subdet.subdetId() == StripSubdetector::TIB) || (subdet.subdetId() == StripSubdetector::TID)) {
        TIBTIDinDAQ = true;
        break;
      } else if (subdet.subdetId() == StripSubdetector::TOB) {
        TOBinDAQ = true;
        break;
      } else if (subdet.subdetId() == StripSubdetector::TEC) {
        if (tTopo.tecSide(detId) == 2)
          TECFinDAQ = true;
        else if (tTopo.tecSide(detId) == 1)
          TECBinDAQ = true;
        break;
      }
    }
    if (TIBTIDinDAQ && TOBinDAQ && TECFinDAQ && TECBinDAQ)
      break;
  }
  initialised = true;
  if (!TIBTIDinDAQ && !TOBinDAQ && !TECFinDAQ && !TECBinDAQ)
    trackerAbsent = true;
}

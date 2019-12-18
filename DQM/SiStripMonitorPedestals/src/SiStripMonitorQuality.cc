// -*- C++ -*-
//
// Package:    SiStripMonitorQuality
// Class:      SiStripMonitorQuality
//
/**\class SiStripMonitorDigi SiStripMonitorDigi.cc
 DQM/SiStripMonitorDigi/src/SiStripMonitorDigi.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Suchandra Dutta
//         Created:  Fri Dec  7 20:50 CET 2007
//
//

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <FWCore/Framework/interface/EventSetup.h>

#include "DQM/SiStripMonitorPedestals/interface/SiStripMonitorQuality.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

// std
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>

SiStripMonitorQuality::SiStripMonitorQuality(edm::ParameterSet const &iConfig)
    : dqmStore_(edm::Service<DQMStore>().operator->()),
      conf_(iConfig),
      m_cacheID_(0)

{
  edm::LogInfo("SiStripMonitorQuality") << "SiStripMonitorQuality  "
                                        << " Constructing....... ";
}

SiStripMonitorQuality::~SiStripMonitorQuality() {
  edm::LogInfo("SiStripMonitorQuality") << "SiStripMonitorQuality  "
                                        << " Destructing....... ";
}
//
void SiStripMonitorQuality::bookHistograms(DQMStore::IBooker &ibooker,
                                           const edm::Run &run,
                                           const edm::EventSetup &eSetup) {
  unsigned long long cacheID = eSetup.get<SiStripQualityRcd>().cacheIdentifier();
  if (m_cacheID_ == cacheID)
    return;

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology *const tTopo = tTopoHandle.product();

  m_cacheID_ = cacheID;

  std::string quality_label = conf_.getParameter<std::string>("StripQualityLabel");
  eSetup.get<SiStripQualityRcd>().get(quality_label, stripQuality_);
  eSetup.get<SiStripDetCablingRcd>().get(detCabling_);

  edm::LogInfo("SiStripMonitorQuality") << "SiStripMonitorQuality::analyze: "
                                        << " Reading SiStripQuality " << std::endl;

  SiStripBadStrip::RegistryIterator rbegin = stripQuality_->getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator rend = stripQuality_->getRegistryVectorEnd();
  uint32_t detid;

  if (rbegin == rend)
    return;

  for (SiStripBadStrip::RegistryIterator rp = rbegin; rp != rend; ++rp) {
    detid = rp->detid;
    // Check consistency in DetId
    if (detid == 0 || detid == 0xFFFFFFFF) {
      edm::LogError("SiStripMonitorQuality") << "SiStripMonitorQuality::bookHistograms : "
                                             << "Wrong DetId !!!!!! " << detid << " Neglecting !!!!!! ";
      continue;
    }
    // check if the detid is connected in cabling
    if (!detCabling_->IsConnected(detid)) {
      edm::LogError("SiStripMonitorQuality") << "SiStripMonitorQuality::bookHistograms : "
                                             << " DetId " << detid << " not connected,  Neglecting !!!!!! ";
      continue;
    }

    MonitorElement *det_me;

    int nStrip = detCabling_->nApvPairs(detid) * 256;

    // use SistripHistoId for producing histogram id (and title)
    SiStripHistoId hidmanager;
    // create SiStripFolderOrganizer
    SiStripFolderOrganizer folder_organizer;
    // set appropriate folder using SiStripFolderOrganizer
    folder_organizer.setDetectorFolder(detid,
                                       tTopo);  // pass the detid to this method

    std::string hid;
    hid = hidmanager.createHistoId("StripQualityFromCondDB", "det", detid);

    det_me = ibooker.book1D(hid, hid, nStrip, 0.5, nStrip + 0.5);
    det_me->setAxisTitle("Strip Number", 1);
    det_me->setAxisTitle("Quality Flag from CondDB ", 2);
    QualityMEs.insert(std::make_pair(detid, det_me));
  }
}

// ------------ method called to produce the data  ------------
void SiStripMonitorQuality::analyze(edm::Event const &iEvent, edm::EventSetup const &eSetup) {
  unsigned long long cacheID = eSetup.get<SiStripQualityRcd>().cacheIdentifier();
  if (m_cacheID_ == cacheID)
    return;

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology *const tTopo = tTopoHandle.product();

  m_cacheID_ = cacheID;

  std::string quality_label = conf_.getParameter<std::string>("StripQualityLabel");
  eSetup.get<SiStripQualityRcd>().get(quality_label, stripQuality_);
  eSetup.get<SiStripDetCablingRcd>().get(detCabling_);

  edm::LogInfo("SiStripMonitorQuality") << "SiStripMonitorQuality::analyze: "
                                        << " Reading SiStripQuality " << std::endl;

  SiStripBadStrip::RegistryIterator rbegin = stripQuality_->getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator rend = stripQuality_->getRegistryVectorEnd();
  uint32_t detid;

  if (rbegin == rend)
    return;

  for (SiStripBadStrip::RegistryIterator rp = rbegin; rp != rend; ++rp) {
    detid = rp->detid;
    // Check consistency in DetId
    if (detid == 0 || detid == 0xFFFFFFFF) {
      edm::LogError("SiStripMonitorQuality") << "SiStripMonitorQuality::analyze : "
                                             << "Wrong DetId !!!!!! " << detid << " Neglecting !!!!!! ";
      continue;
    }
    // check if the detid is connected in cabling
    if (!detCabling_->IsConnected(detid)) {
      edm::LogError("SiStripMonitorQuality") << "SiStripMonitorQuality::analyze : "
                                             << " DetId " << detid << " not connected,  Neglecting !!!!!! ";
      continue;
    }
    MonitorElement *me = getQualityME(detid, tTopo);
    SiStripBadStrip::Range range = SiStripBadStrip::Range(stripQuality_->getDataVectorBegin() + rp->ibegin,
                                                          stripQuality_->getDataVectorBegin() + rp->iend);
    SiStripBadStrip::ContainerIterator it = range.first;
    for (; it != range.second; ++it) {
      unsigned int value = (*it);
      short str_start = stripQuality_->decode(value).firstStrip;
      short str_end = str_start + stripQuality_->decode(value).range;
      for (short isr = str_start; isr < str_end + 1; isr++) {
        if (isr <= (me->getNbinsX() - 1))
          me->Fill(isr + 1, 1.0);
      }
    }
  }
}
//
// -- End Run
//
void SiStripMonitorQuality::dqmEndRun(edm::Run const &run, edm::EventSetup const &eSetup) {
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if (outputMEsInRootFile) {
    dqmStore_->save(outputFileName);
  }
}
//
// -- End Job
//
void SiStripMonitorQuality::endJob(void) {
  edm::LogInfo("SiStripMonitorQuality") << "SiStripMonitorQuality::EndJob: "
                                        << " Finishing!! ";
}
//
// -- End Job
//
SiStripMonitorQuality::MonitorElement *SiStripMonitorQuality::getQualityME(uint32_t idet,
                                                                           const TrackerTopology *tTopo) {
  std::map<uint32_t, MonitorElement *>::iterator pos = QualityMEs.find(idet);
  MonitorElement *det_me = nullptr;
  if (pos != QualityMEs.end()) {
    det_me = pos->second;
    det_me->Reset();
  } else {
    // this should never happen because of bookHistograms()
    edm::LogError("SiStripMonitorQuality") << "SiStripMonitorQuality::getQualityME : "
                                           << "Wrong DetId !!!!!! " << idet << " No ME found!";
  }
  return det_me;
}
DEFINE_FWK_MODULE(SiStripMonitorQuality);

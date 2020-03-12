#ifndef SiStripDCSStatus_H
#define SiStripDCSStatus_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

class TrackerTopology;
class TrackerTopologyRcd;
class SiStripFedCabling;
class SiStripFedCablingRcd;

class SiStripDCSStatus {
public:
  SiStripDCSStatus(edm::ConsumesCollector&& iC) : SiStripDCSStatus(iC){};
  SiStripDCSStatus(edm::ConsumesCollector& iC);

  bool getStatus(edm::Event const& e, edm::EventSetup const& eSetup);

private:
  void initialise(edm::Event const& e, edm::EventSetup const& eSetup);

  bool TIBTIDinDAQ, TOBinDAQ, TECFinDAQ, TECBinDAQ;
  bool trackerAbsent;
  bool rawdataAbsent;
  bool initialised;

  edm::EDGetTokenT<DcsStatusCollection> dcsStatusToken_;
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
};

#endif

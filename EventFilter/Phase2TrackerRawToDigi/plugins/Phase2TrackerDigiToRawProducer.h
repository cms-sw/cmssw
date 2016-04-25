#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerDigiToRawProducer_H
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerDigiToRawProducer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>

namespace Phase2Tracker {

  class Phase2TrackerDigiToRawProducer : public edm::EDProducer
  {
  public:
    /// constructor
    Phase2TrackerDigiToRawProducer( const edm::ParameterSet& pset );
    /// default constructor
    ~Phase2TrackerDigiToRawProducer();
    virtual void beginJob() override;
    virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

  private:
    edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D>> token_;
    const Phase2TrackerCabling * cabling_;
    const TrackerTopology* tTopo_;
    const TrackerGeometry* tGeom_;
    std::map< int, std::pair<int,int> > stackMap_;
  };
}
#endif // EventFilter_Phase2TrackerRawToDigi_Phase2TrackerDigiToRawProducer_H

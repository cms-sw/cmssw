#ifndef SiPixelMonitorRawData_SiPixelHLTSource_h
#define SiPixelMonitorRawData_SiPixelHLTSource_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorRawData
// Class  :     SiPixelHLTSource
//
/**

 Description:
 Produces histograms for error information generated at the raw2digi stage for
 the pixel tracker.

 Implementation:
 Takes raw data and error data as input, and uses it to populate three
 histograms indexed by FED id.

*/
//
// Original Author:  Andrew York
//

#include <memory>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class SiPixelHLTSource : public DQMEDAnalyzer {
public:
  explicit SiPixelHLTSource(const edm::ParameterSet &conf);
  ~SiPixelHLTSource() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &, edm::EventSetup const &) override;
  virtual void bookMEs(DQMStore::IBooker &);

private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<FEDRawDataCollection> rawin_;
  edm::EDGetTokenT<edm::DetSetVector<SiPixelRawDataError>> errin_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  bool saveFile;
  bool slowDown;
  std::string dirName_;
  int eventNo;
  MonitorElement *meRawWords_;
  MonitorElement *meNCRCs_;
  MonitorElement *meNErrors_;
  bool firstRun;
};

#endif

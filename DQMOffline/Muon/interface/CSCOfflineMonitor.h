#ifndef RecoLocalMuon_CSCOfflineMonitor_H
#define RecoLocalMuon_CSCOfflineMonitor_H

/** \class CSCOfflineMonitor
 *
 * Simple package for offline CSC DQM based on RecoLocalMuon/CSCValidation:
 *    digis, rechits, segments
 *
 * Andrew Kubik, Northwestern University, Oct 2008
 *
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "EventFilter/CSCRawToDigi/interface/CSCComparatorData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"

class CSCOfflineMonitor : public DQMEDAnalyzer {
public:
  CSCOfflineMonitor(const edm::ParameterSet &pset);
  ~CSCOfflineMonitor() override{};

  enum LabelType { SMALL, EXTENDED };
  enum AxisType { X = 1, Y = 2, Z = 3 };

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  edm::ParameterSet param;

  edm::EDGetTokenT<FEDRawDataCollection> rd_token;
  edm::EDGetTokenT<CSCWireDigiCollection> wd_token;
  edm::EDGetTokenT<CSCStripDigiCollection> sd_token;
  edm::EDGetTokenT<CSCALCTDigiCollection> al_token;
  edm::EDGetTokenT<CSCCLCTDigiCollection> cl_token;
  edm::EDGetTokenT<CSCRecHit2DCollection> rh_token;
  edm::EDGetTokenT<CSCSegmentCollection> se_token;

  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken_;

  const edm::ESGetToken<CSCCrateMap, CSCCrateMapRcd> hcrateToken_;

  // modules
  void doOccupancies(edm::Handle<CSCStripDigiCollection> strips,
                     edm::Handle<CSCWireDigiCollection> wires,
                     edm::Handle<CSCRecHit2DCollection> recHits,
                     edm::Handle<CSCSegmentCollection> cscSegments,
                     edm::Handle<CSCCLCTDigiCollection> clcts);
  void doStripDigis(edm::Handle<CSCStripDigiCollection> strips);
  void doWireDigis(edm::Handle<CSCWireDigiCollection> wires);
  void doRecHits(edm::Handle<CSCRecHit2DCollection> recHits,
                 edm::Handle<CSCStripDigiCollection> strips,
                 edm::ESHandle<CSCGeometry> cscGeom);
  void doPedestalNoise(edm::Handle<CSCStripDigiCollection> strips, edm::ESHandle<CSCGeometry> cscGeom);
  void doSegments(edm::Handle<CSCSegmentCollection> cscSegments, edm::ESHandle<CSCGeometry> cscGeom);
  void doResolution(edm::Handle<CSCSegmentCollection> cscSegments, edm::ESHandle<CSCGeometry> cscGeom);
  void doEfficiencies(edm::Handle<CSCWireDigiCollection> wires,
                      edm::Handle<CSCStripDigiCollection> strips,
                      edm::Handle<CSCRecHit2DCollection> recHits,
                      edm::Handle<CSCSegmentCollection> cscSegments,
                      edm::ESHandle<CSCGeometry> cscGeom);
  void doBXMonitor(edm::Handle<CSCALCTDigiCollection> alcts,
                   edm::Handle<CSCCLCTDigiCollection> clcts,
                   const edm::Event &event,
                   const edm::EventSetup &eventSetup);

  // used by modules
  float fitX(const CLHEP::HepMatrix &sp, const CLHEP::HepMatrix &ep);
  float getSignal(const CSCStripDigiCollection &stripdigis, CSCDetId idRH, int centerStrip);
  int typeIndex(CSCDetId id, int flag = 1);
  int chamberSerial(CSCDetId id);
  void applyCSClabels(MonitorElement *meHisto, LabelType t, AxisType a);

  // for efficiency calculations
  // - these functions handle Stoyan's efficiency code

  void fillEfficiencyHistos(int bin, int flag);
  //  void  getEfficiency(float bin, float Norm, std::vector<float> &eff);
  //  void  histoEfficiency(TH1F *readHisto, MonitorElement *writeHisto);

  double lineParametrization(double z1Position, double z2Position, double z1Direction) {
    double parameterLine = (z2Position - z1Position) / z1Direction;
    return parameterLine;
  }

  double extrapolate1D(double initPosition, double initDirection, double parameterOfTheLine) {
    double extrapolatedPosition = initPosition + initDirection * parameterOfTheLine;
    return extrapolatedPosition;
  }

  bool withinSensitiveRegion(LocalPoint localPos,
                             const std::array<const float, 4> &layerBounds,
                             int station,
                             int ring,
                             float shiftFromEdge,
                             float shiftFromDeadZone);

  // for BX monitor plots
  //  void harvestChamberMeans(MonitorElement* meMean1D, MonitorElement *meMean2D, MonitorElement *hNum, MonitorElement *meDenom);
  //  void normalize(MonitorElement* me);

  // Wire digis
  MonitorElement *hWirenGroupsTotal;
  MonitorElement *hWirenGroupsTotalHi;
  std::vector<MonitorElement *> hWireTBin;
  std::vector<MonitorElement *> hWireNumber;

  // Strip Digis
  MonitorElement *hStripNFired;
  MonitorElement *hStripNFiredHi;
  std::vector<MonitorElement *> hStripNumber;
  std::vector<MonitorElement *> hStripPed;

  // Rechits
  MonitorElement *hRHnrechits;
  MonitorElement *hRHnrechitsHi;
  std::vector<MonitorElement *> hRHGlobal;
  std::vector<MonitorElement *> hRHSumQ;
  std::vector<MonitorElement *> hRHTiming;
  std::vector<MonitorElement *> hRHTimingAnode;
  std::vector<MonitorElement *> hRHRatioQ;
  std::vector<MonitorElement *> hRHstpos;
  std::vector<MonitorElement *> hRHsterr;

  // Segments
  MonitorElement *hSnSegments;
  MonitorElement *hSnhitsAll;
  std::vector<MonitorElement *> hSnhits;
  MonitorElement *hSChiSqAll;
  std::vector<MonitorElement *> hSChiSq;
  MonitorElement *hSChiSqProbAll;
  std::vector<MonitorElement *> hSChiSqProb;
  MonitorElement *hSGlobalTheta;
  MonitorElement *hSGlobalPhi;
  MonitorElement *hSTimeDiff;
  std::vector<MonitorElement *> hSTimeDiffByChamberType;
  MonitorElement *hSTimeAnode;
  std::vector<MonitorElement *> hSTimeAnodeByChamberType;
  MonitorElement *hSTimeCathode;
  std::vector<MonitorElement *> hSTimeCathodeByChamberType;
  MonitorElement *hSTimeCombined;
  std::vector<MonitorElement *> hSTimeCombinedByChamberType;
  MonitorElement *hSTimeDiffSerial;
  MonitorElement *hSTimeAnodeSerial;
  MonitorElement *hSTimeCathodeSerial;
  MonitorElement *hSTimeCombinedSerial;
  MonitorElement *hSTimeVsZ;
  MonitorElement *hSTimeVsTOF;

  // Resolution
  std::vector<MonitorElement *> hSResid;

  // Occupancies
  MonitorElement *hOWires;
  MonitorElement *hOWiresAndCLCT;
  MonitorElement *hOStrips;
  MonitorElement *hOStripsAndWiresAndCLCT;
  MonitorElement *hORecHits;
  MonitorElement *hOSegments;
  MonitorElement *hOWireSerial;
  MonitorElement *hOStripSerial;
  MonitorElement *hORecHitsSerial;
  MonitorElement *hOSegmentsSerial;
  MonitorElement *hCSCOccupancy;

  // Efficiencies
  MonitorElement *hSnum;
  MonitorElement *hSden;
  MonitorElement *hRHnum;
  MonitorElement *hRHden;
  //  MonitorElement *hSEff;
  //  MonitorElement *hRHEff;
  MonitorElement *hSSTE2;
  MonitorElement *hRHSTE2;
  MonitorElement *hStripSTE2;
  MonitorElement *hWireSTE2;
  //  MonitorElement *hSEff2;
  //  MonitorElement *hRHEff2;
  //  MonitorElement *hStripEff2;
  //  MonitorElement *hWireEff2;
  //  MonitorElement *hStripReadoutEff2;
  MonitorElement *hEffDenominator;
  MonitorElement *hSensitiveAreaEvt;

  // BX monitoring
  MonitorElement *hALCTgetBX;
  MonitorElement *hALCTgetBXSerial;
  //  MonitorElement *hALCTgetBXChamberMeans;
  //  MonitorElement *hALCTgetBX2DMeans;
  MonitorElement *hALCTgetBX2Denominator;
  MonitorElement *hALCTgetBX2DNumerator;

  MonitorElement *hALCTMatch;
  MonitorElement *hALCTMatchSerial;
  //  MonitorElement *hALCTMatchChamberMeans;
  //  MonitorElement *hALCTMatch2DMeans;
  MonitorElement *hALCTMatch2Denominator;
  MonitorElement *hALCTMatch2DNumerator;

  MonitorElement *hCLCTL1A;
  MonitorElement *hCLCTL1ASerial;
  //  MonitorElement *hCLCTL1AChamberMeans;
  //  MonitorElement *hCLCTL1A2DMeans;
  MonitorElement *hCLCTL1A2Denominator;
  MonitorElement *hCLCTL1A2DNumerator;
};
#endif

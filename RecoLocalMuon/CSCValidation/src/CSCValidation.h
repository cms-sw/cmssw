#ifndef RecoLocalMuon_CSCValidation_H
#define RecoLocalMuon_CSCValidation_H

/** \class CSCValidation
 *
 * Package to validate local CSC reconstruction:
 *    DIGIS
 *    recHits
 *    segments
 *    L1 trigger
 *    CSC STA muons
 *    Various efficiencies
 *
 * Responsible:
 *    Andy Kubik, Northwestern University
 */

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCLCTData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCRPCData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"
#include <EventFilter/CSCRawToDigi/interface/CSCMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
//FEDRawData
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "RecoLocalMuon/CSCValidation/src/CSCValHists.h"
#include "TVector3.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"
#include "TProfile2D.h"

class CSCValidation : public edm::EDAnalyzer {
public:
  /// Constructor
  CSCValidation(const edm::ParameterSet &pset);

  /// Destructor
  ~CSCValidation() override;

  /// Perform the analysis
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;
  void endJob() override;

  // for noise module
  struct ltrh {
    bool operator()(const CSCRecHit2D &rh1, const CSCRecHit2D &rh2) const {
      return ((rh1.localPosition()).x() - (rh2.localPosition()).x()) < 0;
    }
  };

protected:
private:
  // these are the "modules"
  // if you would like to add code to CSCValidation, please do so by adding an
  // extra module in the form of an additional private member function
  void doOccupancies(const edm::Handle<CSCStripDigiCollection> &strips,
                     const edm::Handle<CSCWireDigiCollection> &wires,
                     const edm::Handle<CSCRecHit2DCollection> &recHits,
                     const edm::Handle<CSCSegmentCollection> &cscSegments);
  void doStripDigis(const edm::Handle<CSCStripDigiCollection> &strips);
  void doWireDigis(const edm::Handle<CSCWireDigiCollection> &wires);
  void doRecHits(const edm::Handle<CSCRecHit2DCollection> &recHits, const edm::ESHandle<CSCGeometry> &cscGeom);
  void doSimHits(const edm::Handle<CSCRecHit2DCollection> &recHits, const edm::Handle<edm::PSimHitContainer> &simHits);
  void doPedestalNoise(const edm::Handle<CSCStripDigiCollection> &strips);
  void doSegments(const edm::Handle<CSCSegmentCollection> &cscSegments, const edm::ESHandle<CSCGeometry> &cscGeom);
  void doResolution(const edm::Handle<CSCSegmentCollection> &cscSegments, const edm::ESHandle<CSCGeometry> &cscGeom);
  void doEfficiencies(const edm::Handle<CSCWireDigiCollection> &wires,
                      const edm::Handle<CSCStripDigiCollection> &strips,
                      const edm::Handle<CSCRecHit2DCollection> &recHits,
                      const edm::Handle<CSCSegmentCollection> &cscSegments,
                      const edm::ESHandle<CSCGeometry> &cscGeom);
  void doGasGain(const CSCWireDigiCollection &, const CSCStripDigiCollection &, const CSCRecHit2DCollection &);
  void doCalibrations(const edm::EventSetup &eventSetup);
  void doAFEBTiming(const CSCWireDigiCollection &);
  void doCompTiming(const CSCComparatorDigiCollection &);
  void doADCTiming(const CSCRecHit2DCollection &);
  void doNoiseHits(const edm::Handle<CSCRecHit2DCollection> &recHits,
                   const edm::Handle<CSCSegmentCollection> &cscSegments,
                   edm::ESHandle<CSCGeometry> cscGeom,
                   const edm::Handle<CSCStripDigiCollection> &strips);
  bool doTrigger(const edm::Handle<L1MuGMTReadoutCollection> &pCollection);
  void doStandalone(const edm::Handle<reco::TrackCollection> &saMuons);
  void doTimeMonitoring(const edm::Handle<CSCRecHit2DCollection> &recHits,
                        const edm::Handle<CSCSegmentCollection> &cscSegments,
                        const edm::Handle<CSCALCTDigiCollection> &alcts,
                        const edm::Handle<CSCCLCTDigiCollection> &clcts,
                        const edm::Handle<CSCCorrelatedLCTDigiCollection> &correlatedlcts,
                        const edm::Handle<L1MuGMTReadoutCollection> &pCollection,
                        const edm::ESHandle<CSCGeometry> &cscGeom,
                        const edm::EventSetup &eventSetup,
                        const edm::Event &event);
  bool doHLT(const edm::Handle<edm::TriggerResults> &hltResults);

  // some useful functions
  bool filterEvents(const edm::Handle<CSCRecHit2DCollection> &recHits,
                    const edm::Handle<CSCSegmentCollection> &cscSegments,
                    const edm::Handle<reco::TrackCollection> &saMuons);
  float fitX(const CLHEP::HepMatrix &sp, const CLHEP::HepMatrix &ep);
  float getSignal(const CSCStripDigiCollection &stripdigis, CSCDetId idRH, int centerStrip);
  float getthisSignal(const CSCStripDigiCollection &stripdigis, CSCDetId idRH, int centerStrip);
  int getWidth(const CSCStripDigiCollection &stripdigis, CSCDetId idRH, int centerStrip);
  void findNonAssociatedRecHits(const edm::ESHandle<CSCGeometry> &cscGeom,
                                const edm::Handle<CSCStripDigiCollection> &strips);
  int chamberSerial(CSCDetId id);
  int ringSerial(CSCDetId id);

  // these functions handle Stoyan's efficiency code
  void fillEfficiencyHistos(int bin, int flag);
  void getEfficiency(float bin, float Norm, std::vector<float> &eff);
  void histoEfficiency(TH1F *readHisto, TH1F *writeHisto);
  double lineParametrization(double z1Position, double z2Position, double z1Direction) {
    double parameterLine = (z2Position - z1Position) / z1Direction;
    return parameterLine;
  }
  double extrapolate1D(double initPosition, double initDirection, double parameterOfTheLine) {
    double extrapolatedPosition = initPosition + initDirection * parameterOfTheLine;
    return extrapolatedPosition;
  }
  bool withinSensitiveRegion(const LocalPoint &localPos,
                             const std::array<const float, 4> &layerBounds,
                             int station,
                             int ring,
                             float shiftFromEdge,
                             float shiftFromDeadZone);

  // counters
  int nEventsAnalyzed;
  int rhTreeCount;
  int segTreeCount;
  bool firstEvent;
  bool cleanEvent;

  //
  //
  // The root file for the histograms.
  TFile *theFile;

  //
  //
  // input parameters for this module
  bool makePlots;
  bool makeComparisonPlots;
  std::string refRootFile;
  bool writeTreeToFile;
  bool isSimulation;
  std::string rootFileName;
  bool detailedAnalysis;
  bool useDigis;
  bool makeHLTPlots;

  // filters
  bool useQualityFilter;
  bool useTriggerFilter;

  // quality filter parameters
  double pMin;
  double chisqMax;
  int nCSCHitsMin, nCSCHitsMax;
  double lengthMin, lengthMax;
  double deltaPhiMax;
  double polarMin, polarMax;

  edm::EDGetTokenT<FEDRawDataCollection> rd_token;
  edm::EDGetTokenT<CSCWireDigiCollection> wd_token;
  edm::EDGetTokenT<CSCStripDigiCollection> sd_token;
  edm::EDGetTokenT<CSCComparatorDigiCollection> cd_token;
  edm::EDGetTokenT<CSCALCTDigiCollection> al_token;
  edm::EDGetTokenT<CSCCLCTDigiCollection> cl_token;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> co_token;
  edm::EDGetTokenT<CSCRecHit2DCollection> rh_token;
  edm::EDGetTokenT<CSCSegmentCollection> se_token;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> l1_token;
  edm::EDGetTokenT<edm::TriggerResults> tr_token;
  edm::EDGetTokenT<reco::TrackCollection> sa_token;
  edm::EDGetTokenT<edm::PSimHitContainer> sh_token;

  // module on/off switches
  bool makeOccupancyPlots;
  bool makeTriggerPlots;
  bool makeStripPlots;
  bool makeWirePlots;
  bool makeRecHitPlots;
  bool makeSimHitPlots;
  bool makeSegmentPlots;
  bool makeResolutionPlots;
  bool makePedNoisePlots;
  bool makeEfficiencyPlots;
  bool makeGasGainPlots;
  bool makeAFEBTimingPlots;
  bool makeCompTimingPlots;
  bool makeADCTimingPlots;
  bool makeRHNoisePlots;
  bool makeCalibPlots;
  bool makeStandalonePlots;
  bool makeTimeMonitorPlots;

  // The histo managing object
  CSCValHists *histos;

  // tmp histos for Efficiency
  TH1F *hSSTE;
  TH1F *hRHSTE;
  TH1F *hSEff;
  TH1F *hRHEff;
  TH2F *hSSTE2;
  TH2F *hRHSTE2;
  TH2F *hStripSTE2;
  TH2F *hWireSTE2;
  TH2F *hSEff2;
  TH2F *hRHEff2;
  TH2F *hStripEff2;
  TH2F *hWireEff2;
  TH2F *hEffDenominator;
  TH2F *hSensitiveAreaEvt;

  // occupancy
  TH2I *hOWires;
  TH2I *hOStrips;
  TH2I *hORecHits;
  TH2I *hOSegments;

  /// Maps and vectors for module doGasGain()
  std::vector<int> nmbhvsegm;
  std::map<int, std::vector<int> > m_wire_hvsegm;
  std::map<int, int> m_single_wire_layer;

  //maps to store the DetId and associated RecHits
  std::multimap<CSCDetId, CSCRecHit2D> AllRechits;
  std::multimap<CSCDetId, CSCRecHit2D> SegRechits;
  std::multimap<CSCDetId, CSCRecHit2D> NonAssociatedRechits;
  std::map<CSCRecHit2D, float, ltrh> distRHmap;

  int typeIndex(CSCDetId id) {
    // linearlized index bases on endcap, station, and ring
    int index = 0;
    if (id.station() == 1) {
      index = id.ring() + 1;
      if (id.ring() == 4)
        index = 1;
    } else
      index = id.station() * 2 + id.ring();
    if (id.endcap() == 1)
      index = index + 10;
    if (id.endcap() == 2)
      index = 11 - index;
    return index;
  }
};
#endif

#ifndef Phase2TrackerMonitorDigi_h
#define Phase2TrackerMonitorDigi_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

class MonitorElement;
class PixelDigi;
class Phase2TrackerDigi;
class TrackerGeometry;
class TrackerDigiGeometryRecord;
class TrackerTopologyRcd;

class Phase2TrackerMonitorDigi : public DQMEDAnalyzer {
public:
  explicit Phase2TrackerMonitorDigi(const edm::ParameterSet&);
  ~Phase2TrackerMonitorDigi() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  std::string getHistoId(uint32_t det_id, bool flag);

  struct DigiMEs {
    MonitorElement* NumberOfDigisPerDet;
    MonitorElement* DigiOccupancyP;
    MonitorElement* DigiOccupancyS;
    MonitorElement* ChargeXYMap;
    MonitorElement* PositionOfDigisP;
    MonitorElement* PositionOfDigisS;
    MonitorElement* ChargeOfDigis;
    MonitorElement* ChargeOfDigisVsWidth;
    MonitorElement* TotalNumberOfDigisPerLayer;
    MonitorElement* NumberOfHitDetectorsPerLayer;
    MonitorElement* NumberOfClustersPerDet;
    MonitorElement* ClusterWidth;
    MonitorElement* ClusterPositionP;
    MonitorElement* ClusterPositionS;
    MonitorElement* FractionOfOvTBits;
    MonitorElement* FractionOfOvTBitsVsEta;
    MonitorElement* EtaOccupancyProfP;
    MonitorElement* EtaOccupancyProfS;
    unsigned int nDigiPerLayer;
    unsigned int nHitDetsPerLayer;
  };

  struct Ph2DigiCluster {
    int charge;
    int position;
    int width;
    int column;
  };

  MonitorElement* XYPositionMap;
  MonitorElement* RZPositionMap;
  MonitorElement* XYOccupancyMap;
  MonitorElement* RZOccupancyMap;

private:
  void bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id);
  void fillITPixelDigiHistos(const edm::Handle<edm::DetSetVector<PixelDigi>> handle);
  void fillOTDigiHistos(const edm::Handle<edm::DetSetVector<Phase2TrackerDigi>> handle);
  void fillDigiClusters(DigiMEs& mes, std::vector<Ph2DigiCluster>& digi_clusters);

  edm::ParameterSet config_;
  std::map<std::string, DigiMEs> layerMEs;
  bool pixelFlag_;
  bool clsFlag_;
  std::string geomType_;
  edm::InputTag otDigiSrc_;
  edm::InputTag itPixelDigiSrc_;
  const edm::EDGetTokenT<edm::DetSetVector<Phase2TrackerDigi>> otDigiToken_;
  const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> itPixelDigiToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
};
#endif

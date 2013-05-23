#ifndef FastSimulation_CaloRecHitsProducer_NoiseCheckBarreler_h
#define FastSimulation_CaloRecHitsProducer_NoiseCheckBarreler_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"


class NoiseCheckBarrel : public edm::EDAnalyzer {

public:
  explicit NoiseCheckBarrel(const edm::ParameterSet&);
  ~NoiseCheckBarrel();

  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  virtual void beginRun(edm::Run const&, edm::EventSetup const& );
  void beginJobAnalyze(const edm::EventSetup & c);
  virtual void endRun();
private:
  DQMStore * dbe;
  std::vector<MonitorElement*> individual_histos;
  MonitorElement * RecHit;
  MonitorElement * NoiseSigma;
  MonitorElement * NoiseSigmaMap;
  MonitorElement * NoiseSigmaClean;
  MonitorElement * NoiseSigmaMapClean;
  MonitorElement * LowSigmaChannelStatus;
  MonitorElement * HighSigmaChannelStatus;
  MonitorElement * NHits;
  MonitorElement * NChannelBad;
  std::vector<uint16_t>  chanStatus_;

  //  CaloGeometryHelper myGeometry;
  unsigned counter;
  bool m_firstTimeAnalyze ;
  std::string rootFileName;
  std::vector<float> IC;
  std::vector<float> ICMC;
  double threshold_;
  float adcToGeV_;
  unsigned Ngood_;
};

#endif

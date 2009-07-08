#ifndef FastSimulation_CaloRecHitsProducer_NoiseChecker_h
#define FastSimulation_CaloRecHitsProducer_NoiseChecker_h

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

class NoiseCheck : public edm::EDAnalyzer {

public:
  explicit NoiseCheck(const edm::ParameterSet&);
  ~NoiseCheck();

  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  virtual void beginRun(edm::Run const&, edm::EventSetup const& );
  void beginJobAnalyze(const edm::EventSetup & c);
  virtual void endRun();
private:
  DQMStore * dbe;
  std::vector<MonitorElement*> individual_histos;
  MonitorElement * EEPlus;
  MonitorElement * EEMinus;
  MonitorElement * EEPP, *EEPN, * EENN, *EENP;

  //  CaloGeometryHelper myGeometry;

  bool m_firstTimeAnalyze ;
};

#endif

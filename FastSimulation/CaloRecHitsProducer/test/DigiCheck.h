#ifndef FastSimulation_CaloRecHitsProducer_DigiChecker_h
#define FastSimulation_CaloRecHitsProducer_DigiChecker_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"

class DigiCheck : public edm::EDAnalyzer {

public:
  explicit DigiCheck(const edm::ParameterSet&);
  ~DigiCheck();

  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  virtual void beginRun(edm::Run const&, edm::EventSetup const& );
  void beginJobAnalyze(const edm::EventSetup & c);
  virtual void endRun();
private:
  DQMStore * dbe;
  MonitorElement* h0b;
  MonitorElement* h0e;
  MonitorElement* h1b;
  MonitorElement* h2b;
  MonitorElement* h3b;
  MonitorElement* h1e;
  MonitorElement* h2e;
  MonitorElement* h3e;
  MonitorElement* h4;
  MonitorElement* h5;
  MonitorElement* h6;
  MonitorElement* h7;
  MonitorElement* h8;
  MonitorElement* h9;

  const EcalTrigTowerConstituentsMap* eTTmap_;  
  CaloGeometryHelper myGeometry;
  std::map<EcalTrigTowerDetId,double> mapTow_sintheta;

  bool m_firstTimeAnalyze ;
};

#endif

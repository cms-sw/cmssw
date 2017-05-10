#ifndef RPCEfficiencyPerRingLayer_H
#define RPCEfficiencyPerRingLayer_H


/** \class RPCEfficiencyPerRingLayer
 * *
 *  RPCEfficiencyPerRingLayer
 *
 *  \author Cesare Calabria
 *   
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include <memory>
#include <string>

#include "DQMServices/Core/interface/DQMStore.h"
//class DQMStore;
//class RPCDetId;


class RPCEfficiencyPerRingLayer:public edm::EDAnalyzer {
public:

  /// Constructor
  RPCEfficiencyPerRingLayer(const edm::ParameterSet& iConfig);
  
  /// Destructor
  virtual ~RPCEfficiencyPerRingLayer();

  /// BeginJob
  void beginJob();

  //Begin Run
   void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
   //End Run
   void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Analyze  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  MonitorElement * EfficiencyPerRing;
  MonitorElement * EfficiencyPerLayer;
  
 private:
  

  int  numberOfDisks_;
  int innermostRings_ ;
  bool SaveFile;

  std::string NameFile;

  DQMStore* dbe_;

  std::string globalFolder_;

};

#endif

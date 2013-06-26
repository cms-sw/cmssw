#ifndef RPCEfficiencyShiftHisto_H
#define RPCEfficiencyShiftHisto_H


/** \class RPCEfficiencyShiftHisto
 * *
 *  RPCEfficiencyShiftHisto
 *
 *  $Date: 2010/08/16 10:16:09 $
 *  $Revision: 1.1 $
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


class DQMStore;
class RPCDetId;


class RPCEfficiencyShiftHisto:public edm::EDAnalyzer {
public:

  /// Constructor
  RPCEfficiencyShiftHisto(const edm::ParameterSet& iConfig);
  
  /// Destructor
  virtual ~RPCEfficiencyShiftHisto();

  /// BeginJob
  void beginJob();

  //Begin Run
   void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
   //End Run
   void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Analyze  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  MonitorElement * EffBarrelRoll;
  MonitorElement * EffEndcapPlusRoll;
  MonitorElement * EffEndcapMinusRoll;
  MonitorElement * RollPercentage;
  
 private:
  
  bool SaveFile;

  std::string NameFile;
  int  numberOfDisks_;
  int effCut_;

  DQMStore* dbe_;

  std::string globalFolder_;

};

#endif

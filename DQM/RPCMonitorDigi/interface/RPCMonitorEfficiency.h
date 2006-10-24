#ifndef RPCMonitorEfficiency_h
#define RPCMonitorEfficiency_h

/** \class RPCMonitor
 *
 * Class for RPC Monitoring using RPCDigi and RPCRecHit.
 *
 *  $Date: 2006/10/14 10:32:01 $
 *  $Revision: 1.5 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include<string>
#include<map>
#include<fstream>

class RPCDetId;
class TFile;
class TH1F;
class TFile;
class TCanvas;
class TH2F;

class RPCMonitorEfficiency : public edm::EDAnalyzer {
 public:
  
  ///Constructor
  explicit RPCMonitorEfficiency( const edm::ParameterSet& );
  
  ///Destructor
  ~RPCMonitorEfficiency();
  
  //Operations
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  //  virtual void beginJob(const edm::EventSetup &);
  virtual void endJob(void);
  
  std::map<std::string, MonitorElement*> bookDetUnitMEEff(RPCDetId & detId);


 private:
  std::string nameInLog;
  bool EffSaveRootFile;
  int  EffSaveRootFileEventsInterval;
  std::string EffRootFileName;
  
  /// back-end interface

  DaqMonitorBEInterface * dbe;
  MonitorElement * h1;

  bool debug;
  std::string theRecHits4DLabel;
  std::string digiLabel;
  std::map<uint32_t, std::map<std::string, MonitorElement*> >  meCollection;
	
  TH1F *hPositionX;
  std::vector<uint32_t> _idList;
	
  std::vector<std::map<RPCDetId, int> > counter;
  std::vector<int> totalcounter;
  std::ofstream ofrej;

};

#endif

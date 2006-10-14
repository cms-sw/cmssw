#ifndef RPCMonitorEfficiency_h
#define RPCMonitorEfficiency_h

/** \class RPCMonitor
 *
 * Class for RPC Monitoring using RPCDigi and RPCRecHit.
 *
 *  $Date: 2006/09/19 07:49:10 $
 *  $Revision: 1.4 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include<string>
#include<map>

/* Base Class Headers */
#include <DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h> 
#include <DataFormats/DTRecHit/interface/DTRecSegment4D.h> 

/* Collaborating Class Declarations */
#include "FWCore/Framework/interface/Handle.h"
#include <Geometry/Surface/interface/Surface.h>
#include <Geometry/Surface/interface/BoundPlane.h>
#include <MagneticField/Engine/interface/MagneticField.h>

#include <TrackingTools/GeomPropagators/interface/Propagator.h>//

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

  const BoundPlane makeSurface(const edm::EventSetup & eventSetup,const DTRecSegment4D & theSegment);
  const MagneticField *makeMagneticField(const edm::EventSetup& eventSetup);
	
 private:
  int counter;
  std::string nameInLog;
  bool EffSaveRootFile;
  int  EffSaveRootFileEventsInterval;
  std::string EffRootFileName;

  /// back-end interface

  DaqMonitorBEInterface * dbe;
  MonitorElement * h1;

  LocalTrajectoryParameters makeLocalTrajectory(DTRecSegment4D theSegment);
  bool debug;
  //  TFile* theFile;
  std::string theRecHits4DLabel;
  std::string digiLabel;
  PropagationDirection theDir;
  //  std::string HistoOutFile;

  std::map<uint32_t, std::map<std::string, MonitorElement*> >  meCollection;
	
  TH1F *hPositionX;
  std::vector<uint32_t> _idList;

};

#endif

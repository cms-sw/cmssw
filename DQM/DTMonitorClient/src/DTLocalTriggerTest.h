#ifndef DTLocalTriggerTest_H
#define DTLocalTriggerTest_H


/** \class DTLocalTriggerTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/05/22 16:10:06 $
 *  $Revision: 1.2 $
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <boost/cstdint.hpp>
#include <string>
#include <map>

class DTChamberId;
class DTGeometry;
class TH1F;

class DTLocalTriggerTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTLocalTriggerTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTLocalTriggerTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// Book the new MEs (for each sector)
  void bookSectorHistos(int wheel, int sector, std::string folder, std::string htype );

  /// Book the new MEs (for each chamber)
  void bookChambHistos(DTChamberId chambId, std::string htype );

  /// Calculate phi range for histograms
  std::pair<float,float> phiRange(const DTChamberId& id);

  /// Compute efficiency plots
  void makeEfficiencyME(TH1F* numerator, TH1F* denominator, MonitorElement* result);

  /// Get the ME name
  std::string getMEName(std::string histoTag, std::string subfolder, const DTChamberId & chambid);

 private:

  int nevents;

  DaqMonitorBEInterface* dbe;
  std::string sourceFolder;
  edm::ParameterSet parameters;
  std::string hwSource;
  edm::ESHandle<DTGeometry> muonGeom;
  std::map<int,std::map<std::string,MonitorElement*> >      secME;
  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;

};

#endif

#ifndef DQMSourceExample_H
#define DQMSourceExample_H

/** \class DQMSourceExample
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/08/29 13:49:00 $
 *  $Revision: 1.2 $
 *  \author  M. Zanetti CERN
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/LuminosityBlock.h>


#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>



//
// class declaration
//

class DQMSourceExample : public edm::EDAnalyzer {
public:
   explicit DQMSourceExample( const edm::ParameterSet& );
   ~DQMSourceExample();
   
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

  virtual void endJob(void);

private:
      // ----------member data ---------------------------

  MonitorElement * h1;
  MonitorElement * h2;
  MonitorElement * h3;
  MonitorElement * h4;
  MonitorElement * h5;
  MonitorElement * h6;
  MonitorElement * h7;
  MonitorElement * h8;
  MonitorElement * h9;
  MonitorElement * i1;
  MonitorElement * f1;
  MonitorElement * s1;
  float XMIN; float XMAX;
  // event counter
  int counter;
  // back-end interface
  DaqMonitorBEInterface * dbe;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//



#endif



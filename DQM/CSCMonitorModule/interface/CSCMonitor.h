#ifndef CSCMonitor_H
#define CSCMonitor_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"


#include <map>
#include <string>

using namespace std;

class CSCMonitor {
public:
   explicit CSCMonitor( const edm::ParameterSet& );
   ~CSCMonitor();
   
 
// scommenta  void process(CSCDCCUnpacker & unpacker)
  
  map<string, MonitorElement*> book_chamber(int chamberID);
  map<string, MonitorElement*> book_common();

private:


  // back-end interface
  DaqMonitorBEInterface * dbe;


  map<int, map<string, MonitorElement *> > meCollection;
  
//  scommenta  CSCDDUEventData dduEvent;

  int nEvents;
  int dataLength;
  int dduBX;
  int L1ANumber;

};


#endif

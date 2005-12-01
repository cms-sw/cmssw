#ifndef CSCMonitor_H
#define CSCMonitor_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/MonitorInterface.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace std;



class CSCMonitor : public MonitorInterface{
public:
   explicit CSCMonitor( const edm::ParameterSet& );
   ~CSCMonitor();
   
 
  void process(CSCDCCEventData & dccData);
  
  void MonitorDDU(const CSCDDUEventData& dduEvent);
  
  
  map<string, MonitorElement*> book_chamber(int chamberID);
  map<string, MonitorElement*> book_common();

private:


  // back-end interface
  DaqMonitorBEInterface * dbe;


  map<int, map<string, MonitorElement *> > meCollection;
  
  bool printout;
  
  int nEvents;
  
  int dataLength;
  int dduBX;
  int L1ANumber;
  
  int FEBUnpacked;

};


#endif

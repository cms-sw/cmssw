#ifndef CSCMonitor_H
#define CSCMonitor_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCMonitorInterface.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace std;



class CSCMonitor : public CSCMonitorInterface{
public:
   explicit CSCMonitor( const edm::ParameterSet& );
   ~CSCMonitor();
   
 
  void process(CSCDCCEventData & dccData);
  
  void MonitorDDU(const CSCDDUEventData& dduEvent, int dduNumber);
  void MonitorDMB(std::vector<CSCEventData>::iterator data);
 
  
  map<string, MonitorElement*> book_chamber(int chamberID);
  map<string, MonitorElement*> book_common(int dduNumber);

  static const int maxDDU=50; 
  static const int maxCMBID=36; 

private:


  // back-end interface
  DaqMonitorBEInterface * dbe;


  map<int, map<string, MonitorElement *> > meDDU;
  map<int, map<string, MonitorElement *> > meChamber;
  bool dduBooked[maxDDU];
  bool cmbBooked[maxCMBID];

  bool printout;
  
  int nEvents;
  
  int dataLength;
  int dduBX;
  int L1ANumber;
  
  int FEBUnpacked;

};


#endif

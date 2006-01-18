#ifndef CSCMonitor_H
#define CSCMonitor_H

/** \class CSCMonitor
 *
 * Class for CSC Detector Monitoring.
 *  
 *  $Date: 2005/12/12 09:49:04 $
 *  $Revision: 1.7 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */


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
  void MonitorDMB(std::vector<CSCEventData>::iterator data, int dduNumber);
  void MonitorCFEB(std::vector<CSCEventData>::iterator data, int  ChamberID);
 
  map<string, MonitorElement*> book_chamber(int chamberID);
  map<string, MonitorElement*> book_common(int dduNumber);

  static const int maxDDU=50; 
  static const int maxCMBID=36; 

  static const int CSC_DMB_ID_MASK  = 0XF;
  static const int CSC_DMB_ID_SHIFT = 0;
  
  static const int CSC_CRATE_ID_MASK  = 0XFFF;
  static const int CSC_CRATE_ID_SHIFT = 4;
 
private:


  // back-end interface
  DaqMonitorBEInterface * dbe;


  map<int, map<string, MonitorElement *> > meDDU;
  map<int, map<string, MonitorElement *> > meChamber;
  bool dduBooked[maxDDU];
  bool cmbBooked[maxCMBID];

  bool printout;
  
  int nEvents;
  
  int dduBX[maxDDU];
  int L1ANumber[maxDDU];
  
  int FEBUnpacked;

};


#endif

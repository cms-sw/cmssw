#ifndef DTDataIntegrityTask_H
#define DTDataIntegrityTask_H

/** \class DTDataIntegrityTask
 *
 * Class for DT Data Integrity.
 *  
 *  $Date: 2006/04/13 17:16:36 $
 *  $Revision: 1.4 $
 *
 * \author Marco Zanetti  - INFN Padova
 *
 */

#include "EventFilter/DTRawToDigi/interface/DTDataMonitorInterface.h"

#include "EventFilter/DTRawToDigi/interface/DTROChainCoding.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace std;

class DTROS25Data;
class DTDDUData;


class DTDataIntegrityTask : public DTDataMonitorInterface {

public:

  explicit DTDataIntegrityTask( const edm::ParameterSet& ps);
  
  virtual ~DTDataIntegrityTask();
   
  void bookHistos(string folder, DTROChainCoding code);

  void processROS25(DTROS25Data & data, int dduID, int ros);
  void processFED(DTDDUData & data, int dduID);


private:

  edm::ParameterSet parameters;

  // back-end interface
  DaqMonitorBEInterface * dbe;
  
  DTROChainCoding coding;

  // Monitor Elements
  // <histoType, <index , histo> >    
  map<string, map<int, MonitorElement*> > dduHistos;
  // <histoType, <index , histo> >    
  map<string, map<int, MonitorElement*> > rosHistos;
  // <histoType, <tdcID, histo> >   
  map<string, map<int, MonitorElement*> > robHistos;

  int neventsDDU;
  int neventsROS25;
  string outputFile;

};


#endif


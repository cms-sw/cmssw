#ifndef DTDataIntegrityTask_H
#define DTDataIntegrityTask_H

/** \class DTDataIntegrityTask
 *
 * Class for DT Data Integrity.
 *  
 *  $Date: 2006/03/24 16:17:22 $
 *  $Revision: 1.2 $
 *
 * \author Marco Zanetti  - INFN Padova
 *
 */

#include "EventFilter/DTRawToDigi/interface/DTDataMonitorInterface.h"

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
   
  void bookHistos(string folder, int index = 0);

  void processROS25(DTROS25Data & data);
  void processFED(DTDDUData & data);


private:

  edm::ParameterSet parameters;

  // back-end interface
  DaqMonitorBEInterface * dbe;
  

  // Monitor Elements
  // <histoName, histo>    
  map<string, MonitorElement*> rosHistos;
  // <histoName, <robID, histo> >   
  map<string, map<int, MonitorElement*> > robHistos;

  int nevents;
  string outputFile;

};


#endif


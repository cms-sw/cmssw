#ifndef DTDataIntegrityTask_H
#define DTDataIntegrityTask_H

/** \class DTDataIntegrityTask
 *
 * Class for DT Data Integrity.
 *  
 *  $Date: 2006/02/21 19:03:11 $
 *  $Revision: 1.1 $
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


class DTDataIntegrityTask : public DTDataMonitorInterface {

public:

  explicit DTDataIntegrityTask( const edm::ParameterSet& ps);
  
  virtual ~DTDataIntegrityTask();
   
  void bookHistos(string folder, int index = 0);

  void process(DTROS25Data & data);
 
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


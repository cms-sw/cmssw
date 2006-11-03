#ifndef DTDataIntegrityTask_H
#define DTDataIntegrityTask_H

/** \class DTDataIntegrityTask
 *
 * Class for DT Data Integrity.
 *  
 *  $Date: 2006/10/18 18:03:50 $
 *  $Revision: 1.6 $
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

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>

class DTROS25Data;
class DTDDUData;


class DTDataIntegrityTask : public DTDataMonitorInterface {

public:

  explicit DTDataIntegrityTask( const edm::ParameterSet& ps,edm::ActivityRegistry& reg);
  
  virtual ~DTDataIntegrityTask();
   
  void bookHistos(std::string folder, DTROChainCoding code);

  void processROS25(DTROS25Data & data, int dduID, int ros);
  void processFED(DTDDUData & data, int dduID);

  void postEndJob();

private:

  bool debug;
  edm::ParameterSet parameters;

  // back-end interface
  DaqMonitorBEInterface * dbe;
  
  DTROChainCoding coding;

  // Monitor Elements
  // <histoType, <index , histo> >    
  std::map<std::string, std::map<int, MonitorElement*> > dduHistos;
  // <histoType, <index , histo> >    
  std::map<std::string, std::map<int, MonitorElement*> > rosHistos;
  // <histoType, <tdcID, histo> >   
  std::map<std::string, std::map<int, MonitorElement*> > robHistos;

  int neventsDDU;
  int neventsROS25;
  std::string outputFile;

};


#endif


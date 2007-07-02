#ifndef DTDataIntegrityTask_H
#define DTDataIntegrityTask_H

/** \class DTDataIntegrityTask
 *
 * Class for DT Data Integrity.
 *  
 *  $Date: 2007/03/27 14:10:22 $
 *  $Revision: 1.10 $
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
#include <list>

class DTROS25Data;
class DTDDUData;


class DTDataIntegrityTask : public DTDataMonitorInterface {

public:

  explicit DTDataIntegrityTask( const edm::ParameterSet& ps,edm::ActivityRegistry& reg);
  
  virtual ~DTDataIntegrityTask();
   
  void bookHistos(std::string folder, DTROChainCoding code);

  void processROS25(DTROS25Data & data, int dduID, int ros);
  void processFED(DTDDUData & dduData, const std::vector<DTROS25Data> & rosData, int dduID);

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
  // <histoType, histo> >    
  std::map<std::string, MonitorElement*> rosSHistos;
  // <histoType, <index , histo> >    
  std::map<std::string, std::map<int, MonitorElement*> > rosHistos;
  // <histoType, <tdcID, histo> >   
  std::map<std::string, std::map<int, MonitorElement*> > robHistos;

  int neventsDDU;
  int neventsROS25;
  std::string outputFile;
  
  //Event counter for the graphs VS time
  int myPrevEv;
  
  //Monitor TTS,ROS,FIFO VS time
  int myPrevTtsVal;
  int myPrevRosVal;
  int myPrevFifoVal[7];
  //list of pair<events, values>
  std::list<std::pair<int,int> > ttsVSTime;
  std::list<std::pair<int,int> > rosVSTime;
  std::list<std::pair<int,int*> > fifoVSTime;
 
};


#endif


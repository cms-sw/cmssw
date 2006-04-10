#ifndef DTDataMonitorInterface_h
#define DTDataMonitorInterface_h

/** \class DTDataMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
 *  $Date: 2006/02/21 19:14:53 $
 *  $Revision: 1.1 $
 *
 * \author M. Zanetti - INFN Padova
 *
 */

#include <EventFilter/DTRawToDigi/interface/DTControlData.h>
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>


class DTDataMonitorInterface{

public:

  DTDataMonitorInterface(){}
  virtual ~DTDataMonitorInterface(){}
  virtual void processROS25(DTROS25Data & data)=0;
  virtual void processFED(DTDDUData & data)=0;
  

private:
  

};

#endif

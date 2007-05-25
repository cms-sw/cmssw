#ifndef DTDataMonitorInterface_h
#define DTDataMonitorInterface_h

/** \class DTDataMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
 *  $Date: 2006/04/13 17:14:28 $
 *  $Revision: 1.3 $
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
  virtual void processROS25(DTROS25Data & data, int ddu, int ros)=0;
  virtual void processFED(DTDDUData & dduData, const std::vector<DTROS25Data> & rosData,int ddu)=0;
  

private:
  

};

#endif

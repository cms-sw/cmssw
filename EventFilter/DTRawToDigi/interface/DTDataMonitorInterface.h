#ifndef DTDataMonitorInterface_h
#define DTDataMonitorInterface_h

/** \class DTDataMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
 *  $Date: 2007/03/29 17:26:01 $
 *  $Revision: 1.4 $
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
  
  virtual void fedEntry(int dduID) = 0;
  virtual void fedFatal(int dduID) = 0;
  virtual void fedNonFatal(int dduID) = 0;


private:
  

};

#endif

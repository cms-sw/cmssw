#ifndef DTDataMonitorInterface_h
#define DTDataMonitorInterface_h

/** \class DTDataMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
 *  $Date: 2005/12/12 09:11:35 $
 *  $Revision: 1.2 $
 *
 * \author M. Zanetti - INFN Padova
 *
 */

#include <EventFilter/DTRawToDigi/interface/DTROS25Data.h>


class DTDataMonitorInterface{

public:

    DTDataMonitorInterface(){}
    virtual ~DTDataMonitorInterface(){}
    virtual void process(DTROS25Data & data)=0;

      

private:


};

#endif

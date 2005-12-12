#ifndef CSCMonitorInterface_h
#define CSCMonitorInterface_h

/** \class CSCMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
 *  $Date: 2005/11/11 10:31:56 $
 *  $Revision: 1.2 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */


class CSCDCCEventData;

class CSCMonitorInterface{

public:

    CSCMonitorInterface(){}
    virtual ~CSCMonitorInterface(){}
    virtual void process(CSCDCCEventData & dccData)=0;

      

private:


};

#endif

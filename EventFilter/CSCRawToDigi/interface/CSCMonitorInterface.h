#ifndef CSCMonitorInterface_h
#define CSCMonitorInterface_h

/** \class CSCMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
 *  $Date: 2007/06/13 15:33:26 $
 *  $Revision: 1.3 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */


class CSCDCCEventData;
class CSCDCCExaminer;

class CSCMonitorInterface{

public:

    CSCMonitorInterface(){}
    virtual ~CSCMonitorInterface(){}
    // virtual void process(CSCDCCEventData & dccData)=0;
    virtual void process(CSCDCCExaminer * examiner, CSCDCCEventData * dccData)=0;      

private:


};

#endif

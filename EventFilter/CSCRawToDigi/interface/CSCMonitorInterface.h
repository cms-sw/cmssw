#ifndef CSCMonitorInterface_h
#define CSCMonitorInterface_h

/** \class CSCMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
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

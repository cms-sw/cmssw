#ifndef CSCTFMonitorInterface_h
#define CSCTFMonitorInterface_h

/** \class CSCTFMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
 *  $Date: 2005/12/12 09:11:35 $
 *  $Revision: 1.2 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */


class CSCTFDCCEventData;
class CSCTFTBEventData;

class CSCTFMonitorInterface{

public:

    CSCTFMonitorInterface(){}
    virtual ~CSCTFMonitorInterface(){}
    virtual void process(CSCTFDCCEventData & dccData)=0;
    virtual void process(CSCTFTBEventData & tbdata)=0;
      

private:


};

#endif

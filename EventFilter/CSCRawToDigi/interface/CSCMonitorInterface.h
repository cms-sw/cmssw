#ifndef CSCMonitorInterface_h
#define CSCMonitorInterface_h

class CSCDCCEventData;

class CSCMonitorInterface{

public:

    CSCMonitorInterface(){}
    virtual ~CSCMonitorInterface(){}
    virtual void process(CSCDCCEventData & dccData){}

      

private:


};

#endif

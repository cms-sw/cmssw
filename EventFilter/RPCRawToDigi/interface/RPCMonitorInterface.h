#ifndef RPCMonitorInterface_h
#define RPCMonitorInterface_h

/** \class RPCMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
 *  $Date: 2005/12/12 09:11:35 $
 *  $Revision: 1.1 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */

class RPCEventData;

class RPCMonitorInterface{

public:

    RPCMonitorInterface(){}
    virtual ~RPCMonitorInterface(){}
    virtual void process(RPCEventData & rpcData)=0;

      

private:


};

#endif

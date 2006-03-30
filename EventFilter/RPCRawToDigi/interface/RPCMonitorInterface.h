#ifndef RPCMonitorInterface_h
#define RPCMonitorInterface_h

/** \class RPCMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
 *  $Date: 2005/12/15 17:45:16 $
 *  $Revision: 1.1 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */

class RPCFEDData;

class RPCMonitorInterface{

public:

    RPCMonitorInterface(){}
    virtual ~RPCMonitorInterface(){}
    virtual void process(RPCFEDData & rpcData)=0;

      

private:


};

#endif

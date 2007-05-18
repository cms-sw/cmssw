#ifndef RPCMonitorInterface_h
#define RPCMonitorInterface_h

/** \class RPCMonitorInterface
 *
 * Interface to the Data Quality Monitoring Module.
 *  
 *  $Date: 2006/03/30 14:41:07 $
 *  $Revision: 1.2 $
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

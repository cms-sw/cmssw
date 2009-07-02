#include <iostream>
#include <vector>

/* \class RPCGas
 *
 * Definition of RPC Gas data type for O2O 
 * 
 * \author Davide Pagano (Pavia)
 */


class RPCGas {

  public:

  struct GasItem {
    int dpid;
    float flowin;
    float flowout;
    int day;
    int time;
  };

  std::vector<GasItem> Gas_rpc;

  RPCGas(){
    
    std::cout << ">> creating RPC Gas object" << std::endl;

}
  virtual ~RPCGas(){}
  
};


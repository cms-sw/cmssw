#include <iostream>
#include <vector>

/* \class RPCGasT
 *
 * Definition of RPC Gas & Temp data type for O2O 
 * 
 * \author Davide Pagano (Pavia)
 */


class RPCGasT {

  public:

  struct GasItem {
    int dpid;
    float flowin;
    float flowout;
    int day;
    int time;
  };

  struct TempItem {
    int dpid;
    float value;
    int day;
    int time;
  };

  std::vector<GasItem> Gas_rpc;
  std::vector<TempItem> Temp_rpc;

  RPCGasT(){
    
    std::cout << ">> creating RPC Gas-Temp object" << std::endl;

}
  virtual ~RPCGasT(){}
  
};


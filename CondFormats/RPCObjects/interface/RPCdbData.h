#include <iostream>
#include <vector>

/* \class RPCPattern
 *
 * Definition of RPC data type for O2O 
 * 
 * \author Davide Pagano (Pavia)
 */


class RPCdbData {

  public:

  struct Item {
    int dpid;
    float value;
    int day;
    int time;
  };

  std::vector<Item> Imon_rpc;
  std::vector<Item> Vmon_rpc;
  std::vector<Item> Status_rpc;

  RPCdbData(){
    
    std::cout << ">> creating RPC conditioning object" << std::endl;

}
  virtual ~RPCdbData(){}
  
};


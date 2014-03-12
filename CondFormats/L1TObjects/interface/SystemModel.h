///
/// \class l1t::SystemModel
///
/// Description: map of L1T electronics
///              Includes subsytems, boards, cables
///              And firmware versions
///
/// Implementation:
///    
///
/// \author: Jim Brooke
///

#ifndef SystemModel_h
#define SystemModel_h

#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"

#include <iostream>



namespace l1t {

  class SystemModel {
    
  public:

    struct Board;
    struct Link;
    
    struct Subsystem {
      std::vector<Board> boards;
      std::vector<Link> links;
    };

    struct Board {
      std::vector<Link&> rxLinks;
      std::vector<Link&> txLinks;
    };

    struct Link {
      Board& tx;
      int txPort;
      Board& rx;
      int rxPort;
    };
    

  private:

    



  };

}

#endif

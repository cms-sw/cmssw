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

  // class to interpret a transmitted word on a link in terms of meaningful quantities
  class LinkFormat {
  public:
    LinkFormat();
    ~LinkFormat();
    // getters for all quantities
    // will return zero in case quantity is not transmitted on the link
    int getEt( int data );
    int getIEta( int data );
    int getIPhi( int data );
    int getQual( int data );
  private:    
    // bitmasks, most of which are not meaningful and set to zero
    int etMask_;
    int ietaMask_;
    int iphiMask_;
    int qualMask_;
  }


  // class to interpet position of a word on a link in terms of meaningful quantities
  // note that the word position has two indices, position and time
  class LinkMap {
  public: 
    int getIEta( iPos, iTime );
    int getIPhi( iPos, iTime );
  private:    
    //some data structure to store this info
  }
  

  class Link {
  public:
    Link();
    ~Link();
    Device* getRxDevice();
    Device* getTxDevice();
    int getRxPort();
    int getTxPort();
    LinkFormat* getFormat();
    LinkMap* getMap();
  private: 
    Device* txDevice_;
    int txPort_;
    Device* rxDevice_;
    int rxPort_;
    LinkFormat format_;
    LinkMap map_;
  }


  class Device {
  public:
    Device();
    ~Device();
    Link* getRxLink( int port );
    Link* getTxLink( int port );
  private: 
    std::vector<Link*> rxLinks_;
    std::vector<Link*> txLinks_;
  }


  class Board : public Device {
  public:
    Board();
    ~Board();
  }
  
  
  class SubSystem : public Device {
  public:
    SubSystem();
    ~SubSystem();
    std::string getName();
    Board* getBoard( int i);
  private:
    std::string name_;
    std::vector<Board> boards_;
  }
  
  
  class System {
  public:
    System();
    ~System();
    SubSystem* getSubsystem( std::string name ); // central access point
  private:    
    // physical system
    std::vector<SubSystem> subs_;
    std::vector<Link> links_;
  }
  

}

#endif

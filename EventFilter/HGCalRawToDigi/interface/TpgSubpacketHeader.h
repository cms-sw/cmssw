#ifndef Hgcal10gLinkReceiver_TpgSubpacketHeader_h
#define Hgcal10gLinkReceiver_TpgSubpacketHeader_h

/*
  [ 7: 0] :  8b Subpacket size in num of 64b words
  [15: 8] :  8b Channel ID (starting from 0)
  [19:16] :  4b Buffer occupancy status (4 MSBs)
  [23:20] :  4b Number of words per BX
  [31:24] :  8b Reserved
  [63:32] : 32b Header pattern (CAFECAFE)
*/

#include <iostream>
#include <iomanip>
#include <cstdint>


namespace Hgcal10gLinkReceiver {

  class TpgSubpacketHeader {
  
  public:
    TpgSubpacketHeader(uint32_t header) {
      reset(header);
    }
    
    void reset(uint32_t header) {
      _data = header;
      _data = _data << 32;
      //_data=0xcafecafe00000000;
    }

    bool validPattern(uint32_t header) const {
      //return (_data>>32)==0xcafecafe;
      return (_data>>32)==header;
    }

    uint64_t data() const {
      return _data;
    }
   
    uint32_t pattern() const {
      return _data>>32;
    }

    uint8_t numberOfWordsPerBx() const {
      return (_data>>20)&0xf;
    }
    
    uint8_t bufferOccupancyStatus() const {
      return (_data>>16)&0xf;
    }

    uint8_t channelId() const {
      return (_data>>8)&0xff;
    }

    uint8_t subpacketSize() const {
      return _data&0xff;
    }

    uint8_t numberOfBxs() const {
      if(numberOfWordsPerBx()==0) return 0;
      return subpacketSize()/numberOfWordsPerBx();
    }
    
    const TpgSubpacketHeader* nextSubpacketHeader() const {
      //std::cout << "This is the subpacket size::: "<< uint16_t(subpacketSize()) << std::endl;
      return this+subpacketSize()+1;
    }
    const TpgSubpacketHeader* prevSubpacketHeader(uint8_t prevSubpacketSize) const {
      //std::cout << "This is the subpacket size::: "<< uint16_t(prevSubpacketSize) << std::endl;
      return this - prevSubpacketSize -1;
    }


    void print(std::ostream &o=std::cout, std::string s="") const {
      o << s << "TpgSubpacketHeader::print()  Data = 0x"
        << std::hex << std::setfill('0')
        << std::setw(16) << _data
        << std::dec << std::setfill(' ')
	<< std::endl;
      o << s << " Pattern = 0x"
        << std::hex << std::setfill('0')
        << std::setw(8) << pattern()
        << std::dec << std::setfill(' ')
	<< std::endl;
      o << s << " Number of words per BX = "
	<< std::setw(2) << unsigned(numberOfWordsPerBx())
	<< std::endl;
      o << s << " Buffer occupancy status = "
	<< std::setw(2) << unsigned(bufferOccupancyStatus())
	<< std::endl;
      o << s << " Channel id = "
	<< std::setw(3) << unsigned(channelId()) << ", ";
      if(channelId()==0) {
	o << "main" << std::endl;
      } else {
	o << ((channelId()%2)==0?"RX":"TX") << " link "
	  << std::setw(3) << unsigned(channelId()/2)
	  << std::endl;
      }
      o << s << " Subpacket size = "
	<< std::setw(3) << unsigned(subpacketSize())
	<< " words" << std::endl;
      o << s << " Number of BXs = "
	<< std::setw(3) << unsigned(numberOfBxs())
	<< std::endl;
    }
    
  private:
    uint64_t _data;
  };

}

#endif

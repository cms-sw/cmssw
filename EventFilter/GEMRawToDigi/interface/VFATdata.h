#ifndef GEM_VFATdata
#define GEM_VFATdata

#include <vector>

namespace gem {
  class VFATdata 
  {
  public:    
    VFATdata(){}
    ~VFATdata(){}
  VFATdata(const uint8_t &b1010_, 
	   const uint16_t &BC_,
	   const uint8_t &b1100_, 
	   const uint8_t &EC_,
	   const uint8_t &Flag_, 
	   const uint8_t &b1110_, 
	   const uint16_t &ChipID_, 
	   const uint64_t &lsData_, 
	   const uint64_t &msData_, 
	   const uint16_t &crc_,
	   const uint16_t &crc_calc_,
	   const int &SlotNumber_,
	   const bool &isBlockGood_) : 
    fb1010(b1010_),
      fBC(BC_),
      fb1100(b1100_),
      fEC(EC_),
      fFlag(Flag_),
      fb1110(b1110_),
      fChipID(ChipID_),
      flsData(lsData_),
      fmsData(msData_),
      fcrc(crc_),
      fcrc_calc(crc_calc_),
      fSlotNumber(SlotNumber_),
      fisBlockGood(isBlockGood_){}

    //!Read first word from the block.
    void read_fw(uint64_t word)
    {
      fb1010 = 0x0f & (word >> 60);
      fBC = 0x0fff & (word >> 48);
      fb1100 = 0x0f & (word >> 44);
      fEC = word >> 36;
      fFlag = 0x0f & (word >> 32);
      fb1110 = 0x0f & (word >> 28);
      fChipID = 0x0fff & (word >> 16);
      fmsData = 0xffff000000000000 & (word << 48);
    }
    
    //!Read second word from the block.
    void read_sw(uint64_t word)
    {
      fmsData = fmsData | (0x0000ffffffffffff & word >> 16);
      flsData = 0xffff000000000000 & (word << 48);
    }
    
    //!Read third word from the block.
    void read_tw(uint64_t word)
    {
      flsData = flsData | (0x0000ffffffffffff & word >> 16);
      fcrc = word;
    }
    // make write_word function
    
    
    uint8_t   b1010      (){ return fb1010;      }
    uint16_t  BC         (){ return fBC;         }
    uint8_t   b1100      (){ return fb1100;      }
    uint8_t   EC         (){ return fEC;         }
    uint8_t   Flag       (){ return fFlag;       }
    uint8_t   b1110      (){ return fb1110;      }
    uint16_t  ChipID     (){ return fChipID;     }
    uint64_t  lsData     (){ return flsData;     }
    uint64_t  msData     (){ return fmsData;     }
    uint16_t  crc        (){ return fcrc;        }
    uint16_t  crc_calc   (){ return fcrc_calc;   }
    int       SlotNumber (){ return fSlotNumber; }
    bool      isBlockGood(){ return fisBlockGood;}
    
  private:
    
    uint8_t  fb1010;                   ///<1010:4 Control bits, shoud be 1010
    uint16_t fBC;                      ///<Bunch Crossing number, 12 bits
    uint8_t  fb1100;                   ///<1100:4, Control bits, shoud be 1100
    uint8_t  fEC;                      ///<Event Counter, 8 bits
    uint8_t  fFlag;                    ///<Control Flags: 4 bits, Hamming Error/AFULL/SEUlogic/SUEI2C
    uint8_t  fb1110;                   ///<1110:4 Control bits, shoud be 1110
    uint16_t fChipID;                  ///<Chip ID, 12 bits
    uint64_t flsData;                  ///<channels from 1to64 
    uint64_t fmsData;                  ///<channels from 65to128
    uint16_t fcrc;                     ///<Check Sum value, 16 bits
    uint16_t fcrc_calc;                ///<Check Sum value recalculated, 16 bits
    int      fSlotNumber;              ///<Calculated chip position
    bool     fisBlockGood;             ///<Shows if block is good (control bits, chip ID and CRC checks)
  
  };
}

#endif

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
      fisBlockGood(isBlockGood_){
	fcrc = this->checkCRC();
      }

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
    uint64_t get_fw()
    {
      return
	(static_cast<uint64_t>(fb1010 & 0x0f) <<  60) |
	(static_cast<uint64_t>(fBC & 0x0fff) <<  48) |
	(static_cast<uint64_t>(fb1100 & 0x0f) <<  44) |
	(static_cast<uint64_t>(fEC) <<  36) |
	(static_cast<uint64_t>(fFlag & 0x0f) <<  32) |
	(static_cast<uint64_t>(fb1110 & 0x0f) <<  28) |
	(static_cast<uint64_t>(fChipID & 0x0fff) <<  16) |
	(static_cast<uint64_t>(fmsData & 0xffff000000000000) >> 48);
    }

    //!Read second word from the block.
    void read_sw(uint64_t word)
    {
      fmsData = fmsData | (0x0000ffffffffffff & word >> 16);
      flsData = 0xffff000000000000 & (word << 48);
    }
    uint64_t get_sw()
    {
      return
	(static_cast<uint64_t>(fmsData & 0x0000ffffffffffff) <<  16) |
	(static_cast<uint64_t>(flsData & 0xffff000000000000) >>  48);
    }
    
    //!Read third word from the block.
    void read_tw(uint64_t word)
    {
      flsData = flsData | (0x0000ffffffffffff & word >> 16);
      fcrc = word;
    }
    // make write_word function
    uint64_t get_tw()
    {
      return
	(static_cast<uint64_t>(flsData & 0x0000ffffffffffff) <<  16) |
	(static_cast<uint64_t>(fcrc));
    }
        
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

    uint16_t crc_cal(uint16_t crc_in, uint16_t dato)
    {
      uint16_t v = 0x0001;
      uint16_t mask = 0x0001;
      uint16_t d=0x0000;
      uint16_t crc_temp = crc_in;
      unsigned char datalen = 16;
      for (int i=0; i<datalen; i++){
	if (dato & v) d = 0x0001;
	else d = 0x0000;
	if ((crc_temp & mask)^d) crc_temp = crc_temp>>1 ^ 0x8408;
	else crc_temp = crc_temp>>1;
	v<<=1;
      }
      return(crc_temp);
    }
    
    uint16_t checkCRC()
    {
      uint16_t vfatBlockWords[12];
      vfatBlockWords[11] = ((0x000f & fb1010)<<12) | fBC;
      vfatBlockWords[10] = ((0x000f & fb1100)<<12) | ((0x00ff & fEC) <<4) | (0x000f & fFlag);
      vfatBlockWords[9]  = ((0x000f & fb1110)<<12) | fChipID;
      vfatBlockWords[8]  = (0xffff000000000000 & fmsData) >> 48;
      vfatBlockWords[7]  = (0x0000ffff00000000 & fmsData) >> 32;
      vfatBlockWords[6]  = (0x00000000ffff0000 & fmsData) >> 16;
      vfatBlockWords[5]  = (0x000000000000ffff & fmsData);
      vfatBlockWords[4]  = (0xffff000000000000 & flsData) >> 48;
      vfatBlockWords[3]  = (0x0000ffff00000000 & flsData) >> 32;
      vfatBlockWords[2]  = (0x00000000ffff0000 & flsData) >> 16;
      vfatBlockWords[1]  = (0x000000000000ffff & flsData);

      uint16_t crc_fin = 0xffff;
      for (int i = 11; i >= 1; i--){
	crc_fin = this->crc_cal(crc_fin, vfatBlockWords[i]);
      }  
      return(crc_fin);
    }
    
    uint16_t checkCRC(uint8_t b1010, uint16_t BC, uint8_t b1100,
		      uint8_t EC, uint8_t Flag, uint8_t b1110,
		      uint16_t ChipID, uint64_t msData, uint64_t lsData)
    {
      uint16_t vfatBlockWords[12];
      vfatBlockWords[11] = ((0x000f & b1010)<<12) | BC;
      vfatBlockWords[10] = ((0x000f & b1100)<<12) | ((0x00ff & EC) <<4) | (0x000f & Flag);
      vfatBlockWords[9]  = ((0x000f & b1110)<<12) | ChipID;
      vfatBlockWords[8]  = (0xffff000000000000 & msData) >> 48;
      vfatBlockWords[7]  = (0x0000ffff00000000 & msData) >> 32;
      vfatBlockWords[6]  = (0x00000000ffff0000 & msData) >> 16;
      vfatBlockWords[5]  = (0x000000000000ffff & msData);
      vfatBlockWords[4]  = (0xffff000000000000 & lsData) >> 48;
      vfatBlockWords[3]  = (0x0000ffff00000000 & lsData) >> 32;
      vfatBlockWords[2]  = (0x00000000ffff0000 & lsData) >> 16;
      vfatBlockWords[1]  = (0x000000000000ffff & lsData);

      uint16_t crc_fin = 0xffff;
      for (int i = 11; i >= 1; i--){
	crc_fin = this->crc_cal(crc_fin, vfatBlockWords[i]);
      }
      return(crc_fin);
    }
    
    uint16_t checkCRC(VFATdata * vfatData)
    {
      uint16_t vfatBlockWords[12]; 
      vfatBlockWords[11] = ((0x000f & vfatData->b1010())<<12) | vfatData->BC();
      vfatBlockWords[10] = ((0x000f & vfatData->b1100())<<12) | ((0x00ff & vfatData->EC()) <<4) | (0x000f & vfatData->Flag());
      vfatBlockWords[9]  = ((0x000f & vfatData->b1110())<<12) | vfatData->ChipID();
      vfatBlockWords[8]  = (0xffff000000000000 & vfatData->msData()) >> 48;
      vfatBlockWords[7]  = (0x0000ffff00000000 & vfatData->msData()) >> 32;
      vfatBlockWords[6]  = (0x00000000ffff0000 & vfatData->msData()) >> 16;
      vfatBlockWords[5]  = (0x000000000000ffff & vfatData->msData());
      vfatBlockWords[4]  = (0xffff000000000000 & vfatData->lsData()) >> 48;
      vfatBlockWords[3]  = (0x0000ffff00000000 & vfatData->lsData()) >> 32;
      vfatBlockWords[2]  = (0x00000000ffff0000 & vfatData->lsData()) >> 16;
      vfatBlockWords[1]  = (0x000000000000ffff & vfatData->lsData());

      uint16_t crc_fin = 0xffff;
      for (int i = 11; i >= 1; i--){
	crc_fin = this->crc_cal(crc_fin, vfatBlockWords[i]);
      }
      return(crc_fin);
    }    
    
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

#ifndef gem_readout_GEMDataAMCformat_h
#define gem_readout_GEMDataAMCformat_h

#include "EventFilter/GEMRawToDigi/interface/GEMslotContents.h"

#include <iostream>
#include <iomanip> 
#include <fstream>
#include <string>
#include <vector>

    struct GEMDataAMCformat {
      struct VFATData {
        uint16_t BC;          // 1010:4,   BC:12 
        uint16_t EC;          // 1100:4,   EC:8,      Flags:4
        uint16_t ChipID;      // 1110,     ChipID:12
        uint64_t lsData;      // channels from 1to64
        uint64_t msData;      // channels from 65to128
        uint32_t BXfrOH;      // :32       BX from OH  
        uint16_t crc;         // :16       CRC
      };    
    
      struct GEBData {
        uint64_t header;      // ZSFlag:24 ChamID:12
        uint64_t runhed;      // RunType:4 VT1:8 VT2:8 minTH:8 maxTH:8 Step:8 - Threshold Scan Header
        // RunType:4                                    - Latency Scan Header
        // RunType:4                                    - Cosmic Run Header 
        // RunType:4                                    - Data Takiing
        std::vector<VFATData> vfats;
        uint64_t trailer;     // OHcrc: 16 OHwCount:16  ChamStatus:16
      };

      struct GEMData {
        uint64_t header1;    // AmcNo:4      0000:4     LV1ID:24   BXID:12     DataLgth:20 
        uint64_t header2;    // User:32      OrN:16     BoardID:16
        uint64_t header3;    // DAVList:24   BufStat:24 DAVCount:5 FormatVer:3 MP7BordStat:8 
        std::vector<GEBData> gebs; // we have only one at 2015-Sep
        uint64_t trailer2;   // EventStat:32 GEBerrFlag:24  
        uint64_t trailer1;   // crc:32       LV1IDT:8   0000:4     DataLgth:20 
      };

      /*
       * GEM Data
       */

      static bool writeGEMhd1(std::string file, int event, const GEMData& gem) {
        std::ofstream outf(file.c_str(), std::ios_base::app );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf << std::hex << std::setw(16) << std::setfill('0') << gem.header1 << std::dec << std::endl;
        outf.close();
        return true;
      };	  

      static bool writeGEMhd1Binary(std::string file, int event, const GEMData& gem) {
        std::ofstream outf(file.c_str(), std::ios_base::app | std::ios::binary );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        uint64_t cdfHeader = 0x5fffffffffffffff;
        uint64_t amc13Header1 = 0xff1ffffffffffff0;
        uint64_t amc13Header2 = 0xffffffffffffffff;
        outf.write( (char*)&cdfHeader, sizeof(cdfHeader));
        outf.write( (char*)&amc13Header1, sizeof(amc13Header1));
        outf.write( (char*)&amc13Header2, sizeof(amc13Header2));
        outf.write( (char*)&gem.header1, sizeof(gem.header1));
        outf.close();
        return true;
      };

      static bool readGEMhd1Binary(std::ifstream& inpf, const GEMData& gem) {
        inpf.read( (char*)&gem.header1, sizeof(gem.header1));
        inpf.seekg (0, inpf.cur);
        if(inpf.eof()) return false;
        return true;
      };	  

      static bool readGEMhd1(std::ifstream& inpf, GEMData& gem) {
        inpf >> std::hex >> gem.header1;
        if(inpf.eof()) return false;
        return true;
      };	  

      static bool writeGEMhd2(std::string file, int event, const GEMData& gem) {
        std::ofstream outf(file.c_str(), std::ios_base::app );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf << std::hex << std::setw(16) << std::setfill('0') << gem.header2 << std::dec << std::endl;
        outf.close();
        return true;
      };	  

      static bool writeGEMhd2Binary(std::string file, int event, const GEMData& gem) {
        std::ofstream outf(file.c_str(), std::ios_base::app | std::ios::binary );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf.write( (char*)&gem.header2, sizeof(gem.header2));
        outf.close();
        return true;
      };

      static bool readGEMhd2Binary(std::ifstream& inpf, const GEMData& gem) {
        inpf.read( (char*)&gem.header2, sizeof(gem.header2));
        inpf.seekg (0, inpf.cur);
        if(inpf.eof()) return false;
        return true;
      };

      static bool readGEMhd2(std::ifstream& inpf, GEMData& gem) {
        inpf >> std::hex >> gem.header2;
        if(inpf.eof()) return false;
        return true;
      };	  

      static bool writeGEMhd3(std::string file, int event, const GEMData& gem) {
        std::ofstream outf(file.c_str(), std::ios_base::app );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf << std::hex << std::setw(16) << std::setfill('0') << gem.header3 << std::dec << std::endl;
        outf.close();
        return true;
      };	  

      static bool writeGEMhd3Binary(std::string file, int event, const GEMData& gem) {
        std::ofstream outf(file.c_str(), std::ios_base::app | std::ios::binary );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf.write( (char*)&gem.header3, sizeof(gem.header3));
        outf.close();
        return true;
      };

      static bool readGEMhd3Binary(std::ifstream& inpf, const GEMData& gem) {
        inpf.read( (char*)&gem.header3, sizeof(gem.header3));
        inpf.seekg (0, inpf.cur);
        if(inpf.eof()) return false;
        return true;
      };

      static bool readGEMhd3(std::ifstream& inpf, GEMData& gem) {
        inpf >> std::hex >> gem.header3;
        if(inpf.eof()) return false;
        return true;
      };	  

      /*
       * GEB Data (One GEB board, 24 VFATs)
       */

      static bool writeGEBheader(std::string file, int event, const GEBData& geb) {
        std::ofstream outf(file.c_str(), std::ios_base::app );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf << std::hex << std::setw(16) << std::setfill('0') << geb.header << std::dec << std::endl;
        outf.close();
        return true;
      };	  

      static bool writeGEBheaderBinary(std::string file, int event, const GEBData& geb) {
        std::ofstream outf(file.c_str(), std::ios_base::app | std::ios::binary );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf.write( (char*)&geb.header, sizeof(geb.header));
        outf.close();
        return true;
      };
	  
      static bool readGEBheaderBinary(std::ifstream& inpf, const GEBData& geb) {
        inpf.read( (char*)&geb.header, sizeof(geb.header));
        inpf.seekg (0, inpf.cur);
        if(inpf.eof()) return false;
        return true;
      };

      static bool readGEBheader(std::ifstream& inpf, GEBData& geb) {
        inpf >> std::hex >> geb.header;
        if(inpf.eof()) return false;
        return true;
      };	  

      static bool printGEBheader(int event, const GEBData& geb) {
        if ( event<0) return false;
        std::cout << "Received tracking data word: event " << event << std::endl;
        std::cout << " 0x" << std::setw(8) << std::hex << geb.header << " ChamID " << ((0x000000fff0000000 & geb.header) >> 28) 
                  << std::dec << " sumVFAT " << (0x000000000fffffff & geb.header) << std::endl;
        return true;
      };	  

      static bool writeGEBrunhed(std::string file, int event, const GEBData& geb) {
        std::ofstream outf(file.c_str(), std::ios_base::app );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf << std::hex << std::setw(16) << std::setfill('0') << geb.runhed << std::dec << std::endl;
        outf.close();
        return true;
      };	  

      static bool writeGEBrunhedBinary(std::string file, int event, const GEBData& geb) {
        std::ofstream outf(file.c_str(), std::ios_base::app | std::ios::binary );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf.write( (char*)&geb.runhed, sizeof(geb.runhed));
        outf.close();
        return true;
      };
	  
      static bool readGEBrunhedBinary(std::ifstream& inpf, const GEBData& geb) {
        inpf.read( (char*)&geb.runhed, sizeof(geb.runhed));
        inpf.seekg (0, inpf.cur);
        if(inpf.eof()) return false;
        return true;
      };

      static bool readGEBrunhed(std::ifstream& inpf, GEBData& geb) {
        inpf >> std::hex >> geb.runhed;
        if(inpf.eof()) return false;
        return true;
      };	  

      static bool writeGEBtrailer(std::string file, int event, const GEBData& geb) {
        std::ofstream outf(file.c_str(), std::ios_base::app );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf << std::hex << std::setw(16) << std::setfill('0') << geb.trailer << std::dec << std::endl;
        outf.close();
        return true;
      };	  

      static bool writeGEBtrailerBinary(std::string file, int event, const GEBData& geb) {
        std::ofstream outf(file.c_str(), std::ios_base::app | std::ios::binary );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf.write( (char*)&geb.trailer, sizeof(geb.trailer));
        outf.close();
        return true;
      };
	  
      static bool readGEBtrailerBinary(std::ifstream& inpf, const GEBData& geb) {
        inpf.read( (char*)&geb.trailer, sizeof(geb.trailer));
        inpf.seekg (0, inpf.cur);
        if(inpf.eof()) return false;
        return true;
      };

      static bool readGEBtrailer(std::ifstream& inpf, GEBData& geb) {
        inpf >> std::hex >> geb.trailer;
        if(inpf.eof()) return false;
        return true;
      };	  

      static bool printGEBtrailer(int event, const GEBData& geb) {
        if ( event<0) return false;
        uint64_t OHcrc      = (0xffff000000000000 & geb.trailer) >> 48; 
        uint64_t OHwCount   = (0x0000ffff00000000 & geb.trailer) >> 32; 
        uint64_t ChamStatus = (0x00000000ffff0000 & geb.trailer) >> 16;
        std::cout << "GEM Camber Treiler: OHcrc " << std::hex << OHcrc << " OHwCount " << OHwCount << " ChamStatus " << ChamStatus << std::dec 
                  << std::endl;
        return true;
      };	  

      static bool writeGEMtr2(std::string file, int event, const GEMData& gem) {
        std::ofstream outf(file.c_str(), std::ios_base::app );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf << std::hex << std::setw(16) << std::setfill('0') << gem.trailer2 << std::dec << std::endl;
        outf.close();
        return true;
      };	  

      static bool writeGEMtr2Binary(std::string file, int event, const GEMData& gem) {
        std::ofstream outf(file.c_str(), std::ios_base::app | std::ios::binary );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf.write( (char*)&gem.trailer2, sizeof(gem.trailer2));
        outf.close();
        return true;
      };

      static bool readGEMtr2Binary(std::ifstream& inpf, const GEMData& gem) {
        inpf.read( (char*)&gem.trailer2, sizeof(gem.trailer2));
        inpf.seekg (0, inpf.cur);
        if(inpf.eof()) return false;
        return true;
      };

      static bool readGEMtr2(std::ifstream& inpf, GEMData& gem) {
        inpf >> std::hex >> gem.trailer2;
        if(inpf.eof()) return false;
        return true;
      };	  

      static bool writeGEMtr1(std::string file, int event, const GEMData& gem) {
        std::ofstream outf(file.c_str(), std::ios_base::app );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        outf << std::hex << std::setw(16) << std::setfill('0') << gem.trailer1 << std::dec << std::endl;
        outf.close();
        return true;
      };	  

      static bool writeGEMtr1Binary(std::string file, int event, const GEMData& gem) {
        std::ofstream outf(file.c_str(), std::ios_base::app | std::ios::binary );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        uint64_t amc13Trailer = 0xbadc0ffeebadcafe;
        uint64_t cdfTrailer = 0xafffffffffffffff;
        outf.write( (char*)&gem.trailer1, sizeof(gem.trailer1));
        outf.write( (char*)&amc13Trailer, sizeof(amc13Trailer));
        outf.write( (char*)&cdfTrailer, sizeof(cdfTrailer));
        outf.close();
        return true;
      };

      static bool readGEMtr1Binary(std::ifstream& inpf, const GEMData& gem) {
        inpf.read( (char*)&gem.trailer1, sizeof(gem.trailer1));
        inpf.seekg (0, inpf.cur);
        if(inpf.eof()) return false;
        return true;
      };

      static bool readGEMtr1(std::ifstream& inpf, GEMData& gem) {
        inpf >> std::hex >> gem.trailer1;
        if(inpf.eof()) return false;
        return true;
      };	  

      static bool writeVFATdata(std::string file, int event, const VFATData& vfat) {
        std::ofstream outf(file.c_str(), std::ios_base::app );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        // have to have 64 bit word lengths
        outf << std::hex << std::setw(16) << std::setfill('0')
             << ((((((uint64_t)vfat.BC<<16)+vfat.EC)<<16)+vfat.ChipID)<<16)+(vfat.msData>>48) << std::endl;
        outf << std::hex << std::setw(16) << std::setfill('0')
             << (((vfat.msData&0x0000ffffffffffff)<<16)+(vfat.lsData>>48)) << std::endl;
        outf << std::hex << std::setw(16) << std::setfill('0')
             << (((vfat.lsData&0x0000ffffffffffff)<<16)+vfat.crc)          << std::endl;
        outf << std::hex << std::setw(16) << std::setfill('0')
             << vfat.BXfrOH << std::dec << std::endl;
        //writeZEROline(file);
        outf.close();
        return true;
      };	  

      static bool printVFATdata(int event, const VFATData& vfat) {
        if ( event<0) return false;
        std::cout << "Received tracking data word:" << std::endl;
        std::cout << "BC      :: 0x" << std::setfill('0') << std::setw(4)  << std::hex << vfat.BC     << std::dec << std::endl;
        std::cout << "EC      :: 0x" << std::setfill('0') << std::setw(4)  << std::hex << vfat.EC     << std::dec << std::endl;
        std::cout << "ChipID  :: 0x" << std::setfill('0') << std::setw(4)  << std::hex << vfat.ChipID << std::dec << std::endl;
        std::cout << "<127:64>:: 0x" << std::setfill('0') << std::setw(16) << std::hex << vfat.msData << std::dec << std::endl;
        std::cout << "<63:0>  :: 0x" << std::setfill('0') << std::setw(16) << std::hex << vfat.lsData << std::dec << std::endl;
        std::cout << "BXfrOH  :: 0x" << std::setfill('0') << std::setw(8)  << std::hex << vfat.BXfrOH << std::dec << std::endl;
        std::cout << "crc     :: 0x" << std::setfill('0') << std::setw(4)  << std::hex << vfat.crc    << std::dec << std::endl;
        return true;
      };

      static bool writeVFATdataBinary(std::string file, int event, const VFATData& vfat) {
        std::ofstream outf(file.c_str(), std::ios_base::app | std::ios::binary );
        if ( event<0) return false;
        if (!outf.is_open()) return false;
        uint64_t w1;
        uint64_t w2;
        uint64_t w3;
        uint64_t bc = vfat.BC;
        uint64_t ec = vfat.EC;
        uint64_t ci = vfat.ChipID;
        w1 = 0xffffffffffffffff & ((bc <<48) | (ec << 32) | (ci << 16) | (vfat.msData >> 48));
        w2 = 0xffffffffffffffff & ((vfat.msData <<16) | (vfat.lsData >> 48));
        w3 = 0xffffffffffffffff & ((vfat.lsData <<16) | (vfat.crc ));
        outf.write( (char*)&w1,     sizeof(w1));
        outf.write( (char*)&w2,     sizeof(w2));
        outf.write( (char*)&w3,     sizeof(w3));
        //outf.write( (char*)&vfat.BC,     sizeof(vfat.BC));
        //outf.write( (char*)&vfat.EC,     sizeof(vfat.EC));
        //outf.write( (char*)&vfat.ChipID, sizeof(vfat.ChipID));
        //outf.write( (char*)&vfat.msData, sizeof(vfat.msData));
        //outf.write( (char*)&vfat.lsData, sizeof(vfat.lsData));  
        //outf.write( (char*)&vfat.crc,    sizeof(vfat.crc));
        //outf.write( (char*)&vfat.BXfrOH, sizeof(vfat.BXfrOH));
        outf.close();
        return true;
      };	  

      static bool readVFATdataBinary(std::ifstream& inpf, int event, VFATData& vfat) {
        if (event<0) return false;
        inpf.read( (char*)&vfat.BC,     sizeof(vfat.BC));
        inpf.read( (char*)&vfat.EC,     sizeof(vfat.EC));
        inpf.read( (char*)&vfat.ChipID, sizeof(vfat.ChipID));
        inpf.read( (char*)&vfat.msData, sizeof(vfat.msData));
        inpf.read( (char*)&vfat.lsData, sizeof(vfat.lsData));
        inpf.read( (char*)&vfat.crc,    sizeof(vfat.crc));
        inpf.read( (char*)&vfat.BXfrOH, sizeof(vfat.BXfrOH));
        inpf.seekg (0, inpf.cur);
        if(inpf.eof()) return false;
        return true;
      };	  

      static bool readVFATdata(std::ifstream& inpf, int event, VFATData& vfat) {
        if (event<0) return false;
        uint16_t msTmpUp, lsTmpUp;
        uint16_t msTmpMid, lsTmpMid;
        uint32_t msTmpLow, lsTmpLow;
        inpf >> std::hex >> vfat.BC  >> vfat.EC >> vfat.ChipID >> msTmpUp;
        inpf >> std::hex >> msTmpMid >> msTmpLow >> lsTmpUp;
        inpf >> std::hex >> lsTmpMid >> lsTmpLow >> vfat.crc;
        inpf >> std::hex >> vfat.BXfrOH;
        vfat.msData = ((((uint64_t)msTmpUp<<16)+msTmpMid)<<32)+msTmpLow;
        vfat.lsData = ((((uint64_t)lsTmpUp<<16)+lsTmpMid)<<32)+lsTmpLow;
        if(inpf.eof()) return false;
        return true;
      };	  

      //
      // Useful printouts 
      //
      static void show4bits(uint8_t x) {
        int i;
        const unsigned long unit = 1;
        for(i=(sizeof(uint8_t)*4)-1; i>=0; i--)
          (x & ((unit)<<i))?putchar('1'):putchar('0');
        //printf("\n");
      }

      static void show16bits(uint16_t x) {
        int i;
        const unsigned long unit = 1;
        for(i=(sizeof(uint16_t)*8)-1; i>=0; i--)
          (x & ((unit)<<i))?putchar('1'):putchar('0');
        printf("\n");
      }

      static void show24bits(uint32_t x) {
        int i;
        const unsigned long unit = 1;
        for(i=(sizeof(uint32_t)*8)-8-1; i>=0; i--)
          (x & ((unit)<<i))?putchar('1'):putchar('0');
        printf("\n");
      }

      static void show32bits(uint32_t x) {
        int i;
        const unsigned long unit = 1;
        for(i=(sizeof(uint32_t)*8)-1; i>=0; i--)
          (x & ((unit)<<i))?putchar('1'):putchar('0');
        printf("\n");
      }

      static void show64bits(uint64_t x) {
        int i;
        const unsigned long unit = 1;
        for(i=(sizeof(uint64_t)*8)-1; i>=0; i--)
          (x & ((unit)<<i))?putchar('1'):putchar('0');
        printf("\n");
      }

      static bool printVFATdataBits(int event, const VFATData& vfat) {
        if ( event<0) return false;
        std::cout << "\nReceived VFAT data : " << event << std::endl;

        uint8_t   b1010 = (0xf000 & vfat.BC) >> 12;
        show4bits(b1010); std::cout << " BC     0x" << std::hex << (0x0fff & vfat.BC) 
                                    << std::setfill('0') << std::setw(8) << "    BX 0x" << vfat.BXfrOH << std::dec << std::endl;

        uint8_t   b1100 = (0xf000 & vfat.EC) >> 12;
        uint16_t   EC   = (0x0ff0 & vfat.EC) >> 4;
        uint8_t   Flag  = (0x000f & vfat.EC);
        show4bits(b1100); std::cout << " EC     0x" << std::hex << EC << std::dec << std::endl; 
        show4bits(Flag);  std::cout << " Flags " << std::endl;

        uint8_t   b1110 = (0xf000 & vfat.ChipID) >> 12;
        uint16_t ChipID = (0x0fff & vfat.ChipID);
        show4bits(b1110); std::cout << " ChipID 0x" << std::hex << ChipID << std::dec << " " << std::endl;

        /* 
        std::cout << "     bxNum  0x" << std::hex << ((0xff00 & vfat.bxNum) >> 8) << " SBit " << (0x00ff & vfat.bxNum) << std::endl;
        */

        std::cout << " <127:64>:: 0x" << std::setfill('0') << std::setw(16) << std::hex << vfat.msData << std::dec << std::endl;
        std::cout << " <63:0>  :: 0x" << std::setfill('0') << std::setw(16) << std::hex << vfat.lsData << std::dec << std::endl;
        std::cout << "     crc    0x" << std::hex << vfat.crc << std::dec << std::endl;

        //std::cout << " " << std::endl; show16bits(vfat.EC);

        return true;
      };

      static bool writeZEROline(std::string file) {
        std::ofstream outf(file.c_str(), std::ios_base::app );
        if (!outf.is_open()) return false;
        outf << "\n" << std::endl;
        outf.close();
        return true;
      };	  
    }; /// end struct GEMDataAMCformat
  typedef GEMDataAMCformat::GEMData  AMCGEMData;
  typedef GEMDataAMCformat::GEBData  AMCGEBData;
  typedef GEMDataAMCformat::VFATData AMCVFATData;
#endif

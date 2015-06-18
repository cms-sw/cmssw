#ifndef EVF_FEDINTERFACE_FED1024_H
#define EVF_FEDINTERFACE_FED1024_H


#include <stddef.h>
#include "EventFilter/FEDInterface/interface/fed_header.h"
#include "EventFilter/FEDInterface/interface/FEDConstants.h"

namespace evf{
  namespace evtn{
    
    union sizes{
      uint64_t sizeword;
      struct{
	const unsigned char headerSize;
	const unsigned char summarySize;
	const unsigned char L1AhistSize;
	const unsigned char BSTSize;
	const unsigned char reserved0;
	const unsigned char reserved1;
	const unsigned char reserved2;
	const unsigned char BGOSize;
      } size;
    };

    class TCDSFEDHeader{
    public:

      union tcdsfedheader{
        uint64_t word;
        struct {
          uint32_t sourceid; 
          uint32_t eventid; 
        } header;
      };
      TCDSFEDHeader(const unsigned char *p) : 
	fh((tcdsfedheader*)(p))
      {

      }
      const tcdsfedheader &getData(){return *fh;}
    private:
      tcdsfedheader *fh;
    };

    class TCDSHeader{
    public:

      union tcdsheader{
	uint64_t words[11];
	struct {
	  uint64_t macAddress;
	  uint32_t sw;
	  uint32_t fw;
	  uint32_t reserved0;
	  uint32_t format;
	  uint32_t runNumber;
	  uint32_t reserved1;
	  uint32_t activePartitions2;
	  uint32_t reserved2;
	  uint32_t activePartitions0;
	  uint32_t activePartitions1;
	  uint32_t nibble;
	  uint32_t lumiSection;
	  uint16_t nibblesPerLumiSection;
	  uint16_t triggerTypeFlags;
	  uint16_t reserved5;
	  uint16_t inputs;
	  uint16_t bcid;
	  uint16_t orbitLow;
	  uint32_t orbitHigh;
	  uint64_t triggerCount;
	  uint64_t eventNumber;
	} header;
      };

      TCDSHeader(const unsigned char *p) : 
	s((sizes*)p),
	h((tcdsheader*)(p+sizeof(uint64_t)))
      {
	
      }
      const sizes &getSizes(){return *s;}
      const tcdsheader &getData(){return *h;}
    private:
      sizes *s;
      tcdsheader *h;
    };
    class TCDSL1AHistory{
    public:
      struct l1a{ 
	uint16_t bxid;
	uint16_t dummy0;
	uint16_t dummy1;
	unsigned char dummy2;
	unsigned char ind0;
	uint32_t orbitlow;
	uint16_t orbithigh;
	unsigned char eventtype;
	unsigned char ind1;
      };
      union l1h{
	uint64_t words[32];
	l1a hist[32];
      };
      TCDSL1AHistory(const unsigned char *p) : hist((l1h*)p){
      }
      const l1h &history(){return *hist;}
    private:
      l1h *hist;
    };

    class TCDSBST{
    public:
      struct bst{
	uint32_t gpstimelow;
	uint32_t gpstimehigh;
	uint32_t low0;
	uint32_t high0;
	uint32_t low1;
	uint32_t high1;
	uint32_t low2;
	uint32_t high2;
	uint32_t low3;
	uint32_t high3;
	uint32_t low4;
	uint32_t high4;
	uint32_t low5;
	uint32_t status;
      };
      TCDSBST(const unsigned char *p) : b((bst*)p){
      }
      const bst &getBST(){return *b;}
    private:
      bst *b;
    };

    class TCDSRecord{
    public:
      TCDSRecord(const unsigned char *p) : 
	fh(p),
	h(p+sizeof(fedh_t)),
	l1h(p+sizeof(fedh_t)+(h.getSizes().size.headerSize+1)*8),
	b(p+sizeof(fedh_t)+(h.getSizes().size.headerSize+1)*8+
	  (h.getSizes().size.L1AhistSize)*8)
      {
	
      }
      TCDSFEDHeader &getFEDHeader(){return fh;} 
      TCDSHeader &getHeader(){return h;} 
      TCDSL1AHistory &getHistory(){return l1h;}
      TCDSBST &getBST(){return b;}
    private:
      TCDSFEDHeader fh;
      TCDSHeader h;
      TCDSL1AHistory l1h;
      TCDSBST b;
    };


  }
}
#endif

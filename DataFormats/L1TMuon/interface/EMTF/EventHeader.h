// Class for Event Record Header

#ifndef __l1t_emtf_EventHeader_h__
#define __l1t_emtf_EventHeader_h__

#include <boost/cstdint.hpp> 

namespace l1t {
  namespace emtf {
    class EventHeader {
    public:
      
      explicit EventHeader(uint64_t dataword);
    
    EventHeader() : 
      l1a(-99), l1a_BXN(-99), sp_TS(-99), sp_ERSV(-99), sp_ADDR(-99), tbin(-99), dataword(-99) 
	{};
      
    EventHeader(int int_l1a, int int_l1a_BXN, int int_sp_TS, int int_sp_ERSV, int int_sp_ADDR, int int_tbin) :
      l1a(int_l1a), l1a_BXN(int_l1a_BXN), sp_TS(int_sp_TS), sp_ERSV(int_sp_ERSV), sp_ADDR(int_sp_ADDR), 
	tbin(int_tbin), dataword(-99)
    	{};
      
      virtual ~EventHeader() {};
      
      void set_l1a(int bits)      { l1a = bits; };
      void set_l1a_BXN(int bits)        { l1a_BXN = bits;   };
      void set_sp_TS(int bits)       { sp_TS = bits;  };
      void set_sp_ERSV(int bits)            { sp_ERSV = bits;       };
      void set_sp_ADDR(int bits)            { sp_ADDR = bits;       };
      void set_tbin(int bits)       { tbin = bits;  };
      void set_dataword(uint64_t bits)  { dataword = bits;  };
      
      const int L1A()      const { return l1a; };
      const int L1A_BXN()        const { return l1a_BXN;   };
      const int SP_TS()       const { return sp_TS;   };
      const int SP_ERSV()            const { return sp_ERSV;       };
      const int SP_ADDR()            const { return sp_ADDR;       };
      const int TBIN()       const { return tbin;  };
      const uint64_t Dataword()  const { return dataword;  };      
      
    private:
      int l1a;
      int l1a_BXN;
      int sp_TS;
      int sp_ERSV;
      int sp_ADDR;
      int tbin;
      uint64_t dataword;
      
    }; // End of class EventHeader
  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_EventHeader_h__ */

// Class for Resistive Plate Chamber (RPC) Data Record

#ifndef __l1t_emtf_RPC_h__
#define __l1t_emtf_RPC_h__

#include <boost/cstdint.hpp> 

namespace l1t {
  namespace emtf {
    class RPC {
    public:
      
      explicit RPC(uint64_t dataword);
    
    RPC() : 
      prt_delay(-99), prt_num(-99), prt_data(-99), bcn(-99), eod(-99), link_num(-99), dataword(-99) 
	{};
      
    RPC(int int_prt_delay, int int_prt_num, int int_prt_data, int int_bcn, int int_eod, int int_link_num) :
      prt_delay(int_prt_delay), prt_num(int_prt_num), prt_data(int_prt_data), bcn(int_bcn), eod(int_eod), 
	link_num(int_link_num), dataword(-99)
    	{};
      
      virtual ~RPC() {};
      
      void set_prt_delay(int bits)      { prt_delay = bits; };
      void set_prt_num(int bits)        { prt_num = bits;   };
      void set_prt_data(int bits)       { prt_data = bits;  };
      void set_bcn(int bits)            { bcn = bits;       };
      void set_eod(int bits)            { eod = bits;       };
      void set_link_num(int bits)       { link_num = bits;  };
      void set_dataword(uint64_t bits)  { dataword = bits;  };
      
      const int Prt_delay()      const { return prt_delay; };
      const int Prt_num()        const { return prt_num;   };
      const int Prt_data()       const { return prt_data;  };
      const int BCN()            const { return bcn;       };
      const int EOD()            const { return eod;       };
      const int Link_num()       const { return link_num;  };
      const uint64_t Dataword()  const { return dataword;  };      
      
    private:
      int prt_delay;
      int prt_num;
      int prt_data;
      int bcn;
      int eod;
      int link_num;
      uint64_t dataword;
      
    }; // End of class RPC

    // Define a vector of RPC
    typedef std::vector<RPC> RPCCollection;

  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_RPC_h__ */

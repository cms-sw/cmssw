// Class for Event Record Trailer

#ifndef __l1t_emtf_EventTrailer_h__
#define __l1t_emtf_EventTrailer_h__

#include <boost/cstdint.hpp> 

namespace l1t {
  namespace emtf {
    class EventTrailer {
    public:
      
      explicit EventTrailer(uint64_t dataword);
    
    EventTrailer() : 
      dd_CSR_LF(-99), l1a(-99), dataword(-99) 
	{};
      
    EventTrailer(int int_l1a, int int_dd_CSR_LF) :
      dd_CSR_LF(int_dd_CSR_LF), l1a(int_l1a), dataword(-99)
    	{};
      
      virtual ~EventTrailer() {};
      
      void set_dd_CSR_LF(int bits)      { dd_CSR_LF = bits; };
      void set_l1a(int bits)            { l1a = bits;       };
      void set_dataword(uint64_t bits)  { dataword = bits;  };
      
      const int DD_CSR_LF()      const { return dd_CSR_LF; };
      const int L1A()            const { return l1a;       };
      const uint64_t Dataword()  const { return dataword;  };      
      
    private:
      int dd_CSR_LF;
      int l1a;
      uint64_t dataword;
      
    }; // End of class EventTrailer
  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_EventTrailer_h__ */

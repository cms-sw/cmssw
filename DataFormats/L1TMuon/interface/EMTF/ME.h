// Class for Muon Endcap (ME) Data Record

#ifndef __l1t_emtf_ME_h__
#define __l1t_emtf_ME_h__

#include <boost/cstdint.hpp> 

namespace l1t {
  namespace emtf {
    class ME {
    public:
      
      explicit ME(uint64_t dataword);
    
    ME() : 
      key_wire_group(-99), quality(-99), clct_pattern(-99), bc0(-99), bxe(-99), l_r(-99), dataword(-99) 
	{};
      
    ME(int int_key_wire_group, int int_quality, int int_clct_pattern, int int_bc0, int int_bxe, int int_l_r) :
      key_wire_group(int_key_wire_group), quality(int_quality), clct_pattern(int_clct_pattern), bc0(int_bc0), 
	bxe(int_bxe), l_r(int_l_r), dataword(-99)
    	{};
      
      virtual ~ME() {};
      
      void set_key_wire_group(int bits)  { key_wire_group = bits; };
      void set_quality(int bits)         { quality = bits;        };
      void set_clct_pattern(int bits)    { clct_pattern = bits;   };
      void set_bc0(int bits)             { bc0 = bits;            };
      void set_bxe(int bits)             { bxe = bits;            };
      void set_l_r(int bits)             { l_r = bits;            };
      void set_dataword(uint64_t bits)   { dataword = bits;       };
      
      const int Key_wire_group() const { return key_wire_group; };
      const int Quality()        const { return quality;        };
      const int CLCT_PATTERN()   const { return clct_pattern;   };
      const int BC0()            const { return bc0;            };
      const int BXE()            const { return bxe;            };
      const int L_R()            const { return l_r;            };
      const uint64_t Dataword()  const { return dataword;       };      
      
    private:
      int key_wire_group;
      int quality;
      int clct_pattern;
      int bc0;
      int bxe;
      int l_r;
      uint64_t dataword;
      
    }; // End of class ME
    
    // Define a vector of ME
    typedef std::vector<ME> MECollection;

  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_ME_h__ */

// Class for AMC to AMC13 Header

#ifndef __l1t_emtf_MTF7Header_h__
#define __l1t_emtf_MTF7Header_h__

#include <vector>
#include <boost/cstdint.hpp>

namespace l1t {
  namespace emtf {
    class MTF7Header {
      
    public:
      explicit MTF7Header(uint64_t dataword); 
      
    // Empty constructor
    MTF7Header() :
      amcNo(-99), lv1_id(-99), dataword(-99)
	{};
      
    // Fill constructor
    MTF7Header(int int_amcNo, int int_lv1_id) :
      amcNo(int_amcNo), lv1_id(int_lv1_id), dataword(-99)
	{};
      
      virtual ~MTF7Header() {};
      
      void set_amcNo(int bits)          { amcNo = bits;       };
      void set_lv1_id(int bits)         { lv1_id = bits;      };
      void set_dataword(uint64_t bits)  { dataword = bits;    };

      const int AmcNo()          const { return amcNo;    };
      const int LV1_id()         const { return lv1_id;   };
      const uint64_t Dataword()  const { return dataword; };
      
    private:
      int amcNo;
      int lv1_id;
      uint64_t dataword; 
      
    }; // End class MTF7Header
  } // End namespace emtf
} // End namespace l1t

#endif /* define __l1t_emtf_MTF7Header_h__ */

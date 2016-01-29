// Class for AMC13 Header

#ifndef __l1t_emtf_AMC13Header_h__
#define __l1t_emtf_AMC13Header_h__

#include <vector>
#include <boost/cstdint.hpp>

namespace l1t {
  namespace emtf {
    class AMC13Header {
      
    public:
      explicit AMC13Header(uint64_t dataword); 
      
    // Empty constructor
    AMC13Header() :
      evt_ty(-99), lv1_id(-99), dataword(-99)
	{};
      
    // Fill constructor
    AMC13Header(int int_evt_ty, int int_lv1_id) :
      evt_ty(int_evt_ty), lv1_id(int_lv1_id), dataword(-99)
	{};
      
      virtual ~AMC13Header() {};
      
      void set_evt_ty(int bits)         { evt_ty = bits;   };
      void set_lv1_id(int bits)         { lv1_id = bits;   };
      void set_dataword(uint64_t bits)  { dataword = bits; };
      
      const int Evt_ty()            const { return evt_ty;   };
      const int LV1_id()            const { return lv1_id;   };
      const uint64_t Dataword()     const { return dataword; };
      
    private:
      int evt_ty;
      int lv1_id;
      uint64_t dataword; 
      
    }; // End class AMC13Header
  } // End namespace emtf
} // End namespace l1t

#endif /* define __l1t_emtf_AMC13Header_h__ */

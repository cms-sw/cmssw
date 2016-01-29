// Class for Sector Processor (SP) Output Data Record

#ifndef __l1t_emtf_SP_h__
#define __l1t_emtf_SP_h__

#include <boost/cstdint.hpp> 

namespace l1t {
  namespace emtf {
    class SP {
    public:
      
      explicit SP(uint64_t dataword);
    
    SP() : 
      phi_full(-99), phi_GMT(-99), bx(-99), quality(-99), eta_GMT(-99), pt(-99), dataword(-99) 
	{};
      
    SP(int int_phi_full, int int_phi_GMT, int int_bx, int int_quality, int int_eta_GMT, int int_pt) :
      phi_full(int_phi_full), phi_GMT(int_phi_GMT), bx(int_bx), quality(int_quality), eta_GMT(int_eta_GMT), 
	pt(int_pt), dataword(-99)
    	{};
      
      virtual ~SP() {};
      
      void set_phi_full(int bits)       { phi_full = bits; };
      void set_phi_GMT(int bits)        { phi_GMT = bits;  };
      void set_bx(int bits)             { bx = bits;       };
      void set_quality(int bits)        { quality = bits;  };
      void set_eta_GMT(int bits)        { eta_GMT = bits;  };
      void set_pt(int bits)             { pt = bits;       };
      void set_dataword(uint64_t bits)  { dataword = bits; };
      
      const int Phi_full()       const { return phi_full; };
      const int Phi_GMT()        const { return phi_GMT;  };
      const int BX()             const { return bx;       };
      const int Quality()        const { return quality;  };
      const int Eta_GMT()        const { return eta_GMT;  };
      const int Pt()             const { return pt;       };
      const uint64_t Dataword()  const { return dataword; };      
      
    private:
      int phi_full;
      int phi_GMT;
      int bx;
      int quality;
      int eta_GMT;
      int pt;
      uint64_t dataword;
      
    }; // End of class SP

    // Define a vector of SP
    typedef std::vector<SP> SPCollection;

  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_SP_h__ */

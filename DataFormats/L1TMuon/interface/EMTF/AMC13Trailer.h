// Class for AMC13 Trailer

#ifndef __l1t_emtf_AMC13Trailer_h__
#define __l1t_emtf_AMC13Trailer_h__

#include <vector>
#include <boost/cstdint.hpp>

namespace l1t {
  namespace emtf {
    class AMC13Trailer {
      
    public:
      explicit AMC13Trailer(uint64_t dataword); 
      
    // Empty constructor
    AMC13Trailer() :
      evt_lgth(-99), crc(-99), dataword(-99)
	{};
      
    // Fill constructor
    AMC13Trailer(int int_evt_lgth, int int_crc) :
      evt_lgth(int_evt_lgth), crc(int_crc), dataword(-99)
	{};
      
      virtual ~AMC13Trailer() {};
      
      void set_evt_lgth(int bits)       { evt_lgth = bits; };
      void set_crc(int bits)            { crc = bits;      };
      void set_dataword(uint64_t bits)  { dataword = bits; };
      
      const int Evt_lgth()      const { return evt_lgth; };
      const int CRC()           const { return crc;      };
      const uint64_t Dataword() const { return dataword; };
      
    private:
      int evt_lgth;
      int crc;
      uint64_t dataword; 
      
    }; // End class AMC13Trailer
  } // End namespace emtf
} // End namespace l1t

#endif /* define __l1t_emtf_AMC13Trailer_h__ */

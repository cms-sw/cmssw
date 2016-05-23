// Class for AMC to AMC13 Trailer

#ifndef __l1t_emtf_MTF7Trailer_h__
#define __l1t_emtf_MTF7Trailer_h__

#include <vector>
#include <boost/cstdint.hpp>

namespace l1t {
  namespace emtf {
    class MTF7Trailer {
      
    public:
      explicit MTF7Trailer(uint64_t dataword); 
      
    // Empty constructor
    MTF7Trailer() :
      crc_32(-99), lv1_id(-99), data_length(-99), format_errors(0), dataword(-99)
	{};
      
    // Fill constructor
    MTF7Trailer(int int_crc_32, int int_lv1_id, int int_data_length) :
      crc_32(int_crc_32), lv1_id(int_lv1_id), data_length(int_data_length), format_errors(0), dataword(-99)
	{};
      
      virtual ~MTF7Trailer() {};
      
      void set_crc_32(int bits)         { crc_32 = bits;      };
      void set_lv1_id(int bits)         { lv1_id = bits;      };
      void set_data_length(int bits)    { data_length = bits; };
      void add_format_error()           { format_errors += 1; };
      void set_dataword(uint64_t bits)  { dataword = bits;    };

      const int CRC_32()         const { return crc_32;   };
      const int LV1_id()         const { return lv1_id;   };
      const int Data_length()    const { return data_length;};
      const int Format_Errors()  const { return format_errors; };
      const uint64_t Dataword()  const { return dataword; };
      
    private:
      int crc_32;
      int lv1_id;
      int data_length;
      int  format_errors;
      uint64_t dataword; 
      
    }; // End class MTF7Trailer
  } // End namespace emtf
} // End namespace l1t

#endif /* define __l1t_emtf_MTF7Trailer_h__ */

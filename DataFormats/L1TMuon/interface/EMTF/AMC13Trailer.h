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
      evt_lgth(-99), crc16(-99), evt_stat(-99), tts(-99), c(-99), f(-99), t(-99), r(-99), format_errors(0), dataword(-99)
	{};
      
    // Fill constructor
    AMC13Trailer(int int_evt_lgth, int int_crc16, int int_evt_stat, int int_tts, int int_c, int int_f, int int_t, int int_r) :
      evt_lgth(int_evt_lgth), crc16(int_crc16), evt_stat(int_evt_stat), tts(int_tts), c(int_c), f(int_f), t(int_t), r(int_r), format_errors(0), dataword(-99)
	{};
      
      virtual ~AMC13Trailer() {};
      
      void set_evt_lgth(int bits)      { evt_lgth = bits;    }
      void set_crc16(int bits)         { crc16 = bits;       }
      void set_evt_stat(int bits)      { evt_stat = bits;    }
      void set_tts(int bits)           { tts = bits;         }
      void set_c(int bits)             { c = bits;           }
      void set_f(int bits)             { f = bits;           }
      void set_t(int bits)             { t = bits;           }
      void set_r(int bits)             { r = bits;           }
      void add_format_error()          { format_errors += 1; }
      void set_dataword(uint64_t bits) { dataword = bits;    }
      
      int      Evt_lgth()      const { return evt_lgth;      }
      int      CRC16()         const { return crc16;         }
      int      Evt_stat()      const { return evt_stat;      }
      int      TTS()           const { return tts;           }
      int      C()             const { return c;             }
      int      F()             const { return f;             }
      int      T()             const { return t;             }
      int      R()             const { return r;             }
      int      Format_Errors() const { return format_errors; }
      uint64_t Dataword()      const { return dataword;      }
      
    private:
      int evt_lgth;
      int crc16;
      int evt_stat;
      int tts; 
      int c;
      int f;
      int t;
      int r;
      int format_errors;
      uint64_t dataword; 
      
    }; // End class AMC13Trailer
  } // End namespace emtf
} // End namespace l1t

#endif /* define __l1t_emtf_AMC13Trailer_h__ */

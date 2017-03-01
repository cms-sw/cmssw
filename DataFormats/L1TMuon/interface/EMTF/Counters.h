// Class for Block of Counters

#ifndef __l1t_emtf_Counters_h__
#define __l1t_emtf_Counters_h__

#include <boost/cstdint.hpp> 

namespace l1t {
  namespace emtf {
    class Counters {
    public:
      
      explicit Counters(uint64_t dataword);
      
    // rpc_counter not yet implemented in FW - AWB 31.01.16
    Counters() : 
      track_counter(-99), orbit_counter(-99), rpc_counter(-99), format_errors(0), dataword(-99) 
	{};
      
    Counters(int int_track_counter, int int_orbit_counter, int int_rpc_counter) :
      track_counter(int_track_counter), orbit_counter(int_orbit_counter), rpc_counter(int_rpc_counter), format_errors(0), dataword(-99)
    	{};
      
      virtual ~Counters() {};
      
      void set_track_counter(int bits) { track_counter = bits; }
      void set_orbit_counter(int bits) { orbit_counter = bits; }
      void set_rpc_counter(int bits)   { rpc_counter = bits;   }
      void add_format_error()          { format_errors += 1;   }
      void set_dataword(uint64_t bits) { dataword = bits;      }
      
      int      Track_counter() const { return track_counter; }
      int      Orbit_counter() const { return orbit_counter; }
      int      RPC_counter()   const { return rpc_counter;   }
      int      Format_Errors() const { return format_errors; }
      uint64_t Dataword()      const { return dataword;      }      
      
    private:
      int track_counter;
      int orbit_counter;
      int rpc_counter;
      int format_errors;
      uint64_t dataword;
      
    }; // End of class Counters

  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_Counters_h__ */

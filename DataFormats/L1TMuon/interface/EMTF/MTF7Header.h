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
      amc_number(-99), bx_id(-99), orbit_number(-99), board_id(-99), lv1_id(-99), 
	data_length(-99), user_id(-99), format_errors(0), dataword(-99)
	{};
      
    // Fill constructor
    MTF7Header(int int_amc_number, int int_bx_id, int int_orbit_number, int int_board_id, int int_lv1_id, 
	       int int_data_length, int int_user_id) :
      amc_number(int_amc_number), bx_id(int_bx_id), orbit_number(int_orbit_number), board_id(int_board_id), lv1_id(int_lv1_id), 
	data_length(int_data_length), user_id(int_user_id), format_errors(0), dataword(-99)
	{};
      
      virtual ~MTF7Header() {};
      
      void set_amc_number(int bits)     {  amc_number = bits; };
      void set_bx_id(int bits)          {  bx_id = bits; };
      void set_orbit_number(int bits)   {  orbit_number = bits; };
      void set_board_id(int bits)       {  board_id = bits; };
      void set_lv1_id(int bits)         {  lv1_id = bits; };
      void set_data_length(int bits)    {  data_length = bits; };
      void set_user_id(int bits)        {  user_id = bits; };
      void add_format_error()           { format_errors += 1; };
      void set_dataword(uint64_t bits)  { dataword = bits;    };

      const int AMC_number()     const { return  amc_number ; };
      const int BX_id()          const { return  bx_id ; };
      const int Orbit_number()   const { return  orbit_number ; };
      const int Board_id()       const { return  board_id ; };
      const int LV1_id()         const { return  lv1_id ; };
      const int Data_length()    const { return  data_length ; };
      const int User_id()        const { return  user_id ; };   
      const int Format_Errors()  const { return format_errors; };
      const uint64_t Dataword()  const { return dataword; };
      
    private:
      int  amc_number;
      int  bx_id;
      int  orbit_number;
      int  board_id;
      int  lv1_id;
      int  data_length;
      int  user_id;
      int  format_errors;
      uint64_t dataword; 
      
    }; // End class MTF7Header
  } // End namespace emtf
} // End namespace l1t

#endif /* define __l1t_emtf_MTF7Header_h__ */

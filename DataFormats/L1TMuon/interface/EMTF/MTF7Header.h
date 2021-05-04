// Class for AMC to AMC13 Header

#ifndef __l1t_emtf_MTF7Header_h__
#define __l1t_emtf_MTF7Header_h__

#include <vector>
#include <cstdint>

namespace l1t {
  namespace emtf {
    class MTF7Header {
    public:
      explicit MTF7Header(uint64_t dataword);

      MTF7Header()
          : amc_number(-99),
            bx_id(-99),
            orbit_number(-99),
            board_id(-99),
            lv1_id(-99),
            data_length(-99),
            user_id(-99),
            format_errors(0),
            dataword(-99){};

      virtual ~MTF7Header(){};

      void set_amc_number(int bits) { amc_number = bits; }
      void set_bx_id(int bits) { bx_id = bits; }
      void set_orbit_number(int bits) { orbit_number = bits; }
      void set_board_id(int bits) { board_id = bits; }
      void set_lv1_id(int bits) { lv1_id = bits; }
      void set_data_length(int bits) { data_length = bits; }
      void set_user_id(int bits) { user_id = bits; }
      void add_format_error() { format_errors += 1; }
      void set_dataword(uint64_t bits) { dataword = bits; }

      int AMC_number() const { return amc_number; }
      int BX_id() const { return bx_id; }
      int Orbit_number() const { return orbit_number; }
      int Board_id() const { return board_id; }
      int LV1_id() const { return lv1_id; }
      int Data_length() const { return data_length; }
      int User_id() const { return user_id; }
      int Format_errors() const { return format_errors; }
      uint64_t Dataword() const { return dataword; }

    private:
      int amc_number;
      int bx_id;
      int orbit_number;
      int board_id;
      int lv1_id;
      int data_length;
      int user_id;
      int format_errors;
      uint64_t dataword;

    };  // End class MTF7Header
  }     // End namespace emtf
}  // End namespace l1t

#endif /* define __l1t_emtf_MTF7Header_h__ */

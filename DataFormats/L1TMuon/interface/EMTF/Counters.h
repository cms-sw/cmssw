#include <cstdint>
// Class for Block of Counters

#ifndef __l1t_emtf_Counters_h__
#define __l1t_emtf_Counters_h__

namespace l1t {
  namespace emtf {
    class Counters {
    public:
      explicit Counters(uint64_t dataword);

      // rpc_counter not yet implemented in FW - AWB 31.01.16
      Counters()
          : me1a_1(-99),
            me1a_2(-99),
            me1a_3(-99),
            me1a_4(-99),
            me1a_5(-99),
            me1a_6(-99),
            me1a_7(-99),
            me1a_8(-99),
            me1a_9(-99),
            me1b_1(-99),
            me1b_2(-99),
            me1b_3(-99),
            me1b_4(-99),
            me1b_5(-99),
            me1b_6(-99),
            me1b_7(-99),
            me1b_8(-99),
            me1b_9(-99),
            me2_1(-99),
            me2_2(-99),
            me2_3(-99),
            me2_4(-99),
            me2_5(-99),
            me2_6(-99),
            me2_7(-99),
            me2_8(-99),
            me2_9(-99),
            me3_1(-99),
            me3_2(-99),
            me3_3(-99),
            me3_4(-99),
            me3_5(-99),
            me3_6(-99),
            me3_7(-99),
            me3_8(-99),
            me3_9(-99),
            me4_1(-99),
            me4_2(-99),
            me4_3(-99),
            me4_4(-99),
            me4_5(-99),
            me4_6(-99),
            me4_7(-99),
            me4_8(-99),
            me4_9(-99),
            me1n_3(-99),
            me1n_6(-99),
            me1n_9(-99),
            me2n_3(-99),
            me2n_9(-99),
            me3n_3(-99),
            me3n_9(-99),
            me4n_3(-99),
            me4n_9(-99),
            me1a_all(-99),
            me1b_all(-99),
            me2_all(-99),
            me3_all(-99),
            me4_all(-99),
            meN_all(-99),
            me_all(-99),
            format_errors(0),
            dataword(-99){};

      Counters(int int_me1a_all,
               int int_me1b_all,
               int int_me2_all,
               int int_me3_all,
               int int_me4_all,
               int int_meN_all,
               int int_me_all)
          : me1a_all(int_me1a_all),
            me1b_all(int_me1b_all),
            me2_all(int_me2_all),
            me3_all(int_me3_all),
            me4_all(int_me4_all),
            meN_all(int_meN_all),
            me_all(int_me_all),
            dataword(-99){};

      virtual ~Counters(){};

      void set_me1a_1(int bits) { me1a_1 = bits; }
      void set_me1a_2(int bits) { me1a_2 = bits; }
      void set_me1a_3(int bits) { me1a_3 = bits; }
      void set_me1a_4(int bits) { me1a_4 = bits; }
      void set_me1a_5(int bits) { me1a_5 = bits; }
      void set_me1a_6(int bits) { me1a_6 = bits; }
      void set_me1a_7(int bits) { me1a_7 = bits; }
      void set_me1a_8(int bits) { me1a_8 = bits; }
      void set_me1a_9(int bits) { me1a_9 = bits; }
      void set_me1b_1(int bits) { me1b_1 = bits; }
      void set_me1b_2(int bits) { me1b_2 = bits; }
      void set_me1b_3(int bits) { me1b_3 = bits; }
      void set_me1b_4(int bits) { me1b_4 = bits; }
      void set_me1b_5(int bits) { me1b_5 = bits; }
      void set_me1b_6(int bits) { me1b_6 = bits; }
      void set_me1b_7(int bits) { me1b_7 = bits; }
      void set_me1b_8(int bits) { me1b_8 = bits; }
      void set_me1b_9(int bits) { me1b_9 = bits; }
      void set_me2_1(int bits) { me2_1 = bits; }
      void set_me2_2(int bits) { me2_2 = bits; }
      void set_me2_3(int bits) { me2_3 = bits; }
      void set_me2_4(int bits) { me2_4 = bits; }
      void set_me2_5(int bits) { me2_5 = bits; }
      void set_me2_6(int bits) { me2_6 = bits; }
      void set_me2_7(int bits) { me2_7 = bits; }
      void set_me2_8(int bits) { me2_8 = bits; }
      void set_me2_9(int bits) { me2_9 = bits; }
      void set_me3_1(int bits) { me3_1 = bits; }
      void set_me3_2(int bits) { me3_2 = bits; }
      void set_me3_3(int bits) { me3_3 = bits; }
      void set_me3_4(int bits) { me3_4 = bits; }
      void set_me3_5(int bits) { me3_5 = bits; }
      void set_me3_6(int bits) { me3_6 = bits; }
      void set_me3_7(int bits) { me3_7 = bits; }
      void set_me3_8(int bits) { me3_8 = bits; }
      void set_me3_9(int bits) { me3_9 = bits; }
      void set_me4_1(int bits) { me4_1 = bits; }
      void set_me4_2(int bits) { me4_2 = bits; }
      void set_me4_3(int bits) { me4_3 = bits; }
      void set_me4_4(int bits) { me4_4 = bits; }
      void set_me4_5(int bits) { me4_5 = bits; }
      void set_me4_6(int bits) { me4_6 = bits; }
      void set_me4_7(int bits) { me4_7 = bits; }
      void set_me4_8(int bits) { me4_8 = bits; }
      void set_me4_9(int bits) { me4_9 = bits; }
      void set_me1n_3(int bits) { me1n_3 = bits; }
      void set_me1n_6(int bits) { me1n_6 = bits; }
      void set_me1n_9(int bits) { me1n_9 = bits; }
      void set_me2n_3(int bits) { me2n_3 = bits; }
      void set_me2n_9(int bits) { me2n_9 = bits; }
      void set_me3n_3(int bits) { me3n_3 = bits; }
      void set_me3n_9(int bits) { me3n_9 = bits; }
      void set_me4n_3(int bits) { me4n_3 = bits; }
      void set_me4n_9(int bits) { me4n_9 = bits; }

      void set_me1a_all(int bits) { me1a_all = bits; }
      void set_me1b_all(int bits) { me1b_all = bits; }
      void set_me2_all(int bits) { me2_all = bits; }
      void set_me3_all(int bits) { me3_all = bits; }
      void set_me4_all(int bits) { me4_all = bits; }
      void set_meN_all(int bits) { meN_all = bits; }
      void set_me_all(int bits) { me_all = bits; }
      void add_format_error() { format_errors += 1; }
      void set_dataword(uint64_t bits) { dataword = bits; }

      int ME1a_1() const { return me1a_1; }
      int ME1a_2() const { return me1a_2; }
      int ME1a_3() const { return me1a_3; }
      int ME1a_4() const { return me1a_4; }
      int ME1a_5() const { return me1a_5; }
      int ME1a_6() const { return me1a_6; }
      int ME1a_7() const { return me1a_7; }
      int ME1a_8() const { return me1a_8; }
      int ME1a_9() const { return me1a_9; }
      int ME1b_1() const { return me1b_1; }
      int ME1b_2() const { return me1b_2; }
      int ME1b_3() const { return me1b_3; }
      int ME1b_4() const { return me1b_4; }
      int ME1b_5() const { return me1b_5; }
      int ME1b_6() const { return me1b_6; }
      int ME1b_7() const { return me1b_7; }
      int ME1b_8() const { return me1b_8; }
      int ME1b_9() const { return me1b_9; }
      int ME2_1() const { return me2_1; }
      int ME2_2() const { return me2_2; }
      int ME2_3() const { return me2_3; }
      int ME2_4() const { return me2_4; }
      int ME2_5() const { return me2_5; }
      int ME2_6() const { return me2_6; }
      int ME2_7() const { return me2_7; }
      int ME2_8() const { return me2_8; }
      int ME2_9() const { return me2_9; }
      int ME3_1() const { return me3_1; }
      int ME3_2() const { return me3_2; }
      int ME3_3() const { return me3_3; }
      int ME3_4() const { return me3_4; }
      int ME3_5() const { return me3_5; }
      int ME3_6() const { return me3_6; }
      int ME3_7() const { return me3_7; }
      int ME3_8() const { return me3_8; }
      int ME3_9() const { return me3_9; }
      int ME4_1() const { return me4_1; }
      int ME4_2() const { return me4_2; }
      int ME4_3() const { return me4_3; }
      int ME4_4() const { return me4_4; }
      int ME4_5() const { return me4_5; }
      int ME4_6() const { return me4_6; }
      int ME4_7() const { return me4_7; }
      int ME4_8() const { return me4_8; }
      int ME4_9() const { return me4_9; }
      int ME1n_3() const { return me1n_3; }
      int ME1n_6() const { return me1n_6; }
      int ME1n_9() const { return me1n_9; }
      int ME2n_3() const { return me2n_3; }
      int ME2n_9() const { return me2n_9; }
      int ME3n_3() const { return me3n_3; }
      int ME3n_9() const { return me3n_9; }
      int ME4n_3() const { return me4n_3; }
      int ME4n_9() const { return me4n_9; }

      int ME1a_all() const { return me1a_all; }
      int ME1b_all() const { return me1b_all; }
      int ME2_all() const { return me2_all; }
      int ME3_all() const { return me3_all; }
      int ME4_all() const { return me4_all; }
      int MEN_all() const { return meN_all; }
      int ME_all() const { return me_all; }
      int Format_errors() const { return format_errors; }
      uint64_t Dataword() const { return dataword; }

    private:
      int me1a_1;
      int me1a_2;
      int me1a_3;
      int me1a_4;
      int me1a_5;
      int me1a_6;
      int me1a_7;
      int me1a_8;
      int me1a_9;
      int me1b_1;
      int me1b_2;
      int me1b_3;
      int me1b_4;
      int me1b_5;
      int me1b_6;
      int me1b_7;
      int me1b_8;
      int me1b_9;
      int me2_1;
      int me2_2;
      int me2_3;
      int me2_4;
      int me2_5;
      int me2_6;
      int me2_7;
      int me2_8;
      int me2_9;
      int me3_1;
      int me3_2;
      int me3_3;
      int me3_4;
      int me3_5;
      int me3_6;
      int me3_7;
      int me3_8;
      int me3_9;
      int me4_1;
      int me4_2;
      int me4_3;
      int me4_4;
      int me4_5;
      int me4_6;
      int me4_7;
      int me4_8;
      int me4_9;
      int me1n_3;
      int me1n_6;
      int me1n_9;
      int me2n_3;
      int me2n_9;
      int me3n_3;
      int me3n_9;
      int me4n_3;
      int me4n_9;

      int me1a_all;
      int me1b_all;
      int me2_all;
      int me3_all;
      int me4_all;
      int meN_all;
      int me_all;
      int format_errors;
      uint64_t dataword;

    };  // End of class Counters

  }  // End of namespace emtf
}  // End of namespace l1t

#endif /* define __l1t_emtf_Counters_h__ */

#include <cstdint>
// Class for Event Record Trailer

#ifndef __l1t_emtf_EventTrailer_h__
#define __l1t_emtf_EventTrailer_h__

namespace l1t {
  namespace emtf {
    class EventTrailer {
    public:
      explicit EventTrailer(uint64_t dataword);

      EventTrailer()
          : crc22(-99),
            lp(-99),
            hp(-99),
            ddcsr_bid(-99),
            ddcsr_lf(-99),
            spcsr_scc(-99),
            l1a(-99),
            yy(-99),
            mm(-99),
            dd(-99),
            sp_ladr(-99),
            sp_ersv(-99),
            sp_padr(-99),
            lfff(-99),
            bb(-99),
            format_errors(0),
            dataword(-99){};

      virtual ~EventTrailer(){};

      void set_crc22(int bits) { crc22 = bits; }
      void set_lp(int bits) { lp = bits; }
      void set_hp(int bits) { hp = bits; }
      void set_ddcsr_bid(int bits) { ddcsr_bid = bits; }
      void set_ddcsr_lf(int bits) { ddcsr_lf = bits; }
      void set_spcsr_scc(int bits) { spcsr_scc = bits; }
      void set_l1a(int bits) { l1a = bits; }
      void set_yy(int bits) { yy = bits; }
      void set_mm(int bits) { mm = bits; }
      void set_dd(int bits) { dd = bits; }
      void set_sp_ladr(int bits) { sp_ladr = bits; }
      void set_sp_ersv(int bits) { sp_ersv = bits; }
      void set_sp_padr(int bits) { sp_padr = bits; }
      void set_lfff(int bits) { lfff = bits; }
      void set_bb(int bits) { bb = bits; }
      void add_format_error() { format_errors += 1; }
      void set_dataword(uint64_t bits) { dataword = bits; }

      int CRC22() const { return crc22; }
      int LP() const { return lp; }
      int HP() const { return hp; }
      int DDCRC_bid() const { return ddcsr_bid; }
      int DDCRC_lf() const { return ddcsr_lf; }
      int SPCSR_scc() const { return spcsr_scc; }
      int L1a() const { return l1a; }
      int YY() const { return yy; }
      int MM() const { return mm; }
      int DD() const { return dd; }
      int SP_ladr() const { return sp_ladr; }
      int SP_ersv() const { return sp_ersv; }
      int SP_padr() const { return sp_padr; }
      int LFFF() const { return lfff; }
      int BB() const { return bb; }
      int Format_errors() const { return format_errors; }
      uint64_t Dataword() const { return dataword; }

    private:
      int crc22;
      int lp;
      int hp;
      int ddcsr_bid;
      int ddcsr_lf;
      int spcsr_scc;
      int l1a;
      int yy;
      int mm;
      int dd;
      int sp_ladr;
      int sp_ersv;
      int sp_padr;
      int lfff;
      int bb;
      int format_errors;
      uint64_t dataword;

    };  // End of class EventTrailer
  }     // End of namespace emtf
}  // End of namespace l1t

#endif /* define __l1t_emtf_EventTrailer_h__ */

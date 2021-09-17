// Class for Muon Endcap (ME) Data Record

#ifndef __l1t_emtf_ME_h__
#define __l1t_emtf_ME_h__

#include <vector>
#include <cstdint>

namespace l1t {
  namespace emtf {
    class ME {
    public:
      explicit ME(uint64_t dataword);

      ME()
          : wire(-99),
            quality(-99),
            clct_pattern(-99),
            bc0(-99),
            bxe(-99),
            lr(-99),
            csc_ID(-99),
            strip(-99),
            afff(-99),
            cik(-99),
            nit(-99),
            me_bxn(-99),
            afef(-99),
            se(-99),
            sm(-99),
            epc(-99),
            af(-99),
            station(-99),
            vp(-99),
            tbin(-99),
            stub_num(-99),
            format_errors(0),
            dataword(-99){};

      virtual ~ME(){};

      void set_wire(int bits) { wire = bits; }
      void set_quality(int bits) { quality = bits; }
      void set_clct_pattern(int bits) { clct_pattern = bits; }
      void set_bc0(int bits) { bc0 = bits; }
      void set_bxe(int bits) { bxe = bits; }
      void set_lr(int bits) { lr = bits; }
      void set_csc_ID(int bits) { csc_ID = bits; }
      void set_strip(int bits) { strip = bits; }
      void set_afff(int bits) { afff = bits; }
      void set_cik(int bits) { cik = bits; }
      void set_nit(int bits) { nit = bits; }
      void set_me_bxn(int bits) { me_bxn = bits; }
      void set_afef(int bits) { afef = bits; }
      void set_se(int bits) { se = bits; }
      void set_sm(int bits) { sm = bits; }
      void set_epc(int bits) { epc = bits; }
      void set_af(int bits) { af = bits; }
      void set_station(int bits) { station = bits; }
      void set_vp(int bits) { vp = bits; }
      void set_tbin(int bits) { tbin = bits; }
      void set_stub_num(int bits) { stub_num = bits; }
      void add_format_error() { format_errors += 1; }
      void set_dataword(uint64_t bits) { dataword = bits; }

      int Wire() const { return wire; }
      int Quality() const { return quality; }
      int CLCT_pattern() const { return clct_pattern; }
      int BC0() const { return bc0; }
      int BXE() const { return bxe; }
      int LR() const { return lr; }
      int CSC_ID() const { return csc_ID; }
      int Strip() const { return strip; }
      int AFFF() const { return afff; }
      int CIK() const { return cik; }
      int NIT() const { return nit; }
      int ME_BXN() const { return me_bxn; }
      int AFEF() const { return afef; }
      int SE() const { return se; }
      int SM() const { return sm; }
      int EPC() const { return epc; }
      int AF() const { return af; }
      int Station() const { return station; }
      int VP() const { return vp; }
      int TBIN() const { return tbin; }
      int Stub_num() const { return stub_num; }
      int Format_errors() const { return format_errors; }
      uint64_t Dataword() const { return dataword; }

    private:
      int wire;
      int quality;
      int clct_pattern;
      int bc0;
      int bxe;
      int lr;
      int csc_ID;
      int strip;
      int afff;
      int cik;
      int nit;
      int me_bxn;
      int afef;
      int se;
      int sm;
      int epc;
      int af;
      int station;
      int vp;
      int tbin;
      int stub_num;
      int format_errors;
      uint64_t dataword;

    };  // End of class ME

    // Define a vector of ME
    typedef std::vector<ME> MECollection;

  }  // End of namespace emtf
}  // End of namespace l1t

#endif /* define __l1t_emtf_ME_h__ */

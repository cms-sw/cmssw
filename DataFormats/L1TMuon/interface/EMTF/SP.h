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
      pt_lut_address(-99), phi_full(-99), phi_GMT(-99), eta_GMT(-99), pt(-99), quality(-99), bx(-99), me4_ID(-99), me3_ID(-99), me2_ID(-99), me1_ID(-99), me4_TBIN(-99), me3_TBIN(-99), me2_TBIN(-99), me1_TBIN(-99), tbin_num(-99), hl(-99), c(-99), vc(-99), vt(-99), se(-99), bc0(-99), dataword(-99) 
	{};
      
    SP(int int_pt_lut_address, int int_phi_full, int int_phi_GMT, int int_eta_GMT, int int_pt, int int_quality, int int_bx, int int_me4_ID, int int_me3_ID, int int_me2_ID, int int_me1_ID, int int_me4_TBIN, int int_me3_TBIN, int int_me2_TBIN, int int_me1_TBIN, int int_TBIN_num, int int_hl, int int_c, int int_vc, int int_vt, int int_se, int int_bc0) :
      pt_lut_address(int_pt_lut_address), phi_full(int_phi_full), phi_GMT(int_phi_GMT), eta_GMT(int_eta_GMT), pt(int_pt), quality(int_quality), bx(int_bx), me4_ID(int_me4_ID), me3_ID(int_me3_ID), me2_ID(int_me2_ID), me1_ID(int_me1_ID), me4_TBIN(int_me4_TBIN), me3_TBIN(int_me3_TBIN), me2_TBIN(int_me2_TBIN), me1_TBIN(int_me1_TBIN), tbin_num(int_TBIN_num), hl(int_hl), c(int_c), vc(int_vc), vt(int_vt), se(int_se), bc0(int_bc0), dataword(-99)
    	{};
      
      virtual ~SP() {};
      
      void set_pt_lut_address(int bits)       { pt_lut_address= bits; };
      void set_phi_full      (int bits)       { phi_full      = bits; };
      void set_phi_GMT       (int bits)       { phi_GMT       = bits; };
      void set_eta_GMT       (int bits)       { eta_GMT       = bits; };
      void set_pt            (int bits)       { pt            = bits; };
      void set_quality       (int bits)       { quality       = bits; };
      void set_bx            (int bits)       { bx            = bits; };
      void set_me4_ID        (int bits)       { me4_ID        = bits; };
      void set_me3_ID        (int bits)       { me3_ID        = bits; };
      void set_me2_ID        (int bits)       { me2_ID        = bits; };
      void set_me1_ID        (int bits)       { me1_ID        = bits; };
      void set_me4_TBIN      (int bits)       { me4_TBIN      = bits; };
      void set_me3_TBIN      (int bits)       { me3_TBIN      = bits; };
      void set_me2_TBIN      (int bits)       { me2_TBIN      = bits; };
      void set_me1_TBIN      (int bits)       { me1_TBIN      = bits; };
      void set_TBIN_num      (int bits)       { tbin_num      = bits; };
      void set_hl            (int bits)       { hl            = bits; };
      void set_c             (int bits)       { c             = bits; };
      void set_vc            (int bits)       { vc            = bits; };
      void set_vt            (int bits)       { vt            = bits; };
      void set_se            (int bits)       { se            = bits; };
      void set_bc0           (int bits)       { bc0           = bits; };
      void set_dataword(uint64_t bits)  { dataword = bits; };

      const int Pt_lut_address() const { return pt_lut_address; };
      const int Phi_full()       const { return phi_full      ; };
      const int Phi_GMT()        const { return phi_GMT       ; };
      const int Eta_GMT()        const { return eta_GMT       ; };
      const int Pt()             const { return pt            ; };
      const int Quality()        const { return quality       ; };
      const int BX()             const { return bx            ; };
      const int ME4_ID()         const { return me4_ID        ; };
      const int ME3_ID()         const { return me3_ID        ; };
      const int ME2_ID()         const { return me2_ID        ; };
      const int ME1_ID()         const { return me1_ID        ; };
      const int ME4_TBIN()       const { return me4_TBIN      ; };
      const int ME3_TBIN()       const { return me3_TBIN      ; };
      const int ME2_TBIN()       const { return me2_TBIN      ; };
      const int ME1_TBIN()       const { return me1_TBIN      ; };
      const int TBIN_num()       const { return tbin_num      ; };
      const int HL()             const { return hl            ; };
      const int C()              const { return c             ; };
      const int VC()             const { return vc            ; };
      const int VT()             const { return vt            ; };
      const int SE()             const { return se            ; };
      const int BC0()            const { return bc0           ; };
      const uint64_t Dataword()  const { return dataword; };      

      
    private:
      int pt_lut_address;
      int phi_full      ;
      int phi_GMT       ;
      int eta_GMT       ;
      int pt            ;
      int quality       ;
      int bx            ;
      int me4_ID        ;
      int me3_ID        ;
      int me2_ID        ;
      int me1_ID        ;
      int me4_TBIN      ;
      int me3_TBIN      ;
      int me2_TBIN      ;
      int me1_TBIN      ;
      int tbin_num      ;
      int hl            ;
      int c             ;
      int vc            ;
      int vt            ;
      int se            ;
      int bc0           ;
      uint64_t dataword;
      
    }; // End of class SP

    // Define a vector of SP
    typedef std::vector<SP> SPCollection;

  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_SP_h__ */

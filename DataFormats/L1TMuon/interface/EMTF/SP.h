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
      pt_lut_address(-99), phi_full(-99), phi_gmt(-99), eta(-99), pt(-99), quality(-99), bx(-99), me4_id(-99), me3_id(-99), me2_id(-99), me1_id(-99), me4_tbin(-99), me3_tbin(-99), me2_tbin(-99), me1_tbin(-99), tbin_num(-99), hl(-99), c(-99), vc(-99), vt(-99), se(-99), bc0(-99), dataword(-99) 
	{};
      
    SP(int int_pt_lut_address, int int_phi_full, int int_phi_gmt, int int_eta, int int_pt, int int_quality, int int_bx, int int_me4_id, int int_me3_id, int int_me2_id, int int_me1_id, int int_me4_tbin, int int_me3_tbin, int int_me2_tbin, int int_me1_tbin, int int_tbin_num, int int_hl, int int_c, int int_vc, int int_vt, int int_se, int int_bc0) :
      pt_lut_address(int_pt_lut_address), phi_full(int_phi_full), phi_gmt(int_phi_gmt), eta(int_eta), pt(int_pt), quality(int_quality), bx(int_bx), me4_id(int_me4_id), me3_id(int_me3_id), me2_id(int_me2_id), me1_id(int_me1_id), me4_tbin(int_me4_tbin), me3_tbin(int_me3_tbin), me2_tbin(int_me2_tbin), me1_tbin(int_me1_tbin), tbin_num(int_tbin_num), hl(int_hl), c(int_c), vc(int_vc), vt(int_vt), se(int_se), bc0(int_bc0), dataword(-99)
    	{};
      
      virtual ~SP() {};
      
      void set_pt_lut_address(int bits)       { pt_lut_address= bits; };
      void set_phi_full      (int bits)       { phi_full      = bits; };
      void set_phi_gmt       (int bits)       { phi_gmt       = bits; };
      void set_eta           (int bits)       { eta           = bits; };
      void set_pt            (int bits)       { pt            = bits; };
      void set_quality       (int bits)       { quality       = bits; };
      void set_bx            (int bits)       { bx            = bits; };
      void set_me4_id        (int bits)       { me4_id        = bits; };
      void set_me3_id        (int bits)       { me3_id        = bits; };
      void set_me2_id        (int bits)       { me2_id        = bits; };
      void set_me1_id        (int bits)       { me1_id        = bits; };
      void set_me4_tbin      (int bits)       { me4_tbin      = bits; };
      void set_me3_tbin      (int bits)       { me3_tbin      = bits; };
      void set_me2_tbin      (int bits)       { me2_tbin      = bits; };
      void set_me1_tbin      (int bits)       { me1_tbin      = bits; };
      void set_tbin_num      (int bits)       { tbin_num      = bits; };
      void set_hl            (int bits)       { hl            = bits; };
      void set_c             (int bits)       { c             = bits; };
      void set_vc            (int bits)       { vc            = bits; };
      void set_vt            (int bits)       { vt            = bits; };
      void set_se            (int bits)       { se            = bits; };
      void set_bc0           (int bits)       { bc0           = bits; };
      void set_dataword(uint64_t bits)  { dataword = bits; };

      const int Pt_lut_address() const { return pt_lut_address; };
      const int Phi_full()       const { return phi_full      ; };
      const int Phi_gmt()        const { return phi_gmt       ; };
      const int Eta()            const { return eta           ; };
      const int Pt()             const { return pt            ; };
      const int Quality()        const { return quality       ; };
      const int BX()             const { return bx            ; };
      const int ME4_id()         const { return me4_id        ; };
      const int ME3_id()         const { return me3_id        ; };
      const int ME2_id()         const { return me2_id        ; };
      const int ME1_id()         const { return me1_id        ; };
      const int ME4_tbin()       const { return me4_tbin      ; };
      const int ME3_tbin()       const { return me3_tbin      ; };
      const int ME2_tbin()       const { return me2_tbin      ; };
      const int ME1_tbin()       const { return me1_tbin      ; };
      const int Tbin_num()       const { return tbin_num      ; };
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
      int phi_gmt       ;
      int eta           ;
      int pt            ;
      int quality       ;
      int bx            ;
      int me4_id        ;
      int me3_id        ;
      int me2_id        ;
      int me1_id        ;
      int me4_tbin      ;
      int me3_tbin      ;
      int me2_tbin      ;
      int me1_tbin      ;
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

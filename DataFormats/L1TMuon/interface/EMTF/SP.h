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
      pt_lut_address(-99), phi_full_int(-99), phi_GMT_int(-99), eta_GMT_int(-99), pt_int(-99), 
	quality(-99), bx(-99), me4_ID(-99), me3_ID(-99), me2_ID(-99), me1_ID(-99), 
	me4_TBIN(-99), me3_TBIN(-99), me2_TBIN(-99), me1_TBIN(-99), tbin_num(-99), 
	hl(-99), c(-99), vc(-99), vt(-99), se(-99), bc0(-99), 
	pt(-99), phi_full(-99), phi_full_rad(-99), phi_GMT(-99), phi_GMT_corr(-99), 
	phi_GMT_rad(-99), eta_GMT(-99), format_errors(0), dataword(-99) 
	{};

    /* Could we have the fill constructor take the "true" eta/phi/pt and fill the integers? - AWB 02.02.16 */
    SP(int int_pt_lut_address, int int_phi_full_int, int int_phi_GMT_int, int int_eta_GMT_int, int int_pt_int, 
       int int_quality, int int_bx, int int_me4_ID, int int_me3_ID, int int_me2_ID, int int_me1_ID, 
       int int_me4_TBIN, int int_me3_TBIN, int int_me2_TBIN, int int_me1_TBIN, int int_TBIN_num, 
       int int_hl, int int_c, int int_vc, int int_vt, int int_se, int int_bc0) :
       /* float flt_pt, float flt_phi_full, float flt_phi_full_rad, float flt_phi_GMT, float flt_phi_GMT_rad, float flt_eta_GMT) : */
      pt_lut_address(int_pt_lut_address), phi_full_int(int_phi_full_int), phi_GMT_int(int_phi_GMT_int), eta_GMT_int(int_eta_GMT_int), pt_int(int_pt_int), 
	quality(int_quality), bx(int_bx), me4_ID(int_me4_ID), me3_ID(int_me3_ID), me2_ID(int_me2_ID), me1_ID(int_me1_ID), 
	me4_TBIN(int_me4_TBIN), me3_TBIN(int_me3_TBIN), me2_TBIN(int_me2_TBIN), me1_TBIN(int_me1_TBIN), tbin_num(int_TBIN_num), 
	hl(int_hl), c(int_c), vc(int_vc), vt(int_vt), se(int_se), bc0(int_bc0), 
	/* pt(flt_pt), phi_full(flt_phi_full), phi_full_rad(flt_phi_full_rad), phi_GMT(flt_phi_GMT), phi_GMT_rad(flt_phi_GMT_rad), eta_GMT(flt_eta_GMT), */
	format_errors(0), dataword(-99)
    	{};
      
      virtual ~SP() {};

      float pi = 3.141592653589793238;

      // phi_full gives the exact phi value
      // phi_GMT (the value used by GMT) is a rough estimate, with offsets of 1-2 degrees for some phi values
      // The conversion used is: phi_GMT =        (360/576)*phi_GMT_int +        (180/576)
      // More accurate would be: phi_GMT = 1.0208*(360/576)*phi_GMT_int + 1.0208*(180/576) + 0.552

      float calc_pt           (int  bits)  { return (bits - 1) * 0.5;                   };
      int   calc_pt_int       (float val)  { return (val * 2)  + 1;                     };
      float calc_phi_full     (int  bits)  { return (bits / 60.0)        - 2.0;         };  
      float calc_phi_full_rad (int  bits)  { return (bits * pi / 10800)  - (pi / 90);   };
      int   calc_phi_full_int (float val)  { return (val + 2)            * 60;          };  
      float calc_phi_GMT      (int  bits)  { return (bits * 0.625)       + 0.3125;      };  /* x (360/576)  + (180/576) */
      float calc_phi_GMT_corr (int  bits)  { return (bits * 0.625 * 1.0208) + 0.3125 * 1.0208 + 0.552; };  /* AWB mod 09.02.16 */
      float calc_phi_GMT_rad  (int  bits)  { return (bits * pi / 288)    + (pi / 576);  };  /* x (2*pi/576) + (pi/576)  */
      int   calc_phi_GMT_int  (float val)  { return (val - 0.3125)       / 0.625;       };  /* - (180/576)  / (360/576) */
      float calc_eta_GMT      (int  bits)  { return bits * 0.010875;                    };
      int   calc_eta_GMT_int  (float val)  { return val  / 0.010875;                    };


      // Setting pt, phi_full, phi_GMT, or eta_GMT automatically sets all formats (integer, degrees, radians) 
      void set_pt_int       (int  bits)  { pt_int        = bits;               
	                                   set_pt_only           ( calc_pt           (pt_int      ) ); };
      void set_pt           (float val)  { pt            = val;               
	                                   set_pt_int_only       ( calc_pt_int       (pt          ) ); };

      void set_phi_full_int (int  bits)  { phi_full_int  = bits;               
	                                   set_phi_full_only     ( calc_phi_full     (phi_full_int) ); 
                                           set_phi_full_rad_only ( calc_phi_full_rad (phi_full_int) ); };
      void set_phi_full     (float val)  { phi_full      = val;               
	                                   set_phi_full_int_only ( calc_phi_full_int (phi_full    ) ); 
                                           set_phi_full_rad_only ( calc_phi_full_rad (phi_full_int) ); };
      void set_phi_full_rad (float val)  { phi_full_rad  = val;               
	                                   set_phi_full_only     ( val * 180 / pi                   ); 
                                           set_phi_full_int_only ( calc_phi_full_int (phi_full    ) ); };

      void set_phi_GMT_int  (int  bits)  { phi_GMT_int   = bits;               
	                                   set_phi_GMT_only      ( calc_phi_GMT      (phi_GMT_int ) ); 
	                                   set_phi_GMT_corr_only ( calc_phi_GMT_corr (phi_GMT_int ) ); 
                                           set_phi_GMT_rad_only  ( calc_phi_GMT_rad  (phi_GMT_int ) ); };
      void set_phi_GMT      (float val)  { phi_GMT       = val;               
	                                   set_phi_GMT_int_only  ( calc_phi_GMT_int  (phi_GMT     ) ); 
	                                   set_phi_GMT_corr_only ( calc_phi_GMT_corr (phi_GMT_int ) ); 
                                           set_phi_GMT_rad_only  ( calc_phi_GMT_rad  (phi_GMT_int ) ); };
      void set_phi_GMT_rad  (float val)  { phi_GMT_rad   = val;               
	                                   set_phi_GMT_only      ( val * 180 / pi                   ); 
                                           set_phi_GMT_int_only  ( calc_phi_GMT_int  (phi_GMT     ) ); 
                                           set_phi_GMT_corr_only ( calc_phi_GMT_corr (phi_GMT_int ) ); };


      void set_eta_GMT_int  (int  bits)  { eta_GMT_int   = bits;
                                           set_eta_GMT_only      ( calc_eta_GMT      (eta_GMT_int ) ); };
      void set_eta_GMT      (float val)  { eta_GMT       = val;
                                           set_eta_GMT_int_only  ( calc_eta_GMT_int  (eta_GMT     ) ); };


      void set_pt_lut_address(int bits)       { pt_lut_address= bits; };
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
      void add_format_error()           { format_errors += 1; };
      void set_dataword(uint64_t bits)  { dataword = bits; };


      const int Pt_lut_address()  const { return pt_lut_address; };
      const int Phi_full_int()    const { return phi_full_int  ; };
      const int Phi_GMT_int()     const { return phi_GMT_int   ; };
      const int Eta_GMT_int()     const { return eta_GMT_int   ; };
      const int Pt_int()          const { return pt_int        ; };
      const int Quality()         const { return quality       ; };
      const int BX()              const { return bx            ; };
      const int ME4_ID()          const { return me4_ID        ; };
      const int ME3_ID()          const { return me3_ID        ; };
      const int ME2_ID()          const { return me2_ID        ; };
      const int ME1_ID()          const { return me1_ID        ; };
      const int ME4_TBIN()        const { return me4_TBIN      ; };
      const int ME3_TBIN()        const { return me3_TBIN      ; };
      const int ME2_TBIN()        const { return me2_TBIN      ; };
      const int ME1_TBIN()        const { return me1_TBIN      ; };
      const int TBIN_num()        const { return tbin_num      ; };
      const int HL()              const { return hl            ; };
      const int C()               const { return c             ; };
      const int VC()              const { return vc            ; };
      const int VT()              const { return vt            ; };
      const int SE()              const { return se            ; };
      const int BC0()             const { return bc0           ; };
      const float Pt()            const { return pt            ; };
      const float Phi_full()      const { return phi_full      ; };
      const float Phi_full_rad()  const { return phi_full_rad  ; };
      const float Phi_GMT()       const { return phi_GMT       ; };
      const float Phi_GMT_corr()  const { return phi_GMT_corr  ; };
      const float Phi_GMT_rad()   const { return phi_GMT_rad   ; };
      const float Eta_GMT()       const { return eta_GMT       ; };
      const int Format_Errors()  const { return format_errors; };
      const uint64_t Dataword()  const { return dataword; };      

      
    private:
      // Set only specific formats of values
      void set_pt_only           (float val)      { pt           = val;  };
      void set_pt_int_only       (float val)      { pt_int       = val;  };
      void set_phi_full_only     (float val)      { phi_full     = val;  };
      void set_phi_full_rad_only (float val)      { phi_full_rad = val;  };
      void set_phi_full_int_only (float val)      { phi_full_int = val;  };
      void set_phi_GMT_only      (float val)      { phi_GMT      = val;  };
      void set_phi_GMT_corr_only (float val)      { phi_GMT_corr = val;  };
      void set_phi_GMT_rad_only  (float val)      { phi_GMT_rad  = val;  };
      void set_phi_GMT_int_only  (float val)      { phi_GMT_int  = val;  };
      void set_eta_GMT_only      (float val)      { eta_GMT      = val;  };
      void set_eta_GMT_int_only  (float val)      { eta_GMT_int  = val;  };

      int pt_lut_address ;
      int phi_full_int   ;
      int phi_GMT_int    ;
      int eta_GMT_int    ;
      int pt_int         ;
      int quality        ;
      int bx             ;
      int me4_ID         ;
      int me3_ID         ;
      int me2_ID         ;
      int me1_ID         ;
      int me4_TBIN       ;
      int me3_TBIN       ;
      int me2_TBIN       ;
      int me1_TBIN       ;
      int tbin_num       ;
      int hl             ;
      int c              ;
      int vc             ;
      int vt             ;
      int se             ;
      int bc0            ;
      float pt           ;
      float phi_full     ;
      float phi_full_rad ;
      float phi_GMT      ;
      float phi_GMT_corr ;
      float phi_GMT_rad  ;
      float eta_GMT      ;
      int  format_errors;
      uint64_t dataword;
      
    }; // End of class SP

    // Define a vector of SP
    typedef std::vector<SP> SPCollection;

  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_SP_h__ */

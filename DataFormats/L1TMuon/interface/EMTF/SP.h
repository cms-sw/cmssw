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
      pt_lut_address(-99), phi_local_int(-99), phi_GMT_int(-99), eta_GMT_int(-99), pt_int(-99), 
	quality(-99), bx(-99), me2_csc_id(-99), me2_trk_stub_num(-99), me3_csc_id(-99), me3_trk_stub_num(-99), me4_csc_id(-99), me4_trk_stub_num(-99), me1_subsector(-99), me1_csc_id(-99), me1_trk_stub_num(-99),
	me4_TBIN(-99), me3_TBIN(-99), me2_TBIN(-99), me1_TBIN(-99), tbin_num(-99), 
	hl(-99), c(-99), vc(-99), vt(-99), se(-99), bc0(-99), 
	pt(-99), phi_local(-99), phi_local_rad(-99), phi_global(-99), phi_GMT(-99), phi_GMT_corr(-99), 
	phi_GMT_rad(-99), phi_GMT_global(-99), eta_GMT(-99), format_errors(0), dataword(-99) 
	{};

    /* Could we have the fill constructor take the "true" eta/phi/pt and fill the integers? - AWB 02.02.16 */
    SP(int int_pt_lut_address, int int_phi_local_int, int int_phi_GMT_int, int int_eta_GMT_int, int int_pt_int, 
       int int_quality, int int_bx, int int_me2_csc_id, int int_me2_trk_stub_num, int int_me3_csc_id, int int_me3_trk_stub_num, 
       int int_me4_csc_id, int int_me4_trk_stub_num, int int_me1_subsector, int int_me1_csc_id, int int_me1_trk_stub_num, 
       int int_me4_TBIN, int int_me3_TBIN, int int_me2_TBIN, int int_me1_TBIN, int int_TBIN_num, 
       int int_hl, int int_c, int int_vc, int int_vt, int int_se, int int_bc0) :
       /* float flt_pt, float flt_phi_local, float flt_phi_local_rad, float flt_phi_GMT, float flt_phi_GMT_rad, float flt_eta_GMT) : */
        pt_lut_address(int_pt_lut_address), phi_local_int(int_phi_local_int), phi_GMT_int(int_phi_GMT_int), eta_GMT_int(int_eta_GMT_int), pt_int(int_pt_int), 
	quality(int_quality), bx(int_bx), me2_csc_id(int_me2_csc_id), me2_trk_stub_num(int_me2_trk_stub_num), me3_csc_id(int_me3_csc_id), 
        me3_trk_stub_num(int_me3_trk_stub_num), me4_csc_id(int_me4_csc_id), me4_trk_stub_num(int_me4_trk_stub_num), 
        me1_subsector(int_me1_subsector), me1_csc_id(int_me1_csc_id), me1_trk_stub_num(int_me1_trk_stub_num), 
        me4_TBIN(int_me4_TBIN), me3_TBIN(int_me3_TBIN), me2_TBIN(int_me2_TBIN), me1_TBIN(int_me1_TBIN), tbin_num(int_TBIN_num), 
	hl(int_hl), c(int_c), vc(int_vc), vt(int_vt), se(int_se), bc0(int_bc0), 
	/* pt(flt_pt), phi_local(flt_phi_local), phi_local_rad(flt_phi_local_rad), phi_GMT(flt_phi_GMT), phi_GMT_rad(flt_phi_GMT_rad), eta_GMT(flt_eta_GMT), */
	format_errors(0), dataword(-99)
    	{};
      
      virtual ~SP() {};

      float pi = 3.141592653589793238;

      // phi_local gives the exact phi value (marked "phi_full" in the format document)
      // phi_GMT (the value used by GMT) is a rough estimate, with offsets of 1-2 degrees for some phi values
      // The conversion used is: phi_GMT =        (360/576)*phi_GMT_int +        (180/576)
      // More accurate would be: phi_GMT = 1.0208*(360/576)*phi_GMT_int + 1.0208*(180/576) + 0.552

      float calc_pt            (int  bits)  { return (bits - 1) * 0.5;                   };
      int   calc_pt_int        (float val)  { return (val * 2)  + 1;                     };
      float calc_phi_local     (int  bits)  { return (bits / 60.0)        - 2.0;         };  
      float calc_phi_local_rad (int  bits)  { return (bits * pi / 10800)  - (pi / 90);   };
      int   calc_phi_local_int (float val)  { return (val + 2)            * 60;          };  
      float calc_phi_GMT       (int  bits)  { return (bits * 0.625)       + 0.3125;      };  /* x (360/576)  + (180/576) */
      float calc_phi_GMT_corr  (int  bits)  { return (bits * 0.625 * 1.0208) + 0.3125 * 1.0208 + 0.552; };  /* AWB mod 09.02.16 */
      float calc_phi_GMT_rad   (int  bits)  { return (bits * pi / 288)    + (pi / 576);  };  /* x (2*pi/576) + (pi/576)  */
      int   calc_phi_GMT_int   (float val)  { return (val - 0.3125)       / 0.625;       };  /* - (180/576)  / (360/576) */
      float calc_eta_GMT       (int  bits)  { return bits * 0.010875;                    };
      int   calc_eta_GMT_int   (float val)  { return val  / 0.010875;                    };
      float calc_phi_global    (float loc, int sect) { return loc + 15 + (sect - 1) * 60;};  


      // Setting pt, phi_local, phi_GMT, or eta_GMT automatically sets all formats (integer, degrees, radians) 
      void set_pt_int        (int  bits)  { pt_int        = bits;               
	                                    set_pt_only           ( calc_pt           (pt_int      ) ); };
      void set_pt            (float val)  { pt            = val;               
	                                    set_pt_int_only       ( calc_pt_int       (pt          ) ); };

      void set_phi_local_int (int  bits)  { phi_local_int  = bits;               
	                                    set_phi_local_only     ( calc_phi_local     (phi_local_int) ); 
                                            set_phi_local_rad_only ( calc_phi_local_rad (phi_local_int) ); };
      void set_phi_local     (float val)  { phi_local      = val;               
	                                    set_phi_local_int_only ( calc_phi_local_int (phi_local    ) ); 
                                            set_phi_local_rad_only ( calc_phi_local_rad (phi_local_int) ); };
      void set_phi_local_rad (float val)  { phi_local_rad  = val;               
	                                    set_phi_local_only     ( val * 180 / pi                   ); 
                                            set_phi_local_int_only ( calc_phi_local_int (phi_local    ) ); };

      void set_phi_GMT_int   (int  bits)  { phi_GMT_int   = bits;               
	                                    set_phi_GMT_only      ( calc_phi_GMT      (phi_GMT_int ) ); 
	                                    set_phi_GMT_corr_only ( calc_phi_GMT_corr (phi_GMT_int ) ); 
                                            set_phi_GMT_rad_only  ( calc_phi_GMT_rad  (phi_GMT_int ) ); };
      void set_phi_GMT       (float val)  { phi_GMT       = val;               
	                                    set_phi_GMT_int_only  ( calc_phi_GMT_int  (phi_GMT     ) ); 
	                                    set_phi_GMT_corr_only ( calc_phi_GMT_corr (phi_GMT_int ) ); 
                                            set_phi_GMT_rad_only  ( calc_phi_GMT_rad  (phi_GMT_int ) ); };
      void set_phi_GMT_rad   (float val)  { phi_GMT_rad   = val;               
	                                    set_phi_GMT_only      ( val * 180 / pi                   ); 
                                            set_phi_GMT_int_only  ( calc_phi_GMT_int  (phi_GMT     ) ); 
                                            set_phi_GMT_corr_only ( calc_phi_GMT_corr (phi_GMT_int ) ); };

      void set_phi_global    (float loc, int sect) { set_phi_global_only     ( calc_phi_global (loc, sect) ); };
      void set_phi_GMT_global(float loc, int sect) { set_phi_GMT_global_only ( calc_phi_global (loc, sect) ); };

      void set_eta_GMT_int   (int  bits)  { eta_GMT_int   = bits;
                                            set_eta_GMT_only      ( calc_eta_GMT      (eta_GMT_int ) ); };
      void set_eta_GMT       (float val)  { eta_GMT       = val;
                                            set_eta_GMT_int_only  ( calc_eta_GMT_int  (eta_GMT     ) ); };


      void set_pt_lut_address   (int bits)       { pt_lut_address   = bits; };
      void set_quality          (int bits)       { quality          = bits; };
      void set_bx               (int bits)       { bx               = bits; };
      void set_me2_csc_id       (int bits)       { me2_csc_id       = bits; };
      void set_me2_trk_stub_num (int bits)       { me2_trk_stub_num = bits; };
      void set_me3_csc_id       (int bits)       { me3_csc_id       = bits; };
      void set_me3_trk_stub_num (int bits)       { me3_trk_stub_num = bits; };
      void set_me4_csc_id       (int bits)       { me4_csc_id       = bits; };
      void set_me4_trk_stub_num (int bits)       { me4_trk_stub_num = bits; };
      void set_me1_subsector    (int bits)       { me1_subsector    = bits; };
      void set_me1_csc_id       (int bits)       { me1_csc_id       = bits; };
      void set_me1_trk_stub_num (int bits)       { me1_trk_stub_num = bits; };      
      void set_me4_TBIN         (int bits)       { me4_TBIN         = bits; };
      void set_me3_TBIN         (int bits)       { me3_TBIN         = bits; };
      void set_me2_TBIN         (int bits)       { me2_TBIN         = bits; };
      void set_me1_TBIN         (int bits)       { me1_TBIN         = bits; };
      void set_TBIN_num         (int bits)       { tbin_num         = bits; };
      void set_hl               (int bits)       { hl               = bits; };
      void set_c                (int bits)       { c                = bits; };
      void set_vc               (int bits)       { vc               = bits; };
      void set_vt               (int bits)       { vt               = bits; };
      void set_se               (int bits)       { se               = bits; };
      void set_bc0              (int bits)       { bc0              = bits; };
      void add_format_error()           { format_errors += 1; };
      void set_dataword(uint64_t bits)  { dataword = bits; };


      const int Pt_lut_address()   const { return pt_lut_address  ; };
      const int Phi_local_int()    const { return phi_local_int   ; };
      const int Phi_GMT_int()      const { return phi_GMT_int     ; };
      const int Eta_GMT_int()      const { return eta_GMT_int     ; };
      const int Pt_int()           const { return pt_int          ; };
      const int Quality()          const { return quality         ; };
      const int BX()               const { return bx              ; };
      const int ME2_csc_id()       const { return me2_csc_id      ; };
      const int ME2_trk_stub_num() const { return me2_trk_stub_num; };
      const int ME3_csc_id()       const { return me3_csc_id      ; };  
      const int ME3_trk_stub_num() const { return me3_trk_stub_num; };
      const int ME4_csc_id()       const { return me4_csc_id      ; };
      const int ME4_trk_stub_num() const { return me4_trk_stub_num; };
      const int ME1_subsector()    const { return me1_subsector   ; };
      const int ME1_csc_id()       const { return me1_csc_id      ; };
      const int ME1_trk_stub_num() const { return me1_trk_stub_num; };      
      const int ME4_TBIN()         const { return me4_TBIN        ; };
      const int ME3_TBIN()         const { return me3_TBIN        ; };
      const int ME2_TBIN()         const { return me2_TBIN        ; };
      const int ME1_TBIN()         const { return me1_TBIN        ; };
      const int TBIN_num()         const { return tbin_num        ; };
      const int HL()               const { return hl              ; };
      const int C()                const { return c               ; };
      const int VC()               const { return vc              ; };
      const int VT()               const { return vt              ; };
      const int SE()               const { return se              ; };
      const int BC0()              const { return bc0             ; };
      const float Pt()             const { return pt              ; };
      const float Phi_local()      const { return phi_local       ; };
      const float Phi_local_rad()  const { return phi_local_rad   ; };
      const float Phi_global()     const { return phi_global      ; };
      const float Phi_GMT()        const { return phi_GMT         ; };
      const float Phi_GMT_corr()   const { return phi_GMT_corr    ; };
      const float Phi_GMT_rad()    const { return phi_GMT_rad     ; };
      const float Phi_GMT_global() const { return phi_GMT_global  ; };
      const float Eta_GMT()        const { return eta_GMT         ; };
      const int Format_Errors()  const { return format_errors; };
      const uint64_t Dataword()  const { return dataword; };      

      
    private:
      // Set only specific formats of values
      void set_pt_only            (float val)      { pt             = val; };
      void set_pt_int_only        (float val)      { pt_int         = val; };
      void set_phi_local_only     (float val)      { phi_local      = val; };
      void set_phi_local_rad_only (float val)      { phi_local_rad  = val; };
      void set_phi_local_int_only (float val)      { phi_local_int  = val; };
      void set_phi_global_only    (float val)      { phi_global     = val; };
      void set_phi_GMT_only       (float val)      { phi_GMT        = val; };
      void set_phi_GMT_corr_only  (float val)      { phi_GMT_corr   = val; };
      void set_phi_GMT_rad_only   (float val)      { phi_GMT_rad    = val; };
      void set_phi_GMT_int_only   (float val)      { phi_GMT_int    = val; };
      void set_phi_GMT_global_only(float val)      { phi_GMT_global = val; };
      void set_eta_GMT_only       (float val)      { eta_GMT        = val; };
      void set_eta_GMT_int_only   (float val)      { eta_GMT_int    = val; };

      int pt_lut_address  ;
      int phi_local_int   ;
      int phi_GMT_int     ;
      int eta_GMT_int     ;
      int pt_int          ;
      int quality         ;
      int bx              ;
      int me2_csc_id      ;
      int me2_trk_stub_num;
      int me3_csc_id      ;
      int me3_trk_stub_num;
      int me4_csc_id      ;      
      int me4_trk_stub_num;
      int me1_subsector   ;
      int me1_csc_id      ;      
      int me1_trk_stub_num;      
      int me4_TBIN        ;
      int me3_TBIN        ;
      int me2_TBIN        ;
      int me1_TBIN        ;
      int tbin_num        ;
      int hl              ;
      int c               ;
      int vc              ;
      int vt              ;
      int se              ;
      int bc0             ;
      float pt            ;
      float phi_local     ;
      float phi_local_rad ;
      float phi_global    ;
      float phi_GMT       ;
      float phi_GMT_corr  ;
      float phi_GMT_rad   ;
      float phi_GMT_global;
      float eta_GMT       ;
      int  format_errors;
      uint64_t dataword;
      
    }; // End of class SP

    // Define a vector of SP
    typedef std::vector<SP> SPCollection;

  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_SP_h__ */

// Class for Muon Endcap (ME) Data Record

#ifndef __l1t_emtf_ME_h__
#define __l1t_emtf_ME_h__

#include <boost/cstdint.hpp> 

namespace l1t {
  namespace emtf {
    class ME {
    public:
      
      explicit ME(uint64_t dataword);
    
    ME() : 
      me_bxn(-99), key_wire_group(-99), clct_key_half_strip(-99), quality(-99), clct_pattern(-99), 
	csc_ID(-99), epc(-99), station(-99), tbin_num(-99), bc0(-99), bxe(-99), lr(-99), afff(-99), 
	cik(-99), nit(-99), afef(-99), se(-99), sm(-99), af(-99), vp(-99), format_errors(0), dataword(-99) 
	{};
      
    ME(int int_me_bxn, int int_key_wire_group, int int_clct_key_half_strip, int int_quality, int int_clct_pattern, 
       int int_csc_ID, int int_epc, int int_station, int int_tbin_num, int int_bc0, int int_bxe, int int_lr, int int_afff, 
       int int_cik, int int_nit, int int_afef, int int_se, int int_sm, int int_af, int int_vp) :
      me_bxn(int_me_bxn), key_wire_group(int_key_wire_group), clct_key_half_strip(int_clct_key_half_strip), quality(int_quality), clct_pattern(int_clct_pattern), 
	csc_ID(int_csc_ID), epc(int_epc), station(int_station), tbin_num(int_tbin_num), bc0(int_bc0), bxe(int_bxe), lr(int_lr), afff(int_afff), 
	cik(int_cik), nit(int_nit), afef(int_afef), se(int_se), sm(int_sm), af(int_af), vp(int_vp), format_errors(0), dataword(-99)
    	{};
      
      virtual ~ME() {};
      
      void set_me_bxn(int bits)               {  me_bxn = bits; };
      void set_key_wire_group(int bits)       {  key_wire_group = bits; };
      void set_clct_key_half_strip(int bits)  {  clct_key_half_strip = bits; };
      void set_quality(int bits)              {  quality = bits; };
      void set_clct_pattern(int bits)         {  clct_pattern = bits; };
      void set_csc_ID(int bits)               {  csc_ID = bits; };
      void set_epc(int bits)                  {  epc = bits; };
      void set_station(int bits)              {  station = bits; };
      void set_tbin_num(int bits)             {  tbin_num = bits; };
      void set_bc0(int bits)                  {  bc0 = bits; };
      void set_bxe(int bits)                  {  bxe = bits; };
      void set_lr(int bits)                   {  lr = bits; };
      void set_afff(int bits)                 {  afff = bits; };
      void set_cik(int bits)                  {  cik = bits; };
      void set_nit(int bits)                  {  nit = bits; };
      void set_afef(int bits)                 {  afef = bits; };
      void set_se(int bits)                   {  se = bits; };
      void set_sm(int bits)                   {  sm = bits; };
      void set_af(int bits)                   {  af = bits; };
      void set_vp(int bits)                   {  vp = bits; };
      void add_format_error()                 { format_errors += 1; };
      void set_dataword(uint64_t bits)        { dataword = bits;       };

      const int ME_BXN()               const { return  me_bxn ; };
      const int Key_wire_group()       const { return  key_wire_group ; };
      const int CLCT_key_half_strip()  const { return  clct_key_half_strip ; };
      const int Quality()              const { return  quality ; };
      const int CLCT_pattern()         const { return  clct_pattern ; };
      const int CSC_ID()               const { return  csc_ID ; };
      const int EPC()                  const { return  epc ; };
      const int Station()              const { return  station ; };
      const int Tbin_num()             const { return  tbin_num ; };
      const int BC0()                  const { return  bc0 ; };
      const int BXE()                  const { return  bxe ; };
      const int LR()                   const { return  lr ; };
      const int AFFF()                 const { return  afff ; };
      const int CIK()                  const { return  cik ; };
      const int NIT()                  const { return  nit ; };
      const int AFEF()                 const { return  afef ; };
      const int SE()                   const { return  se ; };
      const int SM()                   const { return  sm ; };
      const int AF()                   const { return  af ; };
      const int VP()                   const { return  vp ; };      
      const int Format_Errors()        const { return format_errors; };
      const uint64_t Dataword()        const { return dataword;       };      
      
    private:
      int  me_bxn;
      int  key_wire_group;
      int  clct_key_half_strip;
      int  quality;
      int  clct_pattern;
      int  csc_ID;
      int  epc; 
      int  station;
      int  tbin_num;
      int  bc0; 
      int  bxe; 
      int  lr;
      int  afff;
      int  cik; 
      int  nit; 
      int  afef;
      int  se;
      int  sm;
      int  af;
      int  vp;
      int  format_errors;
      uint64_t dataword;
      
    }; // End of class ME
    
    // Define a vector of ME
    typedef std::vector<ME> MECollection;

  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_ME_h__ */

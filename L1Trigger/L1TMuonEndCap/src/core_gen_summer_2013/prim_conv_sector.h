// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#ifndef __prim_conv_sector_h_file__
#define __prim_conv_sector_h_file__
#include "vppc_sim_lib.h"
#include "prim_conv11.h"
#include "prim_conv.h"
#include "prim_conv.h"

class prim_conv_sector
{
 public:
	prim_conv_sector(){built = false; glbl_gsr = true; defparam();}
	void defparam();
	void build();
	bool built;
	bool glbl_gsr;
		unsigned station;
		unsigned cscid;
		// segments per chamber
	unsigned seg_ch;
		// bit widths of ph and th outputs, reduced precision
// have to be derived from pattern width on top level
	unsigned bw_ph;
		unsigned bw_th;
		// bit widths of ph and th, full precision
	unsigned bw_fph;
		unsigned bw_fth;
		// wiregroup input bit width (0..111)
	unsigned bw_wg;
		// bit width of dblstrip input (max 80 for ME234/1 with double-width strips)
	unsigned bw_ds;
		// width of halfstrip input
	unsigned bw_hs;
		// pattern half-width for stations 3,4
	unsigned pat_w_st3; //4;
		// pattern half-width for station 1
	unsigned pat_w_st1;
		// number of input bits for stations 3,4
	unsigned full_pat_w_st3;
		// number of input bits for st 1
	unsigned full_pat_w_st1;
		// width of zero padding for station copies
	unsigned padding_w_st1;
		unsigned padding_w_st3;
		// full pattern widths (aka reduced pattern)
	unsigned red_pat_w_st3;
		unsigned red_pat_w_st1;
		// number of folds for pattern detectors, do not set to 1
	unsigned fold;
		// number of th outputs for ME1/1
	unsigned th_ch11;
		unsigned bw_q;
		unsigned bw_addr;
		// strips per section, calculated so ph pattern would cover +/- 8 deg in st 1
	unsigned ph_raw_w; // kludge to fix synth error, need to understand
		unsigned th_raw_w;
		// max possible drifttime
	unsigned max_drift;
		// bit widths of precise phi and eta outputs
	unsigned bw_phi;
		unsigned bw_eta;
		// width of ph raw hits, max coverage +8 to cover possible chamber displacement 
	unsigned ph_hit_w; //80 + 8;
		// for 20 deg chambers
	unsigned ph_hit_w20;
		// for 10 deg chambers
	unsigned ph_hit_w10; //40 + 8;  
		// width of th raw hits, max coverage +8 to cover possible chamber displacement 
	unsigned th_hit_w;
		unsigned endcap;
		unsigned n_strips;
		unsigned n_wg;
		// theta range (take +1 because th_coverage contains max th value starting from 0)
	unsigned th_coverage;
		// phi range
	unsigned ph_coverage; //80 : 40;
		// number of th outputs takes ME1/1 th duplication into account
	unsigned th_ch;
		// is this chamber mounted in reverse direction?
	unsigned ph_reverse;
		unsigned th_mem_sz;
		unsigned th_corr_mem_sz;
		// multiplier bit width (phi + factor)
	unsigned mult_bw;
		// ph zone boundaries for chambers that cover more than one zone
// hardcoded boundaries must match boundaries in ph_th_match module
	unsigned ph_zone_bnd1;
		unsigned ph_zone_bnd2;
		unsigned zone_overlap;
		// sorter parameters
	unsigned bwr; // rank width
		unsigned bpow; // (1 << bpow) is count of input ranks
		unsigned cnr; // internal rank count
		unsigned cnrex; // actual input rank count, must be even
	
		signal_ q;
		signal_ wg;
		signal_ hstr;
		signal_ cpat;
		signal_ cs;
		signal_ sel;
		signal_ addr;
		signal_ r_in; // input data for memory or register
		signal_ we; // write enable for memory or register
		signal_ clk; // write clock
		signal_ control_clk; // control interface clock
		signal_ ph;
		signal_ th11;
		signal_ th;
		signal_ vl;
		signal_ phzvl;
		signal_ me11a;
		signal_ cpatr;
		signal_ ph_hit;
		signal_ th_hit;
		signal_ r_out; // output data from memory or register
	
			// wires for read data for each module,
// have to be muxed to r_out according to cs
signal_storage r_out_w__storage;  signal_ r_out_w;
			// wires for we signals for each module
signal_storage we_w__storage;  signal_ we_w;
		signal_storage dummy__storage;  signal_ dummy;
	
			// mux LUT outputs to r_out
signal_storage s__storage;  signal_ s;
		signal_storage c__storage;  signal_ c;
		unsigned i;
		unsigned j;
	
	void init ();
	void operator()
	(
	signal_& q__io,
	signal_& wg__io,
	signal_& hstr__io,
	signal_& cpat__io,
	signal_& ph__io,
	signal_& th11__io,
	signal_& th__io,
	signal_& vl__io,
	signal_& phzvl__io,
	signal_& me11a__io,
	signal_& cpatr__io,
	signal_& ph_hit__io,
	signal_& th_hit__io,
	signal_& cs__io,
	signal_& sel__io,
	signal_& addr__io,
	signal_& r_in__io,
	signal_& r_out__io,
	signal_& we__io,
	signal_& clk__io,
	signal_& control_clk__io
	);
	class genblk__class
	{
	 public:
		class station11__class
		{
		 public:
			class csc11__class
			{
			 public:
				prim_conv11 pc11;

				void init();
			};
			map <ull, csc11__class> csc11;

			void init();
		};
		map <ull, station11__class> station11;
		class station12__class
		{
		 public:
			class csc12__class
			{
			 public:
				prim_conv pc12;

				void init();
			};
			map <ull, csc12__class> csc12;

			void init();
		};
		map <ull, station12__class> station12;
		class station__class
		{
		 public:
			class csc__class
			{
			 public:
				prim_conv pc;

				void init();
			};
			map <ull, csc__class> csc;

			void init();
		};
		map <ull, station__class> station;

		void init();
	};
	genblk__class genblk;
};
#endif

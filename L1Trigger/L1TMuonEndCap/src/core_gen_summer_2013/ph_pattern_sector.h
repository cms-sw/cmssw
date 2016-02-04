// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#ifndef __ph_pattern_sector_h_file__
#define __ph_pattern_sector_h_file__
#include "vppc_sim_lib.h"
#include "ph_pattern.h"

class ph_pattern_sector
{
 public:
	ph_pattern_sector(){built = false; glbl_gsr = true; defparam();}
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
	
		signal_ st;
		signal_ drifttime;
		signal_ foldn;
		signal_ clk;
		signal_ qcode;
			// left, right and center qualities
signal_storage ql__storage;  signal_ ql;
		signal_storage qr__storage;  signal_ qr;
		signal_storage qc__storage;  signal_ qc;
	
			// copy of stations with padding of zeroes on sides, so it's easier to pass to detectors
// station numbering is different here:
// 0 == st1
// 1 == st3
// 2 == st4
// st 2 does not need padding
signal_storage stp__storage;  signal_ stp;
			// quality codes from pattern detectors [zone][key_strip]
signal_storage qcode_p__storage;  signal_ qcode_p;
	
		unsigned i;
		unsigned z;
		// quality code pattern numbers [zone][key_strip]
	unsigned zi;
		unsigned ri;
	
	void init ();
	void operator()
	(
	signal_& st__io,
	signal_& drifttime__io,
	signal_& foldn__io,
	signal_& qcode__io,
	signal_& clk__io
	);
	class gb__class
	{
	 public:
		class ph_pat_zone__class
		{
		 public:
			class ph_pat_hit__class
			{
			 public:
				ph_pattern php;

				void init();
			};
			map <ull, ph_pat_hit__class> ph_pat_hit;

			void init();
		};
		map <ull, ph_pat_zone__class> ph_pat_zone;

		void init();
	};
	gb__class gb;
};
#endif

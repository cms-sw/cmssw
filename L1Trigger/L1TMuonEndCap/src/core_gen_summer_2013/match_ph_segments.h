// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#ifndef __match_ph_segments_h_file__
#define __match_ph_segments_h_file__
#include "vppc_sim_lib.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"
#include "find_segment.h"

class match_ph_segments
{
 public:
	match_ph_segments(){built = false; glbl_gsr = true; defparam();}
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
	
		signal_ ph_num;
		signal_ ph_q;
		signal_ ph;
		signal_ vl;
		signal_ th11;
		signal_ th;
		signal_ cpat;
		signal_ clk;
		signal_ vi; // valid (for each segment in chamber, so we can identify which th to use later)
		signal_ hi; // bx index
		signal_ ci; // chamber
		signal_ si; // segment which has matching ph
		signal_ ph_match; // matching phi
		signal_ th_match; // matching th, 2 segments 
		signal_ th_match11; // matching th for ME11 (station 0 only), 4 segments (due to th duplication)
		signal_ cpat_match; // matching patterns
		signal_ ph_qr;
			// segments rerouted for find_segment inputs
// see find_segment_reroute.xlsx for details
// indexes are: [bx_history][chamber][segment]
signal_storage ph_seg___z0_s0__storage;  signal_ ph_seg___z0_s0;
		signal_storage ph_seg_v_z0_s0__storage;  signal_ ph_seg_v_z0_s0;
		signal_storage ph_seg___z0_s1__storage;  signal_ ph_seg___z0_s1;
		signal_storage ph_seg_v_z0_s1__storage;  signal_ ph_seg_v_z0_s1;
		signal_storage ph_seg___z0_s2__storage;  signal_ ph_seg___z0_s2;
		signal_storage ph_seg_v_z0_s2__storage;  signal_ ph_seg_v_z0_s2;
		signal_storage ph_seg___z0_s3__storage;  signal_ ph_seg___z0_s3;
		signal_storage ph_seg_v_z0_s3__storage;  signal_ ph_seg_v_z0_s3;
		signal_storage ph_seg___z1_s0__storage;  signal_ ph_seg___z1_s0;
		signal_storage ph_seg_v_z1_s0__storage;  signal_ ph_seg_v_z1_s0;
		signal_storage ph_seg___z1_s1__storage;  signal_ ph_seg___z1_s1;
		signal_storage ph_seg_v_z1_s1__storage;  signal_ ph_seg_v_z1_s1;
		signal_storage ph_seg___z1_s2__storage;  signal_ ph_seg___z1_s2;
		signal_storage ph_seg_v_z1_s2__storage;  signal_ ph_seg_v_z1_s2;
		signal_storage ph_seg___z1_s3__storage;  signal_ ph_seg___z1_s3;
		signal_storage ph_seg_v_z1_s3__storage;  signal_ ph_seg_v_z1_s3;
		signal_storage ph_seg___z2_s0__storage;  signal_ ph_seg___z2_s0;
		signal_storage ph_seg_v_z2_s0__storage;  signal_ ph_seg_v_z2_s0;
		signal_storage ph_seg___z2_s1__storage;  signal_ ph_seg___z2_s1;
		signal_storage ph_seg_v_z2_s1__storage;  signal_ ph_seg_v_z2_s1;
		signal_storage ph_seg___z2_s2__storage;  signal_ ph_seg___z2_s2;
		signal_storage ph_seg_v_z2_s2__storage;  signal_ ph_seg_v_z2_s2;
		signal_storage ph_seg___z2_s3__storage;  signal_ ph_seg___z2_s3;
		signal_storage ph_seg_v_z2_s3__storage;  signal_ ph_seg_v_z2_s3;
		signal_storage ph_seg___z3_s0__storage;  signal_ ph_seg___z3_s0;
		signal_storage ph_seg_v_z3_s0__storage;  signal_ ph_seg_v_z3_s0;
		signal_storage ph_seg___z3_s1__storage;  signal_ ph_seg___z3_s1;
		signal_storage ph_seg_v_z3_s1__storage;  signal_ ph_seg_v_z3_s1;
		signal_storage ph_seg___z3_s2__storage;  signal_ ph_seg___z3_s2;
		signal_storage ph_seg_v_z3_s2__storage;  signal_ ph_seg_v_z3_s2;
		signal_storage ph_seg___z3_s3__storage;  signal_ ph_seg___z3_s3;
		signal_storage ph_seg_v_z3_s3__storage;  signal_ ph_seg_v_z3_s3;
		signal_storage th_seg___z0_s0__storage;  signal_ th_seg___z0_s0;
		signal_storage th_seg___z0_s1__storage;  signal_ th_seg___z0_s1;
		signal_storage th_seg___z0_s2__storage;  signal_ th_seg___z0_s2;
		signal_storage th_seg___z0_s3__storage;  signal_ th_seg___z0_s3;
		signal_storage th_seg___z1_s0__storage;  signal_ th_seg___z1_s0;
		signal_storage th_seg___z1_s1__storage;  signal_ th_seg___z1_s1;
		signal_storage th_seg___z1_s2__storage;  signal_ th_seg___z1_s2;
		signal_storage th_seg___z1_s3__storage;  signal_ th_seg___z1_s3;
		signal_storage th_seg___z2_s0__storage;  signal_ th_seg___z2_s0;
		signal_storage th_seg___z2_s1__storage;  signal_ th_seg___z2_s1;
		signal_storage th_seg___z2_s2__storage;  signal_ th_seg___z2_s2;
		signal_storage th_seg___z2_s3__storage;  signal_ th_seg___z2_s3;
		signal_storage th_seg___z3_s0__storage;  signal_ th_seg___z3_s0;
		signal_storage th_seg___z3_s1__storage;  signal_ th_seg___z3_s1;
		signal_storage th_seg___z3_s2__storage;  signal_ th_seg___z3_s2;
		signal_storage th_seg___z3_s3__storage;  signal_ th_seg___z3_s3;
		signal_storage cpat_seg___z0_s0__storage;  signal_ cpat_seg___z0_s0;
		signal_storage cpat_seg___z0_s1__storage;  signal_ cpat_seg___z0_s1;
		signal_storage cpat_seg___z0_s2__storage;  signal_ cpat_seg___z0_s2;
		signal_storage cpat_seg___z0_s3__storage;  signal_ cpat_seg___z0_s3;
		signal_storage cpat_seg___z1_s0__storage;  signal_ cpat_seg___z1_s0;
		signal_storage cpat_seg___z1_s1__storage;  signal_ cpat_seg___z1_s1;
		signal_storage cpat_seg___z1_s2__storage;  signal_ cpat_seg___z1_s2;
		signal_storage cpat_seg___z1_s3__storage;  signal_ cpat_seg___z1_s3;
		signal_storage cpat_seg___z2_s0__storage;  signal_ cpat_seg___z2_s0;
		signal_storage cpat_seg___z2_s1__storage;  signal_ cpat_seg___z2_s1;
		signal_storage cpat_seg___z2_s2__storage;  signal_ cpat_seg___z2_s2;
		signal_storage cpat_seg___z2_s3__storage;  signal_ cpat_seg___z2_s3;
		signal_storage cpat_seg___z3_s0__storage;  signal_ cpat_seg___z3_s0;
		signal_storage cpat_seg___z3_s1__storage;  signal_ cpat_seg___z3_s1;
		signal_storage cpat_seg___z3_s2__storage;  signal_ cpat_seg___z3_s2;
		signal_storage cpat_seg___z3_s3__storage;  signal_ cpat_seg___z3_s3;
	
	
		unsigned i;
		unsigned j;
		unsigned k;
		unsigned ki;
	
	void init ();
	void operator()
	(
	signal_& ph_num__io,
	signal_& ph_q__io,
	signal_& ph__io,
	signal_& vl__io,
	signal_& th11__io,
	signal_& th__io,
	signal_& cpat__io,
	signal_& vi__io,
	signal_& hi__io,
	signal_& ci__io,
	signal_& si__io,
	signal_& ph_match__io,
	signal_& th_match__io,
	signal_& th_match11__io,
	signal_& cpat_match__io,
	signal_& ph_qr__io,
	signal_& clk__io
	);
	class gb__class
	{
	 public:
		class fs_loop__class
		{
		 public:
			find_segment fs_00;
			find_segment fs_01;
			find_segment fs_02;
			find_segment fs_03;
			find_segment fs_10;
			find_segment fs_11;
			find_segment fs_12;
			find_segment fs_13;
			find_segment fs_20;
			find_segment fs_21;
			find_segment fs_22;
			find_segment fs_23;
			find_segment fs_30;
			find_segment fs_31;
			find_segment fs_32;
			find_segment fs_33;

			void init();
		};
		map <ull, fs_loop__class> fs_loop;

		void init();
	};
	gb__class gb;
};
#endif

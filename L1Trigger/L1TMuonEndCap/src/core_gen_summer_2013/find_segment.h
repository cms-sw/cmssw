// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#ifndef __find_segment_h_file__
#define __find_segment_h_file__
#include "vppc_sim_lib.h"

class find_segment
{
 public:

class comp3_class
{
 public:
	comp3_class(){built = false; glbl_gsr = true; defparam();}
	void defparam();
	void build();
	bool built;
	bool glbl_gsr;
	
		signal_ a;
		signal_ b;
		signal_ c;
		signal_storage r__storage;  signal_ r;
		signal_storage comp3_retval__storage;  signal_ comp3_retval;  // vppc generated: function return value holder
	
	
	
	void init ();
	signal_& operator()
	(
	signal_& a__io,
	signal_& b__io,
	signal_& c__io
	);
} comp3;
	find_segment(){built = false; glbl_gsr = true; defparam();}
	void defparam();
	void build();
	bool built;
	bool glbl_gsr;
		unsigned station;
		unsigned cscid;
		unsigned zone_cham; // 6 chambers in this zone and station
		unsigned zone_seg; // segments per chamber in this zone and station
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
		unsigned max_ph_diff; // max phi difference between pattern and segment
		unsigned bw_phdiff; // ph difference bit width
		unsigned tot_diff;
		//	`localpar nodiff = ((1 << (bpow+1)) - 1);
	unsigned nodiff; // invalid difference
	
		signal_ ph_pat_p; // ph detected in pattern
		signal_ ph_pat_q_p; // pattern valid
		signal_ ph_seg_p;
		signal_ ph_seg_v_p;
		signal_ th_seg_p; // theta
		signal_ cpat_seg_p; // patterns
		signal_ clk;
		signal_ vid; // match valid, each flag shows validity of th coord
		signal_ hid; // history id
		signal_ cid; // chamber id
		signal_ sid; // segment id
		signal_ ph_match; // ph from matching segment
		signal_ th_match;
		signal_ cpat_match; // pattern from matching segment
		signal_storage ph_pat__storage;  signal_ ph_pat; // ph detected in pattern
		signal_storage ph_pat_v__storage;  signal_ ph_pat_v; // pattern valid
		signal_storage ph_seg__storage;  signal_ ph_seg;
		signal_storage ph_seg_v__storage;  signal_ ph_seg_v;
		signal_storage th_seg__storage;  signal_ th_seg;
		signal_storage cpat_seg__storage;  signal_ cpat_seg;
		signal_storage ph_segr__storage;  signal_ ph_segr;
		signal_storage ph_diff_tmp__storage;  signal_ ph_diff_tmp;
		signal_storage ph_diff__storage;  signal_ ph_diff;
		signal_storage rcomp__storage;  signal_ rcomp;
		signal_storage diffi0__storage;  signal_ diffi0;
		signal_storage cmp1__storage;  signal_ cmp1;
		signal_storage diffi1__storage;  signal_ diffi1;
		signal_storage cmp2__storage;  signal_ cmp2;
		signal_storage diffi2__storage;  signal_ diffi2;
		signal_storage cmp3__storage;  signal_ cmp3;
		signal_storage diffi3__storage;  signal_ diffi3;
		signal_storage cmp4__storage;  signal_ cmp4;
		signal_storage diffi4__storage;  signal_ diffi4;
		signal_storage ri__storage;  signal_ ri;
		signal_storage rj__storage;  signal_ rj;
		signal_storage rk__storage;  signal_ rk;
	
	
		unsigned i;
		unsigned j;
		unsigned k;
		unsigned di;
	
	void init ();
	void operator()
	(
	signal_& ph_pat_p__io,
	signal_& ph_pat_q_p__io,
	signal_& ph_seg_p__io,
	signal_& ph_seg_v_p__io,
	signal_& th_seg_p__io,
	signal_& cpat_seg_p__io,
	signal_& vid__io,
	signal_& hid__io,
	signal_& cid__io,
	signal_& sid__io,
	signal_& ph_match__io,
	signal_& th_match__io,
	signal_& cpat_match__io,
	signal_& clk__io
	);
};
#endif

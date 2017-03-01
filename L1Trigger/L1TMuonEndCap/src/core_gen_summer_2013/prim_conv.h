// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#ifndef __prim_conv_h_file__
#define __prim_conv_h_file__
#include "vppc_sim_lib.h"

class prim_conv
{
 public:
	prim_conv(){built = false; glbl_gsr = true; defparam();}
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
	
		signal_ quality; // quality inputs
		signal_ wiregroup; // wiregroup numbers
		signal_ hstrip; // halfstrip numbers
		signal_ clctpat; // clct pattern numbers
		signal_ sel; // binary address of the register or memory to access
		signal_ addr; // address in memory to access. For registers, set to 0
		signal_ r_in; // input data for memory or register
		signal_ we; // write enable for memory or register
		signal_ clk; // write clock
		signal_ control_clk; // control interface clock
		signal_ ph;
		signal_ th;
		signal_ vl;
		signal_ phzvl; // raw hit valid flags for up to 3 ph zones
		signal_ me11a;
		signal_ clctpat_r; // clct pattern numbers
		signal_ ph_hit;
		signal_ th_hit;
		signal_ r_out; // output data from memory or register
		signal_storage eight_str__storage;  signal_ eight_str; // eighth-strip
		signal_storage mult__storage;  signal_ mult;
		signal_storage ph_tmp__storage;  signal_ ph_tmp;
		signal_storage wg__storage;  signal_ wg;
		signal_storage th_tmp__storage;  signal_ th_tmp;
			// theta lut, takes wiregroup, returns theta
signal_storage th_mem__storage;  signal_ th_mem; // make memory size such that any address will be in range
		signal_storage params__storage;  signal_ params; // programmable parameters [0] = ph_init, [1] = th_init, [2] = ph_displacement, [3] = th_displacement
			// initial ph for this chamber scaled to 0.1333 deg/strip
signal_storage fph__storage;  signal_ fph;
		signal_storage factor__storage;  signal_ factor; // strip width factor
		signal_storage me11a_w__storage;  signal_ me11a_w; // flag showing that we're working on ME11a region
		signal_storage clct_pat_corr__storage;  signal_ clct_pat_corr; // phi correction derived from clct pattern
		signal_storage clct_pat_sign__storage;  signal_ clct_pat_sign; // phi correction sign
	
		signal_storage pc_id__storage;  signal_ pc_id; // prim converter ID
	
		unsigned i;
		// signals only for ME1/1
// have to declare them here since vppc does not support declarations in generate blocks yet
	unsigned j;
	
	void init ();
	void operator()
	(
	signal_& quality__io,
	signal_& wiregroup__io,
	signal_& hstrip__io,
	signal_& clctpat__io,
	signal_& ph__io,
	signal_& th__io,
	signal_& vl__io,
	signal_& phzvl__io,
	signal_& me11a__io,
	signal_& clctpat_r__io,
	signal_& ph_hit__io,
	signal_& th_hit__io,
	signal_& sel__io,
	signal_& addr__io,
	signal_& r_in__io,
	signal_& r_out__io,
	signal_& we__io,
	signal_& clk__io,
	signal_& control_clk__io
	);
};
#endif

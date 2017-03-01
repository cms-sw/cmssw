// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#ifndef __best_tracks_h_file__
#define __best_tracks_h_file__
#include "vppc_sim_lib.h"

class best_tracks
{
 public:
	best_tracks(){built = false; glbl_gsr = true; defparam();}
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
	
		signal_ phi;
		signal_ theta;
		signal_ cpattern;
		signal_ delta_ph;
		signal_ delta_th;
		signal_ sign_ph;
		signal_ sign_th;
		signal_ rank;
		signal_ vi; // valid
		signal_ hi; // bx index
		signal_ ci; // chamber
		signal_ si; // segment
		signal_ clk;
		signal_ bt_phi;
		signal_ bt_theta;
		signal_ bt_cpattern;
		signal_ bt_delta_ph;
		signal_ bt_delta_th;
		signal_ bt_sign_ph;
		signal_ bt_sign_th;
		signal_ bt_rank;
		signal_ bt_vi; // valid
		signal_ bt_hi; // bx index
		signal_ bt_ci; // chamber
		signal_ bt_si; // segment
			// segment ids reformatted to chamber ids in sector
//[zone][pattern_num][station 0-4]
signal_storage cn_vi__storage;  signal_ cn_vi; // valid
		signal_storage cn_hi__storage;  signal_ cn_hi; // bx index
		signal_storage cn_ci__storage;  signal_ cn_ci; // chamber
		signal_storage cn_si__storage;  signal_ cn_si; // segment
		signal_storage larger__storage;  signal_ larger;
		signal_storage ri__storage;  signal_ ri;
		signal_storage rj__storage;  signal_ rj;
		signal_storage exists__storage;  signal_ exists;
		signal_storage kill1__storage;  signal_ kill1;
		signal_storage winner__storage;  signal_ winner;
		signal_storage gt__storage;  signal_ gt;
		signal_storage eq__storage;  signal_ eq;
		signal_storage cham__storage;  signal_ cham;
		signal_storage real_ch__storage;  signal_ real_ch;
		signal_storage real_st__storage;  signal_ real_st;
		signal_storage sum__storage;  signal_ sum;
		signal_storage sh_segs__storage;  signal_ sh_segs;
	
	
		unsigned z;
		unsigned n;
		unsigned s;
		unsigned bn;
		unsigned i;
		unsigned j;
	
	void init ();
	void operator()
	(
	signal_& phi__io,
	signal_& theta__io,
	signal_& cpattern__io,
	signal_& delta_ph__io,
	signal_& delta_th__io,
	signal_& sign_ph__io,
	signal_& sign_th__io,
	signal_& rank__io,
	signal_& vi__io,
	signal_& hi__io,
	signal_& ci__io,
	signal_& si__io,
	signal_& bt_phi__io,
	signal_& bt_theta__io,
	signal_& bt_cpattern__io,
	signal_& bt_delta_ph__io,
	signal_& bt_delta_th__io,
	signal_& bt_sign_ph__io,
	signal_& bt_sign_th__io,
	signal_& bt_rank__io,
	signal_& bt_vi__io,
	signal_& bt_hi__io,
	signal_& bt_ci__io,
	signal_& bt_si__io,
	signal_& clk__io
	);
};
#endif

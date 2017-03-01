// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#ifndef __sp_h_file__
#define __sp_h_file__
#include "vppc_sim_lib.h"
#include "prim_conv_sector.h"
#include "zones.h"
#include "extend_sector.h"
#include "ph_pattern_sector.h"
#include "sort_sector.h"
#include "coord_delay.h"
#include "match_ph_segments.h"
#include "deltas_sector.h"
#include "best_tracks.h"

class sp
{
 public:
	sp(){built = false; glbl_gsr = true; defparam();}
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
		signal_ pcs_cs;
		signal_ pps_cs;
		signal_ sel;
		signal_ addr;
		signal_ r_in; // input data for memory or register
		signal_ we; // write enable for memory or register
		signal_ clk;
		signal_ control_clk;
		signal_ r_out; // output data from memory or register
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
	
			// coordinates [station][chamber][segment]
signal_storage ph__storage;  signal_ ph;
		signal_storage th11__storage;  signal_ th11;
		signal_storage th__storage;  signal_ th;
		signal_storage vl__storage;  signal_ vl;
		signal_storage phzvl__storage;  signal_ phzvl;
		signal_storage me11a__storage;  signal_ me11a;
		signal_storage cpatr__storage;  signal_ cpatr;
			// numbers of best ranks [zone][num]
signal_storage ph_num__storage;  signal_ ph_num;
			// best ranks [zone][num]
signal_storage ph_q__storage;  signal_ ph_q;
		signal_storage ph_qr__storage;  signal_ ph_qr;
			// ph and th raw hits [station][chamber]
signal_storage ph_hito__storage;  signal_ ph_hito;
		signal_storage th_hito__storage;  signal_ th_hito;
			// ph zones [zone][station]
signal_storage ph_zone__storage;  signal_ ph_zone;
			// ph zones extended [zone][station]
signal_storage ph_ext__storage;  signal_ ph_ext;
			// hardcoded (at this time) inputs 
// drifttime and th_window have to become inputs eventually
signal_storage drifttime__storage;  signal_ drifttime;
		signal_storage th_window__storage;  signal_ th_window;
			// fold numbers for multiplexed pattern detectors
signal_storage ph_foldn__storage;  signal_ ph_foldn;
		signal_storage th_foldn__storage;  signal_ th_foldn;
			// ph quality codes output [zone][key_strip]
signal_storage ph_rank__storage;  signal_ ph_rank;
			// coordinate outputs delayed and with history [bx_history][station][chamber][segment]
// most recent in bx = 0
signal_storage phd__storage;  signal_ phd;
		signal_storage th11d__storage;  signal_ th11d;
		signal_storage thd__storage;  signal_ thd;
		signal_storage vld__storage;  signal_ vld;
		signal_storage me11ad__storage;  signal_ me11ad;
		signal_storage cpatd__storage;  signal_ cpatd;
			// find_segment outputs, in terms of segments match in zones [zone][pattern_num][station 0-3]
signal_storage patt_ph_vi__storage;  signal_ patt_ph_vi; // valid
		signal_storage patt_ph_hi__storage;  signal_ patt_ph_hi; // bx index
		signal_storage patt_ph_ci__storage;  signal_ patt_ph_ci; // chamber
		signal_storage patt_ph_si__storage;  signal_ patt_ph_si; // segment
		signal_storage ph_match__storage;  signal_ ph_match; // matching ph
		signal_storage th_match__storage;  signal_ th_match; // matching th, 2 segments 
		signal_storage th_match11__storage;  signal_ th_match11; // matching th for ME11 (station 0 only), 4 segments (due to th duplication)
		signal_storage cpat_match__storage;  signal_ cpat_match; // matching pattern
		signal_storage phi__storage;  signal_ phi;
		signal_storage theta__storage;  signal_ theta;
		signal_storage cpattern__storage;  signal_ cpattern;
			// ph and th deltas from best stations
// [zone][pattern_num], last index: [0] - best pair of stations, [1] - second best pair
signal_storage delta_ph__storage;  signal_ delta_ph;
		signal_storage delta_th__storage;  signal_ delta_th;
		signal_storage sign_ph__storage;  signal_ sign_ph;
		signal_storage sign_th__storage;  signal_ sign_th;
			// updated ranks [zone][pattern_num]
signal_storage rank__storage;  signal_ rank;
			//[zone][pattern_num][station 0-3]
signal_storage vir__storage;  signal_ vir; // valid
		signal_storage hir__storage;  signal_ hir; // bx index
		signal_storage cir__storage;  signal_ cir; // chamber
		signal_storage sir__storage;  signal_ sir; // segment
	
	
	void init ();
	void operator()
	(
	signal_& q__io,
	signal_& wg__io,
	signal_& hstr__io,
	signal_& cpat__io,
	signal_& pcs_cs__io,
	signal_& pps_cs__io,
	signal_& sel__io,
	signal_& addr__io,
	signal_& r_in__io,
	signal_& r_out__io,
	signal_& we__io,
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
	signal_& clk__io,
	signal_& control_clk__io
	);
	prim_conv_sector pcs;
	zones zns;
	extend_sector exts;
	ph_pattern_sector phps;
	sort_sector srts;
	coord_delay cdl;
	match_ph_segments mphseg;
	deltas_sector ds;
	best_tracks bt;
};
#endif


#ifndef __sp_wrap_h_file__
#define __sp_wrap_h_file__
#include "vppc_sim_lib.h"
#include "sp.h"

class sp_wrap
{
 public:
	void run
	(
		unsigned q__io[5][9][2],
		unsigned wg__io[5][9][2],
		unsigned hstr__io[5][9][2],
		unsigned cpat__io[5][9][2],

		unsigned bt_phi__io [3],
		unsigned bt_theta__io [3],
		unsigned bt_cpattern__io [3],
		// ph and th deltas from best stations
		// [best_track_num], last index: [0] - best pair of stations, [1] - second best pair
		unsigned bt_delta_ph__io [3][2],
		unsigned bt_delta_th__io [3][2], 
		unsigned bt_sign_ph__io[3][2],
		unsigned bt_sign_th__io[3][2],
		// ranks [best_track_num]
		unsigned bt_rank__io [3],
		// segment IDs
		// [best_track_num][station 0-3]
		unsigned bt_vi__io [3][5], // valid
		unsigned bt_hi__io [3][5], // bx index
		unsigned bt_ci__io [3][5], // chamber
		unsigned bt_si__io [3][5] // segment

	);
	sp_wrap(){built = false; glbl_gsr = true; defparam();}

	void defparam();
	void build();
	bool built;
	bool glbl_gsr;
		unsigned station;
		unsigned cscid;
		unsigned max_ev;
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
	unsigned ph_raw_w;
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
	
			// io
signal_storage qi__storage;  signal_ qi;
		signal_storage wgi__storage;  signal_ wgi;
		signal_storage hstri__storage;  signal_ hstri;
		signal_storage cpati__storage;  signal_ cpati;
			// precise parameters
/*
	 * wire [bw_phi-1:0] phi [2:0];
    wire [bw_eta-1:0] eta [2:0];
    wire [12:0] 	  pt [2:0];
    wire [2:0] 		  sign;
    wire [3:0] 		  modeMem [2:0];
    wire [4:0] 		  etaPT [2:0];
    wire [2:0] 		  FR;
	 */signal_storage csi__storage;  signal_ csi;
		signal_storage pps_csi__storage;  signal_ pps_csi;
		signal_storage seli__storage;  signal_ seli;
		signal_storage addri__storage;  signal_ addri;
		signal_storage r_ini__storage;  signal_ r_ini; // reg data for memory or register
		signal_storage wei__storage;  signal_ wei; // write enable for memory or register
		signal_storage clki__storage;  signal_ clki; // write clock
			// ph_init storage one bit wider, because ph_init is scaled at 0.1333 deg per bit
signal_storage ph_init__storage;  signal_ ph_init;
		signal_storage th_init__storage;  signal_ th_init;
		signal_storage ph_disp__storage;  signal_ ph_disp;
		signal_storage th_disp__storage;  signal_ th_disp;
			// event storage
signal_storage quality__storage;  signal_ quality;
		signal_storage wiregroup__storage;  signal_ wiregroup;
		signal_storage hstrip__storage;  signal_ hstrip;
		signal_storage clctpat__storage;  signal_ clctpat;
		signal_storage v0__storage;  signal_ v0;
		signal_storage v1__storage;  signal_ v1;
		signal_storage v2__storage;  signal_ v2;
		signal_storage v3__storage;  signal_ v3;
		signal_storage v4__storage;  signal_ v4;
		signal_storage v5__storage;  signal_ v5;
		signal_storage pr_cnt__storage;  signal_ pr_cnt;
		signal_storage _event__storage;  signal_ _event;
		signal_storage _bx_jitter__storage;  signal_ _bx_jitter;
		signal_storage _endcap__storage;  signal_ _endcap;
		signal_storage _sector__storage;  signal_ _sector;
		signal_storage _subsector__storage;  signal_ _subsector;
		signal_storage _station__storage;  signal_ _station;
		signal_storage _cscid__storage;  signal_ _cscid;
		signal_storage _bend__storage;  signal_ _bend;
		signal_storage _halfstrip__storage;  signal_ _halfstrip;
		signal_storage _valid__storage;  signal_ _valid;
		signal_storage _quality__storage;  signal_ _quality;
		signal_storage _pattern__storage;  signal_ _pattern;
		signal_storage _wiregroup__storage;  signal_ _wiregroup;
		signal_storage line__storage;  signal_ line;
		signal_storage ev__storage;  signal_ ev;
		signal_storage good_ev__storage;  signal_ good_ev;
		signal_storage tphi__storage;  signal_ tphi;
		signal_storage a__storage;  signal_ a;
		signal_storage b__storage;  signal_ b;
		signal_storage d__storage;  signal_ d;
		signal_storage pts__storage;  signal_ pts;
	
		signal_storage r_outo__storage;  signal_ r_outo; // output data from memory or register
			// ph quality codes output [zone][key_strip]
signal_storage ph_ranko__storage;  signal_ ph_ranko;
		signal_storage ph__storage;  signal_ ph;
		signal_storage th11__storage;  signal_ th11;
		signal_storage th__storage;  signal_ th;
		signal_storage vl__storage;  signal_ vl;
		signal_storage phzvl__storage;  signal_ phzvl;
		signal_storage me11a__storage;  signal_ me11a;
		signal_storage ph_zone__storage;  signal_ ph_zone;
		signal_storage patt_vi__storage;  signal_ patt_vi; // valid
		signal_storage patt_hi__storage;  signal_ patt_hi; // bx index
		signal_storage patt_ci__storage;  signal_ patt_ci; // chamber
		signal_storage patt_si__storage;  signal_ patt_si; // segment
			// numbers of best ranks [zone][num]
signal_storage ph_num__storage;  signal_ ph_num;
			// best ranks [zone][num]
signal_storage ph_q__storage;  signal_ ph_q;
		signal_storage ph_match__storage;  signal_ ph_match; // matching ph
		signal_storage th_match__storage;  signal_ th_match; // matching th, 2 segments 
		signal_storage th_match11__storage;  signal_ th_match11; // matching th for ME11 (station 0 only), 4 segments (due to th duplication)
			// precise phi and theta of best tracks
// [best_track_num]
signal_storage bt_phi__storage;  signal_ bt_phi;
		signal_storage bt_theta__storage;  signal_ bt_theta;
		signal_storage bt_cpattern__storage;  signal_ bt_cpattern;
			// ph and th deltas from best stations
// [best_track_num], last index: [0] - best pair of stations, [1] - second best pair
signal_storage bt_delta_ph__storage;  signal_ bt_delta_ph;
		signal_storage bt_delta_th__storage;  signal_ bt_delta_th;
		signal_storage bt_sign_ph__storage;  signal_ bt_sign_ph;
		signal_storage bt_sign_th__storage;  signal_ bt_sign_th;
			// ranks [best_track_num]
signal_storage bt_rank__storage;  signal_ bt_rank;
			// segment IDs
// [best_track_num][station 0-3]
signal_storage bt_vi__storage;  signal_ bt_vi; // valid
		signal_storage bt_hi__storage;  signal_ bt_hi; // bx index
		signal_storage bt_ci__storage;  signal_ bt_ci; // chamber
		signal_storage bt_si__storage;  signal_ bt_si; // segment
		signal_storage begin_time__storage;  signal_ begin_time;
		signal_storage end_time__storage;  signal_ end_time;
		signal_storage elapsed_time__storage;  signal_ elapsed_time;
	
		signal_storage iadr__storage;  signal_ iadr;
		signal_storage s__storage;  signal_ s;
		signal_storage i__storage;  signal_ i;
		signal_storage ii__storage;  signal_ ii;
		signal_storage pi__storage;  signal_ pi;
		signal_storage j__storage;  signal_ j;
		signal_storage sn__storage;  signal_ sn;
		signal_storage ist__storage;  signal_ ist;
		signal_storage icid__storage;  signal_ icid;
		signal_storage ipr__storage;  signal_ ipr;
		signal_storage code__storage;  signal_ code;
		signal_storage iev__storage;  signal_ iev;
		signal_storage im__storage;  signal_ im;
		signal_storage iz__storage;  signal_ iz;
		signal_storage ir__storage;  signal_ ir;
		signal_storage in__storage;  signal_ in;
		signal_storage best_tracks__storage;  signal_ best_tracks;
		signal_storage stat__storage;  signal_ stat;
		signal_storage good_ev_cnt__storage;  signal_ good_ev_cnt;
		signal_storage found_tr__storage;  signal_ found_tr;
		signal_storage found_cand__storage;  signal_ found_cand;
		signal_storage st__storage;  signal_ st;
		signal_storage st_cnt__storage;  signal_ st_cnt; // station count
		signal_storage iseg__storage;  signal_ iseg;
		signal_storage zi__storage;  signal_ zi;
		signal_storage si__storage;  signal_ si;
		signal_storage ip__storage;  signal_ ip;
		signal_storage ibx__storage;  signal_ ibx;
		signal_storage ich__storage;  signal_ ich;
		signal_storage isg__storage;  signal_ isg;
		unsigned k;
	
	void init ();
	void operator()
	(
	);
	sp uut;


};
#endif

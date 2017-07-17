// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#ifndef __coord_delay_h_file__
#define __coord_delay_h_file__
#include "vppc_sim_lib.h"

class coord_delay
{
 public:
	coord_delay(){built = false; glbl_gsr = true; defparam();}
	void defparam();
	void build();
	bool built;
	bool glbl_gsr;
		unsigned station;
		unsigned cscid;
		// pulse length
	unsigned pulse_l;
		unsigned latency;
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
		// width of combined memory data
	unsigned mem_ph_bw;
		unsigned mem_th_bw;
		unsigned mem_th11_bw;
		unsigned mem_vl_bw;
		unsigned mem_me11a_bw;
		unsigned mem_cpat_bw;
	
		signal_ phi;
		signal_ th11i;
		signal_ thi;
		signal_ vli;
		signal_ me11ai;
		signal_ cpati;
		signal_ clk;
		signal_ pho;
		signal_ th11o;
		signal_ tho;
		signal_ vlo;
		signal_ me11ao;
		signal_ cpato;
			// combined signals with all values merged
signal_storage mem_ph_in__storage;  signal_ mem_ph_in;
		signal_storage mem_ph_out__storage;  signal_ mem_ph_out;
		signal_storage mem_th_in__storage;  signal_ mem_th_in;
		signal_storage mem_th_out__storage;  signal_ mem_th_out;
		signal_storage mem_th11_in__storage;  signal_ mem_th11_in;
		signal_storage mem_th11_out__storage;  signal_ mem_th11_out;
		signal_storage mem_vl_in__storage;  signal_ mem_vl_in;
		signal_storage mem_vl_out__storage;  signal_ mem_vl_out;
		signal_storage mem_me11a_in__storage;  signal_ mem_me11a_in;
		signal_storage mem_me11a_out__storage;  signal_ mem_me11a_out;
		signal_storage mem_cpat_in__storage;  signal_ mem_cpat_in;
		signal_storage mem_cpat_out__storage;  signal_ mem_cpat_out;
			// BRAM
signal_storage mem_ph__storage;  signal_ mem_ph;
		signal_storage mem_th__storage;  signal_ mem_th;
		signal_storage mem_th11__storage;  signal_ mem_th11;
		signal_storage mem_vl__storage;  signal_ mem_vl;
		signal_storage mem_me11a__storage;  signal_ mem_me11a;
		signal_storage mem_cpat__storage;  signal_ mem_cpat;
			// read address
signal_storage ra__storage;  signal_ ra;
			// write address
signal_storage wa__storage;  signal_ wa;
	
	
		// merge inputs
	unsigned i;
		unsigned j;
		unsigned k;
		unsigned d;
	
	void init ();
	void operator()
	(
	signal_& phi__io,
	signal_& th11i__io,
	signal_& thi__io,
	signal_& vli__io,
	signal_& me11ai__io,
	signal_& cpati__io,
	signal_& pho__io,
	signal_& th11o__io,
	signal_& tho__io,
	signal_& vlo__io,
	signal_& me11ao__io,
	signal_& cpato__io,
	signal_& clk__io
	);
};
#endif

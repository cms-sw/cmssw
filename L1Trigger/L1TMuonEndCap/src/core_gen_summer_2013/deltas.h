// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#ifndef __deltas_h_file__
#define __deltas_h_file__
#include "vppc_sim_lib.h"
#include "best_delta.h"
#include "best_delta.h"
#include "best_delta.h"
#include "best_delta.h"
#include "best_delta.h"
#include "best_delta.h"

class deltas
{
 public:
	deltas(){built = false; glbl_gsr = true; defparam();}
	void defparam();
	void build();
	bool built;
	bool glbl_gsr;
		unsigned station;
		unsigned cscid;
		unsigned me11;
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
		unsigned seg1; // number of segments station 1
		unsigned bw_nm1;
		unsigned bw_nm2;
	
		signal_ vi; // valid
		signal_ hi; // bx index
		signal_ ci; // chamber
		signal_ si; // segment
		signal_ ph_match; // matching ph
		signal_ th_match;
		signal_ th_match11;
		signal_ cpat_match; // matching pattern
		signal_ ph_q; // pattern rank, carries straigtness and ph station information
		signal_ th_window; // max th diff
		signal_ clk;
		signal_ phi;
		signal_ theta;
		signal_ cpattern;
		signal_ delta_ph;
		signal_ delta_th;
		signal_ sign_ph;
		signal_ sign_th;
		signal_ rank; // output rank, to be used for sorting
		signal_ vir; // valid
		signal_ hir; // bx index
		signal_ cir; // chamber
		signal_ sir; // segment
		signal_storage vstat__storage;  signal_ vstat; // valid stations based on th coordinates
		signal_storage thA__storage;  signal_ thA;
		signal_storage thB__storage;  signal_ thB;
		signal_storage dth__storage;  signal_ dth;
		signal_storage dvl__storage;  signal_ dvl;
		signal_storage dth12__storage;  signal_ dth12;
		signal_storage dth13__storage;  signal_ dth13;
		signal_storage dth14__storage;  signal_ dth14;
		signal_storage dth23__storage;  signal_ dth23;
		signal_storage dth24__storage;  signal_ dth24;
		signal_storage dth34__storage;  signal_ dth34;
		signal_storage dvl12__storage;  signal_ dvl12;
		signal_storage dvl13__storage;  signal_ dvl13;
		signal_storage dvl14__storage;  signal_ dvl14;
		signal_storage dvl23__storage;  signal_ dvl23;
		signal_storage dvl24__storage;  signal_ dvl24;
		signal_storage dvl34__storage;  signal_ dvl34;
		signal_storage sth12__storage;  signal_ sth12;
		signal_storage sth13__storage;  signal_ sth13;
		signal_storage sth14__storage;  signal_ sth14;
		signal_storage sth23__storage;  signal_ sth23;
		signal_storage sth24__storage;  signal_ sth24;
		signal_storage sth34__storage;  signal_ sth34;
		signal_storage bnm12__storage;  signal_ bnm12;
		signal_storage bnm13__storage;  signal_ bnm13;
		signal_storage bnm14__storage;  signal_ bnm14;
		signal_storage bnm23__storage;  signal_ bnm23;
		signal_storage bnm24__storage;  signal_ bnm24;
		signal_storage bnm34__storage;  signal_ bnm34;
		signal_storage phA__storage;  signal_ phA;
		signal_storage phB__storage;  signal_ phB;
		signal_storage dph__storage;  signal_ dph;
		signal_storage sph__storage;  signal_ sph;
		signal_storage dph12__storage;  signal_ dph12;
		signal_storage dph13__storage;  signal_ dph13;
		signal_storage dph14__storage;  signal_ dph14;
		signal_storage dph23__storage;  signal_ dph23;
		signal_storage dph24__storage;  signal_ dph24;
		signal_storage dph34__storage;  signal_ dph34;
		signal_storage sph12__storage;  signal_ sph12;
		signal_storage sph13__storage;  signal_ sph13;
		signal_storage sph14__storage;  signal_ sph14;
		signal_storage sph23__storage;  signal_ sph23;
		signal_storage sph24__storage;  signal_ sph24;
		signal_storage sph34__storage;  signal_ sph34;
		signal_storage bsg12__storage;  signal_ bsg12;
		signal_storage bsg13__storage;  signal_ bsg13;
		signal_storage bsg14__storage;  signal_ bsg14;
		signal_storage bsg23__storage;  signal_ bsg23;
		signal_storage bsg24__storage;  signal_ bsg24;
		signal_storage bsg34__storage;  signal_ bsg34;
		signal_storage bvl12__storage;  signal_ bvl12;
		signal_storage bvl13__storage;  signal_ bvl13;
		signal_storage bvl14__storage;  signal_ bvl14;
		signal_storage bvl23__storage;  signal_ bvl23;
		signal_storage bvl24__storage;  signal_ bvl24;
		signal_storage bvl34__storage;  signal_ bvl34;
		signal_storage sth__storage;  signal_ sth;
		signal_storage vmask1__storage;  signal_ vmask1;
		signal_storage vmask2__storage;  signal_ vmask2;
		signal_storage vmask3__storage;  signal_ vmask3;
	
		signal_storage bth12__storage;  signal_ bth12;
		signal_storage bth13__storage;  signal_ bth13;
		signal_storage bth14__storage;  signal_ bth14;
		signal_storage bth23__storage;  signal_ bth23;
		signal_storage bth24__storage;  signal_ bth24;
		signal_storage bth34__storage;  signal_ bth34;
		signal_storage dvalid__storage;  signal_ dvalid;
	
		signal_storage i1__storage;  signal_ i1;
		signal_storage i2__storage;  signal_ i2;
		unsigned j;
		unsigned k;
	
	void init ();
	void operator()
	(
	signal_& vi__io,
	signal_& hi__io,
	signal_& ci__io,
	signal_& si__io,
	signal_& ph_match__io,
	signal_& th_match__io,
	signal_& th_match11__io,
	signal_& cpat_match__io,
	signal_& ph_q__io,
	signal_& th_window__io,
	signal_& phi__io,
	signal_& theta__io,
	signal_& cpattern__io,
	signal_& delta_ph__io,
	signal_& delta_th__io,
	signal_& sign_ph__io,
	signal_& sign_th__io,
	signal_& rank__io,
	signal_& vir__io,
	signal_& hir__io,
	signal_& cir__io,
	signal_& sir__io,
	signal_& clk__io
	);
	best_delta bd12;
	best_delta bd13;
	best_delta bd14;
	best_delta bd23;
	best_delta bd24;
	best_delta bd34;
};
#endif

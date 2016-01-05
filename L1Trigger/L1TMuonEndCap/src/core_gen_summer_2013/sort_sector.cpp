// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#include "sort_sector.h"

extern size_t __glob_alwaysn__;

void sort_sector::operator()
(
	signal_& ph_rank__io,
	signal_& ph_num__io,
	signal_& ph_q__io,
	signal_& clk__io
)
{
	if (!built)
	{
		seg_ch = 2;
		bw_ph = 8;
		bw_th = 7;
		bw_fph = 12;
		bw_fth = 8;
		bw_wg = 7;
		bw_ds = 7;
		bw_hs = 8;
		pat_w_st3 = 3;
		pat_w_st1 = pat_w_st3 + 1;
		full_pat_w_st3 = (1 << (pat_w_st3+1)) - 1;
		full_pat_w_st1 = (1 << (pat_w_st1+1)) - 1;
		padding_w_st1 = full_pat_w_st1 / 2;
		padding_w_st3 = full_pat_w_st3 / 2;
		red_pat_w_st3 = pat_w_st3 * 2 + 1;
		red_pat_w_st1 = pat_w_st1 * 2 + 1;
		fold = 4;
		th_ch11 = seg_ch*seg_ch;
		bw_q = 4;
		bw_addr = 7;
		ph_raw_w = (1 << pat_w_st3) * 15 + 2;
		th_raw_w = (1 << bw_th);
		max_drift = 3;
		bw_phi = 12;
		bw_eta = 7;
		ph_hit_w = 40+4;
		ph_hit_w20 = ph_hit_w;
		ph_hit_w10 = 20+4;
		th_hit_w = 56 + 8;
		endcap = 1;
		n_strips = (station <= 1 && cscid <= 2) ? 64 :
						 (station <= 1 && cscid >= 6) ? 64 : 80;
		n_wg = (station <= 1 && cscid <= 3) ? 48  :
					 (station <= 1 && cscid >= 6) ? 32  :
					 (station == 2 && cscid <= 3) ? 112 :
					 (station >= 3 && cscid <= 3) ? 96  : 64;
		th_coverage = (station <= 1 && cscid <= 2) ? 45  :
						 (station <= 1 && cscid >= 6) ? 27  :
						 (station <= 1 && cscid >= 3) ? 39  :
						 (station == 2 && cscid <= 2) ? 43  :
						 (station == 2 && cscid >= 3) ? 56  :
						 (station == 3 && cscid <= 2) ? 34  :
						 (station == 3 && cscid >= 3) ? 52  :
						 (station == 4 && cscid <= 2) ? 28  :
						 (station == 4 && cscid >= 3) ? 50  : 0;
		ph_coverage = (station <= 1 && cscid >= 6) ? 15 : //30 :
						   (station >= 2 && cscid <= 2) ? 40 : 20;
		th_ch = (station <= 1 && cscid <= 2) ? (seg_ch*seg_ch) : seg_ch;
		ph_reverse = (endcap == 1 && station >= 3) ? 1 : 
			   			   (endcap == 2 && station <  3) ? 1 : 0;
		th_mem_sz = (1 << bw_addr);
		th_corr_mem_sz = (1 << bw_addr);
		mult_bw = bw_fph + 11;
		ph_zone_bnd1 = (station <= 1 && cscid <= 2) ? 41 :
							(station == 2 && cscid <= 2) ? 41 :
							(station == 2 && cscid >  2) ? 87 :
							(station == 3 && cscid >  2) ? 49 :
							(station == 4 && cscid >  2) ? 49 : 127;
		ph_zone_bnd2 = (station == 3 && cscid >  2) ? 87 : 127;
		zone_overlap = 2;
		bwr = 6;
		bpow = 6;
		cnr = (1 << bpow);
		cnrex = ph_raw_w;
		build();
		// ph pattern ranks [zone][key_ph]
		ph_rank.attach(ph_rank__io);
		clk.attach(clk__io);
		// numbers of best ranks [zone][num]
		ph_num.attach(ph_num__io);
		// best ranks [zone][num]
		ph_q.attach(ph_q__io);
	}

	
		{
			for (i = 0; i < 4; i = i+1)
			{
					gb.ph_zone[i].zb3
	(
		ph_rank[i],
		ph_q[i],
		ph_num[i],
		clk
	);
			}
		}
	
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void sort_sector::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void sort_sector::build()
{
	built = true;
	ph_rank.add_dim(3, 0);
	ph_rank.add_dim(ph_raw_w-1, 0);
	ph_rank.bw(bwr-1, 0);
	ph_rank.build();
	clk.bw(0, 0);
	ph_num.add_dim(3, 0);
	ph_num.add_dim(2, 0);
	ph_num.bw(bpow, 0);
	ph_num.build();
	ph_q.add_dim(3, 0);
	ph_q.add_dim(2, 0);
	ph_q.bw(bwr-1, 0);
	ph_q.build();

}

// vppc: this function checks for changes in any signal on each simulation iteration
void sort_sector::init ()
{
	if (!built)
	{
			}
	else
	{
																																															gb.init();
	}
}
void sort_sector::gb__class::init()
{
	for (map <ull, ph_zone__class>::iterator mit = ph_zone.begin(); mit != ph_zone.end(); mit++)
		mit->second.init();
}
void sort_sector::gb__class::ph_zone__class::init()
{
	zb3.init();
}

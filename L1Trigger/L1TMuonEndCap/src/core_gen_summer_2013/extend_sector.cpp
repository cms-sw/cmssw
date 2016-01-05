// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#include "extend_sector.h"

extern size_t __glob_alwaysn__;

void extend_sector::operator()
(
	signal_& ph_zone__io,
	signal_& ph_ext__io,
	signal_& drifttime__io,
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
		// ph zones [zone][station]
		ph_zone.attach(ph_zone__io);
		drifttime.attach(drifttime__io);
		clk.attach(clk__io);
		// ph extended zones [zone][station]
		ph_ext.attach(ph_ext__io);
	}

	
		{
			for (i = 0; i < 4; i = i+1)
			{
				for (j = 1; j < 5; j = j+1)
				{
					 genblk.ph_zone_blk[i].station[j].ext.bit_w = ph_raw_w;
						genblk.ph_zone_blk[i].station[j].ext
	(
		ph_zone[i][j],
		ph_ext[i][j],
		drifttime,
		clk
	);
				}
			}
		}
	
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void extend_sector::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void extend_sector::build()
{
	built = true;
	ph_zone.add_dim(3, 0);
	ph_zone.add_dim(4, 1);
	ph_zone.bw(ph_raw_w-1, 0);
	ph_zone.build();
	drifttime.bw(2, 0);
	clk.bw(0, 0);
	ph_ext.add_dim(3, 0);
	ph_ext.add_dim(4, 1);
	ph_ext.bw(ph_raw_w-1, 0);
	ph_ext.build();

}

// vppc: this function checks for changes in any signal on each simulation iteration
void extend_sector::init ()
{
	if (!built)
	{
			}
	else
	{
																																															genblk.init();
	}
}
void extend_sector::genblk__class::init()
{
	for (map <ull, ph_zone_blk__class>::iterator mit = ph_zone_blk.begin(); mit != ph_zone_blk.end(); mit++)
		mit->second.init();
}
void extend_sector::genblk__class::ph_zone_blk__class::init()
{
	for (map <ull, station__class>::iterator mit = station.begin(); mit != station.end(); mit++)
		mit->second.init();
}
void extend_sector::genblk__class::ph_zone_blk__class::station__class::init()
{
	ext.init();
}

// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#include "zone_best3.h"

extern size_t __glob_alwaysn__;

void zone_best3::operator()
(
	signal_& rank_ex__io,
	signal_& winner__io,
	signal_& wini__io,
	signal_& clk_nx__io
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
		rank_ex.attach(rank_ex__io);
		clk_nx.attach(clk_nx__io);
		winner.attach(winner__io);
		wini.attach(wini__io);
	}

	
	{
		// implement pair OR between neighboring input ranks.
		// neighbors cannot contain valid ranks because of ghost cancellation
		for (i = 0; i < cnrex/2; i = i+1)
		{
			 rankr[0][i] = rank_ex[i*2] | rank_ex[i*2+1];
		}

		// unused inputs
		for (i = cnrex/2; i < cnr; i = i+1)
			 rankr[0][i] = 0;
		
		for (i = 0; i < 3; i = i+1) //  3 sorters in chain
		{
				gb.zone_best_loop[i].zb
	(
		rankr[i],
		winnerw[i],
		winiw[i],
		rankr[i+1],
		clk_nx
	);
		}
	}
	

	beginalways();

	// winner delay line
	if (posedge (clk_nx))
	{


		//  winner outputs from delay lines for each winner
		// delay lines are of different length to compensate for sorter stages
		winner[0] = winnerd[1][0]; wini[0](bpow,1) = winid[1][0];
		winner[1] = winnerd[0][1]; wini[1](bpow,1) = winid[0][1];
		winner[2] = winnerw[2];    wini[2](bpow,1) = winiw[2];

		wini[0][0] = 0;
		wini[1][0] = 0;
		wini[2][0] = 0;
		
		// find LSBs of wini outputs using valid bits
		for (j = 0; j < 3; j = j+1)
			wini[j][0] = !valid[2][wini[j]];
		
		// winner delay line
		winnerd[1] = winnerd[0];   winid[1] = winid[0];
		winnerd[0] = winnerw;      winid[0] = winiw;

		// delay line for valid bits
		valid[2] = valid[1];
		valid[1] = valid[0];
		for (j = 0; j < cnrex; j = j+1)
			valid[0][j] = rank_ex[j] != 0;
	}
	endalways();
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void zone_best3::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void zone_best3::build()
{
	built = true;
	rank_ex.add_dim(cnrex-1, 0);
	rank_ex.bw(bwr-1, 0);
	rank_ex.build();
	clk_nx.bw(0, 0);
	winner.add_dim(2, 0);
	winner.bw(bwr-1, 0);
	winner.build();
	wini.add_dim(2, 0);
	wini.bw(bpow, 0);
	wini.build();
	winnerd__storage.add_dim(1, 0);
	winnerd__storage.add_dim(2, 0);
	winnerd__storage.bw(bwr-1, 0);
	winnerd__storage.build();
	winnerd.add_dim(1, 0);
	winnerd.add_dim(2, 0);
	winnerd.bw(bwr-1, 0);
	winnerd.build();
	winnerd.set_storage (&winnerd__storage);
	winid__storage.add_dim(1, 0);
	winid__storage.add_dim(2, 0);
	winid__storage.bw(bpow-1, 0);
	winid__storage.build();
	winid.add_dim(1, 0);
	winid.add_dim(2, 0);
	winid.bw(bpow-1, 0);
	winid.build();
	winid.set_storage (&winid__storage);
	winnerw__storage.add_dim(2, 0);
	winnerw__storage.bw(bwr-1, 0);
	winnerw__storage.build();
	winnerw.add_dim(2, 0);
	winnerw.bw(bwr-1, 0);
	winnerw.build();
	winnerw.set_storage (&winnerw__storage);
	winiw__storage.add_dim(2, 0);
	winiw__storage.bw(bpow-1, 0);
	winiw__storage.build();
	winiw.add_dim(2, 0);
	winiw.bw(bpow-1, 0);
	winiw.build();
	winiw.set_storage (&winiw__storage);
	valid__storage.add_dim(2, 0);
	valid__storage.bw(cnrex-1, 0);
	valid__storage.build();
	valid.add_dim(2, 0);
	valid.bw(cnrex-1, 0);
	valid.build();
	valid.set_storage (&valid__storage);
	rankr__storage.add_dim(3, 0);
	rankr__storage.add_dim(cnr-1, 0);
	rankr__storage.bw(bwr-1, 0);
	rankr__storage.build();
	rankr.add_dim(3, 0);
	rankr.add_dim(cnr-1, 0);
	rankr.bw(bwr-1, 0);
	rankr.build();
	rankr.set_storage (&rankr__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void zone_best3::init ()
{
	if (!built)
	{
			}
	else
	{
		winnerd__storage.init();
		winid__storage.init();
		winnerw__storage.init();
		winiw__storage.init();
		valid__storage.init();
		rankr__storage.init();
																																															gb.init();
	}
}
void zone_best3::gb__class::init()
{
	for (map <ull, zone_best_loop__class>::iterator mit = zone_best_loop.begin(); mit != zone_best_loop.end(); mit++)
		mit->second.init();
}
void zone_best3::gb__class::zone_best_loop__class::init()
{
	zb.init();
}

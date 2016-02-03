// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#include "zone_best.h"

extern size_t __glob_alwaysn__;

void zone_best::operator()
(
	signal_& rank__io,
	signal_& winner__io,
	signal_& wini__io,
	signal_& rankr__io,
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
		rank.attach(rank__io);
		clk_nx.attach(clk_nx__io);
		winner.attach(winner__io);
		wini.attach(wini__io);
		rankr.attach(rankr__io);
	}


	beginalways();
	if (posedge (clk_nx))
	{

		// put inputs into initial stage
		cmp[0] = rank;

		//  rank indexes first stage
		for (j = 0; j < cnr; j = j+1)
			ranki[0][j] = j;
		
		for (i = 0; i < bpow; i = i+1) // comparator stage loop
		{
			
			ncomp = (1 << (bpow - i - 1));
			for (j = 0; j < ncomp; j = j+1) // comparator loop
			{
				// compare two ranks, advance winner and its index to next stage
				if (cmp[i][j*2] > cmp[i][j*2+1]) 
				{
					cmp  [i+1][j] = cmp  [i][j*2];
					ranki[i+1][j] = ranki[i][j*2];
				}
				else
				{
					cmp  [i+1][j] = cmp  [i][j*2+1];
					ranki[i+1][j] = ranki[i][j*2+1];
				}
			}
		}

		// pick up winner and index from top of the tree
		winner = cmp[bpow][0];
		wini = ranki[bpow][0];

		// put ranks to output for next stage, except the winner
		rankr = rank;
		rankr[wini] = 0;

	}
	endalways();
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void zone_best::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void zone_best::build()
{
	built = true;
	rank.add_dim(cnr-1, 0);
	rank.bw(bwr-1, 0);
	rank.build();
	clk_nx.bw(0, 0);
	winner.bw(bwr-1, 0);
	wini.bw(bpow-1, 0);
	rankr.add_dim(cnr-1, 0);
	rankr.bw(bwr-1, 0);
	rankr.build();
	cmp__storage.add_dim(bpow, 0);
	cmp__storage.add_dim(cnr-1, 0);
	cmp__storage.bw(bwr-1, 0);
	cmp__storage.build();
	cmp.add_dim(bpow, 0);
	cmp.add_dim(cnr-1, 0);
	cmp.bw(bwr-1, 0);
	cmp.build();
	cmp.set_storage (&cmp__storage);
	ranki__storage.add_dim(bpow, 0);
	ranki__storage.add_dim(cnr-1, 0);
	ranki__storage.bw(bpow-1, 0);
	ranki__storage.build();
	ranki.add_dim(bpow, 0);
	ranki.add_dim(cnr-1, 0);
	ranki.bw(bpow-1, 0);
	ranki.build();
	ranki.set_storage (&ranki__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void zone_best::init ()
{
	if (!built)
	{
			}
	else
	{
		cmp__storage.init();
		ranki__storage.init();
																																															}
}

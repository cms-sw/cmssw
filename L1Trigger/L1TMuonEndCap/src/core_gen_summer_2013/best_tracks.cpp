// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#include "best_tracks.h"

extern size_t __glob_alwaysn__;

void best_tracks::operator()
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
		// precise phi and theta of candidates
// [zone][pattern_num]
		phi.attach(phi__io);
		theta.attach(theta__io);
		cpattern.attach(cpattern__io);
		// ph and th deltas from best stations
// [zone][pattern_num], last index: [0] - best pair of stations, [1] - second best pair
		delta_ph.attach(delta_ph__io);
		delta_th.attach(delta_th__io);
		sign_ph.attach(sign_ph__io);
		sign_th.attach(sign_th__io);
		// updated ranks [zone][pattern_num]
		rank.attach(rank__io);
		//[zone][pattern_num][station 0-3]
		vi.attach(vi__io);
		hi.attach(hi__io);
		ci.attach(ci__io);
		si.attach(si__io);
		clk.attach(clk__io);
		// precise phi and theta of best tracks
// [best_track_num]
		bt_phi.attach(bt_phi__io);
		bt_theta.attach(bt_theta__io);
		bt_cpattern.attach(bt_cpattern__io);
		// ph and th deltas from best stations
// [best_track_num], last index: [0] - best pair of stations, [1] - second best pair
		bt_delta_ph.attach(bt_delta_ph__io);
		bt_delta_th.attach(bt_delta_th__io);
		bt_sign_ph.attach(bt_sign_ph__io);
		bt_sign_th.attach(bt_sign_th__io);
		// ranks [best_track_num]
		bt_rank.attach(bt_rank__io);
		//[best_track_num][station 0-3]
		bt_vi.attach(bt_vi__io);
		bt_hi.attach(bt_hi__io);
		bt_ci.attach(bt_ci__io);
		bt_si.attach(bt_si__io);
	}


	beginalways();
	
	if (posedge (clk))
	{

		// zero segment numbers
		for (z = 0; z < 4; z = z+1) // zone loop
		{
			for (n = 0; n < 3; n = n+1) // pattern number
			{
				for (s = 0; s < 5; s = s+1) // station
				{
					cn_vi[z][n][s] = 0;
					cn_hi[z][n][s] = 0;
					cn_si[z][n][s] = 0;
					cn_ci[z][n][s] = 0;
				}
			}
		}
		// input segment numbers are in terms of chambers in zone
		// convert them back into chamber ids in sector
		for (z = 0; z < 4; z = z+1) // zone loop
		{
			for (n = 0; n < 3; n = n+1) // pattern number
			{
				for (s = 0; s < 4; s = s+1) // station
				{
					// calculate real station and chamber numbers
					cham = ci[z][n][s];
					if (s == 0)
					{
						real_st = (cham < 3) ? 0 : 1;
						real_ch = cham % 3; // will this synthesize OK?
						if (z == 2) real_ch = real_ch + const_(3, 3UL);
						if (z == 3) real_ch = real_ch + const_(3, 6UL);
					}
					else
					if (s == 1)
					{
						real_st = s + 1;
						real_ch = cham;
						if (z > 1) real_ch = real_ch + const_(3, 3UL);
					}
					else
					{
						real_st = s + 1;
						real_ch = cham;
						if (z > 0) real_ch = real_ch + const_(3, 3UL);
					}
					
					cn_vi[z][n][real_st] = vi[z][n][s];
					cn_hi[z][n][real_st] = hi[z][n][s];
					cn_si[z][n][real_st] = si[z][n][s];
					cn_ci[z][n][real_st] = real_ch;
				}
			}
		}
	
		// zero outputs initially
		for (n = 0; n < 3; n = n+1)
		{
			winner[n] = 0;
			bt_rank [n] = 0;
			bt_phi[n] = 0;
			bt_theta[n] = 0;
			bt_cpattern[n] = 0;
			for (s = 0; s < 2; s = s+1) // delta loop
			{
				bt_delta_ph [n][s] = 0;
				bt_sign_ph  [n][s] = 0; 
				bt_delta_th [n][s] = 0; 
				bt_sign_th  [n][s] = 0; 
			}
			
			for (s = 0; s < 5; s = s+1) // station loop
			{
				bt_vi[n][s] = 0;
				bt_hi[n][s] = 0;
				bt_si[n][s] = 0;
				bt_ci[n][s] = 0;
			}
		}
	
		// simultaneously compare each rank with each
		for (i = 0; i < 12; i = i+1)
		{
			larger[i] = 0;
			larger[i][i] = 1; // result of comparison with itself
			ri = rank[i%4][i/4]; // first index loops zone, second loops candidate. Zone loops faster, so we give equal priority to zones
			for (j = 0; j < 12; j = j+1)
			{
				// ilgj bits show which rank is larger
				// the comparison scheme below avoids problems
				// when there are two | more tracks with the same rank
				rj = rank[j%4][j/4];
				gt = ri > rj;
				eq = ri == rj;
				if ((i < j && (gt || eq)) || (i > j && gt)) larger[i][j] = 1; 				
			}
			// "larger" array shows the result of comparison for each rank

			// track exists if quality != 0 
			exists[i] = (ri != 0);
		}

		// ghost cancellation, only in the current BX so far
		kill1 = 0;
		
		for (i = 0; i < 12; i = i+1) // candidate loop
		{
			for (j = i+1; j < 12; j = j+1) // comparison candidate loop
			{
				sh_segs = 0;
				// count shared segments
				for (s = 0; s < 5; s = s+1) // station loop
				{
					if (cn_vi[i%4][i/4][s] && cn_vi[j%4][j/4][s] && // both segments valid
						cn_ci[i%4][i/4][s] == cn_ci[j%4][j/4][s] && // from same chamber
						cn_si[i%4][i/4][s] == cn_si[j%4][j/4][s]) // same segment
						sh_segs = sh_segs + const_(3, 0x1UL); // increment shared segment counter
				}

				if (sh_segs > 0) // a single shared segment means const_s(it, ) ghost
				{
					// kill candidate that has lower rank
					if (larger[i][j]) kill1[j] = 1;
					else kill1[i] = 1;
				}
			}
		}

		// remove ghosts according to kill mask
		exists = exists & (~kill1);
		
		for (i = 0; i < 12; i = i+1)
		{
			if  (exists[i]) larger[i] = larger[i] | (~exists); // if this track exists make it larger than all non-existing tracks
			else  larger[i] = 0; // else make it smaller than anything

			// count zeros in the comparison results. The best track will have none, the next will have one, the third will have two.
			// skip the bits corresponding to the comparison of the track with itself
			sum = 0;
			for (j = 0; j < 12; j = j+1) if (larger[i][j] == 0) sum = sum + 1; 
			
			if (sum < 3) winner[sum][i] = 1; //  positional winner codes
		}
	
		// multiplex best tracks to outputs according to winner signals
		for (n = 0; n < 3; n = n+1) // output loop
		{
			for (i = 0; i < 12; i = i+1) // winner bit loop
			{
				if (winner[n][i])
				{
					bt_rank [n] = bt_rank [n] | rank [i%4][i/4];
					bt_phi[n] = bt_phi[n] | phi[i%4][i/4];
					bt_theta[n] = bt_theta[n] | theta[i%4][i/4];
					bt_cpattern[n] = bt_cpattern[n] | cpattern[i%4][i/4];
					
					for (s = 0; s < 2; s = s+1) // delta loop
					{
						bt_delta_ph [n][s] = bt_delta_ph [n][s] | delta_ph [i%4][i/4][s];
						bt_sign_ph  [n][s] = bt_sign_ph  [n][s] | sign_ph  [i%4][i/4][s];
						bt_delta_th [n][s] = bt_delta_th [n][s] | delta_th [i%4][i/4][s];
						bt_sign_th  [n][s] = bt_sign_th  [n][s] | sign_th  [i%4][i/4][s];
					}
					
					for (s = 0; s < 5; s = s+1) // station loop
					{
						bt_vi[n][s] = bt_vi[n][s] | cn_vi[i%4][i/4][s];
						bt_hi[n][s] = bt_hi[n][s] | cn_hi[i%4][i/4][s];
						bt_ci[n][s] = bt_ci[n][s] | cn_ci[i%4][i/4][s];
						bt_si[n][s] = bt_si[n][s] | cn_si[i%4][i/4][s];
					}              
				}
			}
		}
	}
	endalways();
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void best_tracks::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void best_tracks::build()
{
	built = true;
	phi.add_dim(3, 0);
	phi.add_dim(2, 0);
	phi.bw(bw_fph-1, 0);
	phi.build();
	theta.add_dim(3, 0);
	theta.add_dim(2, 0);
	theta.bw(bw_th-1, 0);
	theta.build();
	cpattern.add_dim(3, 0);
	cpattern.add_dim(2, 0);
	cpattern.bw(3, 0);
	cpattern.build();
	delta_ph.add_dim(3, 0);
	delta_ph.add_dim(2, 0);
	delta_ph.add_dim(1, 0);
	delta_ph.bw(bw_fph-1, 0);
	delta_ph.build();
	delta_th.add_dim(3, 0);
	delta_th.add_dim(2, 0);
	delta_th.add_dim(1, 0);
	delta_th.bw(bw_th-1, 0);
	delta_th.build();
	sign_ph.add_dim(3, 0);
	sign_ph.add_dim(2, 0);
	sign_ph.bw(1, 0);
	sign_ph.build();
	sign_th.add_dim(3, 0);
	sign_th.add_dim(2, 0);
	sign_th.bw(1, 0);
	sign_th.build();
	rank.add_dim(3, 0);
	rank.add_dim(2, 0);
	rank.bw(bwr, 0);
	rank.build();
	vi.add_dim(3, 0);
	vi.add_dim(2, 0);
	vi.add_dim(3, 0);
	vi.bw(seg_ch-1, 0);
	vi.build();
	hi.add_dim(3, 0);
	hi.add_dim(2, 0);
	hi.add_dim(3, 0);
	hi.bw(1, 0);
	hi.build();
	ci.add_dim(3, 0);
	ci.add_dim(2, 0);
	ci.add_dim(3, 0);
	ci.bw(2, 0);
	ci.build();
	si.add_dim(3, 0);
	si.add_dim(2, 0);
	si.bw(3, 0);
	si.build();
	clk.bw(0, 0);
	bt_phi.add_dim(2, 0);
	bt_phi.bw(bw_fph-1, 0);
	bt_phi.build();
	bt_theta.add_dim(2, 0);
	bt_theta.bw(bw_th-1, 0);
	bt_theta.build();
	bt_cpattern.add_dim(2, 0);
	bt_cpattern.bw(3, 0);
	bt_cpattern.build();
	bt_delta_ph.add_dim(2, 0);
	bt_delta_ph.add_dim(1, 0);
	bt_delta_ph.bw(bw_fph-1, 0);
	bt_delta_ph.build();
	bt_delta_th.add_dim(2, 0);
	bt_delta_th.add_dim(1, 0);
	bt_delta_th.bw(bw_th-1, 0);
	bt_delta_th.build();
	bt_sign_ph.add_dim(2, 0);
	bt_sign_ph.bw(1, 0);
	bt_sign_ph.build();
	bt_sign_th.add_dim(2, 0);
	bt_sign_th.bw(1, 0);
	bt_sign_th.build();
	bt_rank.add_dim(2, 0);
	bt_rank.bw(bwr, 0);
	bt_rank.build();
	bt_vi.add_dim(2, 0);
	bt_vi.add_dim(4, 0);
	bt_vi.bw(seg_ch-1, 0);
	bt_vi.build();
	bt_hi.add_dim(2, 0);
	bt_hi.add_dim(4, 0);
	bt_hi.bw(1, 0);
	bt_hi.build();
	bt_ci.add_dim(2, 0);
	bt_ci.add_dim(4, 0);
	bt_ci.bw(3, 0);
	bt_ci.build();
	bt_si.add_dim(2, 0);
	bt_si.bw(4, 0);
	bt_si.build();
	cn_vi__storage.add_dim(3, 0);
	cn_vi__storage.add_dim(2, 0);
	cn_vi__storage.add_dim(4, 0);
	cn_vi__storage.bw(seg_ch-1, 0);
	cn_vi__storage.build();
	cn_vi.add_dim(3, 0);
	cn_vi.add_dim(2, 0);
	cn_vi.add_dim(4, 0);
	cn_vi.bw(seg_ch-1, 0);
	cn_vi.build();
	cn_vi.set_storage (&cn_vi__storage);
	cn_hi__storage.add_dim(3, 0);
	cn_hi__storage.add_dim(2, 0);
	cn_hi__storage.add_dim(4, 0);
	cn_hi__storage.bw(1, 0);
	cn_hi__storage.build();
	cn_hi.add_dim(3, 0);
	cn_hi.add_dim(2, 0);
	cn_hi.add_dim(4, 0);
	cn_hi.bw(1, 0);
	cn_hi.build();
	cn_hi.set_storage (&cn_hi__storage);
	cn_ci__storage.add_dim(3, 0);
	cn_ci__storage.add_dim(2, 0);
	cn_ci__storage.add_dim(4, 0);
	cn_ci__storage.bw(3, 0);
	cn_ci__storage.build();
	cn_ci.add_dim(3, 0);
	cn_ci.add_dim(2, 0);
	cn_ci.add_dim(4, 0);
	cn_ci.bw(3, 0);
	cn_ci.build();
	cn_ci.set_storage (&cn_ci__storage);
	cn_si__storage.add_dim(3, 0);
	cn_si__storage.add_dim(2, 0);
	cn_si__storage.bw(4, 0);
	cn_si__storage.build();
	cn_si.add_dim(3, 0);
	cn_si.add_dim(2, 0);
	cn_si.bw(4, 0);
	cn_si.build();
	cn_si.set_storage (&cn_si__storage);
	larger__storage.add_dim(11, 0);
	larger__storage.bw(11, 0);
	larger__storage.build();
	larger.add_dim(11, 0);
	larger.bw(11, 0);
	larger.build();
	larger.set_storage (&larger__storage);
	ri__storage.bw(6, 0);
	ri.bw(6, 0);
	ri.set_storage (&ri__storage);
	rj__storage.bw(6, 0);
	rj.bw(6, 0);
	rj.set_storage (&rj__storage);
	exists__storage.bw(11, 0);
	exists.bw(11, 0);
	exists.set_storage (&exists__storage);
	kill1__storage.bw(11, 0);
	kill1.bw(11, 0);
	kill1.set_storage (&kill1__storage);
	winner__storage.add_dim(2, 0);
	winner__storage.bw(11, 0);
	winner__storage.build();
	winner.add_dim(2, 0);
	winner.bw(11, 0);
	winner.build();
	winner.set_storage (&winner__storage);
	gt__storage.bw(0, 0);
	gt.bw(0, 0);
	gt.set_storage (&gt__storage);
	eq__storage.bw(0, 0);
	eq.bw(0, 0);
	eq.set_storage (&eq__storage);
	cham__storage.bw(3, 0);
	cham.bw(3, 0);
	cham.set_storage (&cham__storage);
	real_ch__storage.bw(3, 0);
	real_ch.bw(3, 0);
	real_ch.set_storage (&real_ch__storage);
	real_st__storage.bw(2, 0);
	real_st.bw(2, 0);
	real_st.set_storage (&real_st__storage);
	sum__storage.bw(3, 0);
	sum.bw(3, 0);
	sum.set_storage (&sum__storage);
	sh_segs__storage.bw(2, 0);
	sh_segs.bw(2, 0);
	sh_segs.set_storage (&sh_segs__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void best_tracks::init ()
{
	if (!built)
	{
			}
	else
	{
		cn_vi__storage.init();
		cn_hi__storage.init();
		cn_ci__storage.init();
		cn_si__storage.init();
		larger__storage.init();
		ri__storage.init();
		rj__storage.init();
		exists__storage.init();
		kill1__storage.init();
		winner__storage.init();
		gt__storage.init();
		eq__storage.init();
		cham__storage.init();
		real_ch__storage.init();
		real_st__storage.init();
		sum__storage.init();
		sh_segs__storage.init();
																																															}
}

// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#include "deltas_sector.h"

extern size_t __glob_alwaysn__;

void deltas_sector::operator()
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
		seg1 = me11 ? th_ch11 : seg_ch;
		build();
		//[zone][pattern_num][station 0-3]
		vi.attach(vi__io);
		hi.attach(hi__io);
		ci.attach(ci__io);
		si.attach(si__io);
		ph_match.attach(ph_match__io);
		th_match.attach(th_match__io);
		th_match11.attach(th_match11__io);
		cpat_match.attach(cpat_match__io);
		// best ranks [zone][num]
		ph_q.attach(ph_q__io);
		th_window.attach(th_window__io);
		clk.attach(clk__io);
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
		vir.attach(vir__io);
		hir.attach(hir__io);
		cir.attach(cir__io);
		sir.attach(sir__io);
	}

	
	{
		for (i = 0; i < 2; i = i+1) // zone loop
		{
			for (j = 0; j < 3; j = j+1) // pattern loop
			{
				 gb.zl11[i].pl[j].da.me11 = 1;
					gb.zl11[i].pl[j].da
	(
		vi[i][j],
		hi[i][j],
		ci[i][j],
		si[i][j],
		ph_match[i][j],
		th_match[i][j],
		th_match11[i][j],
		cpat_match[i][j],
		ph_q[i][j],
		th_window,
		phi[i][j],
		theta[i][j],
		cpattern[i][j],
		delta_ph[i][j],
		delta_th[i][j],
		sign_ph[i][j],
		sign_th[i][j],
		rank[i][j],
		vir[i][j],
		hir[i][j],
		cir[i][j],
		sir[i][j],
		clk
	);
			}
		}
		
		for (i = 2; i < 4; i = i+1) // zone loop
		{
			for (j = 0; j < 3; j = j+1) // pattern loop
			{
				 gb.zl[i].pl[j].da.me11 = 0;
					gb.zl[i].pl[j].da
	(
		vi[i][j],
		hi[i][j],
		ci[i][j],
		si[i][j],
		ph_match[i][j],
		th_match[i][j],
		dummy[i-2][j],
		cpat_match[i][j],
		ph_q[i][j],
		th_window,
		phi[i][j],
		theta[i][j],
		cpattern[i][j],
		delta_ph[i][j],
		delta_th[i][j],
		sign_ph[i][j],
		sign_th[i][j],
		rank[i][j],
		vir[i][j],
		hir[i][j],
		cir[i][j],
		sir[i][j],
		clk
	);
			}
		} // block: zl
	}
	
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void deltas_sector::defparam()
{
	station = 1;
	cscid = 1;
	me11 = 0;
}

// vppc: this function allocates memory for internal signals
void deltas_sector::build()
{
	built = true;
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
	ph_match.add_dim(3, 0);
	ph_match.add_dim(2, 0);
	ph_match.add_dim(3, 0);
	ph_match.bw(bw_fph-1, 0);
	ph_match.build();
	th_match.add_dim(3, 0);
	th_match.add_dim(2, 0);
	th_match.add_dim(3, 0);
	th_match.add_dim(seg_ch-1, 0);
	th_match.bw(bw_th-1, 0);
	th_match.build();
	th_match11.add_dim(1, 0);
	th_match11.add_dim(2, 0);
	th_match11.add_dim(th_ch11-1, 0);
	th_match11.bw(bw_th-1, 0);
	th_match11.build();
	cpat_match.add_dim(3, 0);
	cpat_match.add_dim(2, 0);
	cpat_match.add_dim(3, 0);
	cpat_match.bw(3, 0);
	cpat_match.build();
	ph_q.add_dim(3, 0);
	ph_q.add_dim(2, 0);
	ph_q.bw(bwr-1, 0);
	ph_q.build();
	th_window.bw(bw_th-1, 0);
	clk.bw(0, 0);
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
	vir.add_dim(3, 0);
	vir.add_dim(2, 0);
	vir.add_dim(3, 0);
	vir.bw(seg_ch-1, 0);
	vir.build();
	hir.add_dim(3, 0);
	hir.add_dim(2, 0);
	hir.add_dim(3, 0);
	hir.bw(1, 0);
	hir.build();
	cir.add_dim(3, 0);
	cir.add_dim(2, 0);
	cir.add_dim(3, 0);
	cir.bw(2, 0);
	cir.build();
	sir.add_dim(3, 0);
	sir.add_dim(2, 0);
	sir.bw(3, 0);
	sir.build();
	dummy__storage.add_dim(1, 0);
	dummy__storage.add_dim(2, 0);
	dummy__storage.add_dim(th_ch11-1, 0);
	dummy__storage.bw(bw_th-1, 0);
	dummy__storage.build();
	dummy.add_dim(1, 0);
	dummy.add_dim(2, 0);
	dummy.add_dim(th_ch11-1, 0);
	dummy.bw(bw_th-1, 0);
	dummy.build();
	dummy.set_storage (&dummy__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void deltas_sector::init ()
{
	if (!built)
	{
				}
	else
	{
		dummy__storage.init();
																																																gb.init();
	}
}
void deltas_sector::gb__class::init()
{
	for (map <ull, zl11__class>::iterator mit = zl11.begin(); mit != zl11.end(); mit++)
		mit->second.init();
	for (map <ull, zl__class>::iterator mit = zl.begin(); mit != zl.end(); mit++)
		mit->second.init();
}
void deltas_sector::gb__class::zl11__class::init()
{
	for (map <ull, pl__class>::iterator mit = pl.begin(); mit != pl.end(); mit++)
		mit->second.init();
}
void deltas_sector::gb__class::zl11__class::pl__class::init()
{
	da.init();
}
void deltas_sector::gb__class::zl__class::init()
{
	for (map <ull, pl__class>::iterator mit = pl.begin(); mit != pl.end(); mit++)
		mit->second.init();
}
void deltas_sector::gb__class::zl__class::pl__class::init()
{
	da.init();
}

// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#include "sp.h"

extern size_t __glob_alwaysn__;

void sp::operator()
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
		q.attach(q__io);
		wg.attach(wg__io);
		hstr.attach(hstr__io);
		cpat.attach(cpat__io);
		pcs_cs.attach(pcs_cs__io);
		pps_cs.attach(pps_cs__io);
		sel.attach(sel__io);
		addr.attach(addr__io);
		r_in.attach(r_in__io);
		we.attach(we__io);
		// clock
		clk.attach(clk__io);
		control_clk.attach(control_clk__io);
		r_out.attach(r_out__io);
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
		// segment IDs
// [best_track_num][station 0-3]
		bt_vi.attach(bt_vi__io);
		bt_hi.attach(bt_hi__io);
		bt_ci.attach(bt_ci__io);
		bt_si.attach(bt_si__io);
	}
	drifttime = 2;
	th_window = 4;
	ph_foldn = 0;
	th_foldn = 0;
	pcs
	(
		q,
		wg,
		hstr,
		cpat,
		ph,
		th11,
		th,
		vl,
		phzvl,
		me11a,
		cpatr,
		ph_hito,
		th_hito,
		pcs_cs,
		sel,
		addr,
		r_in,
		r_out,
		we,
		clk,
		control_clk
	);
	zns
	(
		phzvl,
		ph_hito,
		ph_zone,
		clk
	);
	exts
	(
		ph_zone,
		ph_ext,
		drifttime,
		clk
	);
	phps
	(
		ph_ext,
		drifttime,
		ph_foldn,
		ph_rank,
		clk
	);
	srts
	(
		ph_rank,
		ph_num,
		ph_q,
		clk
	);
	cdl
	(
		ph,
		th11,
		th,
		vl,
		me11a,
		cpatr,
		phd,
		th11d,
		thd,
		vld,
		me11ad,
		cpatd,
		clk
	);
	mphseg
	(
		ph_num,
		ph_q,
		phd,
		vld,
		th11d,
		thd,
		cpatd,
		patt_ph_vi,
		patt_ph_hi,
		patt_ph_ci,
		patt_ph_si,
		ph_match,
		th_match,
		th_match11,
		cpat_match,
		ph_qr,
		clk
	);
	ds
	(
		patt_ph_vi,
		patt_ph_hi,
		patt_ph_ci,
		patt_ph_si,
		ph_match,
		th_match,
		th_match11,
		cpat_match,
		ph_qr,
		th_window,
		phi,
		theta,
		cpattern,
		delta_ph,
		delta_th,
		sign_ph,
		sign_th,
		rank,
		vir,
		hir,
		cir,
		sir,
		clk
	);
	bt
	(
		phi,
		theta,
		cpattern,
		delta_ph,
		delta_th,
		sign_ph,
		sign_th,
		rank,
		vir,
		hir,
		cir,
		sir,
		bt_phi,
		bt_theta,
		bt_cpattern,
		bt_delta_ph,
		bt_delta_th,
		bt_sign_ph,
		bt_sign_th,
		bt_rank,
		bt_vi,
		bt_hi,
		bt_ci,
		bt_si,
		clk
	);

}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void sp::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void sp::build()
{
	built = true;
	q.add_dim(4, 0);
	q.add_dim(8, 0);
	q.add_dim(seg_ch-1, 0);
	q.bw(3, 0);
	q.build();
	wg.add_dim(4, 0);
	wg.add_dim(8, 0);
	wg.add_dim(seg_ch-1, 0);
	wg.bw(bw_wg-1, 0);
	wg.build();
	hstr.add_dim(4, 0);
	hstr.add_dim(8, 0);
	hstr.add_dim(seg_ch-1, 0);
	hstr.bw(bw_hs-1, 0);
	hstr.build();
	cpat.add_dim(4, 0);
	cpat.add_dim(8, 0);
	cpat.add_dim(seg_ch-1, 0);
	cpat.bw(3, 0);
	cpat.build();
	pcs_cs.add_dim(4, 0);
	pcs_cs.bw(8, 0);
	pcs_cs.build();
	pps_cs.add_dim(2, 0);
	pps_cs.bw(4, 0);
	pps_cs.build();
	sel.bw(1, 0);
	addr.bw(bw_addr-1, 0);
	r_in.bw(11, 0);
	we.bw(0, 0);
	clk.bw(0, 0);
	control_clk.bw(0, 0);
	r_out.bw(11, 0);
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
	ph__storage.add_dim(4, 0);
	ph__storage.add_dim(8, 0);
	ph__storage.add_dim(seg_ch-1, 0);
	ph__storage.bw(bw_fph-1, 0);
	ph__storage.build();
	ph.add_dim(4, 0);
	ph.add_dim(8, 0);
	ph.add_dim(seg_ch-1, 0);
	ph.bw(bw_fph-1, 0);
	ph.build();
	ph.set_storage (&ph__storage);
	th11__storage.add_dim(1, 0);
	th11__storage.add_dim(2, 0);
	th11__storage.add_dim(th_ch11-1, 0);
	th11__storage.bw(bw_th-1, 0);
	th11__storage.build();
	th11.add_dim(1, 0);
	th11.add_dim(2, 0);
	th11.add_dim(th_ch11-1, 0);
	th11.bw(bw_th-1, 0);
	th11.build();
	th11.set_storage (&th11__storage);
	th__storage.add_dim(4, 0);
	th__storage.add_dim(8, 0);
	th__storage.add_dim(seg_ch-1, 0);
	th__storage.bw(bw_th-1, 0);
	th__storage.build();
	th.add_dim(4, 0);
	th.add_dim(8, 0);
	th.add_dim(seg_ch-1, 0);
	th.bw(bw_th-1, 0);
	th.build();
	th.set_storage (&th__storage);
	vl__storage.add_dim(4, 0);
	vl__storage.add_dim(8, 0);
	vl__storage.bw(seg_ch-1, 0);
	vl__storage.build();
	vl.add_dim(4, 0);
	vl.add_dim(8, 0);
	vl.bw(seg_ch-1, 0);
	vl.build();
	vl.set_storage (&vl__storage);
	phzvl__storage.add_dim(4, 0);
	phzvl__storage.add_dim(8, 0);
	phzvl__storage.bw(2, 0);
	phzvl__storage.build();
	phzvl.add_dim(4, 0);
	phzvl.add_dim(8, 0);
	phzvl.bw(2, 0);
	phzvl.build();
	phzvl.set_storage (&phzvl__storage);
	me11a__storage.add_dim(1, 0);
	me11a__storage.add_dim(2, 0);
	me11a__storage.bw(seg_ch-1, 0);
	me11a__storage.build();
	me11a.add_dim(1, 0);
	me11a.add_dim(2, 0);
	me11a.bw(seg_ch-1, 0);
	me11a.build();
	me11a.set_storage (&me11a__storage);
	cpatr__storage.add_dim(4, 0);
	cpatr__storage.add_dim(8, 0);
	cpatr__storage.add_dim(seg_ch-1, 0);
	cpatr__storage.bw(3, 0);
	cpatr__storage.build();
	cpatr.add_dim(4, 0);
	cpatr.add_dim(8, 0);
	cpatr.add_dim(seg_ch-1, 0);
	cpatr.bw(3, 0);
	cpatr.build();
	cpatr.set_storage (&cpatr__storage);
	ph_num__storage.add_dim(3, 0);
	ph_num__storage.add_dim(2, 0);
	ph_num__storage.bw(bpow, 0);
	ph_num__storage.build();
	ph_num.add_dim(3, 0);
	ph_num.add_dim(2, 0);
	ph_num.bw(bpow, 0);
	ph_num.build();
	ph_num.set_storage (&ph_num__storage);
	ph_q__storage.add_dim(3, 0);
	ph_q__storage.add_dim(2, 0);
	ph_q__storage.bw(bwr-1, 0);
	ph_q__storage.build();
	ph_q.add_dim(3, 0);
	ph_q.add_dim(2, 0);
	ph_q.bw(bwr-1, 0);
	ph_q.build();
	ph_q.set_storage (&ph_q__storage);
	ph_qr__storage.add_dim(3, 0);
	ph_qr__storage.add_dim(2, 0);
	ph_qr__storage.bw(bwr-1, 0);
	ph_qr__storage.build();
	ph_qr.add_dim(3, 0);
	ph_qr.add_dim(2, 0);
	ph_qr.bw(bwr-1, 0);
	ph_qr.build();
	ph_qr.set_storage (&ph_qr__storage);
	ph_hito__storage.add_dim(4, 0);
	ph_hito__storage.add_dim(8, 0);
	ph_hito__storage.bw(ph_hit_w-1, 0);
	ph_hito__storage.build();
	ph_hito.add_dim(4, 0);
	ph_hito.add_dim(8, 0);
	ph_hito.bw(ph_hit_w-1, 0);
	ph_hito.build();
	ph_hito.set_storage (&ph_hito__storage);
	th_hito__storage.add_dim(4, 0);
	th_hito__storage.add_dim(8, 0);
	th_hito__storage.bw(th_hit_w-1, 0);
	th_hito__storage.build();
	th_hito.add_dim(4, 0);
	th_hito.add_dim(8, 0);
	th_hito.bw(th_hit_w-1, 0);
	th_hito.build();
	th_hito.set_storage (&th_hito__storage);
	ph_zone__storage.add_dim(3, 0);
	ph_zone__storage.add_dim(4, 1);
	ph_zone__storage.bw(ph_raw_w-1, 0);
	ph_zone__storage.build();
	ph_zone.add_dim(3, 0);
	ph_zone.add_dim(4, 1);
	ph_zone.bw(ph_raw_w-1, 0);
	ph_zone.build();
	ph_zone.set_storage (&ph_zone__storage);
	ph_ext__storage.add_dim(3, 0);
	ph_ext__storage.add_dim(4, 1);
	ph_ext__storage.bw(ph_raw_w-1, 0);
	ph_ext__storage.build();
	ph_ext.add_dim(3, 0);
	ph_ext.add_dim(4, 1);
	ph_ext.bw(ph_raw_w-1, 0);
	ph_ext.build();
	ph_ext.set_storage (&ph_ext__storage);
	drifttime__storage.bw(2, 0);
	drifttime.bw(2, 0);
	drifttime.set_storage (&drifttime__storage);
	th_window__storage.bw(bw_th-1, 0);
	th_window.bw(bw_th-1, 0);
	th_window.set_storage (&th_window__storage);
	ph_foldn__storage.bw(2, 0);
	ph_foldn.bw(2, 0);
	ph_foldn.set_storage (&ph_foldn__storage);
	th_foldn__storage.bw(2, 0);
	th_foldn.bw(2, 0);
	th_foldn.set_storage (&th_foldn__storage);
	ph_rank__storage.add_dim(3, 0);
	ph_rank__storage.add_dim(ph_raw_w-1, 0);
	ph_rank__storage.bw(5, 0);
	ph_rank__storage.build();
	ph_rank.add_dim(3, 0);
	ph_rank.add_dim(ph_raw_w-1, 0);
	ph_rank.bw(5, 0);
	ph_rank.build();
	ph_rank.set_storage (&ph_rank__storage);
	phd__storage.add_dim(max_drift-1, 0);
	phd__storage.add_dim(4, 0);
	phd__storage.add_dim(8, 0);
	phd__storage.add_dim(seg_ch-1, 0);
	phd__storage.bw(bw_fph-1, 0);
	phd__storage.build();
	phd.add_dim(max_drift-1, 0);
	phd.add_dim(4, 0);
	phd.add_dim(8, 0);
	phd.add_dim(seg_ch-1, 0);
	phd.bw(bw_fph-1, 0);
	phd.build();
	phd.set_storage (&phd__storage);
	th11d__storage.add_dim(max_drift-1, 0);
	th11d__storage.add_dim(1, 0);
	th11d__storage.add_dim(2, 0);
	th11d__storage.add_dim(th_ch11-1, 0);
	th11d__storage.bw(bw_th-1, 0);
	th11d__storage.build();
	th11d.add_dim(max_drift-1, 0);
	th11d.add_dim(1, 0);
	th11d.add_dim(2, 0);
	th11d.add_dim(th_ch11-1, 0);
	th11d.bw(bw_th-1, 0);
	th11d.build();
	th11d.set_storage (&th11d__storage);
	thd__storage.add_dim(max_drift-1, 0);
	thd__storage.add_dim(4, 0);
	thd__storage.add_dim(8, 0);
	thd__storage.add_dim(seg_ch-1, 0);
	thd__storage.bw(bw_th-1, 0);
	thd__storage.build();
	thd.add_dim(max_drift-1, 0);
	thd.add_dim(4, 0);
	thd.add_dim(8, 0);
	thd.add_dim(seg_ch-1, 0);
	thd.bw(bw_th-1, 0);
	thd.build();
	thd.set_storage (&thd__storage);
	vld__storage.add_dim(max_drift-1, 0);
	vld__storage.add_dim(4, 0);
	vld__storage.add_dim(8, 0);
	vld__storage.bw(seg_ch-1, 0);
	vld__storage.build();
	vld.add_dim(max_drift-1, 0);
	vld.add_dim(4, 0);
	vld.add_dim(8, 0);
	vld.bw(seg_ch-1, 0);
	vld.build();
	vld.set_storage (&vld__storage);
	me11ad__storage.add_dim(max_drift-1, 0);
	me11ad__storage.add_dim(1, 0);
	me11ad__storage.add_dim(2, 0);
	me11ad__storage.bw(seg_ch-1, 0);
	me11ad__storage.build();
	me11ad.add_dim(max_drift-1, 0);
	me11ad.add_dim(1, 0);
	me11ad.add_dim(2, 0);
	me11ad.bw(seg_ch-1, 0);
	me11ad.build();
	me11ad.set_storage (&me11ad__storage);
	cpatd__storage.add_dim(max_drift-1, 0);
	cpatd__storage.add_dim(4, 0);
	cpatd__storage.add_dim(8, 0);
	cpatd__storage.add_dim(seg_ch-1, 0);
	cpatd__storage.bw(3, 0);
	cpatd__storage.build();
	cpatd.add_dim(max_drift-1, 0);
	cpatd.add_dim(4, 0);
	cpatd.add_dim(8, 0);
	cpatd.add_dim(seg_ch-1, 0);
	cpatd.bw(3, 0);
	cpatd.build();
	cpatd.set_storage (&cpatd__storage);
	patt_ph_vi__storage.add_dim(3, 0);
	patt_ph_vi__storage.add_dim(2, 0);
	patt_ph_vi__storage.add_dim(3, 0);
	patt_ph_vi__storage.bw(seg_ch-1, 0);
	patt_ph_vi__storage.build();
	patt_ph_vi.add_dim(3, 0);
	patt_ph_vi.add_dim(2, 0);
	patt_ph_vi.add_dim(3, 0);
	patt_ph_vi.bw(seg_ch-1, 0);
	patt_ph_vi.build();
	patt_ph_vi.set_storage (&patt_ph_vi__storage);
	patt_ph_hi__storage.add_dim(3, 0);
	patt_ph_hi__storage.add_dim(2, 0);
	patt_ph_hi__storage.add_dim(3, 0);
	patt_ph_hi__storage.bw(1, 0);
	patt_ph_hi__storage.build();
	patt_ph_hi.add_dim(3, 0);
	patt_ph_hi.add_dim(2, 0);
	patt_ph_hi.add_dim(3, 0);
	patt_ph_hi.bw(1, 0);
	patt_ph_hi.build();
	patt_ph_hi.set_storage (&patt_ph_hi__storage);
	patt_ph_ci__storage.add_dim(3, 0);
	patt_ph_ci__storage.add_dim(2, 0);
	patt_ph_ci__storage.add_dim(3, 0);
	patt_ph_ci__storage.bw(2, 0);
	patt_ph_ci__storage.build();
	patt_ph_ci.add_dim(3, 0);
	patt_ph_ci.add_dim(2, 0);
	patt_ph_ci.add_dim(3, 0);
	patt_ph_ci.bw(2, 0);
	patt_ph_ci.build();
	patt_ph_ci.set_storage (&patt_ph_ci__storage);
	patt_ph_si__storage.add_dim(3, 0);
	patt_ph_si__storage.add_dim(2, 0);
	patt_ph_si__storage.bw(3, 0);
	patt_ph_si__storage.build();
	patt_ph_si.add_dim(3, 0);
	patt_ph_si.add_dim(2, 0);
	patt_ph_si.bw(3, 0);
	patt_ph_si.build();
	patt_ph_si.set_storage (&patt_ph_si__storage);
	ph_match__storage.add_dim(3, 0);
	ph_match__storage.add_dim(2, 0);
	ph_match__storage.add_dim(3, 0);
	ph_match__storage.bw(bw_fph-1, 0);
	ph_match__storage.build();
	ph_match.add_dim(3, 0);
	ph_match.add_dim(2, 0);
	ph_match.add_dim(3, 0);
	ph_match.bw(bw_fph-1, 0);
	ph_match.build();
	ph_match.set_storage (&ph_match__storage);
	th_match__storage.add_dim(3, 0);
	th_match__storage.add_dim(2, 0);
	th_match__storage.add_dim(3, 0);
	th_match__storage.add_dim(seg_ch-1, 0);
	th_match__storage.bw(bw_th-1, 0);
	th_match__storage.build();
	th_match.add_dim(3, 0);
	th_match.add_dim(2, 0);
	th_match.add_dim(3, 0);
	th_match.add_dim(seg_ch-1, 0);
	th_match.bw(bw_th-1, 0);
	th_match.build();
	th_match.set_storage (&th_match__storage);
	th_match11__storage.add_dim(1, 0);
	th_match11__storage.add_dim(2, 0);
	th_match11__storage.add_dim(th_ch11-1, 0);
	th_match11__storage.bw(bw_th-1, 0);
	th_match11__storage.build();
	th_match11.add_dim(1, 0);
	th_match11.add_dim(2, 0);
	th_match11.add_dim(th_ch11-1, 0);
	th_match11.bw(bw_th-1, 0);
	th_match11.build();
	th_match11.set_storage (&th_match11__storage);
	cpat_match__storage.add_dim(3, 0);
	cpat_match__storage.add_dim(2, 0);
	cpat_match__storage.add_dim(3, 0);
	cpat_match__storage.bw(3, 0);
	cpat_match__storage.build();
	cpat_match.add_dim(3, 0);
	cpat_match.add_dim(2, 0);
	cpat_match.add_dim(3, 0);
	cpat_match.bw(3, 0);
	cpat_match.build();
	cpat_match.set_storage (&cpat_match__storage);
	phi__storage.add_dim(3, 0);
	phi__storage.add_dim(2, 0);
	phi__storage.bw(bw_fph-1, 0);
	phi__storage.build();
	phi.add_dim(3, 0);
	phi.add_dim(2, 0);
	phi.bw(bw_fph-1, 0);
	phi.build();
	phi.set_storage (&phi__storage);
	theta__storage.add_dim(3, 0);
	theta__storage.add_dim(2, 0);
	theta__storage.bw(bw_th-1, 0);
	theta__storage.build();
	theta.add_dim(3, 0);
	theta.add_dim(2, 0);
	theta.bw(bw_th-1, 0);
	theta.build();
	theta.set_storage (&theta__storage);
	cpattern__storage.add_dim(3, 0);
	cpattern__storage.add_dim(2, 0);
	cpattern__storage.bw(3, 0);
	cpattern__storage.build();
	cpattern.add_dim(3, 0);
	cpattern.add_dim(2, 0);
	cpattern.bw(3, 0);
	cpattern.build();
	cpattern.set_storage (&cpattern__storage);
	delta_ph__storage.add_dim(3, 0);
	delta_ph__storage.add_dim(2, 0);
	delta_ph__storage.add_dim(1, 0);
	delta_ph__storage.bw(bw_fph-1, 0);
	delta_ph__storage.build();
	delta_ph.add_dim(3, 0);
	delta_ph.add_dim(2, 0);
	delta_ph.add_dim(1, 0);
	delta_ph.bw(bw_fph-1, 0);
	delta_ph.build();
	delta_ph.set_storage (&delta_ph__storage);
	delta_th__storage.add_dim(3, 0);
	delta_th__storage.add_dim(2, 0);
	delta_th__storage.add_dim(1, 0);
	delta_th__storage.bw(bw_th-1, 0);
	delta_th__storage.build();
	delta_th.add_dim(3, 0);
	delta_th.add_dim(2, 0);
	delta_th.add_dim(1, 0);
	delta_th.bw(bw_th-1, 0);
	delta_th.build();
	delta_th.set_storage (&delta_th__storage);
	sign_ph__storage.add_dim(3, 0);
	sign_ph__storage.add_dim(2, 0);
	sign_ph__storage.bw(1, 0);
	sign_ph__storage.build();
	sign_ph.add_dim(3, 0);
	sign_ph.add_dim(2, 0);
	sign_ph.bw(1, 0);
	sign_ph.build();
	sign_ph.set_storage (&sign_ph__storage);
	sign_th__storage.add_dim(3, 0);
	sign_th__storage.add_dim(2, 0);
	sign_th__storage.bw(1, 0);
	sign_th__storage.build();
	sign_th.add_dim(3, 0);
	sign_th.add_dim(2, 0);
	sign_th.bw(1, 0);
	sign_th.build();
	sign_th.set_storage (&sign_th__storage);
	rank__storage.add_dim(3, 0);
	rank__storage.add_dim(2, 0);
	rank__storage.bw(bwr, 0);
	rank__storage.build();
	rank.add_dim(3, 0);
	rank.add_dim(2, 0);
	rank.bw(bwr, 0);
	rank.build();
	rank.set_storage (&rank__storage);
	vir__storage.add_dim(3, 0);
	vir__storage.add_dim(2, 0);
	vir__storage.add_dim(3, 0);
	vir__storage.bw(seg_ch-1, 0);
	vir__storage.build();
	vir.add_dim(3, 0);
	vir.add_dim(2, 0);
	vir.add_dim(3, 0);
	vir.bw(seg_ch-1, 0);
	vir.build();
	vir.set_storage (&vir__storage);
	hir__storage.add_dim(3, 0);
	hir__storage.add_dim(2, 0);
	hir__storage.add_dim(3, 0);
	hir__storage.bw(1, 0);
	hir__storage.build();
	hir.add_dim(3, 0);
	hir.add_dim(2, 0);
	hir.add_dim(3, 0);
	hir.bw(1, 0);
	hir.build();
	hir.set_storage (&hir__storage);
	cir__storage.add_dim(3, 0);
	cir__storage.add_dim(2, 0);
	cir__storage.add_dim(3, 0);
	cir__storage.bw(2, 0);
	cir__storage.build();
	cir.add_dim(3, 0);
	cir.add_dim(2, 0);
	cir.add_dim(3, 0);
	cir.bw(2, 0);
	cir.build();
	cir.set_storage (&cir__storage);
	sir__storage.add_dim(3, 0);
	sir__storage.add_dim(2, 0);
	sir__storage.bw(3, 0);
	sir__storage.build();
	sir.add_dim(3, 0);
	sir.add_dim(2, 0);
	sir.bw(3, 0);
	sir.build();
	sir.set_storage (&sir__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void sp::init ()
{
	if (!built)
	{
			}
	else
	{
		ph__storage.init();
		th11__storage.init();
		th__storage.init();
		vl__storage.init();
		phzvl__storage.init();
		me11a__storage.init();
		cpatr__storage.init();
		ph_num__storage.init();
		ph_q__storage.init();
		ph_qr__storage.init();
		ph_hito__storage.init();
		th_hito__storage.init();
		ph_zone__storage.init();
		ph_ext__storage.init();
		drifttime__storage.init();
		th_window__storage.init();
		ph_foldn__storage.init();
		th_foldn__storage.init();
		ph_rank__storage.init();
		phd__storage.init();
		th11d__storage.init();
		thd__storage.init();
		vld__storage.init();
		me11ad__storage.init();
		cpatd__storage.init();
		patt_ph_vi__storage.init();
		patt_ph_hi__storage.init();
		patt_ph_ci__storage.init();
		patt_ph_si__storage.init();
		ph_match__storage.init();
		th_match__storage.init();
		th_match11__storage.init();
		cpat_match__storage.init();
		phi__storage.init();
		theta__storage.init();
		cpattern__storage.init();
		delta_ph__storage.init();
		delta_th__storage.init();
		sign_ph__storage.init();
		sign_th__storage.init();
		rank__storage.init();
		vir__storage.init();
		hir__storage.init();
		cir__storage.init();
		sir__storage.init();
																																															pcs.init();

	zns.init();

	exts.init();

	phps.init();

	srts.init();

	cdl.init();

	mphseg.init();

	ds.init();

	bt.init();

	}
}

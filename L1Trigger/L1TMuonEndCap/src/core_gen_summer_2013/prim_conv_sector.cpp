// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:01 2015

#include "prim_conv_sector.h"

extern size_t __glob_alwaysn__;

void prim_conv_sector::operator()
(
	signal_& q__io,
	signal_& wg__io,
	signal_& hstr__io,
	signal_& cpat__io,
	signal_& ph__io,
	signal_& th11__io,
	signal_& th__io,
	signal_& vl__io,
	signal_& phzvl__io,
	signal_& me11a__io,
	signal_& cpatr__io,
	signal_& ph_hit__io,
	signal_& th_hit__io,
	signal_& cs__io,
	signal_& sel__io,
	signal_& addr__io,
	signal_& r_in__io,
	signal_& r_out__io,
	signal_& we__io,
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
		// lct parameters [station][chamber][segment]
		q.attach(q__io);
		wg.attach(wg__io);
		hstr.attach(hstr__io);
		cpat.attach(cpat__io);
		cs.attach(cs__io);
		sel.attach(sel__io);
		addr.attach(addr__io);
		r_in.attach(r_in__io);
		we.attach(we__io);
		clk.attach(clk__io);
		control_clk.attach(control_clk__io);
		ph.attach(ph__io);
		// special th outputs for ME11 because of duplication
		th11.attach(th11__io);
		th.attach(th__io);
		vl.attach(vl__io);
		phzvl.attach(phzvl__io);
		// me11a flags only for ME11 (stations 1,0, chambers 2:0)
		me11a.attach(me11a__io);
		cpatr.attach(cpatr__io);
		// ph and th raw hits
		ph_hit.attach(ph_hit__io);
		th_hit.attach(th_hit__io);
		r_out.attach(r_out__io);
	}


	beginalways();
	if (posedge (control_clk))
	{
		r_out = const_(8, 0x0UL);
		for (s = 0; s < 5; s = s+1) // station loop
			for (c = 0; c < 9; c = c+1) // chamber loop
			{
				if (cs[s][c]) r_out = r_out | r_out_w[s][c];
			}
	}
	endalways();
	
	{
		
		// set we for selected modules
		for (i = 0; i < 5; i = i+1)
		{
			 we_w[i] = (we) ? cs[i] : 0;
		}
		
		// 
		for (i = 0; i < 2; i = i+1)
		{
			for (j = 0; j < 3; j = j+1)
			{
				 genblk.station11[i].csc11[j].pc11.station = i;
				 genblk.station11[i].csc11[j].pc11.cscid = j;
					genblk.station11[i].csc11[j].pc11
	(
		q  [i][j],
		wg [i][j],
		hstr[i][j],
		cpat[i][j],
		ph [i][j],
		th11 [i][j],
		vl [i][j],
		phzvl[i][j],
		me11a[i][j],
		cpatr[i][j],
		ph_hit [i][j],
		th_hit [i][j],
		sel,
		addr,
		r_in,
		r_out_w[i][j],
		we_w   [i][j],
		clk,
		control_clk
	);
			}
		} // block: station11
		
		for (i = 0; i < 2; i = i+1)
		{
			for (j = 3; j < 9; j = j+1)
			{
				 genblk.station12[i].csc12[j].pc12.station = i;
				 genblk.station12[i].csc12[j].pc12.cscid = j;
					genblk.station12[i].csc12[j].pc12
	(
		q  [i][j],
		wg [i][j],
		hstr[i][j],
		cpat[i][j],
		ph [i][j],
		th [i][j],
		vl [i][j],
		phzvl[i][j],
		dummy[i][j],
		cpatr[i][j],
		ph_hit [i][j],
		th_hit [i][j],
		sel,
		addr,
		r_in,
		r_out_w[i][j],
		we_w	  [i][j],
		clk,
		control_clk
	);
			}
		} // block: station12

		for (i = 2; i < 5; i = i+1)
		{
			for (j = 0; j < 9; j = j+1)
			{
				 genblk.station[i].csc[j].pc.station = i;
				 genblk.station[i].csc[j].pc.cscid = j;
					genblk.station[i].csc[j].pc
	(
		q  [i][j],
		wg [i][j],
		hstr[i][j],
		cpat[i][j],
		ph [i][j],
		th [i][j],
		vl [i][j],
		phzvl[i][j],
		dummy[i][j],
		cpatr[i][j],
		ph_hit [i][j],
		th_hit [i][j],
		sel,
		addr,
		r_in,
		r_out_w[i][j],
		we_w	  [i][j],
		clk,
		control_clk
	);
			}
		} // block: station

	} // block: genblk
		
	
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void prim_conv_sector::defparam()
{
	station = 1;
	cscid = 1;
}

// vppc: this function allocates memory for internal signals
void prim_conv_sector::build()
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
	cs.add_dim(4, 0);
	cs.bw(8, 0);
	cs.build();
	sel.bw(1, 0);
	addr.bw(bw_addr-1, 0);
	r_in.bw(11, 0);
	we.bw(0, 0);
	clk.bw(0, 0);
	control_clk.bw(0, 0);
	ph.add_dim(4, 0);
	ph.add_dim(8, 0);
	ph.add_dim(seg_ch-1, 0);
	ph.bw(bw_fph-1, 0);
	ph.build();
	th11.add_dim(1, 0);
	th11.add_dim(2, 0);
	th11.add_dim(th_ch11-1, 0);
	th11.bw(bw_th-1, 0);
	th11.build();
	th.add_dim(4, 0);
	th.add_dim(8, 0);
	th.add_dim(seg_ch-1, 0);
	th.bw(bw_th-1, 0);
	th.build();
	vl.add_dim(4, 0);
	vl.add_dim(8, 0);
	vl.bw(seg_ch-1, 0);
	vl.build();
	phzvl.add_dim(4, 0);
	phzvl.add_dim(8, 0);
	phzvl.bw(2, 0);
	phzvl.build();
	me11a.add_dim(1, 0);
	me11a.add_dim(2, 0);
	me11a.bw(seg_ch-1, 0);
	me11a.build();
	cpatr.add_dim(4, 0);
	cpatr.add_dim(8, 0);
	cpatr.add_dim(seg_ch-1, 0);
	cpatr.bw(3, 0);
	cpatr.build();
	ph_hit.add_dim(4, 0);
	ph_hit.add_dim(8, 0);
	ph_hit.bw(ph_hit_w-1, 0);
	ph_hit.build();
	th_hit.add_dim(4, 0);
	th_hit.add_dim(8, 0);
	th_hit.bw(th_hit_w-1, 0);
	th_hit.build();
	r_out.bw(11, 0);
	r_out_w__storage.add_dim(4, 0);
	r_out_w__storage.add_dim(8, 0);
	r_out_w__storage.bw(11, 0);
	r_out_w__storage.build();
	r_out_w.add_dim(4, 0);
	r_out_w.add_dim(8, 0);
	r_out_w.bw(11, 0);
	r_out_w.build();
	r_out_w.set_storage (&r_out_w__storage);
	we_w__storage.add_dim(4, 0);
	we_w__storage.bw(8, 0);
	we_w__storage.build();
	we_w.add_dim(4, 0);
	we_w.bw(8, 0);
	we_w.build();
	we_w.set_storage (&we_w__storage);
	dummy__storage.add_dim(4, 0);
	dummy__storage.add_dim(8, 0);
	dummy__storage.bw(seg_ch-1, 0);
	dummy__storage.build();
	dummy.add_dim(4, 0);
	dummy.add_dim(8, 0);
	dummy.bw(seg_ch-1, 0);
	dummy.build();
	dummy.set_storage (&dummy__storage);
	s__storage.bw(31, 0);
	s.bw(31, 0);
	s.set_storage (&s__storage);
	c__storage.bw(31, 0);
	c.bw(31, 0);
	c.set_storage (&c__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void prim_conv_sector::init ()
{
	if (!built)
	{
			}
	else
	{
		r_out_w__storage.init();
		we_w__storage.init();
		dummy__storage.init();
																																															genblk.init();
	}
}
void prim_conv_sector::genblk__class::init()
{
	for (map <ull, station11__class>::iterator mit = station11.begin(); mit != station11.end(); mit++)
		mit->second.init();
	for (map <ull, station12__class>::iterator mit = station12.begin(); mit != station12.end(); mit++)
		mit->second.init();
	for (map <ull, station__class>::iterator mit = station.begin(); mit != station.end(); mit++)
		mit->second.init();
}
void prim_conv_sector::genblk__class::station11__class::init()
{
	for (map <ull, csc11__class>::iterator mit = csc11.begin(); mit != csc11.end(); mit++)
		mit->second.init();
}
void prim_conv_sector::genblk__class::station11__class::csc11__class::init()
{
	pc11.init();
}
void prim_conv_sector::genblk__class::station12__class::init()
{
	for (map <ull, csc12__class>::iterator mit = csc12.begin(); mit != csc12.end(); mit++)
		mit->second.init();
}
void prim_conv_sector::genblk__class::station12__class::csc12__class::init()
{
	pc12.init();
}
void prim_conv_sector::genblk__class::station__class::init()
{
	for (map <ull, csc__class>::iterator mit = csc.begin(); mit != csc.end(); mit++)
		mit->second.init();
}
void prim_conv_sector::genblk__class::station__class::csc__class::init()
{
	pc.init();
}

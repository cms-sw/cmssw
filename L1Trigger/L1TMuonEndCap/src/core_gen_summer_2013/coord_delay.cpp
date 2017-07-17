// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#include "coord_delay.h"

extern size_t __glob_alwaysn__;

void coord_delay::operator()
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
		mem_ph_bw = bw_fph*5*9*seg_ch;
		mem_th_bw = bw_th*5*9*seg_ch;
		mem_th11_bw = bw_th*2*3*th_ch11;
		mem_vl_bw = 5*9*seg_ch;
		mem_me11a_bw = 2*3*seg_ch;
		mem_cpat_bw = 4*5*9*seg_ch;
		build();
		phi.attach(phi__io);
		th11i.attach(th11i__io);
		thi.attach(thi__io);
		vli.attach(vli__io);
		me11ai.attach(me11ai__io);
		cpati.attach(cpati__io);
		clk.attach(clk__io);
		// outputs show not only delayed values, but also history of max_drift clocks
		pho.attach(pho__io);
		th11o.attach(th11o__io);
		tho.attach(tho__io);
		vlo.attach(vlo__io);
		me11ao.attach(me11ao__io);
		cpato.attach(cpato__io);
	}


	beginalways();
	if (posedge (clk))
	{
		
		// merge inputs
		for (i = 0; i < 5; i = i+1) // station loop
			for (j = 0; j < 9; j = j+1) // chamber loop
				for (k = 0; k < seg_ch; k = k+1) // segment loop
				{
				    mem_ph_in.bp((i*9*seg_ch+j*seg_ch+k)*bw_fph  , bw_fph) = phi[i][j][k]; 
					mem_th_in.bp((i*9*seg_ch+j*seg_ch+k)*bw_th  , bw_th) = thi[i][j][k]; 
					mem_vl_in[i*9*seg_ch+j*seg_ch+k] = vli[i][j][k]; 
				    mem_cpat_in.bp((i*9*seg_ch+j*seg_ch+k)*4  , 4) = cpati[i][j][k]; 
				}
		// ME1/1 merge inputs
		for (i = 0; i < 2; i = i+1) // station loop
			for (j = 0; j < 3; j = j+1) // chamber loop
			{
				for (k = 0; k < th_ch11; k = k+1) // segment loop
				{
					mem_th11_in.bp((i*3*th_ch11+j*th_ch11+k)*bw_th  , bw_th) = th11i[i][j][k];
					if (k < 2) mem_me11a_in[i*3*seg_ch+j*seg_ch+k] = me11ai[i][j][k]; 
				}
			}

		// read the outputs
		mem_ph_out   = mem_ph  [ra];
		mem_th_out   = mem_th  [ra];
		mem_th11_out = mem_th11[ra];
		mem_vl_out   = mem_vl  [ra];
		mem_me11a_out= mem_me11a  [ra];
		mem_cpat_out = mem_cpat   [ra];
		
		// write all input bits into memory on each clock
		mem_ph  [wa] = mem_ph_in;
		mem_th  [wa] = mem_th_in;
		mem_th11[wa] = mem_th11_in;
		mem_vl  [wa] = mem_vl_in;
		mem_me11a [wa] = mem_me11a_in;
		mem_cpat  [wa] = mem_cpat_in;

		wa = (ra + latency + 1);
		ra = (ra + 1);
		
		// shift history
		for (d = max_drift-1; d > 0; d = d-1) // history bx loop
		{
			for (i = 0; i < 5; i = i+1) // station loop
				for (j = 0; j < 9; j = j+1) // chamber loop
					for (k = 0; k < seg_ch; k = k+1) // segment loop
					{
						pho[d][i][j][k] = pho[d-1][i][j][k]; 
						tho[d][i][j][k] = tho[d-1][i][j][k]; 
						vlo[d][i][j][k] = vlo[d-1][i][j][k];
						cpato[d][i][j][k] = cpato[d-1][i][j][k]; 
					}                 

			for (i = 0; i < 2; i = i+1) // station loop
				for (j = 0; j < 3; j = j+1) // chamber loop
					for (k = 0; k < th_ch11; k = k+1) // segment loop
					{
						th11o[d][i][j][k] = th11o[d-1][i][j][k];
						if (k < 2) me11ao[d][i][j][k] = me11ao[d-1][i][j][k];
					}
		}
		
		
		// split memory outputs into word[0] of the coordinate history 
		for (i = 0; i < 5; i = i+1) // station loop
			for (j = 0; j < 9; j = j+1) // chamber loop
				for (k = 0; k < seg_ch; k = k+1) // segment loop
				{
				    pho[0][i][j][k] = mem_ph_out.bp((i*9*seg_ch+j*seg_ch+k)*bw_fph  , bw_fph); 
					tho[0][i][j][k] = mem_th_out.bp((i*9*seg_ch+j*seg_ch+k)*bw_th  , bw_th); 
					vlo[0][i][j][k] = mem_vl_out[i*9*seg_ch+j*seg_ch+k]; 
				    cpato[0][i][j][k] = mem_cpat_out.bp((i*9*seg_ch+j*seg_ch+k)*4  , 4); 
				}
		// split outputs
		for (i = 0; i < 2; i = i+1) // station loop
			for (j = 0; j < 3; j = j+1) // chamber loop
				for (k = 0; k < th_ch11; k = k+1) // segment loop
				{
					th11o[0][i][j][k] = mem_th11_out.bp((i*3*th_ch11+j*th_ch11+k)*bw_th  , bw_th);
					if (k < 2) me11ao[0][i][j][k] = mem_me11a_out[i*3*seg_ch+j*seg_ch+k]; 
				}
	}
	endalways();
}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void coord_delay::defparam()
{
	station = 1;
	cscid = 1;
	pulse_l = 3;
	latency = 6;
}

// vppc: this function allocates memory for internal signals
void coord_delay::build()
{
	built = true;
	phi.add_dim(4, 0);
	phi.add_dim(8, 0);
	phi.add_dim(seg_ch-1, 0);
	phi.bw(bw_fph-1, 0);
	phi.build();
	th11i.add_dim(1, 0);
	th11i.add_dim(2, 0);
	th11i.add_dim(th_ch11-1, 0);
	th11i.bw(bw_th-1, 0);
	th11i.build();
	thi.add_dim(4, 0);
	thi.add_dim(8, 0);
	thi.add_dim(seg_ch-1, 0);
	thi.bw(bw_th-1, 0);
	thi.build();
	vli.add_dim(4, 0);
	vli.add_dim(8, 0);
	vli.bw(seg_ch-1, 0);
	vli.build();
	me11ai.add_dim(1, 0);
	me11ai.add_dim(2, 0);
	me11ai.bw(seg_ch-1, 0);
	me11ai.build();
	cpati.add_dim(4, 0);
	cpati.add_dim(8, 0);
	cpati.add_dim(seg_ch-1, 0);
	cpati.bw(3, 0);
	cpati.build();
	clk.bw(0, 0);
	pho.add_dim(max_drift-1, 0);
	pho.add_dim(4, 0);
	pho.add_dim(8, 0);
	pho.add_dim(seg_ch-1, 0);
	pho.bw(bw_fph-1, 0);
	pho.build();
	th11o.add_dim(max_drift-1, 0);
	th11o.add_dim(1, 0);
	th11o.add_dim(2, 0);
	th11o.add_dim(th_ch11-1, 0);
	th11o.bw(bw_th-1, 0);
	th11o.build();
	tho.add_dim(max_drift-1, 0);
	tho.add_dim(4, 0);
	tho.add_dim(8, 0);
	tho.add_dim(seg_ch-1, 0);
	tho.bw(bw_th-1, 0);
	tho.build();
	vlo.add_dim(max_drift-1, 0);
	vlo.add_dim(4, 0);
	vlo.add_dim(8, 0);
	vlo.bw(seg_ch-1, 0);
	vlo.build();
	me11ao.add_dim(max_drift-1, 0);
	me11ao.add_dim(1, 0);
	me11ao.add_dim(2, 0);
	me11ao.bw(seg_ch-1, 0);
	me11ao.build();
	cpato.add_dim(max_drift-1, 0);
	cpato.add_dim(4, 0);
	cpato.add_dim(8, 0);
	cpato.add_dim(seg_ch-1, 0);
	cpato.bw(3, 0);
	cpato.build();
	mem_ph_in__storage.bw(mem_ph_bw-1, 0);
	mem_ph_in.bw(mem_ph_bw-1, 0);
	mem_ph_in.set_storage (&mem_ph_in__storage);
	mem_ph_out__storage.bw(mem_ph_bw-1, 0);
	mem_ph_out.bw(mem_ph_bw-1, 0);
	mem_ph_out.set_storage (&mem_ph_out__storage);
	mem_th_in__storage.bw(mem_th_bw-1, 0);
	mem_th_in.bw(mem_th_bw-1, 0);
	mem_th_in.set_storage (&mem_th_in__storage);
	mem_th_out__storage.bw(mem_th_bw-1, 0);
	mem_th_out.bw(mem_th_bw-1, 0);
	mem_th_out.set_storage (&mem_th_out__storage);
	mem_th11_in__storage.bw(mem_th11_bw-1, 0);
	mem_th11_in.bw(mem_th11_bw-1, 0);
	mem_th11_in.set_storage (&mem_th11_in__storage);
	mem_th11_out__storage.bw(mem_th11_bw-1, 0);
	mem_th11_out.bw(mem_th11_bw-1, 0);
	mem_th11_out.set_storage (&mem_th11_out__storage);
	mem_vl_in__storage.bw(mem_vl_bw-1, 0);
	mem_vl_in.bw(mem_vl_bw-1, 0);
	mem_vl_in.set_storage (&mem_vl_in__storage);
	mem_vl_out__storage.bw(mem_vl_bw-1, 0);
	mem_vl_out.bw(mem_vl_bw-1, 0);
	mem_vl_out.set_storage (&mem_vl_out__storage);
	mem_me11a_in__storage.bw(mem_me11a_bw-1, 0);
	mem_me11a_in.bw(mem_me11a_bw-1, 0);
	mem_me11a_in.set_storage (&mem_me11a_in__storage);
	mem_me11a_out__storage.bw(mem_me11a_bw-1, 0);
	mem_me11a_out.bw(mem_me11a_bw-1, 0);
	mem_me11a_out.set_storage (&mem_me11a_out__storage);
	mem_cpat_in__storage.bw(mem_cpat_bw-1, 0);
	mem_cpat_in.bw(mem_cpat_bw-1, 0);
	mem_cpat_in.set_storage (&mem_cpat_in__storage);
	mem_cpat_out__storage.bw(mem_cpat_bw-1, 0);
	mem_cpat_out.bw(mem_cpat_bw-1, 0);
	mem_cpat_out.set_storage (&mem_cpat_out__storage);
	mem_ph__storage.add_dim(511, 0);
	mem_ph__storage.bw(mem_ph_bw-1, 0);
	mem_ph__storage.build();
	mem_ph.add_dim(511, 0);
	mem_ph.bw(mem_ph_bw-1, 0);
	mem_ph.build();
	mem_ph.set_storage (&mem_ph__storage);
	mem_th__storage.add_dim(511, 0);
	mem_th__storage.bw(mem_th_bw-1, 0);
	mem_th__storage.build();
	mem_th.add_dim(511, 0);
	mem_th.bw(mem_th_bw-1, 0);
	mem_th.build();
	mem_th.set_storage (&mem_th__storage);
	mem_th11__storage.add_dim(511, 0);
	mem_th11__storage.bw(mem_th11_bw-1, 0);
	mem_th11__storage.build();
	mem_th11.add_dim(511, 0);
	mem_th11.bw(mem_th11_bw-1, 0);
	mem_th11.build();
	mem_th11.set_storage (&mem_th11__storage);
	mem_vl__storage.add_dim(511, 0);
	mem_vl__storage.bw(mem_vl_bw-1, 0);
	mem_vl__storage.build();
	mem_vl.add_dim(511, 0);
	mem_vl.bw(mem_vl_bw-1, 0);
	mem_vl.build();
	mem_vl.set_storage (&mem_vl__storage);
	mem_me11a__storage.add_dim(511, 0);
	mem_me11a__storage.bw(mem_me11a_bw-1, 0);
	mem_me11a__storage.build();
	mem_me11a.add_dim(511, 0);
	mem_me11a.bw(mem_me11a_bw-1, 0);
	mem_me11a.build();
	mem_me11a.set_storage (&mem_me11a__storage);
	mem_cpat__storage.add_dim(511, 0);
	mem_cpat__storage.bw(mem_cpat_bw-1, 0);
	mem_cpat__storage.build();
	mem_cpat.add_dim(511, 0);
	mem_cpat.bw(mem_cpat_bw-1, 0);
	mem_cpat.build();
	mem_cpat.set_storage (&mem_cpat__storage);
	ra__storage.bw(8, 0);
	ra.bw(8, 0);
	ra.set_storage (&ra__storage);
	wa__storage.bw(8, 0);
	wa.bw(8, 0);
	wa.set_storage (&wa__storage);
	ra = 0;
	wa = latency;

}

// vppc: this function checks for changes in any signal on each simulation iteration
void coord_delay::init ()
{
	if (!built)
	{
					}
	else
	{
		mem_ph_in__storage.init();
		mem_ph_out__storage.init();
		mem_th_in__storage.init();
		mem_th_out__storage.init();
		mem_th11_in__storage.init();
		mem_th11_out__storage.init();
		mem_vl_in__storage.init();
		mem_vl_out__storage.init();
		mem_me11a_in__storage.init();
		mem_me11a_out__storage.init();
		mem_cpat_in__storage.init();
		mem_cpat_out__storage.init();
		mem_ph__storage.init();
		mem_th__storage.init();
		mem_th11__storage.init();
		mem_vl__storage.init();
		mem_me11a__storage.init();
		mem_cpat__storage.init();
		ra__storage.init();
		wa__storage.init();
																																																					}
}

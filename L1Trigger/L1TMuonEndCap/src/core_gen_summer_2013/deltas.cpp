// This C++ source file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Thu Mar 12 14:54:00 2015

#include "deltas.h"

extern size_t __glob_alwaysn__;

void deltas::operator()
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
		bw_nm1 = me11 ? 3 : 2;
		bw_nm2 = 2;
		build();
		// input parameters [station]
		vi.attach(vi__io);
		hi.attach(hi__io);
		ci.attach(ci__io);
		si.attach(si__io);
		ph_match.attach(ph_match__io);
		// theta coordinates [station][segment]
		th_match.attach(th_match__io);
		// ME11 duplicated thetas [segment]
		th_match11.attach(th_match11__io);
		cpat_match.attach(cpat_match__io);
		ph_q.attach(ph_q__io);
		th_window.attach(th_window__io);
		clk.attach(clk__io);
		// precise phi and theta
		phi.attach(phi__io);
		theta.attach(theta__io);
		cpattern.attach(cpattern__io);
		// ph and th deltas from best stations [0] - best pair of stations, [1] - second best pair
		delta_ph.attach(delta_ph__io);
		delta_th.attach(delta_th__io);
		sign_ph.attach(sign_ph__io);
		sign_th.attach(sign_th__io);
		rank.attach(rank__io);
		vir.attach(vir__io);
		hir.attach(hir__io);
		cir.attach(cir__io);
		sir.attach(sir__io);
	}
	dvalid = dvl12 != 0 || dvl13 != 0 || dvl14 != 0 || dvl23 != 0 || dvl24 != 0 || dvl34 != 0;


	// difference sorters
	 bd12.nseg = seg1*seg_ch;
	 bd13.nseg = seg1*seg_ch;
	 bd14.nseg = seg1*seg_ch;
	 bd23.nseg = seg_ch*seg_ch;
	 bd24.nseg = seg_ch*seg_ch;
	 bd34.nseg = seg_ch*seg_ch;
	
	if (true)
	{

		for (i1 = 0; i1 < 3; i1 = i1+1) // station 1 loop
		{
			for (i2 = i1+1; i2 < 4; i2 = i2+1) // station 2 loop
			{
				
				// calculate theta deltas
				for (j = 0; j < ((i1==0) ? seg1 : seg_ch); j = j+1) // segment st A loop
				{
					for (k = 0; k < seg_ch; k = k+1) // segment st B loop
					{
						
						if (me11)
							thA = (i1 == 0) ? th_match11[j] : th_match[i1][j];
						else
							thA = th_match[i1][j];
						
						thB = th_match[i2][k];
						dth = (thA > thB) ? thA - thB : thB - thA;
						sth = (thA > thB); // sign
						// if one of the segments not valid, delta not valid
						dvl = vi[i1][j%seg_ch] & vi[i2][k];
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0x1UL))) {  { dth12[j*seg_ch + k] = dth; dvl12[j*seg_ch + k] = dvl; sth12[j*seg_ch + k] = sth; } } else 
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0x2UL))) {  { dth13[j*seg_ch + k] = dth; dvl13[j*seg_ch + k] = dvl; sth13[j*seg_ch + k] = sth; } } else 
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0x3UL))) {  { dth14[j*seg_ch + k] = dth; dvl14[j*seg_ch + k] = dvl; sth14[j*seg_ch + k] = sth; } } else 
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0x6UL))) {  { dth23[j*seg_ch + k] = dth; dvl23[j*seg_ch + k] = dvl; sth23[j*seg_ch + k] = sth; } } else 
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0x7UL))) {  { dth24[j*seg_ch + k] = dth; dvl24[j*seg_ch + k] = dvl; sth24[j*seg_ch + k] = sth; } } else 
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0xbUL))) {  { dth34[j*seg_ch + k] = dth; dvl34[j*seg_ch + k] = dvl; sth34[j*seg_ch + k] = sth; } } 
					}
				} // for (j = 0; j < ((i1==0) ? seg1 : seg_ch); j = j+1)

				// calculate phi deltas
				phA = ph_match[i1];
				phB = ph_match[i2];
				dph = (phA > phB) ? phA - phB : phB - phA;
				sph = (phA > phB);
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0x1UL))) {  { dph12 = dph; sph12 = sph; } } else 
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0x2UL))) {  { dph13 = dph; sph13 = sph; } } else 
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0x3UL))) {  { dph14 = dph; sph14 = sph; } } else 
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0x6UL))) {  { dph23 = dph; sph23 = sph; } } else 
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0x7UL))) {  { dph24 = dph; sph24 = sph; } } else 
 if (( ((i1(1,0), i2(1,0))) == const_(4, 0xbUL))) {  { dph34 = dph; sph34 = sph; } } 
				
				
			}
		} // for (i = 0; i < 3; i = i+1)
	}

	beginalways();
		
	if (posedge (clk))
	{
		// find valid segments
		vmask1 = const_(4, 0x0UL);
		vmask2 = const_(4, 0x0UL);
		vmask3 = const_(4, 0x0UL);

		// vmask contains valid station mask = (ME4,ME3,ME2,ME1)
		if (bth12 <= th_window && bvl12) vmask1 = vmask1 | const_(4, 0x3UL);
		if (bth13 <= th_window && bvl13) vmask1 = vmask1 | const_(4, 0x5UL);
		if (bth14 <= th_window && bvl14) vmask1 = vmask1 | const_(4, 0x9UL);
		if (bth23 <= th_window && bvl23) vmask2 = vmask2 | const_(4, 0x6UL);
		if (bth24 <= th_window && bvl24) vmask2 = vmask2 | const_(4, 0xaUL);
		if (bth34 <= th_window && bvl34) vmask3 = vmask3 | const_(4, 0xcUL);

		// merge station masks only if they share bits
		// could try here to find two tracks with identical ph
		// for example vmask1 = 1001 and vmask2 = 0110
		// not done so far, just select one with better station combination
		vstat = vmask1;
		if ((vstat & vmask2) != const_(4, 0x0UL) || vstat == const_(4, 0x0UL)) vstat = vstat | vmask2;
		if ((vstat & vmask3) != const_(4, 0x0UL) || vstat == const_(4, 0x0UL)) vstat = vstat | vmask3;
		if (( (vstat) == const_(4, 0xcUL))) {  { delta_ph[0] = dph34; delta_ph[1] = dph34; delta_th[0] = bth34; delta_th[1] = bth34; } } else 
		if (( (vstat) == const_(4, 0xaUL))) {  { delta_ph[0] = dph24; delta_ph[1] = dph24; delta_th[0] = bth24; delta_th[1] = bth24; } } else 
		if (( (vstat) == const_(4, 0x6UL))) {  { delta_ph[0] = dph23; delta_ph[1] = dph23; delta_th[0] = bth23; delta_th[1] = bth23; } } else 
		if (( (vstat) == const_(4, 0xeUL))) {  { delta_ph[0] = dph23; delta_ph[1] = dph34; delta_th[0] = bth23; delta_th[1] = bth34; } } else 
		if (( (vstat) == const_(4, 0x9UL))) {  { delta_ph[0] = dph14; delta_ph[1] = dph14; delta_th[0] = bth14; delta_th[1] = bth14; } } else 
		if (( (vstat) == const_(4, 0x5UL))) {  { delta_ph[0] = dph13; delta_ph[1] = dph13; delta_th[0] = bth13; delta_th[1] = bth13; } } else 
		if (( (vstat) == const_(4, 0xdUL))) {  { delta_ph[0] = dph13; delta_ph[1] = dph34; delta_th[0] = bth13; delta_th[1] = bth34; } } else 
		if (( (vstat) == const_(4, 0x3UL))) {  { delta_ph[0] = dph12; delta_ph[1] = dph12; delta_th[0] = bth12; delta_th[1] = bth12; } } else 
		if (( (vstat) == const_(4, 0xbUL))) {  { delta_ph[0] = dph12; delta_ph[1] = dph24; delta_th[0] = bth12; delta_th[1] = bth24; } } else 
		if (( (vstat) == const_(4, 0x7UL))) {  { delta_ph[0] = dph12; delta_ph[1] = dph23; delta_th[0] = bth12; delta_th[1] = bth23; } } else 
		if (( (vstat) == const_(4, 0xfUL))) {  { delta_ph[0] = dph12; delta_ph[1] = dph23; delta_th[0] = bth12; delta_th[1] = bth23; } }
 		if (( (vstat) == const_(4, 0xcUL))) {  { sign_ph[0] = sph34; sign_ph[1] = sph34; sign_th[0] = bsg34; sign_th[1] = bsg34; } } else 
 		if (( (vstat) == const_(4, 0xaUL))) {  { sign_ph[0] = sph24; sign_ph[1] = sph24; sign_th[0] = bsg24; sign_th[1] = bsg24; } } else 
 		if (( (vstat) == const_(4, 0x6UL))) {  { sign_ph[0] = sph23; sign_ph[1] = sph23; sign_th[0] = bsg23; sign_th[1] = bsg23; } } else 
 		if (( (vstat) == const_(4, 0xeUL))) {  { sign_ph[0] = sph23; sign_ph[1] = sph34; sign_th[0] = bsg23; sign_th[1] = bsg34; } } else 
 		if (( (vstat) == const_(4, 0x9UL))) {  { sign_ph[0] = sph14; sign_ph[1] = sph14; sign_th[0] = bsg14; sign_th[1] = bsg14; } } else 
 		if (( (vstat) == const_(4, 0x5UL))) {  { sign_ph[0] = sph13; sign_ph[1] = sph13; sign_th[0] = bsg13; sign_th[1] = bsg13; } } else 
 		if (( (vstat) == const_(4, 0xdUL))) {  { sign_ph[0] = sph13; sign_ph[1] = sph34; sign_th[0] = bsg13; sign_th[1] = bsg34; } } else 
 		if (( (vstat) == const_(4, 0x3UL))) {  { sign_ph[0] = sph12; sign_ph[1] = sph12; sign_th[0] = bsg12; sign_th[1] = bsg12; } } else 
 		if (( (vstat) == const_(4, 0xbUL))) {  { sign_ph[0] = sph12; sign_ph[1] = sph24; sign_th[0] = bsg12; sign_th[1] = bsg24; } } else 
 		if (( (vstat) == const_(4, 0x7UL))) {  { sign_ph[0] = sph12; sign_ph[1] = sph23; sign_th[0] = bsg12; sign_th[1] = bsg23; } } else 
 		if (( (vstat) == const_(4, 0xfUL))) {  { sign_ph[0] = sph12; sign_ph[1] = sph23; sign_th[0] = bsg12; sign_th[1] = bsg23; } } 


		// segment ids
		vir = vi;
		hir = hi;
		cir = ci;
		sir = si;

		// remove some valid flags if th did not line up
		for (j = 0; j < 4; j = j+1)
			if (vstat[j] == const_(1, 0x0UL)) vir[j] = 0;

		//  precise phi and theta
		phi = 0;
		theta = 0;
		if      (vstat[1] == const_(1, 0x1UL)) // ME2 present
		{
			// phi is simple, we have it
			phi = ph_match[1];

			// for theta, select delta to best station, use winner number as index
			if      (bvl12) theta = th_match[1][bnm12[0]];
			else if (bvl23) theta = th_match[1][bnm23[1]];
			else if (bvl24) theta = th_match[1][bnm24[1]];
		} 
		else if (vstat[2] == const_(1, 0x1UL)) // ME3 present
		{ 
			phi = ph_match[2]; 
			if      (bvl13) theta = th_match[2][bnm13[0]];
			else if (bvl34) theta = th_match[2][bnm34[1]];
		} 
		else if (vstat[3] == const_(1, 0x1UL)) // ME4 present
		{ 
			phi = ph_match[3]; 
			if      (bvl14) theta = th_match[3][bnm14[0]];
		} 

		// update rank taking into account available stations after th deltas
		// keep straightness as it was
		rank = (ph_q, const_s(1, 0x0UL)); // output rank is one bit longer than input, to accommodate ME4 separately
		rank[0] = vstat[3]; // ME4
		rank[1] = vstat[2]; // ME3
		rank[3] = vstat[1]; // ME2
		rank[5] = vstat[0]; // ME1

		// if less than 2 segments, kill rank
		if (vstat == const_(4, 0x1UL) || vstat == const_(4, 0x2UL) || vstat == const_(4, 0x4UL) || vstat == const_(4, 0x8UL) || vstat == const_(4, 0x0UL))
			rank = 0;

		cpattern = cpat_match[0]; // pattern taken from station 1 only at this time

	}
	endalways();	bd12
	(
		dth12,
		sth12,
		dvl12,
		bth12,
		bsg12,
		bvl12,
		bnm12
	);
	bd13
	(
		dth13,
		sth13,
		dvl13,
		bth13,
		bsg13,
		bvl13,
		bnm13
	);
	bd14
	(
		dth14,
		sth14,
		dvl14,
		bth14,
		bsg14,
		bvl14,
		bnm14
	);
	bd23
	(
		dth23,
		sth23,
		dvl23,
		bth23,
		bsg23,
		bvl23,
		bnm23
	);
	bd24
	(
		dth24,
		sth24,
		dvl24,
		bth24,
		bsg24,
		bvl24,
		bnm24
	);
	bd34
	(
		dth34,
		sth34,
		dvl34,
		bth34,
		bsg34,
		bvl34,
		bnm34
	);

}

// vppc: code below this point are service functions, should not require your attention

// vppc: this function assigns initial values to parameters and localparams
void deltas::defparam()
{
	station = 1;
	cscid = 1;
	me11 = 1;
}

// vppc: this function allocates memory for internal signals
void deltas::build()
{
	built = true;
	vi.add_dim(3, 0);
	vi.bw(seg_ch-1, 0);
	vi.build();
	hi.add_dim(3, 0);
	hi.bw(1, 0);
	hi.build();
	ci.add_dim(3, 0);
	ci.bw(2, 0);
	ci.build();
	si.bw(3, 0);
	ph_match.add_dim(3, 0);
	ph_match.bw(bw_fph-1, 0);
	ph_match.build();
	th_match.add_dim(3, 0);
	th_match.add_dim(seg_ch-1, 0);
	th_match.bw(bw_th-1, 0);
	th_match.build();
	th_match11.add_dim(th_ch11-1, 0);
	th_match11.bw(bw_th-1, 0);
	th_match11.build();
	cpat_match.add_dim(3, 0);
	cpat_match.bw(3, 0);
	cpat_match.build();
	ph_q.bw(bwr-1, 0);
	th_window.bw(bw_th-1, 0);
	clk.bw(0, 0);
	phi.bw(bw_fph-1, 0);
	theta.bw(bw_th-1, 0);
	cpattern.bw(3, 0);
	delta_ph.add_dim(1, 0);
	delta_ph.bw(bw_fph-1, 0);
	delta_ph.build();
	delta_th.add_dim(1, 0);
	delta_th.bw(bw_th-1, 0);
	delta_th.build();
	sign_ph.bw(1, 0);
	sign_th.bw(1, 0);
	rank.bw(bwr, 0);
	vir.add_dim(3, 0);
	vir.bw(seg_ch-1, 0);
	vir.build();
	hir.add_dim(3, 0);
	hir.bw(1, 0);
	hir.build();
	cir.add_dim(3, 0);
	cir.bw(2, 0);
	cir.build();
	sir.bw(3, 0);
	vstat__storage.bw(3, 0);
	vstat.bw(3, 0);
	vstat.set_storage (&vstat__storage);
	thA__storage.bw(bw_th-1, 0);
	thA.bw(bw_th-1, 0);
	thA.set_storage (&thA__storage);
	thB__storage.bw(bw_th-1, 0);
	thB.bw(bw_th-1, 0);
	thB.set_storage (&thB__storage);
	dth__storage.bw(bw_th-1, 0);
	dth.bw(bw_th-1, 0);
	dth.set_storage (&dth__storage);
	dvl__storage.bw(0, 0);
	dvl.bw(0, 0);
	dvl.set_storage (&dvl__storage);
	dth12__storage.add_dim(seg1*seg_ch-1, 0);
	dth12__storage.bw(bw_th-1, 0);
	dth12__storage.build();
	dth12.add_dim(seg1*seg_ch-1, 0);
	dth12.bw(bw_th-1, 0);
	dth12.build();
	dth12.set_storage (&dth12__storage);
	dth13__storage.add_dim(seg1*seg_ch-1, 0);
	dth13__storage.bw(bw_th-1, 0);
	dth13__storage.build();
	dth13.add_dim(seg1*seg_ch-1, 0);
	dth13.bw(bw_th-1, 0);
	dth13.build();
	dth13.set_storage (&dth13__storage);
	dth14__storage.add_dim(seg1*seg_ch-1, 0);
	dth14__storage.bw(bw_th-1, 0);
	dth14__storage.build();
	dth14.add_dim(seg1*seg_ch-1, 0);
	dth14.bw(bw_th-1, 0);
	dth14.build();
	dth14.set_storage (&dth14__storage);
	dth23__storage.add_dim(seg_ch*seg_ch-1, 0);
	dth23__storage.bw(bw_th-1, 0);
	dth23__storage.build();
	dth23.add_dim(seg_ch*seg_ch-1, 0);
	dth23.bw(bw_th-1, 0);
	dth23.build();
	dth23.set_storage (&dth23__storage);
	dth24__storage.add_dim(seg_ch*seg_ch-1, 0);
	dth24__storage.bw(bw_th-1, 0);
	dth24__storage.build();
	dth24.add_dim(seg_ch*seg_ch-1, 0);
	dth24.bw(bw_th-1, 0);
	dth24.build();
	dth24.set_storage (&dth24__storage);
	dth34__storage.add_dim(seg_ch*seg_ch-1, 0);
	dth34__storage.bw(bw_th-1, 0);
	dth34__storage.build();
	dth34.add_dim(seg_ch*seg_ch-1, 0);
	dth34.bw(bw_th-1, 0);
	dth34.build();
	dth34.set_storage (&dth34__storage);
	dvl12__storage.bw(seg1*seg_ch-1, 0);
	dvl12.bw(seg1*seg_ch-1, 0);
	dvl12.set_storage (&dvl12__storage);
	dvl13__storage.bw(seg1*seg_ch-1, 0);
	dvl13.bw(seg1*seg_ch-1, 0);
	dvl13.set_storage (&dvl13__storage);
	dvl14__storage.bw(seg1*seg_ch-1, 0);
	dvl14.bw(seg1*seg_ch-1, 0);
	dvl14.set_storage (&dvl14__storage);
	dvl23__storage.bw(seg_ch*seg_ch-1, 0);
	dvl23.bw(seg_ch*seg_ch-1, 0);
	dvl23.set_storage (&dvl23__storage);
	dvl24__storage.bw(seg_ch*seg_ch-1, 0);
	dvl24.bw(seg_ch*seg_ch-1, 0);
	dvl24.set_storage (&dvl24__storage);
	dvl34__storage.bw(seg_ch*seg_ch-1, 0);
	dvl34.bw(seg_ch*seg_ch-1, 0);
	dvl34.set_storage (&dvl34__storage);
	sth12__storage.bw(seg1*seg_ch-1, 0);
	sth12.bw(seg1*seg_ch-1, 0);
	sth12.set_storage (&sth12__storage);
	sth13__storage.bw(seg1*seg_ch-1, 0);
	sth13.bw(seg1*seg_ch-1, 0);
	sth13.set_storage (&sth13__storage);
	sth14__storage.bw(seg1*seg_ch-1, 0);
	sth14.bw(seg1*seg_ch-1, 0);
	sth14.set_storage (&sth14__storage);
	sth23__storage.bw(seg_ch*seg_ch-1, 0);
	sth23.bw(seg_ch*seg_ch-1, 0);
	sth23.set_storage (&sth23__storage);
	sth24__storage.bw(seg_ch*seg_ch-1, 0);
	sth24.bw(seg_ch*seg_ch-1, 0);
	sth24.set_storage (&sth24__storage);
	sth34__storage.bw(seg_ch*seg_ch-1, 0);
	sth34.bw(seg_ch*seg_ch-1, 0);
	sth34.set_storage (&sth34__storage);
	bnm12__storage.bw(bw_nm1-1, 0);
	bnm12.bw(bw_nm1-1, 0);
	bnm12.set_storage (&bnm12__storage);
	bnm13__storage.bw(bw_nm1-1, 0);
	bnm13.bw(bw_nm1-1, 0);
	bnm13.set_storage (&bnm13__storage);
	bnm14__storage.bw(bw_nm1-1, 0);
	bnm14.bw(bw_nm1-1, 0);
	bnm14.set_storage (&bnm14__storage);
	bnm23__storage.bw(bw_nm2-1, 0);
	bnm23.bw(bw_nm2-1, 0);
	bnm23.set_storage (&bnm23__storage);
	bnm24__storage.bw(bw_nm2-1, 0);
	bnm24.bw(bw_nm2-1, 0);
	bnm24.set_storage (&bnm24__storage);
	bnm34__storage.bw(bw_nm2-1, 0);
	bnm34.bw(bw_nm2-1, 0);
	bnm34.set_storage (&bnm34__storage);
	phA__storage.bw(bw_fph-1, 0);
	phA.bw(bw_fph-1, 0);
	phA.set_storage (&phA__storage);
	phB__storage.bw(bw_fph-1, 0);
	phB.bw(bw_fph-1, 0);
	phB.set_storage (&phB__storage);
	dph__storage.bw(bw_fph-1, 0);
	dph.bw(bw_fph-1, 0);
	dph.set_storage (&dph__storage);
	sph__storage.bw(bw_fph-1, 0);
	sph.bw(bw_fph-1, 0);
	sph.set_storage (&sph__storage);
	dph12__storage.bw(bw_fph-1, 0);
	dph12.bw(bw_fph-1, 0);
	dph12.set_storage (&dph12__storage);
	dph13__storage.bw(bw_fph-1, 0);
	dph13.bw(bw_fph-1, 0);
	dph13.set_storage (&dph13__storage);
	dph14__storage.bw(bw_fph-1, 0);
	dph14.bw(bw_fph-1, 0);
	dph14.set_storage (&dph14__storage);
	dph23__storage.bw(bw_fph-1, 0);
	dph23.bw(bw_fph-1, 0);
	dph23.set_storage (&dph23__storage);
	dph24__storage.bw(bw_fph-1, 0);
	dph24.bw(bw_fph-1, 0);
	dph24.set_storage (&dph24__storage);
	dph34__storage.bw(bw_fph-1, 0);
	dph34.bw(bw_fph-1, 0);
	dph34.set_storage (&dph34__storage);
	sph12__storage.bw(0, 0);
	sph12.bw(0, 0);
	sph12.set_storage (&sph12__storage);
	sph13__storage.bw(0, 0);
	sph13.bw(0, 0);
	sph13.set_storage (&sph13__storage);
	sph14__storage.bw(0, 0);
	sph14.bw(0, 0);
	sph14.set_storage (&sph14__storage);
	sph23__storage.bw(0, 0);
	sph23.bw(0, 0);
	sph23.set_storage (&sph23__storage);
	sph24__storage.bw(0, 0);
	sph24.bw(0, 0);
	sph24.set_storage (&sph24__storage);
	sph34__storage.bw(0, 0);
	sph34.bw(0, 0);
	sph34.set_storage (&sph34__storage);
	bsg12__storage.bw(0, 0);
	bsg12.bw(0, 0);
	bsg12.set_storage (&bsg12__storage);
	bsg13__storage.bw(0, 0);
	bsg13.bw(0, 0);
	bsg13.set_storage (&bsg13__storage);
	bsg14__storage.bw(0, 0);
	bsg14.bw(0, 0);
	bsg14.set_storage (&bsg14__storage);
	bsg23__storage.bw(0, 0);
	bsg23.bw(0, 0);
	bsg23.set_storage (&bsg23__storage);
	bsg24__storage.bw(0, 0);
	bsg24.bw(0, 0);
	bsg24.set_storage (&bsg24__storage);
	bsg34__storage.bw(0, 0);
	bsg34.bw(0, 0);
	bsg34.set_storage (&bsg34__storage);
	bvl12__storage.bw(0, 0);
	bvl12.bw(0, 0);
	bvl12.set_storage (&bvl12__storage);
	bvl13__storage.bw(0, 0);
	bvl13.bw(0, 0);
	bvl13.set_storage (&bvl13__storage);
	bvl14__storage.bw(0, 0);
	bvl14.bw(0, 0);
	bvl14.set_storage (&bvl14__storage);
	bvl23__storage.bw(0, 0);
	bvl23.bw(0, 0);
	bvl23.set_storage (&bvl23__storage);
	bvl24__storage.bw(0, 0);
	bvl24.bw(0, 0);
	bvl24.set_storage (&bvl24__storage);
	bvl34__storage.bw(0, 0);
	bvl34.bw(0, 0);
	bvl34.set_storage (&bvl34__storage);
	sth__storage.bw(0, 0);
	sth.bw(0, 0);
	sth.set_storage (&sth__storage);
	vmask1__storage.bw(3, 0);
	vmask1.bw(3, 0);
	vmask1.set_storage (&vmask1__storage);
	vmask2__storage.bw(3, 0);
	vmask2.bw(3, 0);
	vmask2.set_storage (&vmask2__storage);
	vmask3__storage.bw(3, 0);
	vmask3.bw(3, 0);
	vmask3.set_storage (&vmask3__storage);
	bth12__storage.bw(bw_th-1, 0);
	bth12.bw(bw_th-1, 0);
	bth12.set_storage (&bth12__storage);
	bth13__storage.bw(bw_th-1, 0);
	bth13.bw(bw_th-1, 0);
	bth13.set_storage (&bth13__storage);
	bth14__storage.bw(bw_th-1, 0);
	bth14.bw(bw_th-1, 0);
	bth14.set_storage (&bth14__storage);
	bth23__storage.bw(bw_th-1, 0);
	bth23.bw(bw_th-1, 0);
	bth23.set_storage (&bth23__storage);
	bth24__storage.bw(bw_th-1, 0);
	bth24.bw(bw_th-1, 0);
	bth24.set_storage (&bth24__storage);
	bth34__storage.bw(bw_th-1, 0);
	bth34.bw(bw_th-1, 0);
	bth34.set_storage (&bth34__storage);
	dvalid__storage.bw(0, 0);
	dvalid.bw(0, 0);
	dvalid.set_storage (&dvalid__storage);
	i1__storage.bw(31, 0);
	i1.bw(31, 0);
	i1.set_storage (&i1__storage);
	i2__storage.bw(31, 0);
	i2.bw(31, 0);
	i2.set_storage (&i2__storage);

}

// vppc: this function checks for changes in any signal on each simulation iteration
void deltas::init ()
{
	if (!built)
	{
				}
	else
	{
		vstat__storage.init();
		thA__storage.init();
		thB__storage.init();
		dth__storage.init();
		dvl__storage.init();
		dth12__storage.init();
		dth13__storage.init();
		dth14__storage.init();
		dth23__storage.init();
		dth24__storage.init();
		dth34__storage.init();
		dvl12__storage.init();
		dvl13__storage.init();
		dvl14__storage.init();
		dvl23__storage.init();
		dvl24__storage.init();
		dvl34__storage.init();
		sth12__storage.init();
		sth13__storage.init();
		sth14__storage.init();
		sth23__storage.init();
		sth24__storage.init();
		sth34__storage.init();
		bnm12__storage.init();
		bnm13__storage.init();
		bnm14__storage.init();
		bnm23__storage.init();
		bnm24__storage.init();
		bnm34__storage.init();
		phA__storage.init();
		phB__storage.init();
		dph__storage.init();
		sph__storage.init();
		dph12__storage.init();
		dph13__storage.init();
		dph14__storage.init();
		dph23__storage.init();
		dph24__storage.init();
		dph34__storage.init();
		sph12__storage.init();
		sph13__storage.init();
		sph14__storage.init();
		sph23__storage.init();
		sph24__storage.init();
		sph34__storage.init();
		bsg12__storage.init();
		bsg13__storage.init();
		bsg14__storage.init();
		bsg23__storage.init();
		bsg24__storage.init();
		bsg34__storage.init();
		bvl12__storage.init();
		bvl13__storage.init();
		bvl14__storage.init();
		bvl23__storage.init();
		bvl24__storage.init();
		bvl34__storage.init();
		sth__storage.init();
		vmask1__storage.init();
		vmask2__storage.init();
		vmask3__storage.init();
		bth12__storage.init();
		bth13__storage.init();
		bth14__storage.init();
		bth23__storage.init();
		bth24__storage.init();
		bth34__storage.init();
		dvalid__storage.init();
																																																		bd12.init();

	bd13.init();

	bd14.init();

	bd23.init();

	bd24.init();

	bd34.init();

	}
}

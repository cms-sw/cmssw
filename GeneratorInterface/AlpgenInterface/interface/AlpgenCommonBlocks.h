#ifndef GeneratorInterface_AlpgenInterface_AlpgenCommonBlocks_h
#define GeneratorInterface_AlpgenInterface_AlpgenCommonBlocks_h

/// A C/C++ representation of the ALPGEN Common Blocks:
/// AHOPTS, AHPPARA, AHPARS, AHCUTS.
extern "C" {
	extern struct AHOPTS {
		double	etclus;		// needs to be set up
		double	rclus;		// needs to be set up
		double	etaclmax;
		int	iexc;		// needs to be set up
		int	npfst;
		int	nplst;
		int	nljets;
		int	njstart;
		int	njlast;
		int	ickkw;
	} ahopts_;

	extern struct AHPPARA {
		double	masses[6];	// mc,mb,mt,mw,mz,mh (set up these)
		double	ebeam;
		int	ih1, ih2, ihrd;	// ihrd needs to be set up
		int	itopprc;
		int	nw, nz, nh, nph;
		int	ihvy, ihvy2;
		int	npart, ndns, pdftyp;
	} ahppara_;

	extern struct AHPARS {
		static const unsigned int nparam = 200;

		double	parval[nparam];
		char	chpar[nparam][8];
		char	chpdes[nparam][70];
		int	parlen[nparam];
		int	partyp[nparam];
	} ahpars_;

	extern struct AHCUTS {
		double	ptjmin, ptjmax;
		double	etajmax, drjmin;
		double	ptbmin, ptbmax;
		double	etabmax, drbmin;
		double	ptcmin, ptcmax;
		double	etacmax, drcmin;
		double	ptphmin;
		double	etaphmax;
		double	drphjmin, drphmin, drphlmin;
		double	ptlmin;
		double	etalmax, drlmin;
		double	metmin;
		double	mllmin, mllmax;
	} ahcuts_;
}

#endif // GeneratorInterface_AlpgenInterface_AlpgenCommonBlocks_h

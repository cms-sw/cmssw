#ifndef HCALDQMCONSTANTS_H
#define HCALDQMCONSTANTS_H

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"

namespace hcaldqm
{
	namespace constants
	{
		//	RAW Format Constants
		int const RAWDATASIZE_CALIB					=24;
		int const RAWDATASIZE_HEADERTRAILER			= 12;
		int const RAWDATASIZE_EMPTY					= 16;

		//	Generic Constants
		int const MAXCALIBTYPE						=7;
		int const MIN_FIBER_IDLEOFFSET				=-7;
		int const MAX_FIBER_IDLEOFFSET				=	7;
		int const INVALID_FIBER_IDLEOFFSET			=	-1000;
		int const CT_NORMAL							= hc_Null;
		int const CT_PED							= hc_Pedestal;
		int const CT_RADDAM							= hc_RADDAM;
		int const CT_HBHEHPD						= hc_HBHEHPD;
		int const CT_HOSIPM							= hc_HBHEHPD;
		int const CT_HFPMT							= hc_HFPMT;
		int const UTCA_DATAFLAVOR					= 0x5;
		int const VME_DCC_OFFSET					= 700;
		int const PUBLISH_MIN_CALIBEVENTS			= 500;
	
		//	Hcal Specific Constants (not SubSystem specific)
		int const STD_NUMSUBS						=		4;
		int const STD_NUMADC						= 128;
		int const STD_NUMBINSFORPED					= 6;
		double const STD_MINLEDQ					= 20.;
		double const STD_MAXLEDQ					= 10000.;
		int const STD_NUMCAPS						= 4;
		int const STD_NUMIPHIS						= 72;
		int const STD_NUMIETAS						= 83;
		int const STD_NUMDEPTHS						= 4;
	
		//	HF Specific Constants
		int const STD_SUBDET_HF						=	3;
		int const STD_HF_MINIPHI					= 1;
		int const STD_HF_STEPIPHI					= 2;
		int const STD_HF_MAXIPHI					= 72;
		int const STD_HF_MINIETA					= 29;
		int const STD_HF_STEPIETA					= 1;
		int const STD_HF_MAXIETA					= 41;
		int const STD_HF_MINDEPTH					= 1;
		int const STD_HF_STEPDEPTH					= 1;
		int const STD_HF_MAXDEPTH					= 2;
		double const STD_HF_PED						=		2.5;
		int const STD_HF_DIGISIZE_GLOBAL			=		4;
		int const STD_HF_DIGISIZE_LOCAL				=	10;
		double const STD_HF_DIGI_CUT_3TSQg20		=		20.;
		double const STD_HF_DIGI_ZSCUT				= 
			STD_HF_DIGI_CUT_3TSQg20;
		double const STD_HF_RECHIT_CUT_eg2			= 2.;
		double const STD_HF_RECHIT_ZSCUT			= 
			STD_HF_RECHIT_CUT_eg2;
		int const STD_HF_NUMCHS						= 1728;
	
		//	HB Specific
		int const STD_SUBDET_HB						=	0;
		int const STD_HB_MINIPHI					= 1;
		int const STD_HB_STEPIPHI					= 1;
		int const STD_HB_MAXIPHI					= 72;
		int const STD_HB_MINIETA					= 1;
		int const STD_HB_STEPIETA					= 1;
		int const STD_HB_MAXIETA					= 16;
		int const STD_HB_MINDEPTH					= 1;
		int const STD_HB_STEPDEPTH					= 1;
		int const STD_HB_MAXDEPTH					= 2;
		int const STD_HB_DIGISIZE_GLOBAL			=		10;
		double const STD_HB_PED						=		2.5;
		double const STD_HB_DIGI_CUT_3TSQg20		=		20.;
		double const STD_HB_DIGI_ZSCUT				=
			STD_HB_DIGI_CUT_3TSQg20;
		double const STD_HB_RECHIT_CUT_eg2			= 2.;
		double const STD_HB_RECHIT_ZSCUT			= 
			STD_HB_RECHIT_CUT_eg2;
	
		//	HE Specific
		int const STD_SUBDET_HE						=	1;
		int const STD_HE_MINIPHI					= 1;
		int const STD_HE_STEPIPHI					= 1;
		int const STD_HE_MAXIPHI					= 72;
		int const STD_HE_MINIETA					= 16;
		int const STD_HE_STEPIETA					= 1;
		int const STD_HE_MAXIETA					= 29;
		int const STD_HE_MINDEPTH					= 1;
		int const STD_HE_STEPDEPTH					= 1;
		int const STD_HE_MAXDEPTH					= 3;
		int const STD_HE_DIGISIZE_GLOBAL			=		10;
		double const STD_HE_PED						=		0;
		double const STD_HE_DIGI_CUT_3TSQg20		=		20.;
		double const STD_HE_DIGI_ZSCUT				=
			STD_HE_DIGI_CUT_3TSQg20;
		double const STD_HE_RECHIT_CUT_eg2			= 2.;
		double const STD_HE_RECHIT_ZSCUT			= 
			STD_HE_RECHIT_CUT_eg2;

		//	HBHE Common
		double const STD_HBHE_TS5TS4_MEANSDIFF		= 0.05;
	
		//	HO Specific Constants
		int const STD_SUBDET_HO						=	2;
		int const STD_HO_MINIPHI					= 1;
		int const STD_HO_STEPIPHI					= 1;
		int const STD_HO_MAXIPHI					= 72;
		int const STD_HO_MINIETA					= 1;
		int const STD_HO_STEPIETA					= 1;
		int const STD_HO_MAXIETA					= 15;
		int const STD_HO_MINDEPTH					= 4;
		int const STD_HO_STEPDEPTH					= 1;
		int const STD_HO_MAXDEPTH					= 4;
		int const STD_HO_DIGISIZE_GLOBAL			=		10;
		double const STD_HO_PED						=		8.5;
		double const STD_HO_DIGI_CUT_3TSQg30		=		30.;
		double const STD_HO_DIGI_ZSCUT				=
			STD_HO_DIGI_CUT_3TSQg30;
		double const STD_HO_RECHIT_CUT_eg0			= 0;
		double const STD_HO_RECHIT_ZSCUT			= 
			STD_HO_RECHIT_CUT_eg0;

		//	TObject Bits
		int const STD_BIT_KLOGX						= 19;
		int const STD_BIT_KLOGY						= 20;
		int const STD_BIT_KLOGZ						= 21;

		//	Other 
		int const STD_SUBDET_OTHER					=	4;
	
		//	More Hcal Specific
		double const PEDESTALS[STD_NUMSUBS] = {
			STD_HB_PED, STD_HE_PED, STD_HO_PED, STD_HF_PED
		};

		std::string const SUBNAMES[STD_NUMSUBS] = {
			"HB", "HE", "HO", "HF"
		};

		double const DIGISIZE_GLOBAL[STD_NUMSUBS] = {
			STD_HB_DIGISIZE_GLOBAL, STD_HE_DIGISIZE_GLOBAL,
			STD_HO_DIGISIZE_GLOBAL, STD_HF_DIGISIZE_GLOBAL
		};

		double const DIGI_ZSCUT[STD_NUMSUBS] = {
			STD_HB_DIGI_ZSCUT, STD_HE_DIGI_ZSCUT, 
			STD_HO_DIGI_ZSCUT, STD_HF_DIGI_ZSCUT
		};

/*		double const RECHIT_ZSCUT[STD_NUMSUBS] = {
			STD_HB_RECHIT_ZSCUT, STD_HE_RECHIT_ZSCUT,
			STD_HO_RECHIT_ZSCUT, STD_HF_RECHIT_ZSCUT
		};*/
		double const RECHIT_ZSCUT[STD_NUMSUBS] = {
			5., 5., 5., 5.
		};


	}
}

#endif















#ifndef Constants_h
#define Constants_h

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"

#include <vector>

namespace hcaldqm
{
	namespace constants
	{
		/*
		 *	Detector Subsystem Status States
		 */
		double const GOOD = 0.98;
		double const PROBLEMATIC = 0.95;
		double const LOW = 0.75;
		double const VERY_LOW = 0.5;
		double const VERY_LOW_XXX = 0;
		double const NOT_APPLICABLE = -1;
		double const DELTA = 0.005;

		/*
		 *	Electronics Constants
		 */
		//	FED2Crate array and CRATE2FED array
		//	use conversion functions in Utilities.h
		//	For fast look up
		//	This is for uTCA Crates/FEDs only - no other way...
		int const FED_uTCA_MAX_REAL = 50;
		uint16_t const FED2CRATE[FED_uTCA_MAX_REAL] = {
			24, 0, 20, 0, 21, 0, 25, 0, 31, 0,
			35, 0, 37, 0, 34, 0, 30, 0, 22, 0,
			29, 0, 32, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 36, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		};
		uint16_t const CRATE2FED[50] = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			1102, 1104, 1118, 0, 1100, 1106, 0, 0, 0, 1120,
			1116, 1108, 1122, 0, 1114, 1110, 1132, 1112, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		};

		//	FEDs use the first 50 uTCA FED numbers only everywhere
		int const FED_VME_MIN = FEDNumbering::MINHCALFEDID;
		int const FED_VME_MAX = FEDNumbering::MAXHCALFEDID;
		int const FED_VME_DELTA = 1;
		int const FED_VME_NUM = FED_VME_MAX-FED_VME_MIN+1;

		int const FED_uTCA_MIN = FEDNumbering::MINHCALuTCAFEDID;
//		int const FED_uTCA_MAX = FEDNumbering::MAXHCALuTCAFEDID;
		int const FED_uTCA_MAX = FED_uTCA_MIN + FED_uTCA_MAX_REAL-1;
		int const FED_uTCA_NUM = FED_uTCA_MAX - FED_uTCA_MIN + 1;
		int const FED_uTCA_DELTA = 1;
		int const FED_TOTAL_NUM = FED_VME_NUM+FED_uTCA_NUM;
		
		//	Crates
		int const CRATE_VME_MIN = 0;
		int const CRATE_VME_MAX = 18;
		int const CRATE_VME_DELTA = 1;
		int const CRATE_VME_NUM = CRATE_VME_MAX-CRATE_VME_MIN+1;

		int const CRATE_uTCA_MIN = 20;
		int const CRATE_uTCA_MAX = 37;
		int const CRATE_uTCA_DELTA = 1;
		int const CRATE_uTCA_NUM = CRATE_uTCA_MAX-CRATE_uTCA_MIN+1;
		int const CRATE_TOTAL_NUM = CRATE_VME_NUM + CRATE_uTCA_NUM;

		//	Slots
		int const SLOT_uTCA_MIN = 1;
		int const SLOT_uTCA_MAX = 12;
		int const SLOT_uTCA_DELTA = 1;
		int const SLOT_uTCA_NUM = SLOT_uTCA_MAX-SLOT_uTCA_MIN+1;

		int const SLOT_VME_MIN1 = 2;
		int const SLOT_VME_MAX1 = 7;
		int const SLOT_VME_MIN2 = 13;
		int const SLOT_VME_MAX2 = 18;
		int const SLOT_VME_NUM1 = SLOT_VME_MAX1-SLOT_VME_MIN1+1;
		int const SLOT_VME_NUM2 = SLOT_VME_MAX2-SLOT_VME_MIN2+1;
		int const SLOT_VME_NUM = SLOT_VME_NUM1+SLOT_VME_NUM2;

		int const SPIGOT_MIN = 0;
		int const SPIGOT_MAX = 11;
		int const SPIGOT_NUM = SPIGOT_MAX-SPIGOT_MIN+1;

		//	Fibers
		int const FIBER_VME_MIN = 1;
		int const FIBER_VME_MAX = 8;
		int const FIBER_VME_NUM = FIBER_VME_MAX-FIBER_VME_MIN+1;
		int const FIBER_uTCA_MIN1 = 2;
		int const FIBER_uTCA_MAX1 = 9;
		int const FIBER_uTCA_MIN2 = 14;
		int const FIBER_uTCA_MAX2 = 21;
		int const FIBER_uTCA_NUM = FIBER_uTCA_MAX1-FIBER_uTCA_MIN1+1 + 
			FIBER_uTCA_MAX2-FIBER_uTCA_MIN2+1;

		int const FIBERCH_MIN = 0;
		int const FIBERCH_MAX = 2;
		int const FIBERCH_NUM = FIBERCH_MAX-FIBERCH_MIN+1;

		//	TP SLBs, Fibers
		int const SLB_MIN = 1;
		int const SLB_MAX = 6;
		int const SLB_NUM = SLB_MAX-SLB_MIN+1;

		int const TPFIBER_MIN = 0;
		int const TPFIBER_MAX = 5;
		int const TPFIBER_NUM = TPFIBER_MAX-TPFIBER_MIN+1;

		int const SLBCH_MIN = 0;
		int const SLBCH_MAX = 3;
		int const SLBCH_NUM = SLBCH_MAX-SLBCH_MIN+1;

		int const TPFIBERCH_MIN = 0;
		int const TPFIBERCH_MAX = 7;
		int const TPFIBERCH_NUM = TPFIBERCH_MAX-TPFIBERCH_MIN+1;

		/*
		 *	Detector Constants
		 */
		
		//	Hcal Subdetector
		int const HB = 1;
		int const HE = 2;
		int const HO = 3;
		int const HF = 4;
		int const SUBDET_NUM = 4;
		int const TPSUBDET_NUM = 2;
		int const DIGISIZE[SUBDET_NUM] = {10, 10, 10, 4};
		std::string const SUBDET_NAME[SUBDET_NUM]={"HB", "HE", "HO", "HF"};
		std::string const SUBDETPM_NAME[2*SUBDET_NUM] = { "HBM", "HBP",
			"HEM", "HEP", "HOM", "HOP", "HFM", "HFP"};
		std::string const SUBSYSTEM = "Hcal";
		std::string const TPSUBDET_NAME[TPSUBDET_NUM] = {"HBHE", "HF"};
		std::string const TPSUBDETPM_NAME[2*TPSUBDET_NUM] = {
			"HBHEM", "HBHEP", "HFM", "HFP"};

		//	iphis
		int const IPHI_MIN = 1;
		int const IPHI_MAX = 72;
		int const IPHI_NUM = 72;
		int const IPHI_NUM_HF = 36;
		int const IPHI_NUM_TPHF = 18;
		int const IPHI_DELTA = 1;
		int const IPHI_DELTA_HF = 2;
		int const IPHI_DELTA_TPHF = 4;
		
		//	ietas
		int const IETA_MIN=1;
		int const IETA_DELTA = 1;
		int const IETA_MAX=41;
		int const IETA_NUM=2*(IETA_MAX-IETA_MIN+1)+1;
		int const IETA_MIN_HB = 1;
		int const IETA_MAX_HB = 16;
		int const IETA_MIN_HE = 16;
		int const IETA_MAX_HE = 29;
		int const IETA_MIN_HO = 1;
		int const IETA_MAX_HO = 15;
		int const IETA_MIN_HF = 29;
		int const IETA_MAX_HF = 41;

		int const IETA_MAX_TPHBHE = 28;
		int const IETA_MAX_TPHF = 32;

		//	Depth
		int const DEPTH_MIN = 1;
		int const DEPTH_DELTA = 1;
		int const DEPTH_MAX = 4;
		int const DEPTH_NUM = 4;

		//	Caps
		int const CAPS_NUM = 4;

		/*
		 *	Number of Channels Constants
		 */
		int const CHS_NUM[SUBDET_NUM] = { 2592, 2592,2192, 1728};	// HO ??!
		int const TPCHS_NUM[TPSUBDET_NUM] = {2*28*72, 144};

		/*
		 *	Number of Time Samples
		 */
		int const TS_NUM[SUBDET_NUM] = {10, 10, 10, 4};

		/*
		 *	Value Constants
		 */
		double const AXIS_ENERGY_MIN = -10.;
		double const AXIS_ENERGY_MAX = 200;
		int const AXIS_ENERGY_NBINS = 400;
		double const AXIS_TIME_MIN = -50.;
		double const AXIS_TIME_MAX = 50;
		int const AXIS_TIME_NBINS = 200;
		int const AXIS_ADC_NBINS_PED = 100;
		double const AXIS_ADC_MAX_PED = 5;
		int const AXIS_ADC_NBINS = 128;
		double const AXIS_ADC_MIN = 0;
		double const AXIS_ADC_MAX = 128;
		int const AXIS_NOMFC_NBINS_3000 = 300;
		double const AXIS_NOMFC_MAX_3000 = 3000.;
		int const AXIS_NOMFC_NBINS = 300;
		double const AXIS_NOMFC_MIN = 0;
		double const AXIS_NOMFC_MAX = 3000.;
		int const AXIS_TIMETS_NBINS = 10;
		double const AXIS_TIMETS_MIN = 0;
		double const AXIS_TIMETS_MAX = 10;

		int const CALIBEVENTS_MIN = 100;
		int const GARBAGE_VALUE = -1000;
		int const FIBEROFFSET_INVALID = -1000;

		int const RAW_EMPTY = 16;
		int const UTCA_DATAFLAVOR = 0x5;

		/*
		 *	TObject Related. The first 3 bits are set by the Axis Class
		 *	0 - log X axis
		 *	1 - log Y axis
		 *	2 - log Z axis
		 */	
		int const BIT_OFFSET = 19;
		int const BIT_AXIS_XLOG = 0;
		int const BIT_AXIS_YLOG = 1;
		int const BIT_AXIS_ZLOG = 2;
		int const BIT_AXIS_LS = 3;
		int const BIT_AXIS_FLAG = 4;
	}
}

#endif





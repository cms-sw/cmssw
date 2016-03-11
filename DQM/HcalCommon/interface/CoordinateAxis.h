#ifndef CoordinateAxis_h
#define CoordinateAxis_h

/*
 *	file:		CoordinateAxis.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 *		A wrapper between the detector/electronics coordinates to be plotted
 *		and bins. Currently some coordinates do have proper mapping
 *		value -> bin, some do not.
 *
 */

#include "DQM/HcalCommon/interface/Axis.h"

namespace hcaldqm
{
	namespace axis
	{
		using namespace hcaldqm::constants;
		enum CoordinateType
		{
			fSubDet = 0,
			fiphi = 1,
			fieta = 2,
			fdepth = 3,

			fFEDVME = 4,
			fFEDuTCA = 5,
			fFEDComb = 6,
			fCrateVME = 7,
			fCrateuTCA = 8,
			fCrateComb = 9,
			fSlotVME = 10,
			fSlotuTCA = 11,
			fSlotComb = 12,
			fFiberVME = 13,
			fFiberuTCA = 14,
			fFiberComb = 15,
			fFiberCh = 16,

			fTPSubDet = 17,
			fTPieta = 18,

			fSubDetPM = 19,
			fTPSubDetPM = 20,

			fSpigot = 21,

			nCoordinateType = 22
		};

		std::string const ctitle[nCoordinateType] = {
			"Sub Detector", "iphi", "ieta", "depth", 

			"FED", "FED", "FED", "Crate", "Crate", "Crate",
			"Slot", "Slot", "Slot", "Fiber", "Fiber", "Fiber",
			"Fiber Channel",

			"TP Sub Detector", "TP ieta",
			"Sub Detector (+/-)", "TP Sub Detector (+/-)",

			"Spigot"
		};
		double const cmin[nCoordinateType] = {
			HB, IPHI_MIN-0.5, 0, DEPTH_MIN-0.5,

			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FIBERCH_MIN,

			0, 0, 0, 0,

			-.5
		};
		double const cmax[nCoordinateType] = {
			HF+1, IPHI_MAX+0.5, 84, DEPTH_MAX+0.5,

			FED_VME_NUM, FED_uTCA_NUM, FED_VME_NUM+FED_uTCA_NUM,
			CRATE_VME_NUM, CRATE_uTCA_NUM, CRATE_VME_NUM+CRATE_uTCA_NUM,
			SLOT_VME_NUM, SLOT_uTCA_NUM, SLOT_uTCA_NUM, // #uTCA SLOTs=#VME
			2*FIBER_VME_NUM, FIBER_uTCA_NUM, FIBER_uTCA_NUM, //	2*VME for tb
			FIBERCH_MAX+1,

			2, 64, 8, 4,

			SPIGOT_MAX+0.5
		};
		int const cnbins[nCoordinateType] = {
			SUBDET_NUM, IPHI_NUM, 84, DEPTH_NUM, 
			FED_VME_NUM, FED_uTCA_NUM, FED_VME_NUM+FED_uTCA_NUM,
			CRATE_VME_NUM, CRATE_uTCA_NUM, CRATE_VME_NUM+CRATE_uTCA_NUM, 
			SLOT_VME_NUM, SLOT_uTCA_NUM, SLOT_uTCA_NUM,
			2*FIBER_VME_NUM, FIBER_uTCA_NUM, FIBER_uTCA_NUM,  
			FIBERCH_NUM,

			2, 64, 8, 4,

			SPIGOT_NUM
		};

		class CoordinateAxis : public Axis
		{
			public:
				friend class hcaldqm::Container;
				friend class hcaldqm::Container1D;
				friend class hcaldqm::Container2D;
				friend class hcaldqm::ContainerProf1D;
				friend class hcaldqm::ContainerProf2D;
				friend class hcaldqm::ContainerSingle2D;
				friend class hcaldqm::ContainerSingle1D;
				friend class hcaldqm::ContainerSingleProf1D;

			public:
				CoordinateAxis();
				CoordinateAxis(AxisType, CoordinateType, bool log=false);
				CoordinateAxis(AxisType type, CoordinateType ctype, 
					int n, double min, double max, std::string title, 
					bool log=false);
				virtual ~CoordinateAxis() {}
				virtual CoordinateAxis* makeCopy()
				{return new CoordinateAxis(_type, _ctype, _log);}

				virtual int get(HcalDetId const&);
				virtual int get(HcalElectronicsId const&);
				virtual int get(HcalTrigTowerDetId const&);
				virtual int get(int);

				virtual int getBin(HcalDetId const&);
				virtual int getBin(HcalElectronicsId const&);
				virtual int getBin(HcalTrigTowerDetId const&);
				virtual int getBin(int);

			protected:
				virtual void _setup();

			protected:
				CoordinateType _ctype;
		};
	}
}

#endif





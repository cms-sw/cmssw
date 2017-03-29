#ifndef ElectronicsQuantity_h
#define ElectronicsQauntity_h

/**
 *	file:		ElectronicsQuantity.h
 *	Author:		Viktor Khristenko
 */

#include "DQM/HcalCommon/interface/Quantity.h"
#include "boost/unordered_map.hpp"
#include "boost/foreach.hpp"

namespace hcaldqm
{
	using namespace constants;
	namespace quantity
	{
		enum ElectronicsQuantityType
		{
			fFED = 0,
			fFEDuTCA = 1,
			fFEDVME = 2,
			fCrate = 3,
			fCrateuTCA = 4,
			fCrateVME = 5,
			fSlotuTCA = 6,
			fSlotVME = 7,
			fSpigot = 8,
			fFiberuTCA = 9,
			fFiberVME = 10,
			fFiberCh = 11,

			//	Complex Quantities
			fFEDuTCASlot = 12,
			fFEDVMESpigot = 13,
			fFiberuTCAFiberCh = 14,
			fFiberVMEFiberCh = 15,

			//	Adding Trigger Quantities for VME
			fSLB = 16,
			fSLBCh = 17,
			fSLBSLBCh = 18,

			//	Adding Trigger Quantities for uTCA
			fFiberuTCATP = 19,
			fFiberChuTCATP = 20,
			fFiberuTCATPFiberChuTCATP = 21,

			nElectronicsQuantityType = 22
		};

		int getValue_FED(HcalElectronicsId const&);
		int getValue_FEDuTCA(HcalElectronicsId const&);
		int getValue_FEDVME(HcalElectronicsId const&);
		int getValue_Crate(HcalElectronicsId const&);
		int getValue_CrateuTCA(HcalElectronicsId const&);
		int getValue_CrateVME(HcalElectronicsId const&);
		int getValue_SlotuTCA(HcalElectronicsId const&);
		int getValue_SlotVME(HcalElectronicsId const&);
		int getValue_Spigot(HcalElectronicsId const&);
		int getValue_FiberuTCA(HcalElectronicsId const&);
		int getValue_FiberVME(HcalElectronicsId const&);
		int getValue_FiberCh(HcalElectronicsId const&);
		int getValue_FEDuTCASlot(HcalElectronicsId const&);
		int getValue_FEDVMESpigot(HcalElectronicsId const&);
		int getValue_FiberuTCAFiberCh(HcalElectronicsId const&);
		int getValue_FiberVMEFiberCh(HcalElectronicsId const&);
		int getValue_SLB(HcalElectronicsId const&);
		int getValue_SLBCh(HcalElectronicsId const&);
		int getValue_SLBSLBCh(HcalElectronicsId const&);
		int getValue_FiberuTCATP(HcalElectronicsId const&);
		int getValue_FiberChuTCATP(HcalElectronicsId const&);
		int getValue_FiberuTCATPFiberChuTCATP(HcalElectronicsId const&);
		uint32_t getBin_FED(HcalElectronicsId const&);
		uint32_t getBin_FEDuTCA(HcalElectronicsId const&);
		uint32_t getBin_FEDVME(HcalElectronicsId const&);
		uint32_t getBin_Crate(HcalElectronicsId const&);
		uint32_t getBin_CrateuTCA(HcalElectronicsId const&);
		uint32_t getBin_CrateVME(HcalElectronicsId const&);
		uint32_t getBin_SlotuTCA(HcalElectronicsId const&);
		uint32_t getBin_SlotVME(HcalElectronicsId const&);
		uint32_t getBin_Spigot(HcalElectronicsId const&);
		uint32_t getBin_FiberuTCA(HcalElectronicsId const&);
		uint32_t getBin_FiberVME(HcalElectronicsId const&);
		uint32_t getBin_FiberCh(HcalElectronicsId const&);
		uint32_t getBin_FEDuTCASlot(HcalElectronicsId const&);
		uint32_t getBin_FEDVMESpigot(HcalElectronicsId const&);
		uint32_t getBin_FiberuTCAFiberCh(HcalElectronicsId const&);
		uint32_t getBin_FiberVMEFiberCh(HcalElectronicsId const&);
		uint32_t getBin_SLB(HcalElectronicsId const&);
		uint32_t getBin_SLBCh(HcalElectronicsId const&);
		uint32_t getBin_SLBSLBCh(HcalElectronicsId const&);
		uint32_t getBin_FiberuTCATP(HcalElectronicsId const&);
		uint32_t getBin_FiberChuTCATP(HcalElectronicsId const&);
		uint32_t getBin_FiberuTCATPFiberChuTCATP(HcalElectronicsId const&);
		HcalElectronicsId getEid_FED(int);
		HcalElectronicsId getEid_FEDuTCA(int);
		HcalElectronicsId getEid_FEDVME(int);
		HcalElectronicsId getEid_Crate(int);
		HcalElectronicsId getEid_CrateuTCA(int);
		HcalElectronicsId getEid_CrateVME(int);
		HcalElectronicsId getEid_SlotuTCA(int);
		HcalElectronicsId getEid_SlotVME(int);
		HcalElectronicsId getEid_Spigot(int);
		HcalElectronicsId getEid_FiberuTCA(int);
		HcalElectronicsId getEid_FiberVME(int);
		HcalElectronicsId getEid_FiberCh(int);
		HcalElectronicsId getEid_FEDuTCASlot(int);
		HcalElectronicsId getEid_FEDVMESpigot(int);
		HcalElectronicsId getEid_FiberuTCAFiberCh(int);
		HcalElectronicsId getEid_FiberVMEFiberCh(int);
		HcalElectronicsId getEid_SLB(int);
		HcalElectronicsId getEid_SLBCh(int);
		HcalElectronicsId getEid_SLBSLBCh(int);
		HcalElectronicsId getEid_FiberuTCATP(int);
		HcalElectronicsId getEid_FiberChuTCATP(int);
		HcalElectronicsId getEid_FiberuTCATPFiberChuTCATP(int);
		std::vector<std::string> getLabels_FED();
		std::vector<std::string> getLabels_FEDuTCA();
		std::vector<std::string> getLabels_FEDVME();
		std::vector<std::string> getLabels_Crate();
		std::vector<std::string> getLabels_CrateuTCA();
		std::vector<std::string> getLabels_CrateVME();
		std::vector<std::string> getLabels_SlotuTCA();
		std::vector<std::string> getLabels_SlotVME();
		std::vector<std::string> getLabels_Spigot();
		std::vector<std::string> getLabels_FiberuTCA();
		std::vector<std::string> getLabels_FiberVME();
		std::vector<std::string> getLabels_FiberCh();
		std::vector<std::string> getLabels_FEDuTCASlot();
		std::vector<std::string> getLabels_FEDVMESpigot();
		std::vector<std::string> getLabels_FiberuTCAFiberCh();
		std::vector<std::string> getLabels_FiberVMEFiberCh();
		std::vector<std::string> getLabels_SLB();
		std::vector<std::string> getLabels_SLBCh();
		std::vector<std::string> getLabels_SLBSLBCh();
		std::vector<std::string> getLabels_FiberuTCATP();
		std::vector<std::string> getLabels_FiberChuTCATP();
		std::vector<std::string> getLabels_FiberuTCATPFiberChuTCATP();

		typedef int(*getValueType_eid)(HcalElectronicsId const&);
		typedef uint32_t (*getBinType_eid)(HcalElectronicsId const&);
		typedef HcalElectronicsId (*getEid_eid)(int);
		typedef std::vector<std::string> (*getLabels_eid)();
		getValueType_eid const getValue_functions_eid[nElectronicsQuantityType]
			= {
			getValue_FED, getValue_FEDuTCA, getValue_FEDVME,
			getValue_Crate, getValue_CrateuTCA, getValue_CrateVME,
			getValue_SlotuTCA, getValue_SlotVME,
			getValue_Spigot,
			getValue_FiberuTCA, getValue_FiberVME, getValue_FiberCh,
			getValue_FEDuTCASlot, getValue_FEDVMESpigot,
			getValue_FiberuTCAFiberCh, getValue_FiberVMEFiberCh,
			getValue_SLB, getValue_SLBCh, getValue_SLBSLBCh,
			getValue_FiberuTCATP, getValue_FiberChuTCATP,
			getValue_FiberuTCATPFiberChuTCATP
		};
		getBinType_eid const getBin_functions_eid[nElectronicsQuantityType] = {
			getBin_FED, getBin_FEDuTCA, getBin_FEDVME,
			getBin_Crate, getBin_CrateuTCA, getBin_CrateVME,
			getBin_SlotuTCA, getBin_SlotVME, 
			getBin_Spigot,
			getBin_FiberuTCA, getBin_FiberVME, getBin_FiberCh,
			getBin_FEDuTCASlot, getBin_FEDVMESpigot,
			getBin_FiberuTCAFiberCh, getBin_FiberVMEFiberCh,
			getBin_SLB, getBin_SLBCh, getBin_SLBSLBCh,
			getBin_FiberuTCATP, getBin_FiberChuTCATP,
			getBin_FiberuTCATPFiberChuTCATP
		};
		getEid_eid const getEid_functions_eid[nElectronicsQuantityType] = {
			getEid_FED, getEid_FEDuTCA, getEid_FEDVME,
			getEid_Crate, getEid_CrateuTCA, getEid_CrateVME,
			getEid_SlotuTCA, getEid_SlotVME, 
			getEid_Spigot,
			getEid_FiberuTCA, getEid_FiberVME, getEid_FiberCh,
			getEid_FEDuTCASlot, getEid_FEDVMESpigot,
			getEid_FiberuTCAFiberCh, getEid_FiberVMEFiberCh,
			getEid_SLB, getEid_SLBCh, getEid_SLBSLBCh,
			getEid_FiberuTCATP, getEid_FiberChuTCATP,
			getEid_FiberuTCATPFiberChuTCATP
		};
		getLabels_eid const getLabels_functions_eid[nElectronicsQuantityType] = 
		{
			getLabels_FED, getLabels_FEDuTCA, getLabels_FEDVME,
			getLabels_Crate, getLabels_CrateuTCA, getLabels_CrateVME,
			getLabels_SlotuTCA, getLabels_SlotVME, 
			getLabels_Spigot,
			getLabels_FiberuTCA, getLabels_FiberVME, getLabels_FiberCh,
			getLabels_FEDuTCASlot, getLabels_FEDVMESpigot,
			getLabels_FiberuTCAFiberCh, getLabels_FiberVMEFiberCh,
			getLabels_SLB, getLabels_SLBCh, getLabels_SLBSLBCh,
			getLabels_FiberuTCATP, getLabels_FiberChuTCATP,
			getLabels_FiberuTCATPFiberChuTCATP
		};
		std::string const name_eid[nElectronicsQuantityType] = {
			"FED", "FEDuTCA", "FEDVME", 
			"Crate", "CrateuTCA", "CrateVME",
			"SlotuTCA", "SlotVME", 
			"Spigot",
			"FiberuTCA", "FiberVME", 
			"FiberCh",
			"FEDuTCASlot", "FEDVMESpigot",
			"FiberuTCAFiberCh", "FiberVMEFiberCh",
			"SLB", "SLBCh", "SLB-SLBCh",
			"TPFiber", "TPFiberCh",
			"TPF-TPFCh"
		};
		double const min_eid[nElectronicsQuantityType] = {
			0, 0, 0, 
			0, 0, 0,
			0, 0,
			0,
			0, 0,
			0,
			0, 0,
			0, 0,
			0, 0, 0,
			0, 0,
			0
		};
		double const max_eid[nElectronicsQuantityType] = {
			FED_TOTAL_NUM, FED_uTCA_NUM, FED_VME_NUM,
			CRATE_TOTAL_NUM, CRATE_uTCA_NUM, CRATE_VME_NUM,
			SLOT_uTCA_NUM, SLOT_VME_NUM,
			SPIGOT_NUM,
			FIBER_uTCA_NUM, FIBER_VME_NUM,
			FIBERCH_NUM,
			FED_uTCA_NUM*SLOT_uTCA_NUM, FED_VME_NUM*SPIGOT_NUM,
			FIBER_uTCA_NUM*FIBERCH_NUM, FIBER_VME_NUM*FIBERCH_NUM,
			SLB_NUM, SLBCH_NUM, SLB_NUM*SLBCH_NUM,
			TPFIBER_NUM, TPFIBERCH_NUM,
			TPFIBER_NUM*TPFIBERCH_NUM
		};
		int const nbins_eid[nElectronicsQuantityType] = {
			FED_TOTAL_NUM, FED_uTCA_NUM, FED_VME_NUM,
			CRATE_TOTAL_NUM, CRATE_uTCA_NUM, CRATE_VME_NUM,
			SLOT_uTCA_NUM, SLOT_VME_NUM,
			SPIGOT_NUM,
			FIBER_uTCA_NUM, FIBER_VME_NUM,
			FIBERCH_NUM,
			FED_uTCA_NUM*SLOT_uTCA_NUM, FED_VME_NUM*SPIGOT_NUM,
			FIBER_uTCA_NUM*FIBERCH_NUM, FIBER_VME_NUM*FIBERCH_NUM,
			SLB_NUM, SLBCH_NUM, SLB_NUM*SLBCH_NUM,
			TPFIBER_NUM, TPFIBERCH_NUM,
			TPFIBER_NUM*TPFIBERCH_NUM
		};

		class ElectronicsQuantity : public Quantity
		{
			public:
				ElectronicsQuantity() {}
				ElectronicsQuantity(ElectronicsQuantityType type, 
					bool isLog=false) : 
					Quantity(name_eid[type], isLog), _type(type)
				{}
				virtual ~ElectronicsQuantity() {}
				virtual ElectronicsQuantity* makeCopy()
				{return new ElectronicsQuantity(_type, _isLog);}

				virtual int getValue(HcalElectronicsId const& eid)
				{return getValue_functions_eid[_type](eid);}
				virtual uint32_t getBin(HcalElectronicsId const& eid)
				{return getBin_functions_eid[_type](eid);}

				virtual QuantityType type() {return fElectronicsQuantity;}
				virtual int nbins() {return nbins_eid[_type];}
				virtual double min() {return min_eid[_type];}
				virtual double max() {return max_eid[_type];}
				virtual bool isCoordinate() {return true;}
				virtual std::vector<std::string> getLabels()
				{return getLabels_functions_eid[_type]();}

			protected:
				ElectronicsQuantityType _type;
		};

		//	sorted list of FEDs you want to have on the axis
		class FEDQuantity : public ElectronicsQuantity
		{
			public:
				FEDQuantity() {}
				FEDQuantity(std::vector<int> const& vFEDs) :
					ElectronicsQuantity(fFED, false)
				{this->setup(vFEDs);}
				virtual ~FEDQuantity() {}

				virtual void setup(std::vector<int> const& vFEDs);
				virtual int getValue(HcalElectronicsId const&);
				virtual uint32_t getBin(HcalElectronicsId const&);

				virtual int nbins() {return _feds.size();}
				virtual double min() {return 0;}
				virtual double max() {return _feds.size();}
				virtual std::vector<std::string> getLabels();

			protected:
				typedef boost::unordered_map<int, uint32_t> FEDMap;
				FEDMap _feds;

			public:
				virtual FEDQuantity* makeCopy()
				{
					std::vector<int> vfeds;
					BOOST_FOREACH(FEDMap::value_type &p, _feds)
					{
						vfeds.push_back(p.first);
					}

					//	MUST SORT BEFORE EXITING!
					std::sort(vfeds.begin(), vfeds.end());
					return new FEDQuantity(vfeds);
				}
		};
	}
}

#endif

#ifndef ElectronicsQuantity_h
#define ElectronicsQuantity_h

/**
 *	file:		ElectronicsQuantity.h
 *	Author:		Viktor Khristenko
 */

#include "DQM/HcalCommon/interface/Quantity.h"
#include "boost/unordered_map.hpp"

namespace hcaldqm
{
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
		const std::map<ElectronicsQuantityType, getValueType_eid> getValue_functions_eid = {
			{fFED,getValue_FED},
			{fFEDuTCA,getValue_FEDuTCA},
			{fFEDVME,getValue_FEDVME},
			{fCrate,getValue_Crate},
			{fCrateuTCA,getValue_CrateuTCA},
			{fCrateVME,getValue_CrateVME},
			{fSlotuTCA,getValue_SlotuTCA},
			{fSlotVME,getValue_SlotVME},
			{fSpigot,getValue_Spigot},
			{fFiberuTCA,getValue_FiberuTCA},
			{fFiberVME,getValue_FiberVME},
			{fFiberCh,getValue_FiberCh},
			{fFEDuTCASlot,getValue_FEDuTCASlot},
			{fFEDVMESpigot,getValue_FEDVMESpigot},
			{fFiberuTCAFiberCh,getValue_FiberuTCAFiberCh},
			{fFiberVMEFiberCh,getValue_FiberVMEFiberCh},
			{fSLB,getValue_SLB},
			{fSLBCh,getValue_SLBCh},
			{fSLBSLBCh,getValue_SLBSLBCh},
			{fFiberuTCATP,getValue_FiberuTCATP},
			{fFiberChuTCATP,getValue_FiberChuTCATP},
			{fFiberuTCATPFiberChuTCATP,getValue_FiberuTCATPFiberChuTCATP},
		};
		const std::map<ElectronicsQuantityType, getBinType_eid> getBin_functions_eid = {
			{fFED,getBin_FED},
			{fFEDuTCA,getBin_FEDuTCA},
			{fFEDVME,getBin_FEDVME},
			{fCrate,getBin_Crate},
			{fCrateuTCA,getBin_CrateuTCA},
			{fCrateVME,getBin_CrateVME},
			{fSlotuTCA,getBin_SlotuTCA},
			{fSlotVME,getBin_SlotVME},
			{fSpigot,getBin_Spigot},
			{fFiberuTCA,getBin_FiberuTCA},
			{fFiberVME,getBin_FiberVME},
			{fFiberCh,getBin_FiberCh},
			{fFEDuTCASlot,getBin_FEDuTCASlot},
			{fFEDVMESpigot,getBin_FEDVMESpigot},
			{fFiberuTCAFiberCh,getBin_FiberuTCAFiberCh},
			{fFiberVMEFiberCh,getBin_FiberVMEFiberCh},
			{fSLB,getBin_SLB},
			{fSLBCh,getBin_SLBCh},
			{fSLBSLBCh,getBin_SLBSLBCh},
			{fFiberuTCATP,getBin_FiberuTCATP},
			{fFiberChuTCATP,getBin_FiberChuTCATP},
			{fFiberuTCATPFiberChuTCATP,getBin_FiberuTCATPFiberChuTCATP},
		};
		const std::map<ElectronicsQuantityType, getEid_eid> getEid_functions_eid = {
			{fFED,getEid_FED},
			{fFEDuTCA,getEid_FEDuTCA},
			{fFEDVME,getEid_FEDVME},
			{fCrate,getEid_Crate},
			{fCrateuTCA,getEid_CrateuTCA},
			{fCrateVME,getEid_CrateVME},
			{fSlotuTCA,getEid_SlotuTCA},
			{fSlotVME,getEid_SlotVME},
			{fSpigot,getEid_Spigot},
			{fFiberuTCA,getEid_FiberuTCA},
			{fFiberVME,getEid_FiberVME},
			{fFiberCh,getEid_FiberCh},
			{fFEDuTCASlot,getEid_FEDuTCASlot},
			{fFEDVMESpigot,getEid_FEDVMESpigot},
			{fFiberuTCAFiberCh,getEid_FiberuTCAFiberCh},
			{fFiberVMEFiberCh,getEid_FiberVMEFiberCh},
			{fSLB,getEid_SLB},
			{fSLBCh,getEid_SLBCh},
			{fSLBSLBCh,getEid_SLBSLBCh},
			{fFiberuTCATP,getEid_FiberuTCATP},
			{fFiberChuTCATP,getEid_FiberChuTCATP},
			{fFiberuTCATPFiberChuTCATP,getEid_FiberuTCATPFiberChuTCATP},
		};
		const std::map<ElectronicsQuantityType, getLabels_eid> getLabels_functions_eid = {
			{fFED,getLabels_FED},
			{fFEDuTCA,getLabels_FEDuTCA},
			{fFEDVME,getLabels_FEDVME},
			{fCrate,getLabels_Crate},
			{fCrateuTCA,getLabels_CrateuTCA},
			{fCrateVME,getLabels_CrateVME},
			{fSlotuTCA,getLabels_SlotuTCA},
			{fSlotVME,getLabels_SlotVME},
			{fSpigot,getLabels_Spigot},
			{fFiberuTCA,getLabels_FiberuTCA},
			{fFiberVME,getLabels_FiberVME},
			{fFiberCh,getLabels_FiberCh},
			{fFEDuTCASlot,getLabels_FEDuTCASlot},
			{fFEDVMESpigot,getLabels_FEDVMESpigot},
			{fFiberuTCAFiberCh,getLabels_FiberuTCAFiberCh},
			{fFiberVMEFiberCh,getLabels_FiberVMEFiberCh},
			{fSLB,getLabels_SLB},
			{fSLBCh,getLabels_SLBCh},
			{fSLBSLBCh,getLabels_SLBSLBCh},
			{fFiberuTCATP,getLabels_FiberuTCATP},
			{fFiberChuTCATP,getLabels_FiberChuTCATP},
			{fFiberuTCATPFiberChuTCATP,getLabels_FiberuTCATPFiberChuTCATP},
		};
		const std::map<ElectronicsQuantityType, std::string> name_eid = {
			{fFED,"FED"},
			{fFEDuTCA,"FEDuTCA"},
			{fFEDVME,"FEDVME"},
			{fCrate,"Crate"},
			{fCrateuTCA,"CrateuTCA"},
			{fCrateVME,"CrateVME"},
			{fSlotuTCA,"SlotuTCA"},
			{fSlotVME,"SlotVME"},
			{fSpigot,"Spigot"},
			{fFiberuTCA,"FiberuTCA"},
			{fFiberVME,"FiberVME"},
			{fFiberCh,"FiberCh"},
			{fFEDuTCASlot,"FEDuTCASlot"},
			{fFEDVMESpigot,"FEDVMESpigot"},
			{fFiberuTCAFiberCh,"FiberuTCAFiberCh"},
			{fFiberVMEFiberCh,"FiberVMEFiberCh"},
			{fSLB,"SLB"},
			{fSLBCh,"SLBCh"},
			{fSLBSLBCh,"SLB-SLBCh"},
			{fFiberuTCATP,"TPFiber"},
			{fFiberChuTCATP,"TPFiberCh"},
			{fFiberuTCATPFiberChuTCATP,"TPF-TPFCh"},
		};
		const std::map<ElectronicsQuantityType, double> min_eid = {
			{fFED,-0.5},
			{fFEDuTCA,-0.5},
			{fFEDVME,-0.5},
			{fCrate,-0.5},
			{fCrateuTCA,-0.5},
			{fCrateVME,-0.5},
			{fSlotuTCA,0.},
			{fSlotVME,0.},
			{fSpigot,0.},
			{fFiberuTCA,0.},
			{fFiberVME,0.},
			{fFiberCh,0.},
			{fFEDuTCASlot,0.},
			{fFEDVMESpigot,0.},
			{fFiberuTCAFiberCh,0.},
			{fFiberVMEFiberCh,0.},
			{fSLB,0.},
			{fSLBCh,0.},
			{fSLBSLBCh,0.},
			{fFiberuTCATP,0.},
			{fFiberChuTCATP,0.},
			{fFiberuTCATPFiberChuTCATP,0.},
		};
		const std::map<ElectronicsQuantityType, double> max_eid = {
			{fFED,constants::fedList.size() - 0.5},
			{fFEDuTCA,constants::fedListuTCA.size() - 0.5},
			{fFEDVME,constants::fedListVME.size() - 0.5},
			{fCrate,constants::crateList.size() - 0.5},
			{fCrateuTCA,constants::crateListuTCA.size() - 0.5},
			{fCrateVME,constants::crateListVME.size() - 0.5},
			{fSlotuTCA,constants::SLOT_uTCA_NUM},
			{fSlotVME,constants::SLOT_VME_NUM},
			{fSpigot,constants::SPIGOT_NUM},
			{fFiberuTCA,constants::FIBER_uTCA_NUM},
			{fFiberVME,constants::FIBER_VME_NUM},
			{fFiberCh,constants::FIBERCH_NUM},
			{fFEDuTCASlot,constants::FED_uTCA_NUM*constants::SLOT_uTCA_NUM},
			{fFEDVMESpigot,constants::FED_VME_NUM*constants::SPIGOT_NUM},
			{fFiberuTCAFiberCh,constants::FIBER_uTCA_NUM*constants::FIBERCH_NUM},
			{fFiberVMEFiberCh,constants::FIBER_VME_NUM*constants::FIBERCH_NUM},
			{fSLB,constants::SLB_NUM},
			{fSLBCh,constants::SLBCH_NUM},
			{fSLBSLBCh,constants::SLB_NUM*constants::SLBCH_NUM},
			{fFiberuTCATP,constants::TPFIBER_NUM},
			{fFiberChuTCATP,constants::TPFIBERCH_NUM},
			{fFiberuTCATPFiberChuTCATP,constants::TPFIBER_NUM*constants::TPFIBERCH_NUM},
		};
		const std::map<ElectronicsQuantityType, double> nbins_eid = {
			{fFED,constants::fedList.size()},
			{fFEDuTCA,constants::fedListuTCA.size()},
			{fFEDVME,constants::fedListVME.size()},
			{fCrate,constants::crateList.size()},
			{fCrateuTCA,constants::crateListuTCA.size()},
			{fCrateVME,constants::crateListVME.size()},
			{fSlotuTCA,constants::SLOT_uTCA_NUM},
			{fSlotVME,constants::SLOT_VME_NUM},
			{fSpigot,constants::SPIGOT_NUM},
			{fFiberuTCA,constants::FIBER_uTCA_NUM},
			{fFiberVME,constants::FIBER_VME_NUM},
			{fFiberCh,constants::FIBERCH_NUM},
			{fFEDuTCASlot,constants::FED_uTCA_NUM*constants::SLOT_uTCA_NUM},
			{fFEDVMESpigot,constants::FED_VME_NUM*constants::SPIGOT_NUM},
			{fFiberuTCAFiberCh,constants::FIBER_uTCA_NUM*constants::FIBERCH_NUM},
			{fFiberVMEFiberCh,constants::FIBER_VME_NUM*constants::FIBERCH_NUM},
			{fSLB,constants::SLB_NUM},
			{fSLBCh,constants::SLBCH_NUM},
			{fSLBSLBCh,constants::SLB_NUM*constants::SLBCH_NUM},
			{fFiberuTCATP,constants::TPFIBER_NUM},
			{fFiberChuTCATP,constants::TPFIBERCH_NUM},
			{fFiberuTCATPFiberChuTCATP,constants::TPFIBER_NUM*constants::TPFIBERCH_NUM},
		};

		class ElectronicsQuantity : public Quantity
		{
			public:
				ElectronicsQuantity() {}
				ElectronicsQuantity(ElectronicsQuantityType type, 
					bool isLog=false) : 
					Quantity(name_eid.at(type), isLog), _type(type)
				{}
				~ElectronicsQuantity() override {}
				ElectronicsQuantity* makeCopy() override
				{return new ElectronicsQuantity(_type, _isLog);}

				int getValue(HcalElectronicsId const& eid) override
				{return getValue_functions_eid.at(_type)(eid);}
				uint32_t getBin(HcalElectronicsId const& eid) override
				{return getBin_functions_eid.at(_type)(eid);}

				QuantityType type() override {return fElectronicsQuantity;}
				int nbins() override {return nbins_eid.at(_type);}
				double min() override {return min_eid.at(_type);}
				double max() override {return max_eid.at(_type);}
				bool isCoordinate() override {return true;}
				std::vector<std::string> getLabels() override
				{return getLabels_functions_eid.at(_type)();}

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
				~FEDQuantity() override {}

				virtual void setup(std::vector<int> const& vFEDs);
				int getValue(HcalElectronicsId const&) override;
				uint32_t getBin(HcalElectronicsId const&) override;

				int nbins() override {return _feds.size();}
				double min() override {return 0;}
				double max() override {return _feds.size();}
				std::vector<std::string> getLabels() override;

			protected:
				typedef boost::unordered_map<int, uint32_t> FEDMap;
				FEDMap _feds;

			public:
				FEDQuantity* makeCopy() override
				{
					std::vector<int> vfeds;
					for(auto const& p : _feds)
					{
						vfeds.push_back(p.first);
					}

					//	MUST SORT BEFORE EXITING!
					std::sort(vfeds.begin(), vfeds.end());
					return new FEDQuantity(vfeds);
				}
		};

		// Crate quantity, initialized from emap (because it is not easy to turn a VME crate in an EID)
		class CrateQuantity : public ElectronicsQuantity {
			typedef std::map<int, uint32_t> CrateHashMap;
		public:
			CrateQuantity() {}
			CrateQuantity(HcalElectronicsMap const * emap) : ElectronicsQuantity(fCrate, false) {
				this->setup(emap);
			}
			CrateQuantity(std::vector<int> crates, CrateHashMap crateHashes) : ElectronicsQuantity(fCrate, false) {
				this->setup(crates, crateHashes);
			}
			~CrateQuantity() override {}

			virtual void setup(HcalElectronicsMap const * emap);
			virtual void setup(std::vector<int> crates, CrateHashMap crateHashes);
			int getValue(HcalElectronicsId const&) override;
			uint32_t getBin(HcalElectronicsId const&) override;

			int nbins() override {
				return _crates.size();
			}
			double min() override {return 0;}
			double max() override {return _crates.size();}
			std::vector<std::string> getLabels() override;

		protected:
			std::vector<int> _crates;
			CrateHashMap _crateHashes;

			public:
				CrateQuantity* makeCopy() override
				{
					// Make copies of the crate info
					std::vector<int> tmpCrates;
					std::map<int, uint32_t> tmpCrateHashes;
					for (auto& it_crate : _crates) {
						tmpCrates.push_back(it_crate);
						tmpCrateHashes[it_crate] = _crateHashes[it_crate];
					}
					return new CrateQuantity(tmpCrates, tmpCrateHashes);
				}

		};
	}
}

#endif

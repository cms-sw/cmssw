#include "DQM/HcalCommon/interface/HashFunctions.h"
#include "DQM/HcalCommon/interface/Utilities.h"

namespace hcaldqm
{
	using namespace constants;
	namespace hashfunctions
	{
		/**
		 *	HcalDetId
		 */
		uint32_t hash_Subdet(HcalDetId const& did)
		{
			return utilities::hash(HcalDetId(did.subdet(), 1, 1, 1));
		}

		uint32_t hash_Subdetiphi(HcalDetId const& did)
		{
			return utilities::hash(HcalDetId(did.subdet(), 1, did.iphi(), 1));
		}

		uint32_t hash_Subdetieta(HcalDetId const& did)
		{
			return utilities::hash(HcalDetId(did.subdet(), did.ieta(),
				1, 1));
		}

		uint32_t hash_Subdetdepth(HcalDetId const& did)
		{
			return utilities::hash(HcalDetId(did.subdet(), 1,
				1, did.depth()));
		}

		uint32_t hash_SubdetPM(HcalDetId const& did)
		{
			return utilities::hash(HcalDetId(did.subdet(),
				did.ieta()>0 ? 1 : -1, 1, 1));
		}

		uint32_t hash_SubdetPMiphi(HcalDetId const& did)
		{
			return utilities::hash(HcalDetId(did.subdet(),
				did.ieta()>0 ? 1 : -1, did.iphi(), 1));
		}

		uint32_t hash_iphi(HcalDetId const& did)
		{
			return utilities::hash(HcalDetId(HcalBarrel,
				1, did.iphi(), 1));
		}

		uint32_t hash_ieta(HcalDetId const& did)
		{
			return utilities::hash(HcalDetId(HcalBarrel,
				did.ieta(), 1, 1));
		}

		uint32_t hash_depth(HcalDetId const& did)
		{
			return utilities::hash(HcalDetId(HcalBarrel,
				1, 1, did.depth()));
		}

		uint32_t hash_HFPMiphi(HcalDetId const& did)
		{
			return utilities::hash(HcalDetId(HcalForward,
				did.ieta()>0 ? 1 : -1, did.iphi(), 1));
		}

		uint32_t hash_HBHEPartition(HcalDetId const &did)
		{
			int iphi = did.iphi();
			uint32_t hash = 0;
			if (iphi>=3 && iphi<=26)
				hash = utilities::hash(HcalDetId(HcalBarrel,
					1, 3, 1));
			else if (iphi>=27 && iphi<=50)
				hash = utilities::hash(HcalDetId(HcalBarrel,
					1, 27, 1));
			else
				hash = utilities::hash(HcalDetId(HcalBarrel,
					1, 1, 1));

			return hash;
		}

		uint32_t hash_DChannel(HcalDetId const& did)
		{
			return utilities::hash(did);
		}

		std::string name_Subdet(HcalDetId const& did)
		{
			return constants::SUBDET_NAME[did.subdet()-1];
		}

		std::string name_SubdetPM(HcalDetId const& did)
		{
			char name[10];
			sprintf(name, "%s%s",constants::SUBDET_NAME[did.subdet()-1].c_str(),
				did.ieta()>0 ? "P" : "M");
			return std::string(name);
		}

		std::string name_Subdetiphi(HcalDetId const& did)
		{
			char name[10];
			sprintf(name, "%siphi%d", 
				constants::SUBDET_NAME[did.subdet()-1].c_str(),
				did.iphi());
			return std::string(name);
		}

		std::string name_Subdetieta(HcalDetId const& did)
		{
			char name[20];
			sprintf(name, "%sieta%d", 
				constants::SUBDET_NAME[did.subdet()-1].c_str(),
				did.ieta());
			return std::string(name);
		}

		std::string name_Subdetdepth(HcalDetId const& did)
		{
			char name[20];
			sprintf(name, "%sdepth%d", 
				constants::SUBDET_NAME[did.subdet()-1].c_str(),
				did.depth());
			return std::string(name);
		}

		std::string name_SubdetPMiphi(HcalDetId const& did)
		{
			char name[20];
			sprintf(name, "%s%siphi%d", 
				constants::SUBDET_NAME[did.subdet()-1].c_str(), 
				did.ieta()>0 ? "P" : "M", did.iphi());
			return std::string(name);
		}

		std::string name_iphi(HcalDetId const& did)
		{
			char name[10];
			sprintf(name, "iphi%d", did.iphi());
			return std::string(name);
		}

		std::string name_ieta(HcalDetId const& did)
		{
			char name[10];
			sprintf(name, "ieta%d", did.ieta());
			return std::string(name);
		}

		std::string name_depth(HcalDetId const& did)
		{
			char name[10];
			sprintf(name, "depth%d", did.depth());
			return std::string(name);

		}

		std::string name_HFPMiphi(HcalDetId const& did)
		{
			char name[10];
			sprintf(name, "HF%siphi%d", did.ieta()>0 ? "P" : "M", did.iphi());
			return std::string(name);
		}

		std::string name_HBHEPartition(HcalDetId const& did)
		{
			char c;
			if (did.iphi()>=3 && did.iphi()<=26)
				c = 'a';
			else if (did.iphi()>=27 && did.iphi()<=50)
				c = 'b';
			else
				c = 'c';
			char name[10];
			sprintf(name, "HBHE%c", c);
			return std::string(name);
		}

		std::string name_DChannel(HcalDetId const& did)
		{
			char name[40];
			sprintf(name, "%s-%d-%d-%d",
				constants::SUBDET_NAME[did.subdet()-1].c_str(), 
				did.ieta(), did.iphi(), did.depth());
			return std::string(name);
		}

		/**
		 *	by ElectronicsId
		 */
		uint32_t hash_FED(HcalElectronicsId const& eid)
		{
			return eid.isVMEid() ?
				utilities::hash(HcalElectronicsId(
					FIBERCH_MIN, FIBER_VME_MIN, SPIGOT_MIN, eid.dccid())) :
				utilities::hash(HcalElectronicsId(eid.crateId(),
					SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false));
		}

		uint32_t hash_FEDSpigot(HcalElectronicsId const& eid)
		{
			//	note that hashing of uTCA is done by FED-Slot...
			return eid.isVMEid() ?
				utilities::hash(HcalElectronicsId(
					FIBERCH_MIN, FIBER_VME_MIN, eid.spigot(), eid.dccid())) : 
				utilities::hash(HcalElectronicsId(eid.crateId(),
					eid.slot(), FIBER_uTCA_MIN1, FIBERCH_MIN, false));
		}

		uint32_t hash_FEDSlot(HcalElectronicsId const& eid)
		{
			//	note that hashing of VME is done with 
			return eid.isVMEid() ? 
				utilities::hash(HcalElectronicsId(FIBERCH_MIN,
					FIBER_VME_MIN, eid.spigot(), eid.dccid())) :
				utilities::hash(HcalElectronicsId(eid.crateId(),
					eid.slot(), FIBER_uTCA_MIN1, FIBERCH_MIN, false));
		}

		uint32_t hash_Crate(HcalElectronicsId const& eid)
		{
			//	note hashing of VME is done with dccId
			return eid.isVMEid() ? 
				utilities::hash(HcalElectronicsId(FIBERCH_MIN,
					FIBER_VME_MIN, SPIGOT_MIN, eid.dccid())) :
				utilities::hash(HcalElectronicsId(eid.crateId(),
					SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false));
		}

		uint32_t hash_CrateSpigot(HcalElectronicsId const& eid)
		{
			//	note hashing of VME is done with dccid and
			//	uTCA with Slots
			return eid.isVMEid() ?
				utilities::hash(HcalElectronicsId(FIBERCH_MIN,
					FIBER_VME_MIN, eid.spigot(), eid.dccid())) :
				utilities::hash(HcalElectronicsId(eid.crateId(),
					eid.slot(), FIBER_uTCA_MIN1, FIBERCH_MIN, false));
		}

		uint32_t hash_CrateSlot(HcalElectronicsId const& eid)
		{
			return eid.isVMEid() ? 
				utilities::hash(HcalElectronicsId(FIBERCH_MIN,
					FIBER_VME_MIN, eid.spigot(), eid.dccid())) :
				utilities::hash(HcalElectronicsId(eid.crateId(),
					eid.slot(), FIBER_uTCA_MIN1, FIBERCH_MIN, false));
		}

		uint32_t hash_Fiber(HcalElectronicsId const&)
		{
			return 0;
		}

		uint32_t hash_FiberFiberCh(HcalElectronicsId const&)
		{
			return 0;
		}

		uint32_t hash_FiberCh(HcalElectronicsId const& eid)
		{
			return 0;
		}

		uint32_t hash_Electronics(HcalElectronicsId const& eid)
		{
			return eid.isVMEid()?
				utilities::hash(HcalElectronicsId(FIBERCH_MIN,
					FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN)):
				utilities::hash(HcalElectronicsId(CRATE_uTCA_MIN,
					SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false));
/*			NOTE: as an update - should separate Trigger Eid and Det Eid
 *			return eid.isVMEid() ?
				eid.isTriggerChainId()?
					utilities::hash(HcalElectronicsId(SLBCH_MIN,
						SLB_MIN, SPIGOT_MIN, CRATE_VME_MIN, 
						CRATE_VME_MIN, SLOT_VME_MIN1, 0)):
					utilities::hash(HcalElectronicsId(FIBERCH_MIN,
						FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN))
				:
				eid.isTriggerChainId()?
				utilities::hash(HcalElectronicsId(CRATE_uTCA_MIN,
					SLOT_uTCA_MIN, TPFIBER_MIN, TPFIBERCH_MIN, true)):
				utilities::hash(HcalElectronicsId(CRATE_uTCA_MIN,
					SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false));
					*/
		}

		uint32_t hash_EChannel(HcalElectronicsId const& eid)
		{
			return eid.isVMEid() ?
				utilities::hash(HcalElectronicsId(eid.fiberChanId(),
					eid.fiberIndex(), eid.spigot(), eid.dccid())):
				utilities::hash(HcalElectronicsId(eid.crateId(),
					eid.slot(), eid.fiberIndex(), eid.fiberChanId(), false));
		}

		std::string name_FED(HcalElectronicsId const& eid)
		{
			char name[10];
			sprintf(name, "FED%d", eid.isVMEid() ? eid.dccid()+700 :
				utilities::crate2fed(eid.crateId()));
			return std::string(name);
		}

		std::string name_FEDSpigot(HcalElectronicsId const& eid)
		{
			char name[20];
			sprintf(name, "FED%dS%d",
				eid.isVMEid()?eid.dccid()+700:
				utilities::crate2fed(eid.crateId()),
				eid.isVMEid()?eid.spigot():eid.slot());
			return std::string(name);
		}

		std::string name_FEDSlot(HcalElectronicsId const& eid)
		{
			char name[20];
			sprintf(name, "FED%dS%d",
				eid.isVMEid()?eid.dccid()+700:
				utilities::crate2fed(eid.crateId()),
				eid.isVMEid()?eid.spigot():eid.slot());
			return std::string(name);
		}

		std::string name_Crate(HcalElectronicsId const& eid)
		{
			char name[10];
			sprintf(name, "Crate%d", eid.isVMEid()?eid.dccid():eid.crateId());
			return std::string(name);
		}

		std::string name_CrateSpigot(HcalElectronicsId const& eid)
		{
			char name[20];
			sprintf(name, "Crate%dS%d",
				eid.isVMEid()?eid.dccid():eid.crateId(),
				eid.isVMEid()?eid.spigot():eid.slot());
			return std::string(name);
		}

		std::string name_CrateSlot(HcalElectronicsId const& eid)
		{
			char name[20];
			sprintf(name, "Crate%dS%d",
				eid.isVMEid()?eid.dccid():eid.crateId(),
				eid.isVMEid()?eid.spigot():eid.slot());
			return std::string(name);
		}

		std::string name_Fiber(HcalElectronicsId const&)
		{
			return "None";
		}

		std::string name_FiberFiberCh(HcalElectronicsId const&)
		{
			return "None";
		}

		std::string name_FiberCh(HcalElectronicsId const&)
		{
			return "None";
		}

		std::string name_Electronics(HcalElectronicsId const& eid)
		{
			return eid.isVMEid()?std::string("VME"):std::string("uTCA");
		}

		std::string name_EChannel(HcalElectronicsId const& eid)
		{
			char name[20];
			if (eid.isVMEid())
				sprintf(name, "%d-%d-%d-%d", eid.dccid(),
					eid.spigot(), eid.fiberIndex(), eid.fiberChanId());
			else
				sprintf(name, "%d-%d-%d-%d", eid.crateId(),
					eid.slot(), eid.fiberIndex(), eid.fiberChanId());
			return std::string(name);
		}

		/**
		 *	by TrigTowerDetId
		 */
		uint32_t hash_TTSubdet(HcalTrigTowerDetId const& tid)
		{
			return utilities::hash(HcalTrigTowerDetId(
				tid.ietaAbs()>=29?29:1, 1));
		}

		uint32_t hash_TTSubdetPM(HcalTrigTowerDetId const& tid)
		{
			return utilities::hash(HcalTrigTowerDetId(
				tid.ietaAbs()<29? (tid.ieta()<0 ? -1 : 1) : 
				(tid.ieta()<0?-29:29),
				1));
		}

		uint32_t hash_TTSubdetPMiphi(HcalTrigTowerDetId const& tid)
		{
			return utilities::hash(HcalTrigTowerDetId(
				tid.ietaAbs()<29? (tid.ieta()<0 ? 0 : 1) : (tid.ieta()<0?2:3),
				tid.iphi()));
		}

		uint32_t hash_TTSubdetieta(HcalTrigTowerDetId const& tid)
		{
			return 0;
		}

		uint32_t hash_TTdepth(HcalTrigTowerDetId const& tid)
		{
			return utilities::hash(HcalTrigTowerDetId(
				1, 1, tid.depth()));
		}

		uint32_t hash_TChannel(HcalTrigTowerDetId const& tid)
		{
			return utilities::hash(HcalTrigTowerDetId(
				tid.ieta(), tid.iphi(), tid.depth()));
		}

		std::string name_TTSubdet(HcalTrigTowerDetId const& tid)
		{
			return constants::TPSUBDET_NAME[tid.ietaAbs()<29?0:1];
		}

		std::string name_TTSubdetPM(HcalTrigTowerDetId const& tid)
		{
			return constants::TPSUBDETPM_NAME[
				tid.ietaAbs()<29?(tid.ieta()<0?0:1):(tid.ieta()<0?2:3)];
		}

		std::string name_TTSubdetPMiphi(HcalTrigTowerDetId const& tid)
		{
			char name[10];
			sprintf(name, "%siphi%d", name_TTSubdetPM(tid).c_str(),
				tid.iphi());
			return std::string(name);
		}

		std::string name_TTSubdetieta(HcalTrigTowerDetId const& tid)
		{
			return "None";
		}

		std::string name_TTdepth(HcalTrigTowerDetId const& tid)
		{
			char name[10];
			sprintf(name, "depth%d", tid.depth());
			return std::string(name);
		}

		std::string name_TChannel(HcalTrigTowerDetId const& tid)
		{
			std::ostringstream stream;
			stream << tid;
			return std::string(stream.str());
		}
	}
}

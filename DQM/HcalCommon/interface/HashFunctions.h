#ifndef HashFunctions_h
#define HashFunctions_h

/**
 *	file:			HashFunctions.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *		Hash accoring to various Detector Elements
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/Constants.h"

namespace hcaldqm
{
	using namespace constants;
	namespace hashfunctions
	{
		/**
		 *	by HcalDetId
		 */
		uint32_t hash_Subdet(HcalDetId const&);
		uint32_t hash_Subdetiphi(HcalDetId const&);
		uint32_t hash_Subdetieta(HcalDetId const&);
		uint32_t hash_Subdetdepth(HcalDetId const&);
		uint32_t hash_SubdetPM(HcalDetId const&);
		uint32_t hash_SubdetPMiphi(HcalDetId const&);
		uint32_t hash_iphi(HcalDetId const&);
		uint32_t hash_ieta(HcalDetId const&);
		uint32_t hash_depth(HcalDetId const&);
		uint32_t hash_HFPMiphi(HcalDetId const&);
		uint32_t hash_HBHEPartition(HcalDetId const&);
		uint32_t hash_DChannel(HcalDetId const&);

		/**
		 *	by ElectronicsId
		 */
		uint32_t hash_FED(HcalElectronicsId const&);
		uint32_t hash_FEDSpigot(HcalElectronicsId const&);
		uint32_t hash_FEDSlot(HcalElectronicsId const&);
		uint32_t hash_Crate(HcalElectronicsId const&);
		uint32_t hash_CrateSpigot(HcalElectronicsId const&);
		uint32_t hash_CrateSlot(HcalElectronicsId const&);
		uint32_t hash_Fiber(HcalElectronicsId const&);
		uint32_t hash_FiberFiberCh(HcalElectronicsId const&);
		uint32_t hash_FiberCh(HcalElectronicsId const&);
		uint32_t hash_Electronics(HcalElectronicsId const&);
		uint32_t hash_EChannel(HcalElectronicsId const&);

		/**
		 *	by TrigTowerDetId
		 */
		uint32_t hash_TTSubdet(HcalTrigTowerDetId const&);
		uint32_t hash_TTSubdetPM(HcalTrigTowerDetId const&);
		uint32_t hash_TTSubdetPMiphi(HcalTrigTowerDetId const&);
		uint32_t hash_TTSubdetieta(HcalTrigTowerDetId const&);
		uint32_t hash_TTdepth(HcalTrigTowerDetId const&);
		uint32_t hash_TChannel(HcalTrigTowerDetId const&);

		std::string name_Subdet(HcalDetId const&);
		std::string name_Subdetiphi(HcalDetId const&);
		std::string name_Subdetieta(HcalDetId const&);
		std::string name_Subdetdepth(HcalDetId const&);
		std::string name_SubdetPM(HcalDetId const&);
		std::string name_SubdetPMiphi(HcalDetId const&);
		std::string name_iphi(HcalDetId const&);
		std::string name_ieta(HcalDetId const&);
		std::string name_depth(HcalDetId const&);
		std::string name_HFPMiphi(HcalDetId const&);
		std::string name_HBHEPartition(HcalDetId const&);
		std::string name_DChannel(HcalDetId const&);

		uint32_t hash_Subdet(std::string const&);
		uint32_t hash_Subdetiphi(std::string const&);
		uint32_t hash_Subdetieta(std::string const&);
		uint32_t hash_Subdetdepth(std::string const&);
		uint32_t hash_SubdetPM(std::string const&);
		uint32_t hash_SubdetPMiphi(std::string const&);
		uint32_t hash_iphi(std::string const&);
		uint32_t hash_ieta(std::string const&);
		uint32_t hash_depth(std::string const&);
		uint32_t hash_HFPMiphi(std::string const&);
		uint32_t hash_HBHEPartition(std::string const&);
		uint32_t hash_DChannel(std::string const&);

		std::string name_FED(HcalElectronicsId const&);
		std::string name_FEDSpigot(HcalElectronicsId const&);
		std::string name_FEDSlot(HcalElectronicsId const&);
		std::string name_Crate(HcalElectronicsId const&);
		std::string name_CrateSpigot(HcalElectronicsId const&);
		std::string name_CrateSlot(HcalElectronicsId const&);
		std::string name_Fiber(HcalElectronicsId const&);
		std::string name_FiberFiberCh(HcalElectronicsId const&);
		std::string name_FiberCh(HcalElectronicsId const&);
		std::string name_Electronics(HcalElectronicsId const&);
		std::string name_EChannel(HcalElectronicsId const&);

		uint32_t hash_FED(std::string const&);
		uint32_t hash_FEDSpigot(std::string const&);
		uint32_t hash_FEDSlot(std::string const&);
		uint32_t hash_Crate(std::string const&);
		uint32_t hash_CrateSpigot(std::string const&);
		uint32_t hash_CrateSlot(std::string  const&);
		uint32_t hash_Fiber(std::string const&);
		uint32_t hash_FiberFiberCh(std::string const&);
		uint32_t hash_FiberCh(std::string const&);
		uint32_t hash_Electronics(std::string const&);
		uint32_t hash_EChannel(std::string const&);

		std::string name_TTSubdet(HcalTrigTowerDetId const&);
		std::string name_TTSubdetPM(HcalTrigTowerDetId const&);
		std::string name_TTSubdetPMiphi(HcalTrigTowerDetId const&);
		std::string name_TTSubdetieta(HcalTrigTowerDetId const&);
		std::string name_TTdepth(HcalTrigTowerDetId const&);
		std::string name_TChannel(HcalTrigTowerDetId const&);

		uint32_t hash_TTSubdet(std::string const&);
		uint32_t hash_TTSubdetPM(std::string const&);
		uint32_t hash_TTSubdetPMiphi(std::string const&);
		uint32_t hash_TTSubdetieta(std::string const&);
		uint32_t hash_TTdepth(std::string const&);
		uint32_t hash_TChannel(std::string const&);

		enum HashType
		{
			fSubdet = 0,
			fSubdetiphi = 1,
			fSubdetieta = 2,
			fSubdetdepth = 3,
			fSubdetPM = 4,
			fSubdetPMiphi = 5,
			fiphi = 6,
			fieta = 7,
			fdepth = 8,
			fHFPMiphi = 9,
			fHBHEPartition = 10,
			fDChannel = 11,
			nHashType_did = 12,
			fFED = 13,
			fFEDSpigot = 14,
			fFEDSlot = 15,
			fCrate = 16,
			fCrateSpigot = 17,
			fCrateSlot = 18,
			fFiber = 19,
			fFiberFiberCh = 20,
			fFiberCh = 21,
			fElectronics = 22,
			fEChannel = 23, 
			nHashType_eid = 24,
			fTTSubdet = 25,
			fTTSubdetPM = 26,
			fTTSubdetPMiphi = 27,
			fTTSubdetieta = 28,
			fTTdepth = 29,
			fTChannel = 30,
			nHashType_tid = 31,
			nHashType = 32
		};
		typedef uint32_t (*hash_function_did)(HcalDetId const&);
		typedef uint32_t (*hash_function_eid)(HcalElectronicsId const&);
		typedef uint32_t (*hash_function_tid)(HcalTrigTowerDetId const&);
		typedef std::string (*name_function_did)(HcalDetId const&);
		typedef std::string (*name_function_eid)(HcalElectronicsId const&);
		typedef std::string (*name_function_tid)(HcalTrigTowerDetId const&);
		hash_function_did const hash_did[nHashType_did] = {
			hash_Subdet, hash_Subdetiphi, hash_Subdetieta, 
			hash_Subdetdepth, hash_SubdetPM, hash_SubdetPMiphi,
			hash_iphi, hash_ieta, hash_depth, hash_HFPMiphi, 
			hash_HBHEPartition, hash_DChannel
		};
		hash_function_eid const hash_eid[nHashType_eid-nHashType_did-1] = {
			hash_FED, hash_FEDSpigot, hash_FEDSlot, 
			hash_Crate, hash_CrateSpigot, hash_CrateSlot,
			hash_Fiber, hash_FiberFiberCh, hash_FiberCh,
			hash_Electronics, hash_EChannel
		};
		hash_function_tid const hash_tid[nHashType_tid-nHashType_eid-1] = {
			hash_TTSubdet, hash_TTSubdetPM, hash_TTSubdetPMiphi, 
			hash_TTSubdetieta, hash_TTdepth, hash_TChannel
		};
		name_function_did const name_did[nHashType_did] = {
			name_Subdet, name_Subdetiphi, name_Subdetieta, 
			name_Subdetdepth, name_SubdetPM, name_SubdetPMiphi,
			name_iphi, name_ieta, name_depth, name_HFPMiphi, 
			name_HBHEPartition, name_DChannel
		};
		name_function_eid const name_eid[nHashType_eid-nHashType_did-1] = {
			name_FED, name_FEDSpigot, name_FEDSlot,
			name_Crate, name_CrateSpigot, name_CrateSlot,
			name_Fiber, name_FiberFiberCh, name_FiberCh,
			name_Electronics, name_EChannel
		};
		name_function_tid const name_tid[nHashType_tid-nHashType_eid-1] = {
			name_TTSubdet, name_TTSubdetPM, name_TTSubdetPMiphi,
			name_TTSubdetieta, name_TTdepth, name_TChannel
		};
		int const nhashes = nHashType_did + (nHashType_eid-nHashType_did-1) + 
			(nHashType_tid-nHashType_eid-1);
		std::string const hash_names[nhashes] = {
			"Subdet", "Subdetiphi", "Subdetieta", "Subdetdepth",
			"SubdetPM", "SubdetPMiphi", "iphi", "ieta", "depth",
			"HFPMiphi", "HBHEPartition", "DChannel",

			"FED", "FEDSpigot", "FEDSlot",
			"Crate", "CrateSpigot", "CrateSlot",
			"Fiber", "FiberFiberCh", "FiberCh",
			"Electronics", "EChannel",

			"TTSubdet", "TTSubdetPM", "TTSubdetPMiphi",
			"TTSubdetieta", "TTdepth", "TChannel"
		};
	}
}

#endif

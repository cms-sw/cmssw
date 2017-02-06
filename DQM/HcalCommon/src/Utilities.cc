#include "DQM/HcalCommon/interface/Utilities.h"
#include <utility>

namespace hcaldqm
{
	using namespace constants;
	namespace utilities
	{

		/*
		 *	Useful Detector Functions. For Fast Detector Validity Check
		 */
        std::pair<uint16_t, uint16_t> fed2crate(int fed)
		{
			fed-=1100;
			if (fed>=constants::FED_uTCA_MAX_REAL)
				throw cms::Exception("HCALDQM")
					<< "fed2crate::fed index is out of range " 
					<< fed;

            //  uTCA Crate is split in half
            uint16_t slot = fed%2==0 ? SLOT_uTCA_MIN : SLOT_uTCA_MIN+6;
			return std::make_pair<uint16_t const, uint16_t const>((uint16_t const)constants::FED2CRATE[fed],
                (uint16_t const)slot);
		}

	  uint16_t crate2fed(int crate, int slot)
		{
			//	 for the details see Constants.h
			if (crate>=constants::FED_uTCA_MAX_REAL)
				throw cms::Exception("HCALDQM")
					<< "crate2fed::crate index is out of range "
					<< crate;
			int fed = constants::CRATE2FED[crate];
			if (slot > 6 && (crate == 22 || crate == 29 || crate == 32))  //needed to handle dual fed readout
			  ++fed;
			
			return fed;
		}

		uint32_t hash(HcalDetId const& did)
		{
			return did.rawId();
		}
		uint32_t hash(HcalElectronicsId const& eid)
		{
			return eid.rawId();
		}
		uint32_t hash(HcalTrigTowerDetId const& tid)
		{
			return tid.rawId();
		}

		std::vector<int> getFEDList(HcalElectronicsMap const* emap)
		{
			std::vector<int> vfeds;
			std::vector<HcalElectronicsId> vids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator
				it=vids.begin(); it!=vids.end(); ++it)
			{
				int fed = it->isVMEid()?it->dccid()+FED_VME_MIN:
				  crate2fed(it->crateId(),it->slot());
				uint32_t n=0;
				for (std::vector<int>::const_iterator jt=vfeds.begin();
					jt!=vfeds.end(); ++jt)
					if (fed==*jt)
						break;
					else
						n++;
				if (n==vfeds.size())
					vfeds.push_back(fed);
			}

			std::sort(vfeds.begin(), vfeds.end());
			return vfeds;
		}
		std::vector<int> getFEDVMEList(HcalElectronicsMap const* emap)
		{
			std::vector<int> vfeds;
			std::vector<HcalElectronicsId> vids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator
				it=vids.begin(); it!=vids.end(); ++it)
			{
				if (!it->isVMEid())
					continue;
				int fed = it->isVMEid()?it->dccid()+FED_VME_MIN:
				  crate2fed(it->crateId(),it->slot());
				uint32_t n=0;
				for (std::vector<int>::const_iterator jt=vfeds.begin();
					jt!=vfeds.end(); ++jt)
					if (fed==*jt)
						break;
					else
						n++;
				if (n==vfeds.size())
					vfeds.push_back(fed);
			}

			std::sort(vfeds.begin(), vfeds.end());
			return vfeds;
		}
		std::vector<int> getFEDuTCAList(HcalElectronicsMap const* emap)
		{
			std::vector<int> vfeds;
			std::vector<HcalElectronicsId> vids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator
				it=vids.begin(); it!=vids.end(); ++it)
			{
				if (it->isVMEid())
					continue;
				int fed = it->isVMEid()?it->dccid()+FED_VME_MIN:
				  crate2fed(it->crateId(),it->slot());
				uint32_t n=0;
				for (std::vector<int>::const_iterator jt=vfeds.begin();
					jt!=vfeds.end(); ++jt)
					if (fed==*jt)
						break;
					else
						n++;
				if (n==vfeds.size())
					vfeds.push_back(fed);
			}

			std::sort(vfeds.begin(), vfeds.end());
			return vfeds;
		}

		bool isFEDHBHE(HcalElectronicsId const& eid)
		{
			if (eid.isVMEid())
			{
				int fed = eid.dccid()+FED_VME_MIN;	
				if (fed>=700 && fed<=717)
					return true;
				else
					return false;
			}
			else
			{
			  int fed = crate2fed(eid.crateId(),eid.slot());
				if (fed>=1100 && fed<1118)
					return true;
				else
					return false;
			}

			return false;
		}

		bool isFEDHF(HcalElectronicsId const& eid)
		{
			/*
			if (eid.isVMEid())
			{
				int fed = eid.dccid()+FED_VME_MIN;
				if (fed>=718 && fed<=723)
					return true;
				else
					return false;
			}*/
//			else
//			{
			if (eid.isVMEid())
				return false;
			int fed = crate2fed(eid.crateId(),eid.slot());
				if (fed>=1118 && fed<=1123)
					return true;
				else
					return false;
//			}

			return false;
		}

		bool isFEDHO(HcalElectronicsId const& eid)
		{
			if (!eid.isVMEid())
				return false;

			int fed = eid.dccid()+FED_VME_MIN;
			if (fed>=724 && fed<=731)
				return true;
			else
				return false;

			return false;
		}

		/*
		 *	Orbit Gap Related
		 */
		std::string ogtype2string(OrbitGapType type)
		{
			switch (type)
			{
				case tNull : return "Nill";
				case tPedestal : return "Pedestal";
				case tHFRaddam : return "HFRaddam";
				case tHBHEHPD : return "HBHEHPD";
				case tHO : return "HO";
				case tHF : return "HF";
				case tZDC : return "ZDC";
				case tHEPMega : return "HEPMegatile";
				case tHEMMega : return "HEMMegatile";
				case tHBPMega : return "HBPMegatile";
				case tHBMMega : return "HBMMegatile";
				case tCRF : return "CRF";
				case tCalib : return "Calib";
				case tSafe : return "Safe";
				default : return "Unknonw";
			}
		}
	}
}


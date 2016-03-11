#include "DQM/HcalCommon/interface/Utilities.h"

namespace hcaldqm
{
	using namespace constants;
	namespace utilities
	{

		/*
		 *	Useful Detector Functions. For Fast Detector Validity Check
		 */
		uint16_t fed2crate(int fed)
		{
			fed-=1100;
			if (fed>=constants::FED_uTCA_MAX_REAL)
				throw cms::Exception("HCALDQM")
					<< "fed2crate::fed index is out of range " 
					<< fed;

			return constants::FED2CRATE[fed];
		}

		uint16_t crate2fed(int crate)
		{
			//	 for the details see Constants.h
			if (crate>=constants::FED_uTCA_MAX_REAL)
				throw cms::Exception("HCALDQM")
					<< "crate2fed::crate index is out of range "
					<< crate;

			return constants::CRATE2FED[crate];
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
					crate2fed(it->crateId());
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
					crate2fed(it->crateId());
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
					crate2fed(it->crateId());
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
	}
}


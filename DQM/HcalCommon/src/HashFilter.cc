#include "DQM/HcalCommon/interface/HashFilter.h"

namespace hcaldqm
{
	namespace filter
	{
		HashFilter::HashFilter(FilterType ftype, HashType htype) : 
			HashMapper(htype), _ftype(ftype)
		{}

		HashFilter::HashFilter(FilterType ftype, HashType htype,
			std::vector<uint32_t> const& v) :
			HashMapper(htype), _ftype(ftype)
		{
			for (std::vector<uint32_t>::const_iterator it=v.begin();
				it!=v.end(); ++it)
				_ids.insert(*it);		
		}

		HashFilter::HashFilter(HashFilter const& hf) :
			HashMapper(hf._htype), _ftype(hf._ftype)
		{
			_ids = hf._ids;
		}

		/* virtual */ void HashFilter::initialize(FilterType ftype,
			HashType htype, std::vector<uint32_t> const& v)
		{
			HashMapper::initialize(htype);
			_ftype = ftype;
			for (std::vector<uint32_t>::const_iterator it=v.begin();
				it!=v.end(); ++it)
				_ids.insert(*it);		
		}

		/* virtual */ bool HashFilter::filter(HcalDetId const& did) const
		{
			return _ftype==fFilter?
				skip(getHash(did)):preserve(getHash(did));
		}

		/* virtual */ bool HashFilter::filter(HcalElectronicsId const& eid) 
			const
		{
			return _ftype==fFilter?
				skip(getHash(eid)):preserve(getHash(eid));
		}

		/* virtual */ bool HashFilter::filter(HcalTrigTowerDetId const& tid)
			const
		{
			return _ftype==fFilter?
				skip(getHash(tid)):preserve(getHash(tid));
		}

		/* virtual */ bool HashFilter::skip(uint32_t id) const
		{
			return _ids.find(id)==_ids.end()?false:true;
		}

		/* virtual */ bool HashFilter::preserve(uint32_t id) const
		{
			return _ids.find(id)==_ids.end()?true:false;
		}

		/* virtual */ void HashFilter::print()
		{

		}
	}
}

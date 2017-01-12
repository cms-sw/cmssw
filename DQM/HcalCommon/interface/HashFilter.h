#ifndef HashFilter_h
#define HashFilter_h

/**
 *	file:		HashFilter.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 *		Filters out hashes that we do not need
 */

#include "DQM/HcalCommon/interface/HashMapper.h"

#include <vector>
#include <boost/unordered_set.hpp>
#include <boost/foreach.hpp>

namespace hcaldqm
{
	using namespace mapper;
	namespace filter
	{
		enum FilterType
		{
			fFilter = 0,
			fPreserver = 1,
			nFilterType = 2
		};

		class HashFilter : public HashMapper
		{
			public:
				HashFilter() : _ftype(fFilter)
				{}
				//	empty hash
				HashFilter(FilterType ftype, HashType htype);
				//	initialize with a vector of hashes
				HashFilter(FilterType, HashType, 
					std::vector<uint32_t> const&);
				//	copy constructor
				HashFilter(HashFilter const& hf);
				virtual ~HashFilter() {}

				virtual void initialize(FilterType ftype, HashType htype,
					std::vector<uint32_t> const&);

				//	true if should filter out and false if not
				//	true => should skip this hash
				//	false => should keep this hash
				virtual bool filter(HcalDetId const&) const;
				virtual bool filter(HcalElectronicsId const&) const;
				virtual bool filter(HcalTrigTowerDetId const&) const;

				virtual void print();

			protected:	
				FilterType						_ftype;
				typedef boost::unordered_set<uint32_t> FilterMap;
				FilterMap						_ids;

				virtual bool preserve(uint32_t) const;
				virtual bool skip(uint32_t) const;
		};
	}
}

#endif

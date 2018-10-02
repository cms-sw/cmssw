#ifndef ElectronicsMap_h
#define ElectronicsMap_h

/**
 *	file:		ElectronicsMap.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 *		HcalElectronicsMap is slow. Upon beginRun hash what you need from 
 *		emap. Preserve only uint32_t. When you look things up, you know
 *		what is the key and you know what is the output as you define it up
 *		infront.
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/HashMapper.h"
#include "DQM/HcalCommon/interface/HashFilter.h"

#include "boost/unordered_map.hpp"
#include "string"

namespace hcaldqm
{
	namespace electronicsmap
	{
		enum ElectronicsMapType
		{
			fHcalElectronicsMap = 0,
			fD2EHashMap = 1,
			fT2EHashMap = 2,
			fE2DHashMap = 3,
			fE2THashMap = 4,
			nElectronicsMapType = 5
		};

		class ElectronicsMap
		{
			public:
				ElectronicsMap() :
					_emap(nullptr)
				{}
				//	define how to use upon construction
				ElectronicsMap(ElectronicsMapType etype) : 
					_etype(etype), _emap(nullptr)
				{}
				~ElectronicsMap() {}

				void initialize(HcalElectronicsMap const*, ElectronicsMapType
					etype=fHcalElectronicsMap);

				//	filter is to filter things you do not need out
				void initialize(HcalElectronicsMap const*, ElectronicsMapType,
					filter::HashFilter const&);
				uint32_t lookup(DetId const&);
				uint32_t lookup(HcalDetId const&);
				uint32_t lookup(HcalElectronicsId const&);

				void print();
				

			private:
				//	configures how to use emap
				ElectronicsMapType	_etype;

				//	2 choices either use as HcalElectronicsMap or as ur hash
				typedef boost::unordered_map<uint32_t, uint32_t> EMapType;
				EMapType			_ids;

				//	
				HcalElectronicsMap const* _emap;
		};
	}
}

#endif

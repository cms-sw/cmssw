#ifndef HashMapper_h
#define HashMapper_h

/**
 *	file:			HashMapper.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *
 */

#include "DQM/HcalCommon/interface/Mapper.h"
#include "DQM/HcalCommon/interface/HashFunctions.h"

namespace hcaldqm
{
	using namespace hashfunctions;
	namespace mapper
	{
		class HashMapper : public Mapper
		{
			public:
				//	constructors/destructors
				HashMapper() {}
				HashMapper(HashType htype) : Mapper(), _htype(htype) 
				{}
				virtual ~HashMapper() {}

				//	initialize
				virtual void initialize(HashType htype) {_htype = htype;}

				//	get hash
				virtual uint32_t getHash(HcalDetId const& did) const
				{return hash_did[_htype](did);}
				virtual uint32_t getHash(HcalElectronicsId const& eid) const
				{return hash_eid[_htype-nHashType_did-1](eid);}
				virtual uint32_t getHash(HcalTrigTowerDetId const& tid) const
				{return hash_tid[_htype-nHashType_eid-1](tid);}

				//	get name of the hashed element
				virtual std::string getName(HcalDetId const &did) const
				{return hashfunctions::name_did[_htype](did);}
				virtual std::string getName(HcalElectronicsId const& eid) const
				{return hashfunctions::name_eid[_htype-nHashType_did-1](eid);}
				virtual std::string getName(HcalTrigTowerDetId const& tid) const
				{return hashfunctions::name_tid[_htype-nHashType_eid-1](tid);}

				//	get the Hash Type Name
				virtual std::string getHashTypeName() const
				{return hash_names[this->getLinearHashType(_htype)];}
				virtual HashType getHashType() const
				{return _htype;}

				//	determine the type of the hash
				virtual bool isDHash()  const
				{return _htype<nHashType_did ? true : false;}
				virtual bool isEHash() const
				{
					return (_htype>nHashType_did && _htype<nHashType_eid) ? 
						true : false;
				}
				virtual bool isTHash() const
				{
					return (_htype>nHashType_eid && _htype<nHashType_tid) ? 
						true : false;
				}

				//	get the Linear Hash Type
				virtual int getLinearHashType(HashType htype) const
				{
					int l = 0;
					if (htype<nHashType_did)
						l = htype;
					else if (htype<nHashType_eid)
						l = htype - 1;
					else
						l = htype - 2;
					return l;
				}

			protected:
				HashType _htype;
		};
	}
}

#endif

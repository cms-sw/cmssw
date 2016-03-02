#ifndef ContainerXXX_h
#define ContainerXXX_h

/*
 *	file:		ContainerXXX.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 */

#include "DQM/HcalCommon/interface/Container1D.h"
#include <cmath>

namespace hcaldqm
{
	using namespace constants;

	template<typename STDTYPE>
	class ContainerXXX
	{
		public:
			ContainerXXX() {}
			ContainerXXX(hashfunctions::HashType ht) : _hashmap(ht)
			{}
			virtual ~ContainerXXX() {_cmap.clear();}

			virtual void initialize(hashfunctions::HashType,int debug=0);
			virtual void book(HcalElectronicsMap const*);
			virtual void book(HcalElectronicsMap const*,
				filter::HashFilter const&);

			virtual void set(HcalDetId const&, STDTYPE);
			virtual void set(HcalElectronicsId const&, STDTYPE);
			virtual void set(HcalTrigTowerDetId const&, STDTYPE);

			virtual STDTYPE& get(HcalDetId const&);
			virtual STDTYPE& get(HcalElectronicsId const&);
			virtual STDTYPE& get(HcalTrigTowerDetId const&);

			virtual void dump(Container1D*);
			virtual void dump(std::vector<Container1D*> const&);

			virtual void load(Container1D*);
			virtual void reset();
			virtual uint32_t size();
			virtual void print();

		protected:
			typedef boost::unordered_map<uint32_t, STDTYPE> CompactMap;
			CompactMap              _cmap;
			mapper::HashMapper      _hashmap;
			Logger                  _logger;
	};

	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::initialize(hashfunctions::HashType ht,
		int debug)
	{
		_hashmap.initialize(ht);
		_logger.set("XXX", debug);
	}

	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::book(HcalElectronicsMap const* emap)
	{
		if (_hashmap.isDHash())
		{
			std::vector<HcalGenericDetId> dids = 
				emap->allPrecisionId();
			for (std::vector<HcalGenericDetId>::const_iterator it=
				dids.begin(); it!=dids.end(); ++it)
			{
				if (!it->isHcalDetId())
					continue;

				uint32_t hash = it->rawId();
				HcalDetId did = HcalDetId(hash);
				_logger.debug(_hashmap.getName(did));
				typename CompactMap::iterator mit = _cmap.find(hash);
				if (mit!=_cmap.end())
					continue;

				_cmap.insert(
					std::make_pair(hash, STDTYPE(0)));
			}
		}
		else if (_hashmap.isEHash())
		{
			std::vector<HcalElectronicsId> eids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator it=
				eids.begin(); it!=eids.end(); ++it)
			{
				uint32_t hash = it->rawId();
				HcalElectronicsId eid = HcalElectronicsId(hash);
				_logger.debug(_hashmap.getName(eid));
				typename CompactMap::iterator mit = _cmap.find(hash);
				if (mit!=_cmap.end())
					continue;

				_cmap.insert(
					std::make_pair(hash, STDTYPE(0)));
			}
		}
		else if (_hashmap.isTHash())
		{
			std::vector<HcalTrigTowerDetId> tids = emap->allTriggerId();
			for (std::vector<HcalTrigTowerDetId>::const_iterator it=
				tids.begin(); it!=tids.end(); ++it)
			{
				uint32_t hash = it->rawId();
				HcalTrigTowerDetId tid = HcalTrigTowerDetId(hash);
				_logger.debug(_hashmap.getName(tid));
				typename CompactMap::iterator mit = _cmap.find(hash);
				if (mit!=_cmap.end())
					continue;

				_cmap.insert(
					std::make_pair(hash, STDTYPE(0)));
			}
		}
	}

	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::book(HcalElectronicsMap const* emap,
		filter::HashFilter const& filter)
	{
		if (_hashmap.isDHash())
		{
			std::vector<HcalGenericDetId> dids = 
				emap->allPrecisionId();
			for (std::vector<HcalGenericDetId>::const_iterator it=
				dids.begin(); it!=dids.end(); ++it)
			{
				if (!it->isHcalDetId())
					continue;

				uint32_t hash = it->rawId();
				HcalDetId did = HcalDetId(hash);
				_logger.debug(_hashmap.getName(did));
				typename CompactMap::iterator mit = _cmap.find(hash);
				if (mit!=_cmap.end())
					continue;
				if (filter.filter(did))
					continue;

				_cmap.insert(
					std::make_pair(hash, STDTYPE(0)));
			}
		}
		else if (_hashmap.isEHash())
		{
			std::vector<HcalElectronicsId> eids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator it=
				eids.begin(); it!=eids.end(); ++it)
			{
				uint32_t hash = it->rawId();
				HcalElectronicsId eid = HcalElectronicsId(hash);
				_logger.debug(_hashmap.getName(eid));
				typename CompactMap::iterator mit = _cmap.find(hash);
				if (mit!=_cmap.end())
					continue;
				if (filter.filter(eid))
					continue;

				_cmap.insert(
					std::make_pair(hash, STDTYPE(0)));
			}
		}
		else if (_hashmap.isTHash())
		{
			std::vector<HcalTrigTowerDetId> tids = emap->allTriggerId();
			for (std::vector<HcalTrigTowerDetId>::const_iterator it=
				tids.begin(); it!=tids.end(); ++it)
			{
				uint32_t hash = it->rawId();
				HcalTrigTowerDetId tid = HcalTrigTowerDetId(hash);
				_logger.debug(_hashmap.getName(tid));
				typename CompactMap::iterator mit = _cmap.find(hash);
				if (mit!=_cmap.end())
					continue;
				if (filter.filter(tid))
					continue;

				_cmap.insert(
					std::make_pair(hash, STDTYPE(0)));
			}
		}
	}

	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::set(HcalDetId const& did, STDTYPE x)
	{
		_cmap[_hashmap.getHash(did)] = x;
	}

	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::set(HcalElectronicsId const& did, 
		STDTYPE x)
	{
		_cmap[_hashmap.getHash(did)] = x;
	}

	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::set(HcalTrigTowerDetId const& did, 
		STDTYPE x)
	{
		_cmap[_hashmap.getHash(did)] = x;
	}

	template<typename STDTYPE>
	STDTYPE& ContainerXXX<STDTYPE>::get(HcalDetId const& did)
	{
		return _cmap[_hashmap.getHash(did)];
	}
	
	template<typename STDTYPE>
	STDTYPE& ContainerXXX<STDTYPE>::get(HcalElectronicsId const& eid)
	{
		return _cmap[_hashmap.getHash(eid)];
	}

	template<typename STDTYPE>
	STDTYPE& ContainerXXX<STDTYPE>::get(HcalTrigTowerDetId const& tid)
	{
		return _cmap[_hashmap.getHash(tid)];
	}

	template<typename STDTYPE>
	uint32_t ContainerXXX<STDTYPE>::size()
	{
		return (uint32_t)(_cmap.size());
	}

	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::dump(Container1D* c)
	{
		BOOST_FOREACH(typename CompactMap::value_type &p, _cmap)
		{
			STDTYPE &x = p.second;
			uint32_t hash = p.first;
			c->fill(hash, (double)x);
		}
	}
	
	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::dump(std::vector<Container1D*> const &vc)
	{
		BOOST_FOREACH(typename CompactMap::value_type &p, _cmap)
		{
			STDTYPE &x = p.second;
			uint32_t hash = p.first;

			for (std::vector<Container1D*>::const_iterator it=vc.begin();
				it!=vc.end(); ++it)
					(*it)->fill(hash, (double)x);
		}
	}

	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::print()
	{
		std::cout << "Container by " << _hashmap.getHashTypeName() << std::endl;
		BOOST_FOREACH(typename CompactMap::value_type &p, _cmap)
		{
			if (_hashmap.isDHash())
				std::cout << HcalDetId(p.first) << p.second << std::endl;
			else if (_hashmap.isEHash())
				std::cout << HcalElectronicsId(p.first) << p.second 
					<< std::endl;
			else if (_hashmap.isTHash())
				std::cout << HcalTrigTowerDetId(p.first) << p.second 
					<< std::endl;
		}
	}

	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::reset()
	{
		BOOST_FOREACH(typename CompactMap::value_type &p, _cmap)
		{
			p.second = 0;
		}
	}

	template<typename STDTYPE>
	void ContainerXXX<STDTYPE>::load(Container1D* cont)
	{
		BOOST_FOREACH(typename CompactMap::value_type &p, _cmap)
		{
			STDTYPE &x = p.second;
			uint32_t hash = p.first;

			if (_hashmap.isDHash())
				x = cont->getBinContent(HcalDetId(hash));
			else if (_hashmap.isEHash())
				x = cont->getBinContent(HcalElectronicsId(hash));
			else if (_hashmap.isTHash())
				x = cont->getBinContent(HcalTrigTowerDetId(hash));
		}	
	}
}

#endif

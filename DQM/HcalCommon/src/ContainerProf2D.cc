
#include "DQM/HcalCommon/interface/ContainerProf2D.h"

namespace hcaldqm
{
	using namespace mapper;
	using namespace quantity;
	using namespace constants;

	ContainerProf2D::ContainerProf2D()
	{
		_qx = NULL;
		_qy = NULL;
		_qz = NULL;
	}

	ContainerProf2D::ContainerProf2D(std::string const& folder,
		hashfunctions::HashType hashtype, Quantity *qx, Quantity *qy,
		Quantity *qz) :
		Container2D(folder, hashtype, qx, qy, qz)
	{
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
		_qz->setAxisType(quantity::fZAxis);
	}
	
	/* virtual */ void ContainerProf2D::initialize(std::string const& folder,
		hashfunctions::HashType hashtype, Quantity *qx, Quantity *qy,
		Quantity *qz,
		int debug/*=0*/)
	{
		Container2D::initialize(folder, hashtype, qx, qy, qz,debug);
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
		_qz->setAxisType(quantity::fZAxis);
	}

	/* virtual */ void ContainerProf2D::initialize(std::string const& folder,
		std::string const& qname,
		hashfunctions::HashType hashtype, Quantity *qx, Quantity *qy,
		Quantity *qz,
		int debug/*=0*/)
	{
		Container2D::initialize(folder, qname, hashtype, qx, qy, qz, 
			debug);
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
		_qz->setAxisType(quantity::fZAxis);
	}

	/* virtual */ void ContainerProf2D::book(DQMStore::IBooker &ib,
		HcalElectronicsMap const *emap,
		std::string subsystem, std::string aux)
	{
		//	full path as in Container1D.cc
		//
		ib.setCurrentFolder(subsystem+"/"+_folder+"/"+_qname
			+(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName());
		if (_hashmap.isDHash())
		{
			//	for Detector Hashes
			std::vector<HcalGenericDetId> dids = emap->allPrecisionId();
			for (std::vector<HcalGenericDetId>::const_iterator it=
				dids.begin(); it!=dids.end(); ++it)
			{
				//	skip trigger towers and calibration
				if (!it->isHcalDetId())
					continue;

				HcalDetId did = HcalDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(did);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;

				_logger.debug(_hashmap.getName(did));
				_mes.insert(
					std::make_pair(hash, ib.bookProfile2D(
					_hashmap.getName(did), _hashmap.getName(did),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isEHash())
		{	
			//	for Electronics hashes
			std::vector<HcalElectronicsId> eids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator it=
				eids.begin(); it!=eids.end(); ++it)
			{
				HcalElectronicsId eid = HcalElectronicsId(it->rawId());
				uint32_t hash = _hashmap.getHash(eid);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;

				_logger.debug(_hashmap.getName(eid));
				_mes.insert(
					std::make_pair(hash, ib.bookProfile2D(
					_hashmap.getName(eid), _hashmap.getName(eid),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower hashes
			std::vector<HcalTrigTowerDetId> tids = 
				emap->allTriggerId();
			for (std::vector<HcalTrigTowerDetId>::const_iterator it=
				tids.begin(); it!=tids.end(); ++it)
			{
				HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(tid);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;

				_logger.debug(_hashmap.getName(tid));
				_mes.insert(
					std::make_pair(hash, ib.bookProfile2D(
					_hashmap.getName(tid), _hashmap.getName(tid),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
	}

	/* virtual */ void ContainerProf2D::book(DQMStore::IBooker &ib,
		HcalElectronicsMap const *emap, filter::HashFilter const& filter, 
		std::string subsystem, std::string aux)
	{
		//	full path as in Container1D.cc
		//
		ib.setCurrentFolder(subsystem+"/"+_folder+"/"+_qname
			+(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName());
		if (_hashmap.isDHash())
		{
			//	for Detector Hashes
			std::vector<HcalGenericDetId> dids = emap->allPrecisionId();
			for (std::vector<HcalGenericDetId>::const_iterator it=
				dids.begin(); it!=dids.end(); ++it)
			{
				//	skip trigger towers and calibration
				if (!it->isHcalDetId())
					continue;

				HcalDetId did = HcalDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(did);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;
				if (filter.filter(did))
					continue;

				_logger.debug(_hashmap.getName(did));
				_mes.insert(
					std::make_pair(hash, ib.bookProfile2D(
					_hashmap.getName(did), _hashmap.getName(did),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isEHash())
		{	
			//	for Electronics hashes
			std::vector<HcalElectronicsId> eids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator it=
				eids.begin(); it!=eids.end(); ++it)
			{
				HcalElectronicsId eid = HcalElectronicsId(it->rawId());
				uint32_t hash = _hashmap.getHash(eid);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;
				if (filter.filter(eid))
					continue;

				_logger.debug(_hashmap.getName(eid));
				_mes.insert(
					std::make_pair(hash, ib.bookProfile2D(
					_hashmap.getName(eid), _hashmap.getName(eid),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower hashes
			std::vector<HcalTrigTowerDetId> tids = 
				emap->allTriggerId();
			for (std::vector<HcalTrigTowerDetId>::const_iterator it=
				tids.begin(); it!=tids.end(); ++it)
			{
				HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(tid);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;
				if (filter.filter(tid))
					continue;

				_logger.debug(_hashmap.getName(tid));
				_mes.insert(
					std::make_pair(hash, ib.bookProfile2D(
					_hashmap.getName(tid), _hashmap.getName(tid),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
	}

	/* virtual */ void ContainerProf2D::book(DQMStore *store,
		HcalElectronicsMap const *emap,
		std::string subsystem, std::string aux)
	{
		//	full path as in Container1D.cc
		//
		store->setCurrentFolder(subsystem+"/"+_folder+"/"+_qname
			+(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName());
		if (_hashmap.isDHash())
		{
			//	for Detector Hashes
			std::vector<HcalGenericDetId> dids = emap->allPrecisionId();
			for (std::vector<HcalGenericDetId>::const_iterator it=
				dids.begin(); it!=dids.end(); ++it)
			{
				//	skip trigger towers and calibration
				if (!it->isHcalDetId())
					continue;

				HcalDetId did = HcalDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(did);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;

				_logger.debug(_hashmap.getName(did));
				_mes.insert(
					std::make_pair(hash, store->bookProfile2D(
					_hashmap.getName(did), _hashmap.getName(did),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isEHash())
		{	
			//	for Electronics hashes
			std::vector<HcalElectronicsId> eids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator it=
				eids.begin(); it!=eids.end(); ++it)
			{
				HcalElectronicsId eid = HcalElectronicsId(it->rawId());
				uint32_t hash = _hashmap.getHash(eid);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;

				_logger.debug(_hashmap.getName(eid));
				_mes.insert(
					std::make_pair(hash, store->bookProfile2D(
					_hashmap.getName(eid), _hashmap.getName(eid),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower hashes
			std::vector<HcalTrigTowerDetId> tids = 
				emap->allTriggerId();
			for (std::vector<HcalTrigTowerDetId>::const_iterator it=
				tids.begin(); it!=tids.end(); ++it)
			{
				HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(tid);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;

				_logger.debug(_hashmap.getName(tid));
				_mes.insert(
					std::make_pair(hash, store->bookProfile2D(
					_hashmap.getName(tid), _hashmap.getName(tid),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
	}

	/* virtual */ void ContainerProf2D::book(DQMStore *store,
		HcalElectronicsMap const *emap, filter::HashFilter const& filter,
		std::string subsystem, std::string aux)
	{
		//	full path as in Container1D.cc
		//
		store->setCurrentFolder(subsystem+"/"+_folder+"/"+_qname
			+(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName());
		if (_hashmap.isDHash())
		{
			//	for Detector Hashes
			std::vector<HcalGenericDetId> dids = emap->allPrecisionId();
			for (std::vector<HcalGenericDetId>::const_iterator it=
				dids.begin(); it!=dids.end(); ++it)
			{
				//	skip trigger towers and calibration
				if (!it->isHcalDetId())
					continue;

				HcalDetId did = HcalDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(did);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;
				if (filter.filter(did))
					continue;

				_logger.debug(_hashmap.getName(did));
				_mes.insert(
					std::make_pair(hash, store->bookProfile2D(
					_hashmap.getName(did), _hashmap.getName(did),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isEHash())
		{	
			//	for Electronics hashes
			std::vector<HcalElectronicsId> eids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator it=
				eids.begin(); it!=eids.end(); ++it)
			{
				HcalElectronicsId eid = HcalElectronicsId(it->rawId());
				uint32_t hash = _hashmap.getHash(eid);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;
				if (filter.filter(eid))
					continue;

				_logger.debug(_hashmap.getName(eid));
				_mes.insert(
					std::make_pair(hash, store->bookProfile2D(
					_hashmap.getName(eid), _hashmap.getName(eid),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower hashes
			std::vector<HcalTrigTowerDetId> tids = 
				emap->allTriggerId();
			for (std::vector<HcalTrigTowerDetId>::const_iterator it=
				tids.begin(); it!=tids.end(); ++it)
			{
				HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(tid);
				MEMap::iterator mit = _mes.find(hash);
				if (mit!=_mes.end())
					continue;
				if (filter.filter(tid))
					continue;

				_logger.debug(_hashmap.getName(tid));
				_mes.insert(
					std::make_pair(hash, store->bookProfile2D(
					_hashmap.getName(tid), _hashmap.getName(tid),
					_qx->nbins(), _qx->min(), _qx->max(),
					_qy->nbins(), _qy->min(), _qy->max(),
					_qz->min(), _qz->max())));
				customize(_mes[hash]);
			}
		}
	}
}

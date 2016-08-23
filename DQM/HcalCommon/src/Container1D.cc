
#include "DQM/HcalCommon/interface/Container1D.h"

namespace hcaldqm
{
	using namespace mapper;
	using namespace constants;

	Container1D::Container1D():
		_qx(NULL), _qy(NULL)
	{}

	Container1D::Container1D(std::string const& folder,
		hashfunctions::HashType hashtype, Quantity *qx, Quantity *qy) :
		Container(folder, qy->name()+"vs"+qx->name()), _hashmap(hashtype),
		_qx(qx), _qy(qy)
	{
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
	}

	Container1D::~Container1D()
	{
		if (_qx!=NULL)
			delete _qx;
		if (_qy!=NULL)
			delete _qy;
		_qx = NULL;
		_qy = NULL;
	}

	/* virtual */ /*void Container1D::release()
	{
		BOOST_FOREACH(MEMap::value_type &pair, _mes)
		{
			pair.second=NULL;
		}
	}*/

	/* virtuial */ void Container1D::initialize(std::string const& folder, 
		hashfunctions::HashType hashtype, Quantity *qx, Quantity *qy/* = ... */,
		int debug /* =0 */)
	{
		Container::initialize(folder, qy->name()+"vs"+qx->name(), debug);
		_hashmap.initialize(hashtype);
		_qx = qx;
		_qy = qy;
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
	}

	/* virtuial */ void Container1D::initialize(std::string const& folder, 
		std::string const& qname,
		hashfunctions::HashType hashtype, Quantity *qx, Quantity *qy/* = ... */,
		int debug /* =0 */)
	{
		Container::initialize(folder, qname, debug);
		_hashmap.initialize(hashtype);
		_qx = qx;
		_qy = qy;
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
	}

	/* virtual */ void Container1D::reset()
	{
		BOOST_FOREACH(MEMap::value_type &pair, _mes)
		{
			pair.second->Reset();
		}
	}

	/* virtual */ void Container1D::print()
	{	
		std::cout << "Container by " << _hashmap.getHashTypeName() << std::endl;
		BOOST_FOREACH(MEMap::value_type &pair, _mes)
		{
			std::cout << std::hex << pair.first << std::dec << std::endl;
		}
	}

	/* virtual */ void Container1D::fill(uint32_t hash)
	{
		if (_hashmap.isDHash())
			this->fill(HcalDetId(hash));
		else if (_hashmap.isEHash())
			this->fill(HcalElectronicsId(hash));
		else if (_hashmap.isTHash())
			this->fill(HcalTrigTowerDetId(hash));
	}

	/* virtual */ void Container1D::fill(uint32_t hash, int x)
	{
		if (_hashmap.isDHash())
			this->fill(HcalDetId(hash), x);
		else if (_hashmap.isEHash())
			this->fill(HcalElectronicsId(hash), x);
		else if (_hashmap.isTHash())
			this->fill(HcalTrigTowerDetId(hash), x);
	}

	/* virtual */ void Container1D::fill(uint32_t hash, double x)
	{
		if (_hashmap.isDHash())
			this->fill(HcalDetId(hash), x);
		else if (_hashmap.isEHash())
			this->fill(HcalElectronicsId(hash), x);
		else if (_hashmap.isTHash())
			this->fill(HcalTrigTowerDetId(hash), x);
	}

	/* virtual */ void Container1D::fill(uint32_t hash, int x, double y)
	{
		if (_hashmap.isDHash())
			this->fill(HcalDetId(hash), x, y);
		else if (_hashmap.isEHash())
			this->fill(HcalElectronicsId(hash), x, y);
		else if (_hashmap.isTHash())
			this->fill(HcalTrigTowerDetId(hash), x, y);
	}

	/* virtual */ void Container1D::fill(uint32_t hash, int x, int y)
	{
		if (_hashmap.isDHash())
			this->fill(HcalDetId(hash), x, y);
		else if (_hashmap.isEHash())
			this->fill(HcalElectronicsId(hash), x, y);
		else if (_hashmap.isTHash())
			this->fill(HcalTrigTowerDetId(hash), x, y);
	}

	/* virtual */ void Container1D::fill(uint32_t hash, double x, double y)
	{
		if (_hashmap.isDHash())
			this->fill(HcalDetId(hash), x, y);
		else if (_hashmap.isEHash())
			this->fill(HcalElectronicsId(hash), x, y);
		else if (_hashmap.isTHash())
			this->fill(HcalTrigTowerDetId(hash), x, y);
	}

	//	by HcalDetId
	/* virtual */ void Container1D::fill(HcalDetId const& did)
	{
		_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did));
	}
	/* virtual */ void Container1D::fill(HcalDetId const& did, int x)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x));
	}
	/* virtual */ void Container1D::fill(HcalDetId const& did, double x)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
					_qy->getValue(x));
	}
	/* virtual */ void Container1D::fill(HcalDetId const& did, int x, double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), 
			_qy->getValue(y));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x), y);
	}
	/* virtual */ void Container1D::fill(HcalDetId const& did, int x, int y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), 
			_qy->getValue(y));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x), y);
	}
	/* virtual */ void Container1D::fill(HcalDetId const& did, double x , 
			double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), 
			_qy->getValue(y));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x), y);
	}

	/* virtual */ double Container1D::getBinEntries(HcalDetId const& id)
	{
		return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id));
	}

	/* virtual */ double Container1D::getBinEntries(HcalDetId const& id,
		int x)
	{
		return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getBinEntries(HcalDetId const& id,
		double x)
	{
		return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getBinContent(HcalDetId const& 
		tid)
	{
		return _mes[_hashmap.getHash(tid)]->getBinContent(_qx->getBin(tid));
	}

	/* virtual */ double Container1D::getBinContent(HcalDetId const& 
		tid, int x)
	{
		return _mes[_hashmap.getHash(tid)]->getBinContent(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getBinContent(HcalDetId const& 
		tid, double x)
	{
		return _mes[_hashmap.getHash(tid)]->getBinContent(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getMean(HcalDetId const& tid, int axis)
	{
		return _mes[_hashmap.getHash(tid)]->getMean(axis);
	}

	/* virtual */ double Container1D::getRMS(HcalDetId const& id, int axis)
	{
		return _mes[_hashmap.getHash(id)]->getRMS(axis);
	}

	//	setBinContent
	/* virtual */ void Container1D::setBinContent(HcalDetId const& id,
		int x)
	{
		_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalDetId const& id,
		double x)
	{
		_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalDetId const& id,
		int x, int y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalDetId const& id,
		int x, double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalDetId const& id,
		double x, int y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalDetId const& id,
		double x, double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}

	//	by HcalElectronicsId
	/* virtual */ void Container1D::fill(HcalElectronicsId const& did)
	{
		_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did));
	}
	/* virtual */ void Container1D::fill(HcalElectronicsId const& did, int x)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x));
	}
	/* virtual */ void Container1D::fill(HcalElectronicsId const& did, double x)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
					_qy->getValue(x));
	}
	/* virtual */ void Container1D::fill(HcalElectronicsId const& did, 
		int x, double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), 
			_qy->getValue(y));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x), y);
	}
	/* virtual */ void Container1D::fill(HcalElectronicsId const& did, 
		int x, int y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), 
			_qy->getValue(y));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x), y);
	}
	/* virtual */ void Container1D::fill(HcalElectronicsId const& did, 
		double x , double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), 
			_qy->getValue(y));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x), y);
	}

	/* virtual */ double Container1D::getBinEntries(HcalElectronicsId const& id)
	{
		return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id));
	}

	/* virtual */ double Container1D::getBinEntries(HcalElectronicsId const& id,
		int x)
	{
		return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getBinEntries(HcalElectronicsId const& id,
		double x)
	{
		return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getBinContent(HcalElectronicsId const& 
		tid)
	{
		return _mes[_hashmap.getHash(tid)]->getBinContent(_qx->getBin(tid));
	}

	/* virtual */ double Container1D::getBinContent(HcalElectronicsId const& 
		tid, int x)
	{
		return _mes[_hashmap.getHash(tid)]->getBinContent(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getBinContent(HcalElectronicsId const& 
		tid, double x)
	{
		return _mes[_hashmap.getHash(tid)]->getBinContent(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getMean(HcalElectronicsId const& tid,
		int axis)
	{
		return _mes[_hashmap.getHash(tid)]->getMean(axis);
	}
	
	/* virtual */ double Container1D::getRMS(HcalElectronicsId const& id, 
		int axis)
	{
		return _mes[_hashmap.getHash(id)]->getRMS(axis);
	}

	//	setBinContent
	/* virtual */ void Container1D::setBinContent(HcalElectronicsId const& id,
		int x)
	{
		_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalElectronicsId const& id,
		double x)
	{
		_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalElectronicsId const& id,
		int x, int y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalElectronicsId const& id,
		int x, double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalElectronicsId const& id,
		double x, int y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalElectronicsId const& id,
		double x, double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}

	//	by HcaTrigTowerlDetId
	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& did)
	{
		_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did));
	}
	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& did, int x)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x));
	}
	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& did, 
		double x)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
					_qy->getValue(x));
	}
	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& did, 
		int x, double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), 
			_qy->getValue(y));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x), y);
	}
	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& did, 
		int x, int y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), 
			_qy->getValue(y));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x), y);
	}
	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& did, 
		double x, double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), 
			_qy->getValue(y));
		else 
			_mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), 
				_qy->getValue(x), y);
	}

	/* virtual */ double Container1D::getBinEntries(HcalTrigTowerDetId const& 
		id)
	{
		return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id));
	}

	/* virtual */ double Container1D::getBinEntries(HcalTrigTowerDetId const& 
		id, int x)
	{
		return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getBinEntries(HcalTrigTowerDetId const& 
		id, double x)
	{
		return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getBinContent(HcalTrigTowerDetId const& 
		tid)
	{
		return _mes[_hashmap.getHash(tid)]->getBinContent(_qx->getBin(tid));
	}

	/* virtual */ double Container1D::getBinContent(HcalTrigTowerDetId const& 
		tid, int x)
	{
		return _mes[_hashmap.getHash(tid)]->getBinContent(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getBinContent(HcalTrigTowerDetId const& 
		tid, double x)
	{
		return _mes[_hashmap.getHash(tid)]->getBinContent(_qx->getBin(x));
	}

	/* virtual */ double Container1D::getMean(HcalTrigTowerDetId const& tid,
		int axis)
	{
		return _mes[_hashmap.getHash(tid)]->getMean(axis);
	}
	
	/* virtual */ double Container1D::getRMS(HcalTrigTowerDetId const& id, 
		int axis)
	{
		return _mes[_hashmap.getHash(id)]->getRMS(axis);
	}

	//	setBinContent
	/* virtual */ void Container1D::setBinContent(HcalTrigTowerDetId const& id,
		int x)
	{
		_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalTrigTowerDetId const& id,
		double x)
	{
		_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalTrigTowerDetId const& id,
		int x, int y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalTrigTowerDetId const& id,
		int x, double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalTrigTowerDetId const& id,
		double x, int y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void Container1D::setBinContent(HcalTrigTowerDetId const& id,
		double x, double y)
	{
		QuantityType qtype = _qx->type();
		if (qtype==fValueQuantity || qtype==fFlagQuantity)
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), y);
		else
			_mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), x);
	}

	/* virtual */ void Container1D::load(DQMStore *store,
		HcalElectronicsMap const* emap, std::string const &subsystem,
		std::string const& aux, std::string const& prepend,
		DQMStore::OpenRunDirs mode)
	{
		//	full path to where all the plots are living
		//	prepend/subsystem/taskname/QxvsQy_auxilary/HashType
		//	if loaded and not stripped, then 
		//	prepend/subsystem/Run summary/taskname/...
		_logger.debug(_hashmap.getHashTypeName());
		std::string path = (prepend==""?prepend:prepend+"/")+
			subsystem+"/"+(mode==DQMStore::KeepRunDirs?"Run summary/":"")
			+_folder+"/"+_qname+
			(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName();
		_logger.debug("FULLPATH::"+path);

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
					std::make_pair(hash, 
						store->get(path+"/"+_hashmap.getName(did))));
			}
		}
		
		else if (_hashmap.isEHash())
		{
			//	for Electronics Hashes
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
					std::make_pair(hash, 
						store->get(path+"/"+_hashmap.getName(eid))));
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower Hashes
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
					std::make_pair(hash, 
						store->get(path+"/"+_hashmap.getName(tid))));
			}
		}
	}

	/* virtual */ void Container1D::load(DQMStore *store,
		HcalElectronicsMap const* emap, filter::HashFilter const& filter,
		std::string const &subsystem,
		std::string const& aux, std::string const& prepend,
		DQMStore::OpenRunDirs mode)
	{
		//	full path to where all the plots are living
		//	prepend/subsystem/taskname/QxvsQy_auxilary/HashType
		//	if loaded and not stripped, then 
		//	prepend/subsystem/Run summary/taskname/...
		_logger.debug(_hashmap.getHashTypeName());
		std::string path = (prepend==""?prepend:prepend+"/")+
			subsystem+"/"+(mode==DQMStore::KeepRunDirs?"Run summary/":"")
			+_folder+"/"+_qname+
			(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName();
		_logger.debug("FULLPATH::"+path);

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

				//	skip this guy, it's already present
				if (mit!=_mes.end())
					continue;
				//	filter out what's not needed
				if (filter.filter(did))
					continue;

				_logger.debug(_hashmap.getName(did));
				_mes.insert(
					std::make_pair(hash, 
						store->get(path+"/"+_hashmap.getName(did))));
			}
		}
		
		else if (_hashmap.isEHash())
		{
			//	for Electronics Hashes
			std::vector<HcalElectronicsId> eids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator it=
				eids.begin(); it!=eids.end(); ++it)
			{
				HcalElectronicsId eid = HcalElectronicsId(it->rawId());
				uint32_t hash = _hashmap.getHash(eid);
				MEMap::iterator mit = _mes.find(hash);

				//	skip this guy, it's already present
				if (mit!=_mes.end())
					continue;
				//	filter out
				if (filter.filter(eid))
					continue;

				_logger.debug(_hashmap.getName(eid));
				_mes.insert(
					std::make_pair(hash, 
						store->get(path+"/"+_hashmap.getName(eid))));
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower Hashes
			std::vector<HcalTrigTowerDetId> tids = 
				emap->allTriggerId();
			for (std::vector<HcalTrigTowerDetId>::const_iterator it=
				tids.begin(); it!=tids.end(); ++it)
			{
				HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(tid);
				MEMap::iterator mit = _mes.find(hash);

				//	skip if this guy already exists
				if (mit!=_mes.end())
					continue;
				//	 filter out
				if (filter.filter(tid))
					continue;

				_logger.debug(_hashmap.getName(tid));
				_mes.insert(
					std::make_pair(hash, 
						store->get(path+"/"+_hashmap.getName(tid))));
			}
		}
	}

	//	load w/o a filter
	/* virtual */ void Container1D::load(DQMStore::IGetter& ig,
		HcalElectronicsMap const* emap, std::string const& subsystem,
		std::string const& aux)
	{
		//	full path to where all the plots are living
		//	prepend/subsystem/taskname/QxvsQy_auxilary/HashType
		_logger.debug(_hashmap.getHashTypeName());
		std::string path = 
			subsystem+"/"+_folder+"/"+_qname+
			(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName();
		_logger.debug("FULLPATH::"+path);

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

				//	skip this guy, it's already present
				if (mit!=_mes.end())
					continue;

				_logger.debug(_hashmap.getName(did));
				_mes.insert(
					std::make_pair(hash, 
						ig.get(path+"/"+_hashmap.getName(did))));
			}
		}
		
		else if (_hashmap.isEHash())
		{
			//	for Electronics Hashes
			std::vector<HcalElectronicsId> eids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator it=
				eids.begin(); it!=eids.end(); ++it)
			{
				HcalElectronicsId eid = HcalElectronicsId(it->rawId());
				uint32_t hash = _hashmap.getHash(eid);
				MEMap::iterator mit = _mes.find(hash);

				//	skip this guy, it's already present
				if (mit!=_mes.end())
					continue;

				_logger.debug(_hashmap.getName(eid));
				_mes.insert(
					std::make_pair(hash, 
						ig.get(path+"/"+_hashmap.getName(eid))));
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower Hashes
			std::vector<HcalTrigTowerDetId> tids = 
				emap->allTriggerId();
			for (std::vector<HcalTrigTowerDetId>::const_iterator it=
				tids.begin(); it!=tids.end(); ++it)
			{
				HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(tid);
				MEMap::iterator mit = _mes.find(hash);

				//	skip if this guy already exists
				if (mit!=_mes.end())
					continue;

				_logger.debug(_hashmap.getName(tid));
				_mes.insert(
					std::make_pair(hash, 
						ig.get(path+"/"+_hashmap.getName(tid))));
			}
		}
	}

	//	load w/ a filter
	/* virtual */ void Container1D::load(DQMStore::IGetter& ig,
		HcalElectronicsMap const* emap, filter::HashFilter const& filter,
		std::string const& subsystem,
		std::string const& aux)
	{
		//	full path to where all the plots are living
		//	prepend/subsystem/taskname/QxvsQy_auxilary/HashType
		_logger.debug(_hashmap.getHashTypeName());
		std::string path = 
			subsystem+"/"+_folder+"/"+_qname+
			(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName();
		_logger.debug("FULLPATH::"+path);

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

				//	skip this guy, it's already present
				if (mit!=_mes.end())
					continue;
				//	filter out what's not needed
				if (filter.filter(did))
					continue;

				_logger.debug(_hashmap.getName(did));
				_mes.insert(
					std::make_pair(hash, 
						ig.get(path+"/"+_hashmap.getName(did))));
			}
		}
		
		else if (_hashmap.isEHash())
		{
			//	for Electronics Hashes
			std::vector<HcalElectronicsId> eids = 
				emap->allElectronicsIdPrecision();
			for (std::vector<HcalElectronicsId>::const_iterator it=
				eids.begin(); it!=eids.end(); ++it)
			{
				HcalElectronicsId eid = HcalElectronicsId(it->rawId());
				uint32_t hash = _hashmap.getHash(eid);
				MEMap::iterator mit = _mes.find(hash);

				//	skip this guy, it's already present
				if (mit!=_mes.end())
					continue;
				//	filter out
				if (filter.filter(eid))
					continue;

				_logger.debug(_hashmap.getName(eid));
				_mes.insert(
					std::make_pair(hash, 
						ig.get(path+"/"+_hashmap.getName(eid))));
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower Hashes
			std::vector<HcalTrigTowerDetId> tids = 
				emap->allTriggerId();
			for (std::vector<HcalTrigTowerDetId>::const_iterator it=
				tids.begin(); it!=tids.end(); ++it)
			{
				HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
				uint32_t hash = _hashmap.getHash(tid);
				MEMap::iterator mit = _mes.find(hash);

				//	skip if this guy already exists
				if (mit!=_mes.end())
					continue;
				//	 filter out
				if (filter.filter(tid))
					continue;

				_logger.debug(_hashmap.getName(tid));
				_mes.insert(
					std::make_pair(hash, 
						ig.get(path+"/"+_hashmap.getName(tid))));
			}
		}
	}

	//	Book
	/* virtual */ void Container1D::book(DQMStore::IBooker& ib, 
		HcalElectronicsMap const *emap,
		std::string subsystem, std::string aux)
	{
		//	full path to where all the plots are living
		//	subsystem/taskname/QxvsQy_auxilary/HashType
		ib.setCurrentFolder(subsystem+"/"+_folder+"/"+_qname+
			(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName());
		_logger.debug(_hashmap.getHashTypeName());
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
					std::make_pair(hash, ib.book1D(_hashmap.getName(did),
					_hashmap.getName(did), _qx->nbins(), _qx->min(), 
					_qx->max())));
				
				//	customize
				customize(_mes[hash]);
			}
		}
		
		else if (_hashmap.isEHash())
		{
			//	for Electronics Hashes
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
					std::make_pair(hash,
					ib.book1D(_hashmap.getName(eid),
					_hashmap.getName(eid), 
					_qx->nbins(), _qx->min(), _qx->max())));

				//	customize
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower Hashes
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
					std::make_pair(hash,
					ib.book1D(_hashmap.getName(tid),
					_hashmap.getName(tid), 
					_qx->nbins(), _qx->min(), _qx->max())));
				//	customize
				customize(_mes[hash]);
			}
		}
	}

	//	Book
	/* virtual */ void Container1D::book(DQMStore::IBooker& ib, 
		HcalElectronicsMap const *emap, filter::HashFilter const& filter,
		std::string subsystem, std::string aux)
	{
		//	full path to where all the plots are living
		//	subsystem/taskname/QxvsQy_auxilary/HashType
		ib.setCurrentFolder(subsystem+"/"+_folder+"/"+_qname+
			(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName());
		_logger.debug(_hashmap.getHashTypeName());

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
					std::make_pair(hash, ib.book1D(_hashmap.getName(did),
					_hashmap.getName(did), _qx->nbins(), _qx->min(), 
					_qx->max())));
				
				//	customize
				customize(_mes[hash]);
			}
		}
		
		else if (_hashmap.isEHash())
		{
			//	for Electronics Hashes
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
					std::make_pair(hash,
					ib.book1D(_hashmap.getName(eid),
					_hashmap.getName(eid), 
					_qx->nbins(), _qx->min(), _qx->max())));

				//	customize
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower Hashes
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
					std::make_pair(hash,
					ib.book1D(_hashmap.getName(tid),
					_hashmap.getName(tid), 
					_qx->nbins(), _qx->min(), _qx->max())));
				//	customize
				customize(_mes[hash]);
			}
		}
	}

	/* virtual */ void Container1D::book(DQMStore *store, 
		HcalElectronicsMap const *emap,
		std::string subsystem, std::string aux)
	{
		//	full path to where all the plots are living
		//	subsystem/taskname/QxvsQy_auxilary/HashType
		store->setCurrentFolder(subsystem+"/"+_folder+"/"+_qname+
			(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName());
		_logger.debug(_hashmap.getHashTypeName());
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
					std::make_pair(hash, store->book1D(_hashmap.getName(did),
					_hashmap.getName(did), _qx->nbins(), _qx->min(), 
					_qx->max())));
				
				//	customize
				customize(_mes[hash]);
			}
		}
		
		else if (_hashmap.isEHash())
		{
			//	for Electronics Hashes
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
					std::make_pair(hash,
					store->book1D(_hashmap.getName(eid),
					_hashmap.getName(eid), 
					_qx->nbins(), _qx->min(), _qx->max())));

				//	customize
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower Hashes
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
					std::make_pair(hash,
					store->book1D(_hashmap.getName(tid),
					_hashmap.getName(tid), 
					_qx->nbins(), _qx->min(), _qx->max())));
				//	customize
				customize(_mes[hash]);
			}
		}
	}

	/* virtual */ void Container1D::book(DQMStore *store, 
		HcalElectronicsMap const *emap, filter::HashFilter const& filter,
		std::string subsystem, std::string aux)
	{
		//	full path to where all the plots are living
		//	subsystem/taskname/QxvsQy_auxilary/HashType
		store->setCurrentFolder(subsystem+"/"+_folder+"/"+_qname+
			(aux==""?aux:"_"+aux)+"/"+_hashmap.getHashTypeName());
		_logger.debug(_hashmap.getHashTypeName());
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
					std::make_pair(hash, store->book1D(_hashmap.getName(did),
					_hashmap.getName(did), _qx->nbins(), _qx->min(), 
					_qx->max())));
				
				//	customize
				customize(_mes[hash]);
			}
		}
		
		else if (_hashmap.isEHash())
		{
			//	for Electronics Hashes
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
					std::make_pair(hash,
					store->book1D(_hashmap.getName(eid),
					_hashmap.getName(eid), 
					_qx->nbins(), _qx->min(), _qx->max())));

				//	customize
				customize(_mes[hash]);
			}
		}
		else if (_hashmap.isTHash())
		{
			//	for TrigTower Hashes
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
					std::make_pair(hash,
					store->book1D(_hashmap.getName(tid),
					_hashmap.getName(tid), 
					_qx->nbins(), _qx->min(), _qx->max())));
				//	customize
				customize(_mes[hash]);
			}
		}
	}

	/* virtual */ void Container1D::customize(MonitorElement* me)
	{
		//	set labels
		me->setAxisTitle(_qx->name(), 1);
		me->setAxisTitle(_qy->name(), 2);

		//	set bits
		TH1 *h = me->getTH1();	
		_qx->setBits(h);
		_qy->setBits(h);

		//	set labels
		std::vector<std::string> xlabels = _qx->getLabels();
		for (unsigned int i=0; i<xlabels.size(); i++)
			me->setBinLabel(i+1, xlabels[i], 1);
	}

	/* virtual */ void Container1D::extendAxisRange(int l)
	{
		if (l<_qx->nbins())
			return;

		//	inflate all the mes
		BOOST_FOREACH(MEMap::value_type &pair, _mes)
		{
			int x=_qx->nbins();
			while (l>=x)
			{
				pair.second->getTH1()->LabelsInflate();
				x*=2;
				_qx->setMax(x);
			}
		}
	}

	/* virtual */ void Container1D::setLumiFlag()
	{
		BOOST_FOREACH(MEMap::value_type &pair, _mes)
		{
			pair.second->setLumiFlag();
		}
	}
}








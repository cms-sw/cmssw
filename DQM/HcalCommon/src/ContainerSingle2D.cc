
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/Utilities.h"

namespace hcaldqm
{
	ContainerSingle2D::ContainerSingle2D()
	{
		_qx = NULL;
		_qy = NULL;
		_qz = NULL;
	}

	ContainerSingle2D::ContainerSingle2D(std::string const& folder,
		Quantity *qx, Quantity *qy, Quantity *qz):
		Container(folder, qz->name()+"vs"+qy->name()+"vs"+qx->name()),
		_qx(qx), _qy(qy), _qz(qz)
	{
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
		_qz->setAxisType(quantity::fZAxis);
	}
	
	ContainerSingle2D::ContainerSingle2D(std::string const& folder,
		std::string const& qname,
		Quantity *qx, Quantity *qy, Quantity *qz, int debug/*=0*/):
		Container(folder, qname), _qx(qx), _qy(qy), _qz(qz)
	{
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
		_qz->setAxisType(quantity::fZAxis);
	}

	ContainerSingle2D::ContainerSingle2D(ContainerSingle2D const& c) :
		Container(c._folder, c._qname)
	{
		_qx = c._qx->makeCopy();
		_qy = c._qy->makeCopy();
		_qz = c._qz->makeCopy();
	}

	ContainerSingle2D::~ContainerSingle2D()
	{
		if (_qx!=NULL)
			delete _qx;
		if (_qy!=NULL)
			delete _qy;
		if (_qz!=NULL)
			delete _qz;
		_qx=NULL;
		_qy=NULL;
		_qz=NULL;
	}

	/* virtual */ void ContainerSingle2D::initialize(std::string const& folder,
		Quantity *qx, Quantity *qy, Quantity *qz, int debug/*=0*/)
	{
		Container::initialize(folder, qz->name()+"vs"+qy->name()+"vs"+
			qx->name(), debug);
		_qx = qx;
		_qy = qy;
		_qz = qz;
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
		_qz->setAxisType(quantity::fZAxis);
	}

	/* virtual */ void ContainerSingle2D::initialize(std::string const& folder,
		std::string const& qname,
		Quantity *qx, Quantity *qy, Quantity *qz, int debug/*=0*/)
	{
		Container::initialize(folder, qname, debug);
		_qx = qx;
		_qy = qy;
		_qz = qz;
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
		_qz->setAxisType(quantity::fZAxis);
	}
	
	/* virtual */ void ContainerSingle2D::book(DQMStore::IBooker &ib,
		 std::string subsystem, std::string aux)
	{
		ib.setCurrentFolder(subsystem+"/"+_folder+"/"+_qname);
		_me = ib.book2D(_qname+(aux==""?aux:"_"+aux), 
			_qname+(aux==""?aux:" "+aux),
			_qx->nbins(), _qx->min(), _qx->max(),
			_qy->nbins(), _qy->min(), _qy->max());
		customize();
	}

	/* virtual */ void ContainerSingle2D::load(DQMStore::IGetter& ig,
		std::string subsystem, std::string aux)
	{
		_me = ig.get(subsystem+"/"+_folder+"/"+_qname+"/"+
			_qname+(aux==""?aux:"_"+aux));
	}

	/* virtual */ void ContainerSingle2D::book(DQMStore *store,
		 std::string subsystem, std::string aux)
	{
		store->setCurrentFolder(subsystem+"/"+_folder+"/"+_qname);
		_me = store->book2D(_qname+(aux==""?aux:"_"+aux), 
			_qname+(aux==""?aux:" "+aux),
			_qx->nbins(), _qx->min(), _qx->max(),
			_qy->nbins(), _qy->min(), _qy->max());
		customize();
	}

	/* virtual */ void ContainerSingle2D::customize()
	{
		_me->setAxisTitle(_qx->name(), 1);
		_me->setAxisTitle(_qy->name(), 2);
		_me->setAxisTitle(_qz->name(), 3);

		TH1 *h = _me->getTH1();
		_qx->setBits(h);
		_qy->setBits(h);
		_qz->setBits(h);

		std::vector<std::string> xlabels = _qx->getLabels();
		std::vector<std::string> ylabels = _qy->getLabels();
		for (unsigned int i=0; i<xlabels.size(); i++)
			_me->setBinLabel(i+1, xlabels[i], 1);
		for (unsigned int i=0; i<ylabels.size(); i++)
			_me->setBinLabel(i+1, ylabels[i], 2);
	}

	/* virtual */ void ContainerSingle2D::fill(int x, int y)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y));
	}

	/* virtual */ void ContainerSingle2D::fill(int x, double y)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y));
	}

	/* virtual */ void ContainerSingle2D::fill(int x, double y, double z)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y), z);
	}

	/* virtual */ void ContainerSingle2D::fill(double x, int y)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y));
	}

	/* virtual */ void ContainerSingle2D::fill(double x, double y)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y));
	}

	/* virtual */ void ContainerSingle2D::fill(double x, double y, double z)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y), z);
	}

	/* virtual */ void ContainerSingle2D::fill(int x, int y, double z)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y), z);
	}

	/* virtual */ void ContainerSingle2D::fill(int x, int y, int z)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y), z);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& id)
	{
		_me->Fill(_qx->getValue(id), _qy->getValue(id));
	}

	/* virtual */ double ContainerSingle2D::getBinContent(int x, int y)
	{
		return _me->getBinContent(_qx->getBin(x), _qy->getBin(y));
	}
	/* virtual */ double ContainerSingle2D::getBinContent(int x, double y)
	{
		return _me->getBinContent(_qx->getBin(x), _qy->getBin(y));
	}
	/* virtual */ double ContainerSingle2D::getBinContent(double x, int y)
	{
		return _me->getBinContent(_qx->getBin(x), _qy->getBin(y));
	}
	/* virtual */ double ContainerSingle2D::getBinContent(double x, double y)
	{
		return _me->getBinContent(_qx->getBin(x), _qy->getBin(y));
	}
	/* virtual */ double ContainerSingle2D::getBinEntries(int x, int y)
	{
		return _me->getBinEntries(_qx->getBin(x)+_qx->wofnbins()*_qy->getBin(y));
	}
	/* virtual */ double ContainerSingle2D::getBinEntries(int x, double y)
	{
		return _me->getBinEntries(_qx->getBin(x)+_qx->wofnbins()*_qy->getBin(y));
	}
	/* virtual */ double ContainerSingle2D::getBinEntries(double x, int y)
	{
		return _me->getBinEntries(_qx->getBin(x)+_qx->wofnbins()*_qy->getBin(y));
	}
	/* virtual */ double ContainerSingle2D::getBinEntries(double x, double y)
	{
		return _me->getBinEntries(_qx->getBin(x)+_qx->wofnbins()*_qy->getBin(y));
	}

	/* virtual */ void ContainerSingle2D::setBinContent(int x, int y, int z)
	{
		_me->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(int x, double y, int z)
	{
		_me->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(double x, int y, int z)
	{
		_me->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(double x, 
		double y, int z)
	{
		_me->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(int x, int y, double z)
	{
		_me->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(int x, double y, 
		double z)
	{
		_me->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(double x, int y, 
		double z)
	{
		_me->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(double x, 
		double y, double z)
	{
		_me->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& id, double x)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x));
		else if (_qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& id, int x)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x));
		else if (_qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& id, double x,
		double y)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate() && !_qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else if (!_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id), y);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& id, int x,
		int y)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate() && !_qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else if (!_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id), y);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& id, int x,
		double y)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate() && !_qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else if (!_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id), y);
	}

	/* virtula */ double ContainerSingle2D::getBinContent(HcalDetId const& id)
	{
		return _me->getBinContent(_qx->getBin(id), _qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinContent(HcalDetId const& id,
		int x)
	{
		if (_qx->isCoordinate())
			return _me->getBinContent(_qx->getBin(id), _qy->getBin(x));
		else
			return _me->getBinContent(_qx->getBin(x), _qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinContent(HcalDetId const& id,
		double x)
	{
		if (_qx->isCoordinate())
			return _me->getBinContent(_qx->getBin(id), _qy->getBin(x));
		else
			return _me->getBinContent(_qx->getBin(x), _qy->getBin(id));
	}
	/* virtula */ double ContainerSingle2D::getBinEntries(HcalDetId const& id)
	{
		return _me->getBinEntries(_qx->getBin(id)+_qx->wofnbins()* _qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinEntries(HcalDetId const& id,
		int x)
	{
		if (_qx->isCoordinate())
			return _me->getBinEntries(_qx->getBin(id)+_qx->wofnbins()*_qy->getBin(x));
		else
			return _me->getBinEntries(_qx->getBin(x)+_qx->wofnbins()*_qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinEntries(HcalDetId const& id,
		double x)
	{
		if (_qx->isCoordinate())
			return _me->getBinEntries(_qx->getBin(id)+_qx->wofnbins()*_qy->getBin(x));
		else
			return _me->getBinEntries(_qx->getBin(x)+_qx->wofnbins()*_qy->getBin(id));
	}

	/* virtual */ void ContainerSingle2D::setBinContent(HcalDetId const& id, 
		int x)
	{
		_me->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalDetId const& id, 
		double x)
	{
		_me->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalDetId const& id, 
		int x, int y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalDetId const& id, 
		int x, double y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalDetId const& id, 
		double x, int y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalDetId const& id, 
		double x, double y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}

	//	by ElectronicsId
	/* virtual */ void ContainerSingle2D::fill(HcalElectronicsId const& id)
	{
		_me->Fill(_qx->getValue(id), _qy->getValue(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalElectronicsId const& id, 
		double x)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x));
		else if (_qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalElectronicsId const& id, 
		int x)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x));
		else if (_qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalElectronicsId const& id, 
		double x,
		double y)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate() && !_qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else if (!_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id), y);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalElectronicsId const& id, 
		int x,
		int y)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate() && !_qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else if (!_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id), y);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalElectronicsId const& id, 
		int x,
		double y)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate() && !_qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else if (!_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id), y);
	}

	/* virtula */ double ContainerSingle2D::getBinContent(HcalElectronicsId const& id)
	{
		return _me->getBinContent(_qx->getBin(id), _qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinContent(HcalElectronicsId const& id,
		int x)
	{
		if (_qx->isCoordinate())
			return _me->getBinContent(_qx->getBin(id), _qy->getBin(x));
		else
			return _me->getBinContent(_qx->getBin(x), _qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinContent(HcalElectronicsId const& id,
		double x)
	{
		if (_qx->isCoordinate())
			return _me->getBinContent(_qx->getBin(id), _qy->getBin(x));
		else
			return _me->getBinContent(_qx->getBin(x), _qy->getBin(id));
	}
	/* virtula */ double ContainerSingle2D::getBinEntries(HcalElectronicsId const& id)
	{
		return _me->getBinEntries(_qx->getBin(id)+_qx->wofnbins()*_qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinEntries(HcalElectronicsId const& id,
		int x)
	{
		if (_qx->isCoordinate())
			return _me->getBinEntries(_qx->getBin(id)+_qx->wofnbins()*_qy->getBin(x));
		else
			return _me->getBinEntries(_qx->getBin(x)+_qx->wofnbins()*_qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinEntries(HcalElectronicsId const& id,
		double x)
	{
		if (_qx->isCoordinate())
			return _me->getBinEntries(_qx->getBin(id)+_qx->wofnbins()*_qy->getBin(x));
		else
			return _me->getBinEntries(_qx->getBin(x)+_qx->wofnbins()*_qy->getBin(id));
	}

	/* virtual */ void ContainerSingle2D::setBinContent(HcalElectronicsId const& id, 
		int x)
	{
		_me->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalElectronicsId const& id, 
		double x)
	{
		_me->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalElectronicsId const& id, 
		int x, int y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalElectronicsId const& id, 
		int x, double y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalElectronicsId const& id, 
		double x, int y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalElectronicsId const& id, 
		double x, double y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}

	//	by TrigTowerDetId
	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id)
	{
		_me->Fill(_qx->getValue(id), _qy->getValue(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id, 
		double x)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x));
		else if (_qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id, 
		int x)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x));
		else if (_qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id, 
		double x, double y)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate() && !_qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else if (!_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id), y);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id, 
		int x, int y)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate() && !_qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else if (!_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id), y);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id, 
		int x, double y)
	{
		if (_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(id), x);
		else if (_qx->isCoordinate() && !_qy->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else if (!_qx->isCoordinate() && _qy->isCoordinate())
			_me->Fill(_qx->getValue(x), _qy->getValue(id), y);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& did,
		HcalElectronicsId const& eid)
	{
		if (_qx->type()==fDetectorQuantity)
			_me->Fill(_qx->getValue(did), _qy->getValue(eid));
		else
			_me->Fill(_qx->getValue(eid), _qy->getValue(did));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& did,
		HcalElectronicsId const& eid, double x)
	{
		if (_qx->type()==fDetectorQuantity)
			_me->Fill(_qx->getValue(did), _qy->getValue(eid), x);
		else
			_me->Fill(_qx->getValue(eid), _qy->getValue(did), x);
	}

	/* virtula */ double ContainerSingle2D::getBinContent(HcalTrigTowerDetId const& id)
	{
		return _me->getBinContent(_qx->getBin(id), _qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinContent(HcalTrigTowerDetId const& id,
		int x)
	{
		if (_qx->isCoordinate())
			return _me->getBinContent(_qx->getBin(id), _qy->getBin(x));
		else
			return _me->getBinContent(_qx->getBin(x), _qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinContent(HcalTrigTowerDetId const& id,
		double x)
	{
		if (_qx->isCoordinate())
			return _me->getBinContent(_qx->getBin(id), _qy->getBin(x));
		else
			return _me->getBinContent(_qx->getBin(x), _qy->getBin(id));
	}
	/* virtula */ double ContainerSingle2D::getBinEntries(HcalTrigTowerDetId const& id)
	{
		return _me->getBinEntries(_qx->getBin(id)+_qx->wofnbins()*_qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinEntries(HcalTrigTowerDetId const& id,
		int x)
	{
		if (_qx->isCoordinate())
			return _me->getBinEntries(_qx->getBin(id)+_qx->wofnbins()*_qy->getBin(x));
		else
			return _me->getBinEntries(_qx->getBin(x)+_qx->wofnbins()*_qy->getBin(id));
	}
	/* virtual */ double ContainerSingle2D::getBinEntries(HcalTrigTowerDetId const& id,
		double x)
	{
		if (_qx->isCoordinate())
			return _me->getBinEntries(_qx->getBin(id)+_qx->wofnbins()*_qy->getBin(x));
		else
			return _me->getBinEntries(_qx->getBin(x)+_qx->wofnbins()*_qy->getBin(id));
	}

	/* virtual */ void ContainerSingle2D::setBinContent(HcalTrigTowerDetId const& id, 
		int x)
	{
		_me->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalTrigTowerDetId const& id, 
		double x)
	{
		_me->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalTrigTowerDetId const& id, 
		int x, int y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalTrigTowerDetId const& id, 
		int x, double y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalTrigTowerDetId const& id, 
		double x, int y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}
	/* virtual */ void ContainerSingle2D::setBinContent(HcalTrigTowerDetId const& id, 
		double x, double y)
	{
		if (_qx->isCoordinate())
			_me->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
		else
			_me->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
	}

	/* virtual */ void ContainerSingle2D::extendAxisRange(int l)
	{
		if (l<_qx->nbins())
			return;

		int x=_qx->nbins();
		while(l>=x)
		{
			_me->getTH1()->LabelsInflate();
			x*=2;
			_qx->setMax(x);
		}
	}
}


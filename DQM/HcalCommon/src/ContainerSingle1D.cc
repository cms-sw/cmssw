
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"

namespace hcaldqm
{
	using namespace quantity;
	ContainerSingle1D::ContainerSingle1D()
	{
		_qx = NULL;
		_qy = NULL;
	}

	ContainerSingle1D::ContainerSingle1D(ContainerSingle1D const& c):
		Container(c._folder, c._qname)
	{
		_qx = c._qx->makeCopy();
		_qy = c._qy->makeCopy();
	}

	ContainerSingle1D::ContainerSingle1D(std::string const& folder,
		Quantity *qx, Quantity *qy):
		Container(folder, qy->name()+"vs"+qx->name()), _qx(qx), _qy(qy)
	{
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
	}

	ContainerSingle1D::ContainerSingle1D(std::string const& folder,
		std::string const& qname,
		Quantity *qx, Quantity *qy):
		Container(folder, qname), _qx(qx), _qy(qy)
	{
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
	}

	ContainerSingle1D::~ContainerSingle1D()
	{
		if (_qx!=NULL)
			delete _qx;
		if (_qy!=NULL)
			delete _qy;
		_qx = NULL;
		_qy = NULL;
	}

	/* virtual */ void ContainerSingle1D::initialize(std::string const& folder,
		Quantity *qx, Quantity *qy, int debug/*=0*/)
	{
		Container::initialize(folder, qy->name()+"vs"+qx->name(), debug);
		_qx = qx;
		_qy = qy;
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
	}

	/* virtual */ void ContainerSingle1D::initialize(std::string const& folder,
		std::string const& qname,
		Quantity *qx, Quantity *qy, int debug/*=0*/)
	{
		Container::initialize(folder, qname, debug);
		_qx = qx;
		_qy = qy;
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
	}

	/* virtual */ void ContainerSingle1D::book(DQMStore::IBooker &ib,
		 std::string subsystem, std::string aux)
	{
		ib.setCurrentFolder(subsystem+"/"+_folder+"/"+_qname);
		_me = ib.book1D(_qname+(aux==""?aux:"_"+aux), 
			_qname+(aux==""?aux:" "+aux),
			_qx->nbins(), _qx->min(), _qx->max());
		customize();
	}

	/* virtual */ void ContainerSingle1D::book(DQMStore *store,
		 std::string subsystem, std::string aux)
	{
		store->setCurrentFolder(subsystem+"/"+_folder+"/"+_qname);
		_me = store->book1D(_qname+(aux==""?aux:"_"+aux), 
			_qname+(aux==""?aux:" "+aux),
			_qx->nbins(), _qx->min(), _qx->max());
		customize();
	}

	/* virtual */ void ContainerSingle1D::customize()
	{
		_me->setAxisTitle(_qx->name(), 1);
		_me->setAxisTitle(_qy->name(), 2);

		TH1 *h = _me->getTH1();
		_qx->setBits(h);
		_qy->setBits(h);

		std::vector<std::string> xlabels = _qx->getLabels();
		for (unsigned int i=0; i<xlabels.size(); i++)
			_me->setBinLabel(i+1, xlabels[i], 1);
	}

	/* virtual */ void ContainerSingle1D::fill(int x)
	{
		_me->Fill(_qx->getValue(x));
	}

	/* virtual */ void ContainerSingle1D::fill(double x)
	{
		_me->Fill(_qx->getValue(x));
	}

	/* virtual */ void ContainerSingle1D::fill(int x, int y)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y));
	}

	/* virtual */ void ContainerSingle1D::fill(int x, double y)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y));
	}

	/* virtual */ void ContainerSingle1D::fill(double x, int y)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y));
	}

	/* virtual */ void ContainerSingle1D::fill(double x, double y)
	{
		_me->Fill(_qx->getValue(x), _qy->getValue(y));
	}

	/* virtual */ double ContainerSingle1D::getBinContent(int x)
	{
		return _me->getBinContent(_qx->getBin(x));
	}
	/* virtual */ double ContainerSingle1D::getBinContent(double x)
	{
		return _me->getBinContent(_qx->getBin(x));
	}
	/* virtual */ double ContainerSingle1D::getBinEntries(int x)
	{
		return _me->getBinEntries(_qx->getBin(x));
	}
	/* virtual */ double ContainerSingle1D::getBinEntries(double x)
	{
		return _me->getBinEntries(_qx->getBin(x));
	}
	/* virtual */ void ContainerSingle1D::setBinContent(int x, int y)
	{
		_me->setBinContent(_qx->getBin(x), y);
	}
	/* virtual */ void ContainerSingle1D::setBinContent(int x, double y)
	{
		_me->setBinContent(_qx->getBin(x), y);
	}
	/* virtual */ void ContainerSingle1D::setBinContent(double x, int y)
	{
		_me->setBinContent(_qx->getBin(x), y);
	}
	/* virtual */ void ContainerSingle1D::setBinContent(double x, double y)
	{
		_me->setBinContent(_qx->getBin(x), y);
	}

	/* virtual */ void ContainerSingle1D::fill(HcalDetId const& id)
	{
		_me->Fill(_qx->getValue(id));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalDetId const& id, double x)
	{
		if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x));
		else
			_me->Fill(_qx->getValue(x));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalDetId const& id, double x,
		double y)
	{
		if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else
			_me->Fill(_qy->getValue(x), _qy->getValue(y));
	}

	/* virtual */ double ContainerSingle1D::getBinContent(HcalDetId const& id)
	{
		return _me->getBinContent(_qx->getBin(id));
	}
	/* virtual */ double ContainerSingle1D::getBinEntries(HcalDetId const& id)
	{
		return _me->getBinEntries(_qx->getBin(id));
	}

	/* virtual */ void ContainerSingle1D::setBinContent(HcalDetId const& id,
		int x)
	{
		_me->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void ContainerSingle1D::setBinContent(HcalDetId const& id,
		double x)
	{
		_me->setBinContent(_qx->getBin(id), x);
	}

	/* virtual */ void ContainerSingle1D::fill(HcalElectronicsId const& id)
	{
		_me->Fill(_qx->getValue(id));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalElectronicsId const& id, 
		double x)
	{
		if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x));
		else
			_me->Fill(_qx->getValue(x));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalElectronicsId const& id, 
		double x, double y)
	{
		if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else
			_me->Fill(_qy->getValue(x), _qy->getValue(y));
	}

	/* virtual */ double ContainerSingle1D::getBinContent(HcalElectronicsId const& id)
	{
		return _me->getBinContent(_qx->getBin(id));
	}
	/* virtual */ double ContainerSingle1D::getBinEntries(HcalElectronicsId const& id)
	{
		return _me->getBinEntries(_qx->getBin(id));
	}

	/* virtual */ void ContainerSingle1D::setBinContent(HcalElectronicsId const& id,
		int x)
	{
		_me->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void ContainerSingle1D::setBinContent(HcalElectronicsId const& id,
		double x)
	{
		_me->setBinContent(_qx->getBin(id), x);
	}

	/* virtual */ void ContainerSingle1D::fill(HcalTrigTowerDetId const& id)
	{
		_me->Fill(_qx->getValue(id));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalTrigTowerDetId const& id, 
		double x)
	{
		if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x));
		else
			_me->Fill(_qx->getValue(x));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalTrigTowerDetId const& id, 
		double x,
		double y)
	{
		if (_qx->isCoordinate())
			_me->Fill(_qx->getValue(id), _qy->getValue(x), y);
		else
			_me->Fill(_qy->getValue(x), _qy->getValue(y));
	}

	/* virtual */ double ContainerSingle1D::getBinContent(HcalTrigTowerDetId const& id)
	{
		return _me->getBinContent(_qx->getBin(id));
	}
	/* virtual */ double ContainerSingle1D::getBinEntries(HcalTrigTowerDetId const& id)
	{
		return _me->getBinEntries(_qx->getBin(id));
	}

	/* virtual */ void ContainerSingle1D::setBinContent(HcalTrigTowerDetId const& id,
		int x)
	{
		_me->setBinContent(_qx->getBin(id), x);
	}
	/* virtual */ void ContainerSingle1D::setBinContent(HcalTrigTowerDetId const& id,
		double x)
	{
		_me->setBinContent(_qx->getBin(id), x);
	}

	/* virtual */ void ContainerSingle1D::extendAxisRange(int l)
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



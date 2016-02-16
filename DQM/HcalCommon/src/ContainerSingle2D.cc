
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/Utilities.h"

namespace hcaldqm
{
	using namespace axis;

	ContainerSingle2D::ContainerSingle2D()
	{
		_xaxis = NULL;
		_yaxis = NULL;
		_zaxis = NULL;
	}

	/* virtual */ ContainerSingle2D::~ContainerSingle2D()
	{
		delete _xaxis; _xaxis=NULL;
		delete _yaxis; _yaxis=NULL;
		delete _zaxis; _zaxis=NULL;
	}

	ContainerSingle2D::ContainerSingle2D(std::string const& folder,
		std::string const& nametitle,
		Axis *xaxis, Axis *yaxis, Axis *zaxis):
		Container(folder, nametitle),
		_xaxis(xaxis), _yaxis(yaxis), _zaxis(zaxis)
	{}

	/* virtual */ void ContainerSingle2D::initialize(std::string const& folder,
		std::string const& nametitle,
		Axis *xaxis, Axis *yaxis, Axis *zaxis, int debug/*=0*/)
	{
		Container::initialize(folder, nametitle, debug);
		_xaxis =xaxis;
		_yaxis = yaxis;
		_zaxis = zaxis;
	}
	
	/* virtual */ void ContainerSingle2D::book(DQMStore::IBooker &ib,
		 std::string subsystem, std::string aux)
	{
		ib.setCurrentFolder(subsystem+"/"+_folder+aux);
		_me = ib.book2D(_name, _name,
			_xaxis->_nbins, _xaxis->_min, _xaxis->_max,
			_yaxis->_nbins, _yaxis->_min, _yaxis->_max);
		TObject *o = _me->getRootObject();
		_xaxis->setLog(o);
		_yaxis->setLog(o);
		_zaxis->setLog(o);
		_xaxis->setBitAxisLS(o);
		_yaxis->setBitAxisLS(o);
		_xaxis->setBitAxisFlag(o);
		_yaxis->setBitAxisFlag(o);
		_me->setAxisTitle(_xaxis->_title, 1);
		_me->setAxisTitle(_yaxis->_title, 2);
		_me->setAxisTitle(_zaxis->_title, 3);
		for (unsigned int i=0; i<_xaxis->_labels.size(); i++)
			_me->setBinLabel(i+1, _xaxis->_labels[i], 1);
		for (unsigned int i=0; i<_yaxis->_labels.size(); i++)
			_me->setBinLabel(i+1, _yaxis->_labels[i], 2);
	}

	/* virtual */ void ContainerSingle2D::fill(int x, int y)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle2D::fill(int x, double y)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle2D::fill(int x, double y, double z)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y), z);
	}

	/* virtual */ void ContainerSingle2D::fill(double x, int y)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle2D::fill(double x, double y)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle2D::fill(double x, double y, double z)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y), z);
	}

	/* virtual */ void ContainerSingle2D::fill(int x, int y, double z)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y), z);
	}

	/* virtual */ void ContainerSingle2D::fill(int x, int y, int z)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y), z);
	}

	/* virtual */ void ContainerSingle2D::setBinContent(int x, int y, double z)
	{
		_me->setBinContent(_xaxis->getBin(x), _yaxis->getBin(y), z);
	}

	/* virtual */ void ContainerSingle2D::setBinContent(unsigned int x, 
		int y, double z)
	{
		_me->setBinContent(_xaxis->getBin(x), _yaxis->getBin(y), z);
	}

	/* virtual */ void ContainerSingle2D::setBinContent(int x, double y, double z)
	{
		_me->setBinContent(_xaxis->getBin(x), _yaxis->getBin(y), z);
	}

	/* virtual */ void ContainerSingle2D::setBinContent(double x, int y, double z)
	{
		_me->setBinContent(_xaxis->getBin(x), _yaxis->getBin(y), z);
	}

	/* virtual */ void ContainerSingle2D::setBinContent(double x, double y, double z)
	{
		_me->setBinContent(_xaxis->getBin(x), _yaxis->getBin(y), z);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& id)
	{
		_me->Fill(_xaxis->get(id), _yaxis->get(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& id, double x)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(id), x);
		else if (xact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x));
		else if (yact==fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& id, double x,
		double y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(id), y);
		else if (xact!=fCoordinate && yact!=fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalElectronicsId const& id)
	{
		_me->Fill(_xaxis->get(id), _yaxis->get(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalElectronicsId const& id, 
		double x)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(id), x);
		else if (xact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x));
		else if (yact==fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalElectronicsId const& id, 
		double x, double y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(id), y);
		else if (xact!=fCoordinate && yact!=fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& did, 
		HcalElectronicsId const& eid)
	{
		_me->Fill(_xaxis->get(did), _yaxis->get(eid));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalDetId const& did, 
		HcalElectronicsId const& eid, double x)
	{
		_me->Fill(_xaxis->get(did), _yaxis->get(eid), x);
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& tid)
	{
		_me->Fill(_xaxis->get(tid), _yaxis->get(tid));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id, 
		double x)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(id), x);
		else if (xact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x));
		else if (yact==fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id, 
		int x)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(id), x);
		else if (xact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x));
		else if (yact==fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(id));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id, 
		int x, int y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(id), y);
		else if (xact!=fCoordinate && yact!=fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id, 
		int x, double y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(id), y);
		else if (xact!=fCoordinate && yact!=fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle2D::fill(HcalTrigTowerDetId const& id, 
		double x, double y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(id), y);
		else if (xact!=fCoordinate && yact!=fCoordinate)
			_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle2D::fill(int x,
		HcalElectronicsId const& eid)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(eid));
	}

	/* virtual */ void ContainerSingle2D::fill(int x,
		HcalElectronicsId const& eid, int z)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(eid), z);
	}

	/* virtual */ void ContainerSingle2D::fill(int x,
		HcalElectronicsId const& eid, double z)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(eid), z);
	}

	/* virtual */ void ContainerSingle2D::loadLabels(
		std::vector<std::string> const& labels)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact!=fFlag && yact!=fFlag)
			return;
		else if (xact!=fFlag)
			_yaxis->loadLabels(labels);
		else 
			_xaxis->loadLabels(labels);
	}

	/* virtual */ void ContainerSingle2D::reset()
	{
		_me->Reset();
	}
}




#include "DQM/HcalCommon/interface/ContainerSingle1D.h"

namespace hcaldqm
{
	using namespace axis;

	ContainerSingle1D::ContainerSingle1D()
	{
		_xaxis = NULL;
		_yaxis = NULL;
	}

	/* virtual */ContainerSingle1D::~ContainerSingle1D()
	{
		delete _xaxis; _xaxis=NULL;
		delete _yaxis; _yaxis=NULL;
	}

	ContainerSingle1D::ContainerSingle1D(std::string const& folder,
		std::string const& nametitle,
		Axis *xaxis, Axis *yaxis):
		Container(folder, nametitle),
		_xaxis(xaxis), _yaxis(yaxis)
	{}

	ContainerSingle1D::ContainerSingle1D(ContainerSingle1D const& c):
		Container(c._folder, c._name)
	{
		_xaxis = c._xaxis->makeCopy();
		_yaxis = c._yaxis->makeCopy();
	}

	/* virtual */ void ContainerSingle1D::initialize(std::string const& folder,
		std::string const& nametitle,
		Axis *xaxis, Axis *yaxis, int debug/*=0*/)
	{
		Container::initialize(folder, nametitle, debug);
		_xaxis = xaxis;
		_yaxis = yaxis;
	}

	/* virtual */ void ContainerSingle1D::book(DQMStore::IBooker &ib,
		 std::string subsystem, std::string aux)
	{
		ib.setCurrentFolder(subsystem+"/"+_folder+aux);
		_me = ib.book1D(_name, _name,
			_xaxis->_nbins, _xaxis->_min, _xaxis->_max);
		TObject *o = _me->getRootObject();
		_xaxis->setLog(o);
		_yaxis->setLog(o);
		_xaxis->setBitAxisLS(o);
		_yaxis->setBitAxisLS(o);
		_xaxis->setBitAxisFlag(o);
		_yaxis->setBitAxisFlag(o);
		_me->setAxisTitle(_xaxis->_title, 1);
		_me->setAxisTitle(_yaxis->_title, 2);
		for (unsigned int i=0; i<_xaxis->_labels.size(); i++)
			_me->setBinLabel(i+1, _xaxis->_labels[i], 1);
	}

	/* virtual */ void ContainerSingle1D::fill(int x)
	{
		_me->Fill(_xaxis->get(x));
	}

	/* virtual */ void ContainerSingle1D::fill(double x)
	{
		_me->Fill(_xaxis->get(x));
	}

	/* virtual */ void ContainerSingle1D::fill(int x, int y)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle1D::fill(int x, double y)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle1D::fill(double x, int y)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle1D::fill(double x, double y)
	{
		_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalDetId const& id)
	{
		_me->Fill(_xaxis->get(id));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalDetId const& id, double x)
	{
		AxisQType xact = _xaxis->getType();
		if (xact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x));
		else
			_me->Fill(_xaxis->get(x));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalDetId const& id, double x,
		double y)
	{
		AxisQType xact = _xaxis->getType();
		if (xact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x), y);
		else
			_me->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalElectronicsId const& id)
	{
		_me->Fill(_xaxis->get(id));
	}

	/* virtual */ void ContainerSingle1D::fill(HcalElectronicsId const& id, 
		double x)
	{
		AxisQType xact = _xaxis->getType();
		if (xact==fCoordinate)
			_me->Fill(_xaxis->get(id), _yaxis->get(x));
		else
			_me->Fill(_xaxis->get(x));
	}
}



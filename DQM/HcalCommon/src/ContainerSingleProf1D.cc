#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"

namespace hcaldqm
{
	using namespace axis; 

	ContainerSingleProf1D::ContainerSingleProf1D()
	{
		_xaxis = NULL;
		_yaxis = NULL;
	}

	ContainerSingleProf1D::ContainerSingleProf1D(std::string const& folder,
		std::string const& nametitle, axis::Axis *xaxis,
		axis::Axis *yaxis):
		ContainerSingle1D(folder, nametitle, xaxis, yaxis)
	{}
	
	/* virtual */ void ContainerSingleProf1D::initialize(std::string const& 
		folder,
		std::string const& nametitle, axis::Axis *xaxis,
		axis::Axis *yaxis, int debug/*=0*/)
	{
		ContainerSingle1D::initialize(folder, nametitle, xaxis, yaxis, debug);
	}

	/* virtual */ void ContainerSingleProf1D::book(DQMStore::IBooker& ib,
		std::string subsystem, std::string aux)
	{
		ib.setCurrentFolder(subsystem+"/"+_folder+aux);
		_me = ib.bookProfile(_name, _name,
			_xaxis->_nbins, _xaxis->_min, _xaxis->_max,
			_yaxis->_min, _yaxis->_max);
		TObject *o = _me->getRootObject();
		_xaxis->setLog(o);
		_yaxis->setLog(o);
		_xaxis->setBitAxisLS(o);
		_yaxis->setBitAxisLS(o);
		_xaxis->setBitAxisLS(o);
		_yaxis->setBitAxisLS(o);
		_me->setAxisTitle(_xaxis->_title, 1);
		_me->setAxisTitle(_yaxis->_title, 1);
		for (unsigned int i=0; i<_xaxis->_labels.size(); i++)
			_me->setBinLabel(i+1, _xaxis->_labels[i], 1);
	}
}




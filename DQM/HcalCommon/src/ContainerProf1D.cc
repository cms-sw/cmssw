
#include "DQM/HcalCommon/interface/ContainerProf1D.h"


namespace hcaldqm
{
	using namespace mapper;
	using namespace axis;
	using namespace constants;

	ContainerProf1D::ContainerProf1D()
	{
		_xaxis = NULL;
		_yaxis = NULL;
	}

	ContainerProf1D::ContainerProf1D(std::string const& folder,
		std::string const& nametitle, mapper::MapperType mt,
		axis::Axis *xaxis, axis::Axis *yaxis):
		Container1D(folder, nametitle, mt, xaxis, yaxis)
	{}

	/* virtual */ void ContainerProf1D::initialize(std::string const& folder,
		std::string const& nametitle, mapper::MapperType mt,
		axis::Axis *xaxis, axis::Axis *yaxis, int debug/*=0*/)
	{
		Container1D::initialize(folder, nametitle, mt, xaxis, yaxis, debug);
	}

	/* virtual */ void ContainerProf1D::book(DQMStore::IBooker &ib,
		std::string subsystem, std::string aux)
	{
		unsigned int size = _mapper.getSize();
		 ib.setCurrentFolder(subsystem+"/"+_folder+aux);
		 _logger.debug(_folder+aux);
		 _logger.debug("Size:");
		 _logger.debug(size);
		for (unsigned int i=0; i<size; i++)
		{
			_logger.debug(i);
			_logger.debug(_name);
			std::string hname = _mapper.buildName(i);
			MonitorElement *me = ib.bookProfile(_name+"_"+hname, 
				_name+" "+hname, _xaxis->_nbins, _xaxis->_min, _xaxis->_max,
				_yaxis->_min, _yaxis->_max);
			TObject *o = me->getRootObject();
			_xaxis->setLog(o);
			_yaxis->setLog(o);
			_xaxis->setBitAxisLS(o);
			_yaxis->setBitAxisLS(o);
			_xaxis->setBitAxisFlag(o);
			_yaxis->setBitAxisFlag(o);
			me->setAxisTitle(_xaxis->_title, 1);
			me->setAxisTitle(_yaxis->_title, 2);
			for (unsigned int i=0; i<_xaxis->_labels.size(); i++)
				me->setBinLabel(i+1, _xaxis->_labels[i]);
			_mes.push_back(me);
		}
	}
}









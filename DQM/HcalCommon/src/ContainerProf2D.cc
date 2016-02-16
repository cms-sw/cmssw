
#include "DQM/HcalCommon/interface/ContainerProf2D.h"

namespace hcaldqm
{
	using namespace mapper;
	using namespace axis;
	using namespace constants;

	ContainerProf2D::ContainerProf2D()
	{
		_xaxis = NULL;
		_yaxis = NULL;
		_zaxis = NULL;
	}

	ContainerProf2D::ContainerProf2D(std::string const& folder,
		std::string const& nametitle, MapperType mt, Axis *xaxis,
		Axis *yaxis, Axis *zaxis):
		Container2D(folder, nametitle, mt, xaxis, yaxis, zaxis)
	{}
	
	/* virtual */ void ContainerProf2D::initialize(std::string const& folder,
		std::string const& nametitle, MapperType mt, Axis *xaxis,
		Axis *yaxis, Axis *zaxis, int debug/*=0*/)
	{
		Container2D::initialize(folder, nametitle, mt, xaxis, yaxis, zaxis, 
			debug);
	}

	/* virtual */ void ContainerProf2D::book(DQMStore::IBooker &ib,
		std::string subsystem, std::string aux)
	{
		unsigned int size = _mapper.getSize();
		 ib.setCurrentFolder(subsystem+"/"+_folder+aux);
		for (unsigned int i=0; i<size; i++)
		{
			std::string hname = _mapper.buildName(i);
			MonitorElement *me = ib.bookProfile2D(_name+"_"+hname,
				_name+" "+hname, _xaxis->_nbins, _xaxis->_min, _xaxis->_max,
				_yaxis->_nbins, _yaxis->_min, _yaxis->_max, 
				_zaxis->_min, _zaxis->_max);
			TObject *o = me->getRootObject();
			_xaxis->setLog(o);
			_yaxis->setLog(o);
			_zaxis->setLog(o);
			_xaxis->setBitAxisLS(o);
			_yaxis->setBitAxisLS(o);
			_xaxis->setBitAxisFlag(o);
			_yaxis->setBitAxisFlag(o);
			me->setAxisTitle(_xaxis->_title, 1);
			me->setAxisTitle(_yaxis->_title, 2);
			me->setAxisTitle(_zaxis->_title, 3);
			for (unsigned int i=0; i<_xaxis->_labels.size(); i++)
				me->setBinLabel(i+1, _xaxis->_labels[i], 1);
			for (unsigned int i=0; i<_yaxis->_labels.size(); i++)
				me->setBinLabel(i+1, _yaxis->_labels[i], 2);
			_mes.push_back(me);
		}
	}
}











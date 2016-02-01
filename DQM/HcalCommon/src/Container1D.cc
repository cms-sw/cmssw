
#include "DQM/HcalCommon/interface/Container1D.h"

namespace hcaldqm
{
	using namespace mapper;
	using namespace axis;
	using namespace constants;

	Container1D::Container1D()
	{
		_xaxis = NULL;
		_yaxis = NULL;
	}

	/* virtual */ Container1D::~Container1D()
	{
		delete _xaxis; _xaxis=NULL;
		delete _yaxis; _yaxis=NULL;
	}

	Container1D::Container1D(std::string const& folder, 
		std::string const& nametitle, mapper::MapperType mt, axis::Axis *xaxis,
		axis::Axis *yaxis):
		Container(folder, nametitle), _mapper(mt), _xaxis(xaxis), _yaxis(yaxis)
	{}

	/* virtuial */ void Container1D::initialize(std::string const& folder, 
		std::string const& nametitle, mapper::MapperType mt, axis::Axis *xaxis,
		axis::Axis *yaxis, int debug /* =0 */)
	{
		Container::initialize(folder, nametitle, debug);
		_mapper.initialize(mt, debug);
		_xaxis = xaxis;
		_yaxis = yaxis;
	}

	/* virtual */ void Container1D::fill(int id, int x)
	{
		_mes[_mapper.index(id)]->Fill(_xaxis->get(x));
	}
	/* virtual */ void Container1D::fill(int id, double x)
	{
		_mes[_mapper.index(id)]->Fill(_xaxis->get(x));
	}
	/* virtual */ void Container1D::fill(int id, int x, double y)
	{
		_mes[_mapper.index(id)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}
	/* virtual */ void Container1D::fill(int id, double x, double y)
	{
		_mes[_mapper.index(id)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void Container1D::fill(int id, int x, int y, double z)
	{
		_mes[_mapper.index(id)]->Fill(_xaxis->get(x), _yaxis->get(y), z);
	}

	/* virtual */ void Container1D::fill(int id, int x, double y, double z)
	{
		_mes[_mapper.index(id)]->Fill(_xaxis->get(x), _yaxis->get(y), z);
	}

	/* virtual */ void Container1D::fill(int id, double x, double y, double z)
	{
		_mes[_mapper.index(id)]->Fill(_xaxis->get(x), _yaxis->get(y), z);
	}

	/* virtual */ void Container1D::fill(HcalDetId const& did)
	{
		_mes[_mapper.index(did)]->Fill(_xaxis->get(did));
	}
	/* virtual */ void Container1D::fill(HcalDetId const& did, int x)
	{
		AxisQType act = _xaxis->getType();
		if (act==fValue || act==fFlag)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x));
		else if (act==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did), _yaxis->get(x));
	}
	/* virtual */ void Container1D::fill(HcalDetId const& did, double x)
	{
		AxisQType act = _xaxis->getType();
		if (act==fValue || act==fFlag)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x));
		else if (act==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did), _yaxis->get(x));
	}
	/* virtual */ void Container1D::fill(HcalDetId const& did, int x, double y)
	{
		AxisQType act = _xaxis->getType();
		if (act==fValue || act==fFlag)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x), _yaxis->get(y));
		else if (act==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did), _yaxis->get(x), y);
	}
	/* virtual */ void Container1D::fill(HcalDetId const& did, int x, int y)
	{
		AxisQType act = _xaxis->getType();
		if (act==fValue || act==fFlag)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x), _yaxis->get(y));
		else if (act==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did), _yaxis->get(x), y);
	}
	/* virtual */ void Container1D::fill(HcalDetId const& did, double x , 
			double y)
	{
		AxisQType act = _xaxis->getType();
		if (act==fValue || act==fFlag)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x), _yaxis->get(y));
		else if (act==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did), _yaxis->get(x), y);
	}

	/* virtual */ void Container1D::fill(HcalElectronicsId const& eid)
	{
		_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid));
	}
	/* virtual */ void Container1D::fill(HcalElectronicsId const& eid, int x)
	{
		AxisQType act = _xaxis->getType();
		if (act==fValue || act==fFlag)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(x));
		else if (act==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid), _yaxis->get(x));
	}
	/* virtual */ void Container1D::fill(HcalElectronicsId const& eid, double x)
	{
		AxisQType act = _xaxis->getType();
		if (act==fValue || act==fFlag)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(x));
		else if (act==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid), _yaxis->get(x));
	}
	/* virtual */ void Container1D::fill(HcalElectronicsId const& eid, 
		int x, double y)
	{
		AxisQType act = _xaxis->getType();
		if (act==fValue || act==fFlag)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(x), _yaxis->get(y));
		else if (act==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid), _yaxis->get(x), y);
	}
	/* virtual */ void Container1D::fill(HcalElectronicsId const& eid, 
		double x, double y)
	{
		AxisQType act = _xaxis->getType();
		if (act==fValue || act==fFlag)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(x), _yaxis->get(y));
		else if (act==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid), _yaxis->get(x), y);
	}

	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& tid)
	{
		_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid));
	}

	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& tid, int x)
	{
		AxisQType act = _xaxis->getType();
		if (act==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid), _yaxis->get(x));
		else
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x));
	}

	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& tid, double x)
	{
		AxisQType act = _xaxis->getType();
		if (act==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid), _yaxis->get(x));
		else
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x));
	}

	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& tid, int x,
		int y)
	{
		AxisQType act = _xaxis->getType();
		if (act==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid), _yaxis->get(x), y);
		else
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void Container1D::fill(HcalTrigTowerDetId const& tid, int x,
		double y)
	{
		AxisQType act = _xaxis->getType();
		if (act==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid), _yaxis->get(x), y);
		else
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void Container1D::book(DQMStore::IBooker& ib, 
		std::string subsystem, std::string aux)
	{

		unsigned int size = _mapper.getSize();
		ib.setCurrentFolder(subsystem+"/"+_folder+aux);
		for (unsigned int i=0; i<size; i++)
		{
//			utilities::log(_name);			
			std::string hname = _mapper.buildName(i);
			MonitorElement *me = ib.book1D(_name+"_"+hname, _name +" "+hname,
				_xaxis->_nbins, _xaxis->_min, _xaxis->_max);
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

	/* virtual */ double Container1D::getBinContent(unsigned int id, int x)
	{
		return _mes[id]->getBinContent(_xaxis->getBin(x));
	}

	/* virtual */ void Container1D::reset()
	{
		for (MEVector::const_iterator it=_mes.begin(); it!=_mes.end(); ++it)
			(*it)->Reset();
	}

	/* virtual */ MonitorElement* Container1D::at(unsigned int i)
	{
		return _mes[i];
	}

	/* virtual */ MonitorElement* Container1D::at(HcalDetId const& did)
	{
		return _mes[_mapper.index(did)];
	}

	/* virtual */ MonitorElement* Container1D::at(HcalElectronicsId const& eid)
	{
		return _mes[_mapper.index(eid)];
	}

	/* virtual */ MonitorElement* Container1D::at(HcalTrigTowerDetId const& tid)
	{
		return _mes[_mapper.index(tid)];
	}
}








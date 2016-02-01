
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/Utilities.h"

namespace hcaldqm
{
	using namespace hcaldqm::axis;
	using namespace hcaldqm::mapper;
	using namespace constants;

	Container2D::Container2D()
	{
		_xaxis = NULL;
		_yaxis = NULL;
		_zaxis = NULL;
	}

	/* virtual */ Container2D::~Container2D()
	{
		//	Container1D::~Container1D will be called as well
		delete _zaxis; _zaxis=NULL;
	}

	Container2D::Container2D(std::string const& folder, std::string nametitle,
		mapper::MapperType mt, axis::Axis *xaxis, axis::Axis* yaxis,
		axis::Axis *zaxis):
		Container1D(folder, nametitle, mt, xaxis, yaxis), _zaxis(zaxis)
	{}
	
	/* virtual */ void Container2D::initialize(std::string const& folder, 
		std::string nametitle,
		mapper::MapperType mt, axis::Axis *xaxis, axis::Axis* yaxis,
		axis::Axis *zaxis, int debug/*=0*/)
	{
		Container1D::initialize(folder, nametitle, mt, xaxis, yaxis, debug);
		_zaxis = zaxis;
	}

	/* virtual */ void Container2D::fill(HcalDetId const& did)
	{
		_mes[_mapper.index(did)]->Fill(_xaxis->get(did),
			_yaxis->get(did));
	}

	//	HcalDetId based
	/* virtual */ void Container2D::fill(HcalDetId const& did, int x)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did),
				_yaxis->get(did), x);
		else if (xact==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did),
				_yaxis->get(x));
		else if (yact==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x), _yaxis->get(did));
	}

	/* virtual */ void Container2D::fill(HcalDetId const& did, double x)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did),
				_yaxis->get(did), x);
		else if (xact==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did),
				_yaxis->get(x));
		else if (yact==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x), _yaxis->get(did));
	}

	/* virtual */ void Container2D::fill(HcalDetId const& did, 
		int x, double y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x), _yaxis->get(did), y);
		else if (yact!=fCoordinate && xact!=fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void Container2D::fill(HcalDetId const& did, 
		double x, double y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(did), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x), _yaxis->get(did), y);
		else if (yact!=fCoordinate && xact!=fCoordinate)
			_mes[_mapper.index(did)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	//	HcalElectronicsId based
	/* virtual */ void Container2D::fill(HcalElectronicsId const& eid)
	{
		_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid),
			_yaxis->get(eid));
	}

	/* virtual */ void Container2D::fill(HcalElectronicsId const& eid, int x)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid),
				_yaxis->get(eid), x);
		else if (xact==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid),
				_yaxis->get(x));
		else if (yact==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(x), _yaxis->get(eid));
	}

	/* virtual */ void Container2D::fill(HcalElectronicsId const& eid, double x)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid),
				_yaxis->get(eid), x);
		else if (xact==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid),
				_yaxis->get(x));
		else if (yact==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(x), _yaxis->get(eid));
	}

	/* virtual */ void Container2D::fill(HcalElectronicsId const& eid, 
		int x, double y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(x), _yaxis->get(eid), y);
		else if (yact!=fCoordinate && xact!=fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void Container2D::fill(HcalElectronicsId const& eid, 
		double x, double y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(eid), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(x), _yaxis->get(eid), y);
		else if (yact!=fCoordinate && xact!=fCoordinate)
			_mes[_mapper.index(eid)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void Container2D::fill(HcalTrigTowerDetId const& tid)
	{
		_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid), _yaxis->get(tid));
	}

	/* virtual */ void Container2D::fill(HcalTrigTowerDetId const& tid, int x)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid),
				_yaxis->get(tid), x);
		else if (xact==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid),
				_yaxis->get(x));
		else if (yact==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x), _yaxis->get(tid));
	}

	/* virtual */ void Container2D::fill(HcalTrigTowerDetId const& tid, double x)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid),
				_yaxis->get(tid), x);
		else if (xact==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid),
				_yaxis->get(x));
		else if (yact==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x), _yaxis->get(tid));
	}

	/* virtual */ void Container2D::fill(HcalTrigTowerDetId const& tid, 
		int x, int y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x), _yaxis->get(tid), y);
		else if (yact!=fCoordinate && xact!=fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void Container2D::fill(HcalTrigTowerDetId const& tid, 
		int x, double y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x), _yaxis->get(tid), y);
		else if (yact!=fCoordinate && xact!=fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void Container2D::fill(HcalTrigTowerDetId const& tid, 
		double x, double y)
	{
		AxisQType xact = _xaxis->getType();
		AxisQType yact = _yaxis->getType();
		if (xact==fCoordinate && yact!=fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(tid), _yaxis->get(x), y);
		else if (xact!=fCoordinate && yact==fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x), _yaxis->get(tid), y);
		else if (yact!=fCoordinate && xact!=fCoordinate)
			_mes[_mapper.index(tid)]->Fill(_xaxis->get(x), _yaxis->get(y));
	}

	/* virtual */ void Container2D::book(DQMStore::IBooker &ib, 
		std::string subsystem, std::string aux)
	{
		unsigned int size = _mapper.getSize();
		ib.setCurrentFolder(subsystem+"/"+_folder+aux);
		for (unsigned int i=0; i<size; i++)
		{
			std::string hname = _mapper.buildName(i);
			MonitorElement *me = ib.book2D(_name+"_"+hname,
				_name+" "+hname, _xaxis->_nbins, _xaxis->_min, _xaxis->_max,
				_yaxis->_nbins, _yaxis->_min, _yaxis->_max);
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

	/* virtual */ void Container2D::setBinContent(int id, int x, int y, double z)
	{
		_mes[id]->setBinContent(_xaxis->getBin(x),
			_yaxis->getBin(y), z);
	}

	/* virtual */ void Container2D::setBinContent(unsigned int id, 
		int x, int y, double z)
	{
		_mes[id]->setBinContent(_xaxis->getBin(x),
			_yaxis->getBin(y), z);
	}

	/* virtual */ void Container2D::setBinContent(int id, int x, double y, double z)
	{
		_mes[id]->setBinContent(_xaxis->getBin(x),
			_yaxis->getBin(y), z);
	}

	/* virtual */ void Container2D::setBinContent(int id, double x, int y, double z)
	{
		_mes[id]->setBinContent(_xaxis->getBin(x),
			_yaxis->getBin(y), z);
	}

	/* virtual */ void Container2D::setBinContent(int id, double x, double y, double z)
	{
		_mes[id]->setBinContent(_xaxis->getBin(x),
			_yaxis->getBin(y), z);
	}

	/* virtual */ void Container2D::loadLabels(
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
}



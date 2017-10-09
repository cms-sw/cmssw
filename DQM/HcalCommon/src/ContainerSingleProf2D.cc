#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"

namespace hcaldqm
{
	ContainerSingleProf2D::ContainerSingleProf2D()
	{
		_qx = NULL;
		_qy = NULL;
		_qz = NULL;
	}

	ContainerSingleProf2D::ContainerSingleProf2D(std::string const& folder,
		Quantity *qx, Quantity *qy, Quantity *qz):
		ContainerSingle2D(folder, qx, qy, qz)
	{
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
		_qz->setAxisType(quantity::fZAxis);
	}

	/* virtual */ void ContainerSingleProf2D::initialize(std::string const& 
		folder, Quantity *qx, Quantity *qy, Quantity *qz,
		int debug/*=0*/)
	{
		ContainerSingle2D::initialize(folder, qx, qy, qz, debug);
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
		_qz->setAxisType(quantity::fZAxis);
	}
	
	/* virtual */ void ContainerSingleProf2D::initialize(std::string const& 
		folder, std::string const& qname,
		Quantity *qx, Quantity *qy, Quantity *qz,
		int debug/*=0*/)
	{
		ContainerSingle2D::initialize(folder, qname, qx, qy, qz, debug);
		_qx->setAxisType(quantity::fXAxis);
		_qy->setAxisType(quantity::fYAxis);
		_qz->setAxisType(quantity::fZAxis);
	}

	/* virtual */ void ContainerSingleProf2D::book(DQMStore::IBooker& ib,
		std::string subsystem, std::string aux)
	{
		ib.setCurrentFolder(subsystem+"/"+_folder+"/"+_qname);
		_me = ib.bookProfile2D(_qname+(aux==""?aux:"_"+aux), 
			_qname+(aux==""?aux:" "+aux),
			_qx->nbins(), _qx->min(), _qx->max(),
			_qy->nbins(), _qy->min(), _qy->max(),
			_qz->min(), _qz->max());
		customize();
	}

	/* virtual */ void ContainerSingleProf2D::book(DQMStore *store,
		std::string subsystem, std::string aux)
	{
		store->setCurrentFolder(subsystem+"/"+_folder+"/"+_qname);
		_me = store->bookProfile2D(_qname+(aux==""?aux:"_"+aux), 
			_qname+(aux==""?aux:" "+aux),
			_qx->nbins(), _qx->min(), _qx->max(),
			_qy->nbins(), _qy->min(), _qy->max(),
			_qz->min(), _qz->max());
		customize();
	}
}




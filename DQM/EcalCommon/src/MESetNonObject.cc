#include "DQM/EcalCommon/interface/MESetNonObject.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm
{
  MESetNonObject::MESetNonObject(std::string const& _fullpath, MEData const& _data, bool _readOnly/* = false*/) :
    MESet(_fullpath, _data, _readOnly)
  {
  }

  void
  MESetNonObject::book()
  {
    dqmStore_->setCurrentFolder(dir_);

    mes_.clear();

    MonitorElement* me(0);

    BinService::AxisSpecs* xaxis(data_->xaxis);
    BinService::AxisSpecs* yaxis(data_->yaxis);
    BinService::AxisSpecs* zaxis(data_->zaxis);

    switch(data_->kind) {
    case MonitorElement::DQM_KIND_REAL :
      me = dqmStore_->bookFloat(name_);
      break;

    case MonitorElement::DQM_KIND_TH1F :
      {
	if(!xaxis)
	  throw cms::Exception("InvalidCall") << "No xaxis found for MESetNonObject" << std::endl;

	if(!xaxis->edges)
	  me = dqmStore_->book1D(name_, name_, xaxis->nbins, xaxis->low, xaxis->high);
	else{
	  float* edges(new float[xaxis->nbins + 1]);
	  for(int i(0); i < xaxis->nbins + 1; i++)
	    edges[i] = xaxis->edges[i];
	  me = dqmStore_->book1D(name_, name_, xaxis->nbins, edges);
	  delete [] edges;
	}
      }
      break;

    case MonitorElement::DQM_KIND_TPROFILE :
      {
	if(!xaxis)
	  throw cms::Exception("InvalidCall") << "No xaxis found for MESetNonObject" << std::endl;

	double ylow, yhigh;
	if(!yaxis){
	  ylow = -std::numeric_limits<double>::max();
	  yhigh = std::numeric_limits<double>::max();
	}
	else{
	  ylow = yaxis->low;
	  yhigh = yaxis->high;
	}
	if(xaxis->edges)
	  me = dqmStore_->bookProfile(name_, name_, xaxis->nbins, xaxis->edges, ylow, yhigh, "");
	else
	  me = dqmStore_->bookProfile(name_, name_, xaxis->nbins, xaxis->low, xaxis->high, ylow, yhigh, "");

      }
      break;

    case MonitorElement::DQM_KIND_TH2F :
      {
	if(!xaxis || !yaxis)
	  throw cms::Exception("InvalidCall") << "No x/yaxis found for MESetNonObject" << std::endl;

	if(!xaxis->edges || !yaxis->edges)
	  me = dqmStore_->book2D(name_, name_, xaxis->nbins, xaxis->low, xaxis->high, yaxis->nbins, yaxis->low, yaxis->high);
	else{
	  float* xedges(new float[xaxis->nbins + 1]);
	  for(int i(0); i < xaxis->nbins + 1; i++)
	    xedges[i] = xaxis->edges[i];
	  float* yedges(new float[yaxis->nbins + 1]);
	  for(int i(0); i < yaxis->nbins + 1; i++)
	    yedges[i] = yaxis->edges[i];
	  me = dqmStore_->book2D(name_, name_, xaxis->nbins, xedges, yaxis->nbins, yedges);
	  delete [] xedges;
	  delete [] yedges;
	}
      }
      break;

    case MonitorElement::DQM_KIND_TPROFILE2D :
      {
	if(!xaxis || !yaxis)
	  throw cms::Exception("InvalidCall") << "No x/yaxis found for MESetNonObject" << std::endl;
	double high(0.), low(0.);
	if(zaxis){
	  low = zaxis->low;
	  high = zaxis->high;
	}
	else{
	  low = -std::numeric_limits<double>::max();
	  high = std::numeric_limits<double>::max();
	}

	me = dqmStore_->bookProfile2D(name_, name_, xaxis->nbins, xaxis->low, xaxis->high, yaxis->nbins, yaxis->low, yaxis->high, low, high, "");
      }
      break;

    default :
      throw cms::Exception("InvalidCall") << "MESetNonObject of type " << data_->kind << " not implemented" << std::endl;
    }

    mes_.push_back(me);

    active_ = true;
  }

  bool
  MESetNonObject::retrieve() const
  {
    mes_.clear();

    MonitorElement* me(dqmStore_->get(dir_ + "/" + name_));
    if(!me) return false;

    mes_.push_back(me);

    active_ = true;
    return true;
  }

  MESetNonObject::~MESetNonObject()
  {
  }

  void
  MESetNonObject::fill(double _x, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(mes_.size() == 0 || !mes_[0]) return;

    switch(data_->kind) {
    case MonitorElement::DQM_KIND_REAL :
      mes_[0]->Fill(_x);
      break;
    case MonitorElement::DQM_KIND_TH1F :
    case MonitorElement::DQM_KIND_TPROFILE :
      mes_[0]->Fill(_x, _wy);
      break;
    case MonitorElement::DQM_KIND_TH2F :
    case MonitorElement::DQM_KIND_TPROFILE2D :
      mes_[0]->Fill(_x, _wy, _w);
      break;
    default :
      break;
    }
  }

}

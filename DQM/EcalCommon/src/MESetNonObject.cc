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

    switch(data_->kind) {
    case MonitorElement::DQM_KIND_REAL :
      me = dqmStore_->bookFloat(name_);
      break;

    case MonitorElement::DQM_KIND_TH1F :
      {
	if(!data_->xaxis)
	  throw cms::Exception("InvalidCall") << "No xaxis found for MESetNonObject" << std::endl;

	if(!data_->xaxis->edges)
	  me = dqmStore_->book1D(name_, name_, data_->xaxis->nbins, data_->xaxis->low, data_->xaxis->high);
	else
	  me = dqmStore_->book1D(name_, name_, data_->xaxis->nbins, data_->xaxis->edges);
      }
      break;

    case MonitorElement::DQM_KIND_TPROFILE :
      {
	if(!data_->xaxis)
	  throw cms::Exception("InvalidCall") << "No xaxis found for MESetNonObject" << std::endl;

	float ylow, yhigh;
	if(!data_->yaxis) {
	  ylow = -std::numeric_limits<double>::max();
	  yhigh = std::numeric_limits<double>::max();
	}
	else {
	  ylow = data_->yaxis->low;
	  yhigh = data_->yaxis->high;
	}
	if(data_->xaxis->edges){
	  double* xedges(new double[data_->xaxis->nbins + 1]);
	  for(int i = 0; i <= data_->xaxis->nbins; i++) xedges[i] = data_->xaxis->edges[i];
	  me = dqmStore_->bookProfile(name_, name_, data_->xaxis->nbins, xedges, ylow, yhigh);
	  delete [] xedges;
	}
	else
	  me = dqmStore_->bookProfile(name_, name_, data_->xaxis->nbins, data_->xaxis->low, data_->xaxis->high, ylow, yhigh);

      }
      break;

    case MonitorElement::DQM_KIND_TH2F :
      {
	if(!data_->xaxis || !data_->yaxis)
	  throw cms::Exception("InvalidCall") << "No x/yaxis found for MESetNonObject" << std::endl;

	if(!data_->xaxis->edges || !data_->yaxis->edges)
	  me = dqmStore_->book2D(name_, name_, data_->xaxis->nbins, data_->xaxis->low, data_->xaxis->high, data_->yaxis->nbins, data_->yaxis->low, data_->yaxis->high);
	else
	  me = dqmStore_->book2D(name_, name_, data_->xaxis->nbins, data_->xaxis->edges, data_->yaxis->nbins, data_->yaxis->edges);
      }
      break;

    case MonitorElement::DQM_KIND_TPROFILE2D :
      {
	if(!data_->xaxis || !data_->yaxis)
	  throw cms::Exception("InvalidCall") << "No x/yaxis found for MESetNonObject" << std::endl;

	me = dqmStore_->book2D(name_, name_, data_->xaxis->nbins, data_->xaxis->edges, data_->yaxis->nbins, data_->yaxis->edges);
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
  MESetNonObject::fill(float _x, float _wy/* = 1.*/, float _w/* = 1.*/)
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

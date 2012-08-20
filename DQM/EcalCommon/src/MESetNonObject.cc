#include "DQM/EcalCommon/interface/MESetNonObject.h"

namespace ecaldqm
{
  MESetNonObject::MESetNonObject(MEData const& _data) :
    MESet(_data)
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
	  throw_("No xaxis found for MESetNonObject");

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
	  throw_("No xaxis found for MESetNonObject");

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
	  throw_("No x/yaxis found for MESetNonObject");

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
	  throw_("No x/yaxis found for MESetNonObject");
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
      throw_("Unsupported MonitorElement kind");
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
    if(!active_) return;

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

  void
  MESetNonObject::setBinContent(int _bin, double _content)
  {
    if(!active_) return;
    if(data_->kind == MonitorElement::DQM_KIND_REAL) return;

    if(mes_.size() == 0 || !mes_[0]) return;

    mes_[0]->setBinContent(_bin, _content);
  }

  void
  MESetNonObject::setBinError(int _bin, double _error)
  {
    if(!active_) return;
    if(data_->kind == MonitorElement::DQM_KIND_REAL) return;

    if(mes_.size() == 0 || !mes_[0]) return;

    mes_[0]->setBinError(_bin, _error);
  }

  void
  MESetNonObject::setBinEntries(int _bin, double _entries)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    if(mes_.size() == 0 || !mes_[0]) return;

    mes_[0]->setBinEntries(_bin, _entries);
  }

  double
  MESetNonObject::getBinContent(int _bin) const
  {
    if(!active_) return 0.;
    if(data_->kind == MonitorElement::DQM_KIND_REAL) return 0.;

    if(mes_.size() == 0 || !mes_[0]) return 0.;

    return mes_[0]->getBinContent(_bin);
  }

  double
  MESetNonObject::getBinError(int _bin) const
  {
    if(!active_) return 0.;
    if(data_->kind == MonitorElement::DQM_KIND_REAL) return 0.;

    if(mes_.size() == 0 || !mes_[0]) return 0.;

    return mes_[0]->getBinError(_bin);
  }

  double
  MESetNonObject::getBinEntries(int _bin) const
  {
    if(!active_) return 0.;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    if(mes_.size() == 0 || !mes_[0]) return 0.;

    return mes_[0]->getBinEntries(_bin);
  }

  int
  MESetNonObject::findBin(double _x, double _y/* = 0.*/) const
  {
    if(!active_) return 0;

    if(mes_.size() == 0 || !mes_[0]) return 0;

    if(data_->kind == MonitorElement::DQM_KIND_TH1F || data_->kind == MonitorElement::DQM_KIND_TPROFILE)
      return mes_[0]->getTH1()->FindBin(_x);
    else if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D)
      return mes_[0]->getTH1()->FindBin(_x, _y);
    else
      return 0;
  }
}

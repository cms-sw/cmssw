#include "DQM/EcalCommon/interface/MESetDet2D.h"

namespace ecaldqm
{

  MESetDet2D::MESetDet2D(std::string const& _fullpath, MEData const& _data, bool _readOnly/* = false*/) :
    MESetEcal(_fullpath, _data, 2, _readOnly)
  {
  }

  MESetDet2D::~MESetDet2D()
  {
  }

  void
  MESetDet2D::fill(DetId const& _id, double _w/* = 1.*/, double, double)
  {
    find_(_id);

    if(data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      if(getBinEntries(_id) < 0.){
	setBinEntries(_id, 0.);
	setBinContent(_id, 0., 0.);
      }
    }

    fill_(_w);
  }

}

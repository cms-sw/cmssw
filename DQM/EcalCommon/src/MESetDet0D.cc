#include "DQM/EcalCommon/interface/MESetDet0D.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm
{

  MESetDet0D::MESetDet0D(std::string const& _fullpath, MEData const& _data, bool _readOnly/* = false*/) :
    MESetEcal(_fullpath, _data, 0, _readOnly)
  {
  }

  MESetDet0D::~MESetDet0D()
  {
  }

  void
  MESetDet0D::fill(DetId const& _id, float _value, float, float)
  {
    unsigned offset(binService_->findOffset(data_->otype, _id));
    if(offset >= mes_.size() || !mes_[offset])
      throw cms::Exception("InvalidCall") << "ME array index overflow" << std::endl;

    mes_[offset]->Fill(_value);
  }

  void
  MESetDet0D::fill(unsigned _dcctccid, float _value, float, float)
  {
    unsigned offset(binService_->findOffset(data_->otype, data_->btype, _dcctccid));
    if(offset >= mes_.size() || !mes_[offset])
      throw cms::Exception("InvalidCall") << "ME array index overflow" << offset << std::endl;

    mes_[offset]->Fill(_value);
  }

  void
  MESetDet0D::fill(float _value, float, float)
  {
    if(!(data_->otype == BinService::kEcal && mes_.size() == 1))
      throw cms::Exception("InvalidCall") << "ME type incompatible";

    mes_[0]->Fill(_value);
  }

}

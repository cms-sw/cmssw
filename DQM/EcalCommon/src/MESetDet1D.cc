#include "DQM/EcalCommon/interface/MESetDet1D.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

namespace ecaldqm
{

  MESetDet1D::MESetDet1D(std::string const& _fullpath, MEData const& _data, bool _readOnly/* = false*/) :
    MESetEcal(_fullpath, _data, 1, _readOnly)
  {
  }

  MESetDet1D::~MESetDet1D()
  {
  }

  void
  MESetDet1D::fill(DetId const& _id, float _wy/* = 1.*/, float _w/* = 1.*/, float)
  {
    find_(_id);

    fill_(_wy, _w);
  }

  void
  MESetDet1D::fill(unsigned _dcctccid, float _wy/* = 1.*/, float _w/* = 1.*/, float)
  {
    find_(_dcctccid);

    fill_(_wy, _w);
  }

  float
  MESetDet1D::getBinContent(DetId const& _id, int _bin/* = 0*/) const
  {
    find_(_id);

    if(cache_.second.size() == 0) return 0.;

    int bin(cache_.second[0]);

    if(_bin > 0 && (data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D)){
      int nbinsX(mes_[cache_.first]->getTH1()->GetNbinsX());
      bin += (_bin - 1) * nbinsX;
    }

    return getBinContent_(cache_.first, bin);
  }

  float
  MESetDet1D::getBinContent(unsigned _dcctccid, int _bin/* = 0*/) const
  {
    find_(_dcctccid);

    if(cache_.second.size() == 0) return 0.;

    int bin(cache_.second[0]);

    if(_bin > 0 && (data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D)){
      int nbinsX(mes_[cache_.first]->getTH1()->GetNbinsX());
      bin += (_bin - 1) * nbinsX;
    }

    return getBinContent_(cache_.first, bin);
  }

  float
  MESetDet1D::getBinError(DetId const& _id, int _bin/* = 0*/) const
  {
    find_(_id);

    if(cache_.second.size() == 0) return 0.;

    int bin(cache_.second[0]);

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsX(mes_[cache_.first]->getTH1()->GetNbinsX());
      bin += (_bin - 1) * nbinsX;
    }

    return getBinError_(cache_.first, bin);
  }

  float
  MESetDet1D::getBinError(unsigned _dcctccid, int _bin/* = 0*/) const
  {
    find_(_dcctccid);

    if(cache_.second.size() == 0) return 0.;

    int bin(cache_.second[0]);

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsX(mes_[cache_.first]->getTH1()->GetNbinsX());
      bin += (_bin - 1) * nbinsX;
    }

    return getBinError_(cache_.first, bin);
  }

  float
  MESetDet1D::getBinEntries(DetId const& _id, int _bin/* = 0*/) const
  {
    find_(_id);

    if(cache_.second.size() == 0) return 0.;

    int bin(cache_.second[0]);

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsX(mes_[cache_.first]->getTH1()->GetNbinsX());
      bin += (_bin - 1) * nbinsX;
    }

    return getBinEntries_(cache_.first, bin);
  }

  float
  MESetDet1D::getBinEntries(unsigned _dcctccid, int _bin/* = 0*/) const
  {
    find_(_dcctccid);

    if(cache_.second.size() == 0) return 0.;

    int bin(cache_.second[0]);

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsX(mes_[cache_.first]->getTH1()->GetNbinsX());
      bin += (_bin - 1) * nbinsX;
    }

    return getBinEntries_(cache_.first, bin);
  }

  void
  MESetDet1D::find_(uint32_t _id) const
  {
    if(_id == cacheId_) return;

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      DetId id(_id);
      if(id.det() == DetId::Ecal)
	cache_ = binService_->findBinsNoMap(data_->otype, data_->btype, id);
      else
	cache_ = binService_->findBins(data_->otype, data_->btype, _id);
    }
    else{
      DetId id(_id);
      if(id.det() == DetId::Ecal)
	cache_ = binService_->findBins(data_->otype, data_->btype, id);
      else
	cache_ = binService_->findBins(data_->otype, data_->btype, unsigned(_id));
    }

    if(cache_.first >= mes_.size() || !mes_[cache_.first])
      throw cms::Exception("InvalidCall") << "ME array index overflow" << std::endl;

    // some TTs are apparently empty..!
//     if(cache_.second.size() == 0)
//       throw cms::Exception("InvalidCall") << "No bins to get content from" << std::endl;

    cacheId_ = _id;
  }

  void
  MESetDet1D::fill_(float _wy, float _w)
  {
    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      std::vector<int> bins(cache_.second);
      TH1* h(mes_[cache_.first]->getTH1());
      int iy(h->GetYaxis()->FindBin(_wy));
      int nbinsX(h->GetNbinsX());
      for(unsigned iBin(0); iBin < bins.size(); iBin++)
	bins[iBin] += (iy - 1) * nbinsX;

      for(unsigned iBin(0); iBin < bins.size(); iBin++)
	MESet::fill_(cache_.first, bins[iBin], _w);
    }
    else
      MESetEcal::fill_(_wy);
  }

}


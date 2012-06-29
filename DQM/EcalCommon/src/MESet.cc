#include "DQM/EcalCommon/interface/MESet.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TString.h"
#include "TPRegexp.h"

namespace ecaldqm
{
  BinService const* MESet::binService_(0);
  DQMStore* MESet::dqmStore_(0);

  MESet::MESet(std::string const& _fullpath, MEData const& _data, bool _readOnly/* = false*/) :
    mes_(0),
    dir_(_fullpath.substr(0, _fullpath.find_last_of('/'))),
    name_(_fullpath.substr(_fullpath.find_last_of('/') + 1)),
    data_(&_data),
    active_(false),
    readOnly_(_readOnly)
  {
    if (!binService_) {
      binService_ = &(*(edm::Service<EcalDQMBinningService>()));
      if(!binService_)
	throw cms::Exception("Service") << "EcalDQMBinningService not found" << std::endl;
    }

    if (!dqmStore_) {
      dqmStore_ = &(*(edm::Service<DQMStore>()));
      if(!dqmStore_)
	throw cms::Exception("Service") << "DQMStore not found" << std::endl;
    }

    // expand full path into dir & name
    if(_fullpath.size() == 0)
      throw cms::Exception("InvalidConfiguration") << "MonitorElement path empty";
  }

  MESet::~MESet()
  {
  }

  void
  MESet::book()
  {
    clear();
    active_ = true;
  }

  bool
  MESet::retrieve() const
  {
    return false;
  }

  void
  MESet::clear() const
  {
    active_ = false;
    mes_.clear();
  }

  void
  MESet::setAxisTitle(std::string const& _title, int _axis/* = 1*/)
  {
    for(std::vector<MonitorElement*>::iterator meItr(mes_.begin()); meItr != mes_.end(); ++meItr)
      (*meItr)->setAxisTitle(_title, _axis);
  }

  void
  MESet::setBinLabel(unsigned _offset, int _bin, std::string const& _label, int _axis/* = 1*/)
  {
    if(_offset == unsigned(-1)){
      for(std::vector<MonitorElement*>::iterator meItr(mes_.begin()); meItr != mes_.end(); ++meItr)
	(*meItr)->setBinLabel(_bin, _label, _axis);

      return;
    }

    if(_offset >= mes_.size() || !mes_[_offset]) return;
    mes_[_offset]->setBinLabel(_bin, _label, _axis);
  }

  void
  MESet::reset(double _content/* = 0.*/, double _err/* = 0.*/, double _entries/* = 0.*/)
  {
    resetAll(_content, _err, _entries);
  }

  void
  MESet::resetAll(double _content/* = 0.*/, double _err/* = 0.*/, double _entries/* = 0.*/)
  {
    if(data_->kind == MonitorElement::DQM_KIND_REAL){
      for(std::vector<MonitorElement*>::iterator meItr(mes_.begin()); meItr != mes_.end(); ++meItr)
	(*meItr)->Fill(_content);
      return;
    }

    bool simple(true);
    if(_content != 0. || _err != 0. || _entries != 0.) simple = false;

    for(std::vector<MonitorElement*>::iterator meItr(mes_.begin()); meItr != mes_.end(); ++meItr){
      TH1* h((*meItr)->getTH1());
      h->Reset();
      if(simple) continue;

      int nbinsX(h->GetNbinsX());
      int nbinsY(h->GetNbinsY());
      double entries(0.);
      for(int ix(1); ix <= nbinsX; ix++){
	for(int iy(1); iy <= nbinsY; iy++){
	  int bin(h->GetBin(ix, iy));
	  h->SetBinContent(bin, _content);
	  h->SetBinError(bin, _err);
	  if(data_->kind == MonitorElement::DQM_KIND_TPROFILE){
	    static_cast<TProfile*>(h)->SetBinEntries(bin, _entries);
	    entries += _entries;
	  }
	  else if(data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
	    static_cast<TProfile2D*>(h)->SetBinEntries(bin, _entries);
	    entries += _entries;
	  }
	}
      }
      if(entries == 0.) entries = _entries;
      h->SetEntries(_entries);
    }
  }

  void
  MESet::fill(DetId const&, double, double, double)
  {
  }

  void
  MESet::fill(EcalElectronicsId const& _id, double _wx/* = 1.*/,double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    fill(getElectronicsMap()->getDetId(_id), _wx, _wy, _w);
  }

  void
  MESet::fill(unsigned _dcctccid, double _wx/* = 1.*/, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
  }

  void
  MESet::fill(double, double, double)
  {
  }

  void
  MESet::setBinContent(DetId const&, double, double)
  {
  }

  void
  MESet::setBinContent(EcalElectronicsId const& _id, double _content, double _err/* = 0.*/)
  {
    setBinContent(getElectronicsMap()->getDetId(_id), _content, _err);
  }

  void
  MESet::setBinContent(unsigned, double, double)
  {
  }

  void
  MESet::setBinEntries(DetId const&, double)
  {
  }

  void
  MESet::setBinEntries(EcalElectronicsId const& _id, double _entries)
  {
    setBinEntries(getElectronicsMap()->getDetId(_id), _entries);
  }

  void
  MESet::setBinEntries(unsigned, double)
  {
  }

  double
  MESet::getBinContent(DetId const&, int) const
  {
    return 0.;
  }

  double
  MESet::getBinContent(EcalElectronicsId const& _id, int _bin/* = 0*/) const
  {
    return getBinContent(getElectronicsMap()->getDetId(_id), _bin);
  }

  double
  MESet::getBinContent(unsigned, int) const
  {
    return 0.;
  }

  double
  MESet::getBinError(DetId const&, int) const
  {
    return 0.;
  }

  double
  MESet::getBinError(EcalElectronicsId const& _id, int _bin/* = 0*/) const
  {
    return getBinError(getElectronicsMap()->getDetId(_id), _bin);
  }

  double
  MESet::getBinError(unsigned, int) const
  {
    return 0.;
  }

  double
  MESet::getBinEntries(DetId const&, int) const
  {
    return 0.;
  }

  double
  MESet::getBinEntries(EcalElectronicsId const& _id, int _bin/* = 0*/) const
  {
    return getBinEntries(getElectronicsMap()->getDetId(_id), _bin);
  }

  double
  MESet::getBinEntries(unsigned, int) const
  {
    return 0.;
  }

  void
  MESet::name(std::map<std::string, std::string> const& _replacements) const
  {
    TString dir(dir_);
    TString name(name_);

    for(std::map<std::string, std::string>::const_iterator repItr(_replacements.begin()); repItr != _replacements.end(); ++repItr){

      TString pattern("\\%\\(");
      pattern += repItr->first;
      pattern += "\\)s";

      TPRegexp re(pattern);

      re.Substitute(dir, repItr->second, "g");
      re.Substitute(name, repItr->second, "g");
    }

    dir_ = dir.Data();
    name_ = name.Data();
  }

  void
  MESet::fill_(unsigned _index, int _bin, double _w)
  {
    MonitorElement* me(mes_.at(_index));

    TH1* h(me->getTH1());

    int nbinsX(h->GetNbinsX());

    double x(h->GetXaxis()->GetBinCenter((_bin - 1) % nbinsX + 1));

    if((data_->kind < MonitorElement::DQM_KIND_TH2F && data_->kind >= MonitorElement::DQM_KIND_TH1F) || data_->kind == MonitorElement::DQM_KIND_TPROFILE) {
      me->Fill(x, _w);
      return;
    }

    double y(h->GetYaxis()->GetBinCenter((_bin - 1) / nbinsX + 1));

    me->Fill(x, y, _w);
  }

  void
  MESet::fill_(unsigned _offset, double _x, double _wy, double _w)
  {
    if(data_->kind == MonitorElement::DQM_KIND_REAL)
      mes_.at(_offset)->Fill(_x);
    else if(data_->kind < MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE)
      mes_.at(_offset)->Fill(_x, _wy);
    else
      mes_.at(_offset)->Fill(_x, _wy, _w);
  }

  void
  MESet::setBinContent_(unsigned _index, int _bin, double _content, double _err)
  {
    MonitorElement* me(mes_.at(_index));

    if(data_->kind < MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE){
      me->setBinContent(_bin, _content);
      me->setBinError(_bin, _err);
    }
    else{
      TH1* h(me->getTH1());
      int nbinsX(h->GetNbinsX());
      int ix((_bin - 1) % nbinsX + 1);
      int iy((_bin - 1) / nbinsX + 1);
      me->setBinContent(ix, iy, _content);
      me->setBinError(ix, iy, _err);
    }
  }

  void
  MESet::setBinEntries_(unsigned _index, int _bin, double _entries)
  {
    MonitorElement* me(mes_.at(_index));

    if(data_->kind == MonitorElement::DQM_KIND_TPROFILE){
      me->setBinEntries(_bin, _entries);
    }
    else if(data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      TH1* h(me->getTH1());
      int nbinsX(h->GetNbinsX());
      int ix((_bin - 1) % nbinsX + 1);
      int iy((_bin - 1) / nbinsX + 1);
      me->setBinEntries(h->GetBin(ix, iy), _entries);
    }
  }

  double
  MESet::getBinContent_(unsigned _index, int _bin) const
  {
    MonitorElement* me(mes_.at(_index));

    if(data_->kind < MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE)
      return me->getBinContent(_bin);
    else{
      TH1* h(me->getTH1());
      int nbinsX(h->GetNbinsX());
      int ix((_bin - 1) % nbinsX + 1);
      int iy((_bin - 1) / nbinsX + 1);
      return h->GetBinContent(ix, iy);
    }
  }

  double
  MESet::getBinError_(unsigned _index, int _bin) const
  {
    MonitorElement* me(mes_.at(_index));

    if(data_->kind < MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE)
      return me->getBinError(_bin);
    else{
      TH1* h(me->getTH1());
      int nbinsX(h->GetNbinsX());
      int ix((_bin - 1) % nbinsX + 1);
      int iy((_bin - 1) / nbinsX + 1);
      return h->GetBinError(ix, iy);
    }
  }

  double
  MESet::getBinEntries_(unsigned _index, int _bin) const
  {
    MonitorElement* me(mes_.at(_index));

    switch(data_->kind){
    case MonitorElement::DQM_KIND_TH1F:
      return me->getBinContent(_bin);
    case MonitorElement::DQM_KIND_TPROFILE:
      return me->getBinEntries(_bin);
    case MonitorElement::DQM_KIND_TH2F:
      {
	TH1* h(me->getTH1());
	int nbinsX(h->GetNbinsX());
	int ix((_bin - 1) % nbinsX + 1);
	int iy((_bin - 1) / nbinsX + 1);
	int bin(h->GetBin(ix, iy));
	return me->getBinContent(bin);
      }
    case MonitorElement::DQM_KIND_TPROFILE2D:
      {
	TH1* h(me->getTH1());
	int nbinsX(h->GetNbinsX());
	int ix((_bin - 1) % nbinsX + 1);
	int iy((_bin - 1) / nbinsX + 1);
	int bin(h->GetBin(ix, iy));
	return me->getBinEntries(bin);
      }
    default:
      return 0.;
    }
  }

}

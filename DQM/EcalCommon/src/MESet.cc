#include "DQM/EcalCommon/interface/MESet.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TString.h"
#include "TPRegexp.h"

namespace ecaldqm
{
  BinService const* MESet::binService_(0);
  DQMStore* MESet::dqmStore_(0);

  MESet::MESet(MEData const& _data) :
    mes_(0),
    dir_(_data.fullPath.substr(0, _data.fullPath.find_last_of('/'))),
    name_(_data.fullPath.substr(_data.fullPath.find_last_of('/') + 1)),
    data_(&_data),
    active_(false)
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

    if(dir_.find("/") == std::string::npos ||
       (data_->otype != BinService::kChannel && data_->btype != BinService::kReport && name_.size() == 0))
      throw_(_data.fullPath + " cannot be used for ME path name");

    switch(data_->kind){
    case MonitorElement::DQM_KIND_REAL:
    case MonitorElement::DQM_KIND_TH1F:
    case MonitorElement::DQM_KIND_TPROFILE:
    case MonitorElement::DQM_KIND_TH2F:
    case MonitorElement::DQM_KIND_TPROFILE2D:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }

    // expand full path into dir & name
    if(_data.fullPath.size() == 0)
      throw_("MonitorElement path empty");
  }

  MESet::~MESet()
  {
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
    unsigned nME(mes_.size());
    for(unsigned iME(0); iME < nME; iME++)
      mes_[iME]->setAxisTitle(_title, _axis);
  }

  void
  MESet::setBinLabel(unsigned _iME, int _bin, std::string const& _label, int _axis/* = 1*/)
  {
    unsigned nME(mes_.size());

    if(_iME == unsigned(-1)){
      for(unsigned iME(0); iME < nME; iME++)
	mes_[iME]->setBinLabel(_bin, _label, _axis);

      return;
    }

    if(_iME >= nME || !mes_[_iME]) return;
    mes_[_iME]->setBinLabel(_bin, _label, _axis);
  }

  void
  MESet::reset(double _content/* = 0.*/, double _err/* = 0.*/, double _entries/* = 0.*/)
  {
    resetAll(_content, _err, _entries);
  }

  void
  MESet::resetAll(double _content/* = 0.*/, double _err/* = 0.*/, double _entries/* = 0.*/)
  {
    unsigned nME(mes_.size());

    if(data_->kind == MonitorElement::DQM_KIND_REAL){
      for(unsigned iME(0); iME < nME; iME++)
	mes_[iME]->Fill(_content);
      return;
    }

    bool simple(true);
    if(_content != 0. || _err != 0. || _entries != 0.) simple = false;

    for(unsigned iME(0); iME < nME; iME++){
      TH1* h(mes_[iME]->getTH1());
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
  MESet::formName(std::map<std::string, std::string> const& _replacements) const
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
  MESet::fill_(unsigned _iME, int _bin, double _w)
  {
    if(data_->kind == MonitorElement::DQM_KIND_REAL) return;

    MonitorElement* me(mes_.at(_iME));

    TH1* h(me->getTH1());

    int nbinsX(h->GetNbinsX());

    double x(h->GetXaxis()->GetBinCenter(_bin % (nbinsX + 2)));

    if(data_->kind == MonitorElement::DQM_KIND_TH1F || data_->kind == MonitorElement::DQM_KIND_TPROFILE)
      me->Fill(x, _w);
    else{
      double y(h->GetYaxis()->GetBinCenter(_bin / (nbinsX + 2)));
      me->Fill(x, y, _w);
    }
  }

  void
  MESet::fill_(unsigned _iME, int _bin, double _y, double _w)
  {
    if(data_->kind != MonitorElement::DQM_KIND_TH2F && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    MonitorElement* me(mes_.at(_iME));

    TH1* h(me->getTH1());

    int nbinsX(h->GetNbinsX());

    double x(h->GetXaxis()->GetBinCenter(_bin % (nbinsX + 2)));
    me->Fill(x, _y, _w);
  }

  void
  MESet::fill_(unsigned _iME, double _x, double _wy, double _w)
  {
    if(data_->kind == MonitorElement::DQM_KIND_REAL)
      mes_.at(_iME)->Fill(_x);
    else if(data_->kind == MonitorElement::DQM_KIND_TH1F || data_->kind == MonitorElement::DQM_KIND_TPROFILE)
      mes_.at(_iME)->Fill(_x, _wy);
    else
      mes_.at(_iME)->Fill(_x, _wy, _w);
  }


  MESet::ConstBin::ConstBin(MESet const* _meSet, unsigned _iME/* = 0*/, int _iBin/* = 1*/) :
    meSet(_meSet),
    iME(_iME),
    iBin(_iBin),
    otype(BinService::nObjType)
  {
    if(!meSet){
      iME = unsigned(-1);
      iBin = -1;
    }

    MonitorElement::Kind kind(meSet->getKind());
    if(kind != MonitorElement::DQM_KIND_TH1F && kind != MonitorElement::DQM_KIND_TPROFILE &&
       kind != MonitorElement::DQM_KIND_TH2F && kind != MonitorElement::DQM_KIND_TPROFILE2D)
      throw cms::Exception("InvalidOperation")
        << "const_iterator onlye available for MESet of histogram kind";

    MonitorElement const* me(meSet->mes_[iME]);

    if(!me){
      meSet = 0;
      iME = unsigned(-1);
      iBin = -1;
    }

    if(iME == unsigned(-1)) return;

    if(iBin == 1 && (kind == MonitorElement::DQM_KIND_TH2F || kind == MonitorElement::DQM_KIND_TPROFILE2D))
      iBin = me->getNbinsX() + 3;

    otype = binService_->getObject(meSet->getObjType(), iME);
  }

  MESet::ConstBin&
  MESet::ConstBin::operator=(ConstBin const& _rhs)
  {
    if(!meSet) meSet = _rhs.meSet;
    else if(meSet->getObjType() != _rhs.meSet->getObjType() ||
            meSet->getBinType() != _rhs.meSet->getBinType())
      throw cms::Exception("IncompatibleAssignment")
        << "Iterator of otype " << _rhs.meSet->getObjType() << " and btype " << _rhs.meSet->getBinType()
        << " to otype " << meSet->getObjType() << " and btype " << meSet->getBinType();

    iME = _rhs.iME;
    iBin = _rhs.iBin;
    otype = _rhs.otype;
      
    return *this;
  }


  MESet::const_iterator&
  MESet::const_iterator::operator++()
  {
    unsigned& iME(constBin_.iME);
    int& bin(constBin_.iBin);
    MESet const* meSet(constBin_.meSet);
    BinService::ObjectType& otype(constBin_.otype);

    if(!meSet || bin < 0) return *this;

    MonitorElement::Kind kind(meSet->getKind());
    MonitorElement const* me(meSet->mes_[iME]);
      
    bin += 1;
    if(bin == 1){
      iME = 0;
      me = meSet->mes_[iME];
      if(kind == MonitorElement::DQM_KIND_TH2F || kind == MonitorElement::DQM_KIND_TPROFILE2D)
        bin = me->getNbinsX() + 3;
      otype = binService_->getObject(meSet->getObjType(), iME);
    }

    bool overflow(false);
    if(kind == MonitorElement::DQM_KIND_TH1F || kind == MonitorElement::DQM_KIND_TPROFILE)
      overflow = (bin == me->getNbinsX() + 1);
    else
      overflow = (bin == (me->getNbinsX() + 2) * me->getNbinsY() + me->getNbinsX() + 1);

    if(overflow){
      iME += 1;
      me = meSet->mes_[iME];
      if(!me){
        iME = unsigned(-1);
        bin = -1;
        otype = BinService::nObjType;
      }
      else{
        if(kind == MonitorElement::DQM_KIND_TH2F || kind == MonitorElement::DQM_KIND_TPROFILE2D)
          bin = me->getNbinsX() + 3;
        else
          bin = 1;
            
        otype = binService_->getObject(meSet->getObjType(), iME);
      }
    }

    return *this;
  }

  MESet::const_iterator&
  MESet::const_iterator::operator--()
  {
    unsigned& iME(constBin_.iME);
    int& bin(constBin_.iBin);
    MESet const* meSet(constBin_.meSet);
    BinService::ObjectType& otype(constBin_.otype);

    if(!meSet || bin == 0) return *this;

    MonitorElement::Kind kind(meSet->getKind());
    MonitorElement const* me(meSet->mes_[iME]);

    if(bin == -1){
      iME = binService_->getNObjects(meSet->getObjType()) - 1;
      me = meSet->mes_[iME];
      if(kind == MonitorElement::DQM_KIND_TH2F || kind == MonitorElement::DQM_KIND_TPROFILE2D)
        bin = (me->getNbinsX() + 2) * me->getNbinsY() + me->getNbinsX();
      else
        bin = me->getNbinsX();
      otype = binService_->getObject(meSet->getObjType(), iME);
    }
    else
      bin -= 1;

    bool underflow(false);
    if(kind == MonitorElement::DQM_KIND_TH1F || kind == MonitorElement::DQM_KIND_TPROFILE)
      underflow = (bin == 0);
    else
      underflow = (bin == me->getNbinsX() + 2);

    if(underflow){
      iME -= 1;
      me = meSet->mes_[iME];
      if(!me){
        iME = unsigned(-1);
        bin = 0;
        otype = BinService::nObjType;
      }
      else{
        if(kind == MonitorElement::DQM_KIND_TH2F || kind == MonitorElement::DQM_KIND_TPROFILE2D)
          bin = (me->getNbinsX() + 2) * me->getNbinsY() + me->getNbinsX();
        else
          bin = me->getNbinsX();
            
        otype = binService_->getObject(meSet->getObjType(), iME);
      }
    }

    return *this;
  }

  MESet::const_iterator&
  MESet::const_iterator::toNextChannel()
  {
    if(!constBin_.meSet) return *this;
    do operator++();
    while(constBin_.iBin > 0 && !constBin_.isChannel());
    return *this;
  }

  MESet::const_iterator&
  MESet::const_iterator::toPreviousChannel()
  {
    if(!constBin_.meSet) return *this;
    do operator--();
    while(constBin_.iBin > 0 && !constBin_.isChannel());
    return *this;
  }

  bool
  MESet::const_iterator::up()
  {
    if(!constBin_.meSet || constBin_.iBin < 1) return false;

    MonitorElement::Kind kind(constBin_.meSet->getKind());
    if(kind != MonitorElement::DQM_KIND_TH2F && kind != MonitorElement::DQM_KIND_TPROFILE2D) return false;

    MonitorElement const* me(constBin_.meSet->mes_[constBin_.iME]);

    if(constBin_.iBin / (me->getNbinsX() + 2) >= me->getNbinsY()) return false;

    constBin_.iBin += me->getNbinsX() + 2;

    return true;
  }

  bool
  MESet::const_iterator::down()
  {
    if(!constBin_.meSet || constBin_.iBin < 1) return false;

    MonitorElement::Kind kind(constBin_.meSet->getKind());
    if(kind != MonitorElement::DQM_KIND_TH2F && kind != MonitorElement::DQM_KIND_TPROFILE2D) return false;

    MonitorElement const* me(constBin_.meSet->mes_[constBin_.iME]);

    if(constBin_.iBin / (me->getNbinsX() + 2) <= 1) return false;

    constBin_.iBin -= me->getNbinsX() + 2;

    return true;
  }

}

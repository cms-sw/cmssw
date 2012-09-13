#include "DQM/EcalCommon/interface/MESet.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TString.h"
#include "TPRegexp.h"

namespace ecaldqm
{
  BinService const* MESet::binService_(0);
  DQMStore* MESet::dqmStore_(0);

  MESet::MESet(std::string const& _fullPath, BinService::ObjectType _otype, BinService::BinningType _btype, MonitorElement::Kind _kind) :
    mes_(0),
    dir_(_fullPath.substr(0, _fullPath.find_last_of('/'))),
    name_(_fullPath.substr(_fullPath.find_last_of('/') + 1)),
    otype_(_otype),
    btype_(_btype),
    kind_(_kind),
    active_(false)
  {
    if(!binService_){
      binService_ = &(*(edm::Service<EcalDQMBinningService>()));
      if(!binService_)
	throw cms::Exception("Service") << "EcalDQMBinningService not found" << std::endl;
    }

    if(!dqmStore_){
      dqmStore_ = &(*(edm::Service<DQMStore>()));
      if(!dqmStore_)
	throw cms::Exception("Service") << "DQMStore not found" << std::endl;
    }

    if(dir_.find("/") == std::string::npos ||
       (otype_ != BinService::kChannel && btype_ != BinService::kReport && name_.size() == 0))
      throw_(_fullPath + " cannot be used for ME path name");

    switch(kind_){
    case MonitorElement::DQM_KIND_REAL:
    case MonitorElement::DQM_KIND_TH1F:
    case MonitorElement::DQM_KIND_TPROFILE:
    case MonitorElement::DQM_KIND_TH2F:
    case MonitorElement::DQM_KIND_TPROFILE2D:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }
  }

  MESet::MESet(MESet const& _orig) :
    mes_(_orig.mes_),
    dir_(_orig.dir_),
    name_(_orig.name_),
    otype_(_orig.otype_),
    btype_(_orig.btype_),
    kind_(_orig.kind_),
    active_(_orig.active_)
  {
  }

  MESet::~MESet()
  {
  }

  MESet&
  MESet::operator=(MESet const& _rhs)
  {
    mes_ = _rhs.mes_;
    dir_ = _rhs.dir_;
    name_ = _rhs.name_;
    otype_ = _rhs.otype_;
    btype_ = _rhs.btype_;
    kind_ = _rhs.kind_;
    active_ = _rhs.active_;

    return *this;
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
    if(!active_) return;

    unsigned nME(mes_.size());
    for(unsigned iME(0); iME < nME; iME++)
      mes_[iME]->setAxisTitle(_title, _axis);
  }

  void
  MESet::setBinLabel(unsigned _iME, int _bin, std::string const& _label, int _axis/* = 1*/)
  {
    if(!active_) return;

    unsigned nME(mes_.size());

    if(_iME == unsigned(-1)){
      for(unsigned iME(0); iME < nME; iME++)
	mes_[iME]->setBinLabel(_bin, _label, _axis);

      return;
    }

    if(_iME >= nME || !getME(_iME)) return;
    mes_[_iME]->setBinLabel(_bin, _label, _axis);
  }

  void
  MESet::reset(double _content/* = 0.*/, double _err/* = 0.*/, double _entries/* = 0.*/)
  {
    if(!active_) return;

    resetAll(_content, _err, _entries);
  }

  void
  MESet::resetAll(double _content/* = 0.*/, double _err/* = 0.*/, double _entries/* = 0.*/)
  {
    if(!active_) return;

    unsigned nME(mes_.size());

    if(kind_ == MonitorElement::DQM_KIND_REAL){
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
	  if(kind_ == MonitorElement::DQM_KIND_TPROFILE){
	    static_cast<TProfile*>(h)->SetBinEntries(bin, _entries);
	    entries += _entries;
	  }
	  else if(kind_ == MonitorElement::DQM_KIND_TPROFILE2D){
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
  MESet::formPath(std::map<std::string, std::string> const& _replacements) const
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
  MESet::setLumiFlag()
  {
    if(!active_) return;

    for(unsigned iME(0); iME < mes_.size(); ++iME)
      if(mes_[iME]) mes_[iME]->setLumiFlag();
  }

  /*static*/
  MonitorElement::Kind
  MESet::translateKind(std::string const& _kindName)
  {
    if(_kindName == "REAL") return MonitorElement::DQM_KIND_REAL;
    else if(_kindName == "TH1F") return MonitorElement::DQM_KIND_TH1F;
    else if(_kindName == "TProfile") return MonitorElement::DQM_KIND_TPROFILE;
    else if(_kindName == "TH2F") return MonitorElement::DQM_KIND_TH2F;
    else if(_kindName == "TProfile2D") return MonitorElement::DQM_KIND_TPROFILE2D;
    else return MonitorElement::DQM_KIND_INVALID;
  }

  void
  MESet::fill_(unsigned _iME, int _bin, double _w)
  {
    if(kind_ == MonitorElement::DQM_KIND_REAL) return;

    MonitorElement* me(mes_[_iME]);
    if(!me) return;

    TH1* h(me->getTH1());

    int nbinsX(h->GetNbinsX());

    double x(h->GetXaxis()->GetBinCenter(_bin % (nbinsX + 2)));

    if(kind_ == MonitorElement::DQM_KIND_TH1F || kind_ == MonitorElement::DQM_KIND_TPROFILE)
      me->Fill(x, _w);
    else{
      double y(h->GetYaxis()->GetBinCenter(_bin / (nbinsX + 2)));
      me->Fill(x, y, _w);
    }
  }

  void
  MESet::fill_(unsigned _iME, int _bin, double _y, double _w)
  {
    if(kind_ != MonitorElement::DQM_KIND_TH2F && kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return;

    MonitorElement* me(mes_[_iME]);
    if(!me) return;

    TH1* h(me->getTH1());

    int nbinsX(h->GetNbinsX());

    double x(h->GetXaxis()->GetBinCenter(_bin % (nbinsX + 2)));
    me->Fill(x, _y, _w);
  }

  void
  MESet::fill_(unsigned _iME, double _x, double _wy, double _w)
  {
    MonitorElement* me(mes_[_iME]);
    if(!me) return;

    if(kind_ == MonitorElement::DQM_KIND_REAL)
      me->Fill(_x);
    else if(kind_ == MonitorElement::DQM_KIND_TH1F || kind_ == MonitorElement::DQM_KIND_TPROFILE)
      me->Fill(_x, _wy);
    else
      me->Fill(_x, _wy, _w);
  }



  MESet::ConstBin::ConstBin(MESet const* _meSet, unsigned _iME/* = 0*/, int _iBin/* = 1*/) :
    meSet_(_meSet),
    iME(_iME),
    iBin(_iBin),
    otype(BinService::nObjType)
  {
    if(!meSet_){
      iME = unsigned(-1);
      iBin = -1;
      return;
    }

    if(iME == unsigned(-1)) return;

    // current internal bin numbering scheme does not allow 1D histograms (overflow & underflow in each y)
    MonitorElement::Kind kind(meSet_->getKind());
    //    if(kind != MonitorElement::DQM_KIND_TH1F && kind != MonitorElement::DQM_KIND_TPROFILE &&
    if(kind != MonitorElement::DQM_KIND_TH2F && kind != MonitorElement::DQM_KIND_TPROFILE2D)
      throw cms::Exception("InvalidOperation")
        << "const_iterator only available for MESet of 2D histograms";

    MonitorElement const* me(meSet_->getME(iME));

    if(!me)
      throw cms::Exception("InvalidOperation") << "ME " << iME << " does not exist for MESet " << meSet_->getName();

    if(iBin == 1 && (kind == MonitorElement::DQM_KIND_TH2F || kind == MonitorElement::DQM_KIND_TPROFILE2D))
      iBin = me->getNbinsX() + 3;

    otype = binService_->getObject(meSet_->getObjType(), iME);
  }

  MESet::ConstBin&
  MESet::ConstBin::operator=(ConstBin const& _rhs)
  {
    if(!meSet_) meSet_ = _rhs.meSet_;
    else if(meSet_->getObjType() != _rhs.meSet_->getObjType() ||
            meSet_->getBinType() != _rhs.meSet_->getBinType())
      throw cms::Exception("IncompatibleAssignment")
        << "Iterator of otype " << _rhs.meSet_->getObjType() << " and btype " << _rhs.meSet_->getBinType()
        << " to otype " << meSet_->getObjType() << " and btype " << meSet_->getBinType()
        << " (" << _rhs.meSet_->getName() << " to " << meSet_->getName() << ")";

    iME = _rhs.iME;
    iBin = _rhs.iBin;
    otype = _rhs.otype;
      
    return *this;
  }



  MESet::const_iterator::const_iterator(MESet const* _meSet, DetId const& _id) :
    bin_()
  {
    if(!_meSet) return;

    BinService::ObjectType otype(_meSet->getObjType());
    unsigned iME(binService_->findPlot(otype, _id));
    if(iME == unsigned(-1)) return;

    BinService::BinningType btype(_meSet->getBinType());
    int bin(binService_->findBin2D(otype, btype, _id));
    if(bin == 0) return;

    bin_.setMESet(_meSet);
    bin_.iME = iME;
    bin_.iBin = bin;
    bin_.otype = otype;
  }

  MESet::const_iterator&
  MESet::const_iterator::operator++()
  {
    unsigned& iME(bin_.iME);
    MESet const* meSet(bin_.getMESet());

    if(!meSet || iME == unsigned(-1)) return *this;

    int& bin(bin_.iBin);
    BinService::ObjectType& otype(bin_.otype);

    MonitorElement::Kind kind(meSet->getKind());
    MonitorElement const* me(meSet->getME(iME));

    int nbinsX(me->getNbinsX());
      
    ++bin;
    if(bin == 1){
      iME = 0;
      me = meSet->getME(iME);
      nbinsX = me->getNbinsX();
      if(kind == MonitorElement::DQM_KIND_TH2F || kind == MonitorElement::DQM_KIND_TPROFILE2D)
        bin = nbinsX + 3;
      otype = binService_->getObject(meSet->getObjType(), iME);
    }

    if(bin % (nbinsX + 2) == nbinsX + 1){
      if(kind == MonitorElement::DQM_KIND_TH1F || kind == MonitorElement::DQM_KIND_TPROFILE ||
         bin / (nbinsX + 2) == me->getNbinsY()){
        iME += 1;
        me = meSet->getME(iME);
        if(!me){
          iME = unsigned(-1);
          bin = -1;
          otype = BinService::nObjType;
        }
        else{
          nbinsX = me->getNbinsX();
          if(kind == MonitorElement::DQM_KIND_TH2F || kind == MonitorElement::DQM_KIND_TPROFILE2D)
            bin = nbinsX + 3;
          else
            bin = 1;
            
          otype = binService_->getObject(meSet->getObjType(), iME);
        }
      }
      else 
        bin += 2;
    }

    return *this;
  }

  MESet::const_iterator&
  MESet::const_iterator::toNextChannel()
  {
    if(!bin_.getMESet()) return *this;
    do operator++();
    while(bin_.iME != unsigned(-1) && !bin_.isChannel());

    return *this;
  }

  bool
  MESet::const_iterator::up()
  {
    MESet const* meSet(bin_.getMESet());
    if(!meSet || bin_.iME == unsigned(-1) || bin_.iBin < 1) return false;

    MonitorElement::Kind kind(meSet->getKind());
    if(kind != MonitorElement::DQM_KIND_TH2F && kind != MonitorElement::DQM_KIND_TPROFILE2D) return false;

    MonitorElement const* me(meSet->getME(bin_.iME));

    if(bin_.iBin / (me->getNbinsX() + 2) >= me->getNbinsY()) return false;

    bin_.iBin += me->getNbinsX() + 2;

    return true;
  }

  bool
  MESet::const_iterator::down()
  {
    MESet const* meSet(bin_.getMESet());
    if(!meSet || bin_.iME == unsigned(-1) || bin_.iBin < 1) return false;

    MonitorElement::Kind kind(meSet->getKind());
    if(kind != MonitorElement::DQM_KIND_TH2F && kind != MonitorElement::DQM_KIND_TPROFILE2D) return false;

    MonitorElement const* me(meSet->getME(bin_.iME));

    if(bin_.iBin / (me->getNbinsX() + 2) <= 1) return false;

    bin_.iBin -= me->getNbinsX() + 2;

    return true;
  }

  bool
  MESet::const_iterator::left()
  {
    MESet const* meSet(bin_.getMESet());
    if(!meSet || bin_.iME == unsigned(-1) || bin_.iBin < 1) return false;

    MonitorElement::Kind kind(meSet->getKind());
    if(kind != MonitorElement::DQM_KIND_TH2F && kind != MonitorElement::DQM_KIND_TPROFILE2D) return false;

    MonitorElement const* me(meSet->getME(bin_.iME));

    if(bin_.iBin % (me->getNbinsX() + 2) <= 1) return false;

    bin_.iBin -= 1;

    return true;
  }

  bool
  MESet::const_iterator::right()
  {
    MESet const* meSet(bin_.getMESet());
    if(!meSet || bin_.iME == unsigned(-1) || bin_.iBin < 1) return false;

    MonitorElement::Kind kind(meSet->getKind());
    if(kind != MonitorElement::DQM_KIND_TH2F && kind != MonitorElement::DQM_KIND_TPROFILE2D) return false;

    MonitorElement const* me(meSet->getME(bin_.iME));

    if(bin_.iBin % (me->getNbinsX() + 2) >= me->getNbinsX()) return false;

    bin_.iBin += 1;

    return true;
  }
}

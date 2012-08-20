#ifndef MESet_H
#define MESet_H

#include <string>
#include <vector>

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/EcalDQMBinningService.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"

#include "FWCore/Utilities/interface/Exception.h"

typedef EcalDQMBinningService BinService; // prepare for expansion into more than one binning service

namespace ecaldqm
{
  struct MEData {
    std::string pathName;
    std::string fullPath;
    BinService::ObjectType otype;
    BinService::BinningType btype;
    MonitorElement::Kind kind;
    BinService::AxisSpecs *xaxis;
    BinService::AxisSpecs *yaxis;
    BinService::AxisSpecs *zaxis;
    MEData() :
      pathName(""), fullPath(""), otype(BinService::nObjType), btype(BinService::nBinType), kind(MonitorElement::DQM_KIND_INVALID),
      xaxis(0), yaxis(0), zaxis(0)
    {}
    MEData(std::string const& _pathName) :
      pathName(_pathName), fullPath(""), otype(BinService::nObjType), btype(BinService::nBinType), kind(MonitorElement::DQM_KIND_INVALID),
      xaxis(0), yaxis(0), zaxis(0)
    {}
    MEData(std::string const& _pathName, BinService::ObjectType _otype, BinService::BinningType _btype, MonitorElement::Kind _kind,
	   BinService::AxisSpecs const* _xaxis = 0, BinService::AxisSpecs const* _yaxis = 0, BinService::AxisSpecs const* _zaxis = 0) :
      pathName(_pathName), fullPath(""), otype(_otype), btype(_btype), kind(_kind),
      xaxis(_xaxis ? new BinService::AxisSpecs(*_xaxis) : 0),
      yaxis(_yaxis ? new BinService::AxisSpecs(*_yaxis) : 0),
      zaxis(_zaxis ? new BinService::AxisSpecs(*_zaxis) : 0)
    {}
    MEData(MEData const& _orig) :
      pathName(_orig.pathName), fullPath(_orig.fullPath), otype(_orig.otype), btype(_orig.btype), kind(_orig.kind),
      xaxis(_orig.xaxis ? new BinService::AxisSpecs(*_orig.xaxis) : 0),
      yaxis(_orig.yaxis ? new BinService::AxisSpecs(*_orig.yaxis) : 0),
      zaxis(_orig.zaxis ? new BinService::AxisSpecs(*_orig.zaxis) : 0)
    {}
    ~MEData()
    {
      delete xaxis;
      delete yaxis;
      delete zaxis;
    }

    MEData& operator=(MEData const& _rhs)
    {
      pathName = _rhs.pathName;
      fullPath = _rhs.fullPath;
      otype = _rhs.otype;
      btype = _rhs.btype;
      kind = _rhs.kind;
      xaxis = _rhs.xaxis ? new BinService::AxisSpecs(*_rhs.xaxis) : 0;
      yaxis = _rhs.yaxis ? new BinService::AxisSpecs(*_rhs.yaxis) : 0;
      zaxis = _rhs.zaxis ? new BinService::AxisSpecs(*_rhs.zaxis) : 0;
      return *this;
    }
  };

  /* class MESet
     ABC for MonitorElement wrappers
     Interface between ME bins and DetId
  */

  class MESet {
  public :
    MESet(MEData const&);
    virtual ~MESet();

    virtual void book() = 0;
    virtual bool retrieve() const = 0;
    virtual void clear() const;

    // default values are necessary (otherwise fill(DetId) will be interpreted as fill(uint32_t)!!)
    virtual void fill(DetId const&, double = 1., double = 1., double = 1.) {}
    virtual void fill(EcalElectronicsId const&, double = 1., double = 1., double = 1.) {}
    virtual void fill(unsigned, double = 1., double = 1., double = 1.) {}
    virtual void fill(double, double = 1., double = 1.) {}

    virtual void setBinContent(DetId const&, double) {}
    virtual void setBinContent(EcalElectronicsId const&, double) {}
    virtual void setBinContent(unsigned, double) {}
    virtual void setBinContent(DetId const&, int, double) {}
    virtual void setBinContent(EcalElectronicsId const&, int, double) {}
    virtual void setBinContent(unsigned, int, double) {}
    virtual void setBinContent(int, double) {}

    virtual void setBinError(DetId const&, double) {}
    virtual void setBinError(EcalElectronicsId const&, double) {}
    virtual void setBinError(unsigned, double) {}
    virtual void setBinError(DetId const&, int, double) {}
    virtual void setBinError(EcalElectronicsId const&, int, double) {}
    virtual void setBinError(unsigned, int, double) {}
    virtual void setBinError(int, double) {}

    virtual void setBinEntries(DetId const&, double) {}
    virtual void setBinEntries(EcalElectronicsId const&, double) {}
    virtual void setBinEntries(unsigned, double) {}
    virtual void setBinEntries(DetId const&, int, double) {}
    virtual void setBinEntries(EcalElectronicsId const&, int, double) {}
    virtual void setBinEntries(unsigned, int, double) {}
    virtual void setBinEntries(int, double) {}

    virtual double getBinContent(DetId const&, int = 0) const { return 0.; }
    virtual double getBinContent(EcalElectronicsId const&, int = 0) const { return 0.; }
    virtual double getBinContent(unsigned, int = 0) const { return 0.; }
    virtual double getBinContent(int) const { return 0.; }

    virtual double getBinError(DetId const&, int = 0) const { return 0.; }
    virtual double getBinError(EcalElectronicsId const&, int = 0) const { return 0.; }
    virtual double getBinError(unsigned, int = 0) const { return 0.; }
    virtual double getBinError(int) const { return 0.; }

    virtual double getBinEntries(DetId const&, int = 0) const { return 0.; }
    virtual double getBinEntries(EcalElectronicsId const&, int = 0) const { return 0.; }
    virtual double getBinEntries(unsigned, int = 0) const { return 0.; }
    virtual double getBinEntries(int) const { return 0.; }

    virtual int findBin(DetId const&) const { return 0; }
    virtual int findBin(EcalElectronicsId const&) const { return 0; }
    virtual int findBin(unsigned) const { return 0; }
    virtual int findBin(DetId const&, double, double = 0.) const { return 0; }
    virtual int findBin(EcalElectronicsId const&, double, double = 0.) const { return 0; }
    virtual int findBin(unsigned, double, double = 0.) const { return 0; }
    virtual int findBin(double, double = 0.) const { return 0; }

    virtual void setAxisTitle(std::string const&, int _axis = 1);
    virtual void setBinLabel(unsigned, int, std::string const&, int _axis = 1);

    virtual void reset(double _content = 0., double _err = 0., double _entries = 0.);
    virtual void resetAll(double _content = 0., double _err = 0., double _entries = 0.);

    void formName(std::map<std::string, std::string> const&) const;

    std::string const& getDir() const { return dir_; }
    void setDir(std::string const& _dir) { dir_ = _dir; }
    std::string const& getName() const { return name_; }
    void setName(std::string const& _name) { name_ = _name; }
    BinService::ObjectType getObjType() const { return data_->otype; }
    BinService::BinningType getBinType() const { return data_->btype; }
    MonitorElement::Kind getKind() const { return data_->kind; }
    bool isActive() const { return active_; }
    MonitorElement const* getME(unsigned _iME) const { return (_iME < mes_.size() ? mes_[_iME] : 0); }

  protected:
    virtual void fill_(unsigned, int, double);
    virtual void fill_(unsigned, int, double, double);
    virtual void fill_(unsigned, double, double, double);

    void checkME_(unsigned _iME) const
    {
      if(_iME >= mes_.size() || !mes_[_iME])
        throw_("ME array index overflow");
    }

    void throw_(std::string const& _message) const
    {
      throw cms::Exception("EcalDQM") << dir_ << "/" << name_ << ": " << _message;
    }

    static BinService const* binService_;
    static DQMStore* dqmStore_;

    mutable std::vector<MonitorElement*> mes_;

    mutable std::string dir_;
    mutable std::string name_;
    MEData const* data_;

    mutable bool active_;

  public:

    struct ConstBin {
      MESet const* meSet;
      unsigned iME;
      int iBin;
      BinService::ObjectType otype;

      ConstBin() : meSet(0), iME(-1), iBin(-1), otype(BinService::nObjType) {}
      ConstBin(MESet const*, unsigned = 0, int = 1);
      ConstBin(ConstBin const& _orig) : meSet(_orig.meSet), iME(_orig.iME), iBin(_orig.iBin), otype(_orig.otype) {}
      ConstBin& operator=(ConstBin const&);
      bool operator==(ConstBin const& _rhs) const
      {
        return meSet != 0 && meSet == _rhs.meSet && iME == _rhs.iME && iBin == _rhs.iBin;
      }
      bool isChannel() const
      {
        return binService_->isValidIdBin(otype, meSet->getBinType(), iME, iBin);
      }
      uint32_t getId() const
      {
        return binService_->idFromBin(otype, meSet->getBinType(), iME, iBin);
      }
      double getBinContent() const
      {
        if(iBin > 0) return meSet->mes_[iME]->getBinContent(iBin);
        else return 0.;
      }
      double getBinError() const
      {
        if(iBin > 0) return meSet->mes_[iME]->getBinError(iBin);
        else return 0.;
      }
      double getBinEntries() const
      {
        if(iBin > 0) return meSet->mes_[iME]->getBinEntries(iBin);
        else return 0.;
      }
    };

    struct Bin {
      MESet* meSet;
      ConstBin const* constBin;

      Bin() : meSet(0), constBin(0) {}
      Bin(MESet* _set, ConstBin const& _constBin) : meSet(_set), constBin(&_constBin) {}
      Bin(Bin const& _orig) : meSet(_orig.meSet), constBin(_orig.constBin) {}
      Bin& operator=(Bin const& _rhs)
      {
        meSet = _rhs.meSet;
        constBin = _rhs.constBin;
        return *this;
      }

      bool operator==(ConstBin const& _rhs) const { return constBin && constBin->operator==(_rhs); }
      bool operator==(Bin const& _rhs) const { return constBin && _rhs.constBin && meSet && _rhs.meSet && meSet == _rhs.meSet && constBin->operator==(*_rhs.constBin); }
      bool isChannel() const { return constBin && constBin->isChannel(); }
      uint32_t getId() const { return constBin ? constBin->getId() : 0; }
      double getBinContent() const { return constBin ? constBin->getBinContent() : 0.; }
      double getBinError() const { return constBin ? constBin->getBinError() : 0.; }
      double getBinEntries() const { return constBin ? constBin->getBinEntries() : 0.; }
      void fill(double _w = 1.)
      {
        if(meSet && constBin && constBin->iBin > 0) meSet->fill_(constBin->iME, constBin->iBin, _w);
      }
      void fill(double _y, double _w = 1.)
      {
        if(meSet && constBin && constBin->iBin > 0) meSet->fill_(constBin->iME, constBin->iBin, _y, _w);
      }
      void setBinContent(double _content)
      {
        if(meSet && constBin && constBin->iBin > 0) meSet->mes_[constBin->iME]->setBinContent(constBin->iBin, _content);
      }
      void setBinError(double _error)
      {
        if(meSet && constBin && constBin->iBin > 0) meSet->mes_[constBin->iME]->setBinError(constBin->iBin, _error);
      }
      void setBinEntries(double _entries)
      {
        if(meSet && constBin && constBin->iBin > 0) meSet->mes_[constBin->iME]->setBinEntries(constBin->iBin, _entries);
      }
    };

    /* const_iterator
       iterates over bins
       supports automatic transition between MEs in the same set
       underflow -> bin == 0 overflow -> bin == -1
    */
    struct const_iterator {
      const_iterator() : constBin_() {}
      const_iterator(MESet const* _meSet, unsigned _iME = 0, int _iBin = 1) : constBin_(_meSet, _iME, _iBin) {}
      const_iterator(const_iterator const& _orig) : constBin_(_orig.constBin_) {}
      const_iterator& operator=(const_iterator const& _rhs) { constBin_ = _rhs.constBin_; return *this; }
      bool operator==(const_iterator const& _rhs) const { return constBin_ == _rhs.constBin_; }
      bool operator!=(const_iterator const& _rhs) const { return !(constBin_ == _rhs.constBin_); }
      ConstBin const* operator->() const { return &constBin_; }
      const_iterator& operator++();
      const_iterator& operator--();
      const_iterator& toNextChannel();
      const_iterator& toPreviousChannel();
      bool up();
      bool down();

      protected:
      ConstBin constBin_;
    };

    struct iterator : public const_iterator {
      iterator() : const_iterator(), bin_(0, constBin_) {}
      iterator(MESet* _meSet, unsigned _iME = 0, int _iBin = 1) : const_iterator(_meSet, _iME, _iBin), bin_(_meSet, constBin_) {}
      iterator(MESet* _meSet, const_iterator const& _itr) : const_iterator(_itr), bin_(_meSet, constBin_) {}
      iterator(iterator const& _orig) : const_iterator(_orig), bin_(_orig.bin_.meSet, constBin_) {}
      iterator& operator=(const_iterator const& _rhs) { const_iterator::operator=(_rhs); bin_.constBin = &constBin_; return *this; }
      Bin* operator->() { return &bin_; }
      void setMESet(MESet* _meSet) { bin_.meSet = _meSet; }

      private:
      Bin bin_;
    };

    const_iterator begin() const
    {
      return const_iterator(this);
    }

    const_iterator end() const
    {
      return const_iterator(this, -1, -1);
    }

    iterator begin()
    {
      return iterator(this);
    }

    iterator end()
    {
      return iterator(this, -1, -1);
    }
  
  };

}

#endif

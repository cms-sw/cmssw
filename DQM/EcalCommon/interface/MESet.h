#ifndef MESet_H
#define MESet_H

#include <string>
#include <vector>

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/EcalDQMBinningService.h"
#include "DQM/EcalCommon/interface/PtrMap.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"

#include "FWCore/Utilities/interface/Exception.h"

typedef EcalDQMBinningService BinService; // prepare for expansion into more than one binning service

namespace ecaldqm
{
  /* class MESet
     ABC for MonitorElement wrappers
     Interface between ME bins and DetId
  */

  class MESet {
  public :
    MESet(std::string const&, BinService::ObjectType, BinService::BinningType, MonitorElement::Kind);
    MESet(MESet const&);
    virtual ~MESet();

    virtual MESet& operator=(MESet const&);

    virtual MESet* clone() const = 0;

    virtual void book() = 0;
    virtual bool retrieve() const = 0;
    virtual void clear() const;

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

    virtual void setAxisTitle(std::string const&, int = 1);
    virtual void setBinLabel(unsigned, int, std::string const&, int = 1);

    virtual void reset(double = 0., double = 0., double = 0.);
    virtual void resetAll(double = 0., double = 0., double = 0.);

    virtual void formPath(std::map<std::string, std::string> const&) const;

    virtual std::string const& getPath() const { return path_; }
    BinService::ObjectType getObjType() const { return otype_; }
    BinService::BinningType getBinType() const { return btype_; }
    MonitorElement::Kind getKind() const { return kind_; }
    bool isActive() const { return active_; }
    virtual bool isVariableBinning() const { return false; }
    virtual MonitorElement const* getME(unsigned _iME) const { return (_iME < mes_.size() ? mes_[_iME] : 0); }
    virtual MonitorElement* getME(unsigned _iME) { return (_iME < mes_.size() ? mes_[_iME] : 0); }
    virtual void setLumiFlag();
    virtual void softReset();
    virtual void recoverStats();

    static MonitorElement::Kind translateKind(std::string const&);

  protected:
    virtual void fill_(unsigned, int, double);
    virtual void fill_(unsigned, int, double, double);
    virtual void fill_(unsigned, double, double, double);

    virtual void checkME_(unsigned _iME) const
    {
      if(!getME(_iME)){
        std::stringstream ss;
        ss << "ME does not exist at index " << _iME;
        throw_(ss.str());
      }
    }

    void throw_(std::string const& _message) const
    {
      throw cms::Exception("EcalDQM") << path_ << ": " << _message;
    }

    static BinService const* binService_;
    static DQMStore* dqmStore_;

    mutable std::vector<MonitorElement*> mes_;

    mutable std::string path_;
    BinService::ObjectType otype_;
    BinService::BinningType btype_;
    MonitorElement::Kind kind_;

    mutable bool active_;

  public:

    struct ConstBin {
      protected:
      MESet const* meSet_;

      public:
      unsigned iME;
      int iBin;
      BinService::ObjectType otype;

      ConstBin() : meSet_(0), iME(-1), iBin(-1), otype(BinService::nObjType) {}
      ConstBin(MESet const*, unsigned = 0, int = 1);
      ConstBin(ConstBin const& _orig) : meSet_(_orig.meSet_), iME(_orig.iME), iBin(_orig.iBin), otype(_orig.otype) {}
      ConstBin& operator=(ConstBin const&);
      bool operator==(ConstBin const& _rhs) const
      {
        return meSet_ != 0 && meSet_ == _rhs.meSet_ && iME == _rhs.iME && iBin == _rhs.iBin;
      }
      bool isChannel() const
      {
        if(meSet_) return binService_->isValidIdBin(otype, meSet_->getBinType(), iME, iBin);
        else return false;
      }
      uint32_t getId() const
      {
        if(meSet_) return binService_->idFromBin(otype, meSet_->getBinType(), iME, iBin);
        else return 0;
      }
      double getBinContent() const
      {
        if(meSet_ && iME != unsigned(-1)) return meSet_->getME(iME)->getBinContent(iBin);
        else return 0.;
      }
      double getBinError() const
      {
        if(meSet_ && iME != unsigned(-1)) return meSet_->getME(iME)->getBinError(iBin);
        else return 0.;
      }
      double getBinEntries() const
      {
        if(meSet_ && iME != unsigned(-1)) return meSet_->getME(iME)->getBinEntries(iBin);
        else return 0.;
      }
      MonitorElement const* getME() const
      {
        if(meSet_ && iME != unsigned(-1)) return meSet_->getME(iME);
        else return 0;
      }
      void setMESet(MESet const* _meSet) { meSet_ = _meSet; }
      MESet const* getMESet() const { return meSet_; }
    };

    struct Bin : public ConstBin {
      protected:
      MESet* meSet_;

      public:
      Bin() : ConstBin(), meSet_(0) {}
      Bin(MESet* _set, unsigned _iME = 0, int _iBin = 1) : ConstBin(_set, _iME, _iBin), meSet_(_set) {}
      Bin(Bin const& _orig) : ConstBin(_orig), meSet_(_orig.meSet_) {}
      ConstBin& operator=(Bin const& _rhs)
      {
        bool wasNull(ConstBin::meSet_ == 0);
        ConstBin::operator=(_rhs);
        if(wasNull) meSet_ = _rhs.meSet_;
        return *this;
      }
      void fill(double _w = 1.)
      {
        if(meSet_) meSet_->fill_(iME, iBin, _w);
      }
      void fill(double _y, double _w = 1.)
      {
        if(meSet_) meSet_->fill_(iME, iBin, _y, _w);
      }
      void setBinContent(double _content)
      {
        if(meSet_ && iME != unsigned(-1)) meSet_->getME(iME)->setBinContent(iBin, _content);
      }
      void setBinError(double _error)
      {
        if(meSet_ && iME != unsigned(-1)) meSet_->getME(iME)->setBinError(iBin, _error);
      }
      void setBinEntries(double _entries)
      {
        if(meSet_ && iME != unsigned(-1)) meSet_->getME(iME)->setBinEntries(iBin, _entries);
      }
      MonitorElement* getME() const
      {
        if(meSet_ && iME != unsigned(-1)) return meSet_->getME(iME);
        else return 0;
      }
      void setMESet(MESet* _meSet) { ConstBin::meSet_ = _meSet; meSet_ = _meSet; }
      MESet* getMESet() const { return meSet_; }
    };

    /* const_iterator
       iterates over bins
       supports automatic transition between MEs in the same set
       underflow -> bin == 0 overflow -> bin == -1
    */
    struct const_iterator {
      const_iterator() : bin_() {}
      const_iterator(MESet const* _meSet, unsigned _iME = 0, int _iBin = 1) : bin_(_meSet, _iME, _iBin) {}
      const_iterator(MESet const*, DetId const&);
      const_iterator(const_iterator const& _orig) : bin_(_orig.bin_) {}
      const_iterator& operator=(const_iterator const& _rhs) { bin_ = _rhs.bin_; return *this; }
      bool operator==(const_iterator const& _rhs) const { return bin_ == _rhs.bin_; }
      bool operator!=(const_iterator const& _rhs) const { return !(bin_ == _rhs.bin_); }
      ConstBin const* operator->() const { return &bin_; }
      const_iterator& operator++();
      const_iterator& toNextChannel();
      bool up();
      bool down();
      bool left();
      bool right();

      protected:
      ConstBin bin_;
    };

    struct iterator : public const_iterator {
      iterator() : const_iterator(), bin_() {}
      iterator(MESet* _meSet, unsigned _iME = 0, int _iBin = 1) : const_iterator(_meSet, _iME, _iBin), bin_(_meSet) { bin_.ConstBin::operator=(const_iterator::bin_); }
      iterator(MESet* _meSet, DetId const& _id) : const_iterator(_meSet, _id), bin_(_meSet) { bin_.ConstBin::operator=(const_iterator::bin_); }
      iterator(iterator const& _orig) : const_iterator(_orig), bin_(_orig.bin_) {}
      iterator& operator=(const_iterator const& _rhs) { const_iterator::operator=(_rhs); bin_.ConstBin::operator=(const_iterator::bin_); return *this; }
      Bin* operator->() { return &bin_; }
      Bin const* operator->() const { return &bin_; }
      const_iterator& operator++() { const_iterator::operator++(); bin_.ConstBin::operator=(const_iterator::bin_); return *this; }
      const_iterator& toNextChannel() { const_iterator::toNextChannel(); bin_.ConstBin::operator=(const_iterator::bin_); return *this; }
      bool up() { bool res(const_iterator::up()); bin_.ConstBin::operator=(const_iterator::bin_); return res; }
      bool down() { bool res(const_iterator::down()); bin_.ConstBin::operator=(const_iterator::bin_); return res; }
      bool left() { bool res(const_iterator::left()); bin_.ConstBin::operator=(const_iterator::bin_); return res; }
      bool right() { bool res(const_iterator::right()); bin_.ConstBin::operator=(const_iterator::bin_); return res; }

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

    const_iterator beginChannel() const
    {
      const_iterator itr(this, 0, 0);
      return itr.toNextChannel();
    }

    iterator begin()
    {
      return iterator(this);
    }

    iterator end()
    {
      return iterator(this, -1, -1);
    }

    iterator beginChannel()
    {
      iterator itr(this, 0, 0);
      itr.toNextChannel();
      return itr;
    }
  
  };

  typedef PtrMap<std::string, MESet> MESetCollection;
  typedef PtrMap<std::string, MESet const> ConstMESetCollection;

}

#endif

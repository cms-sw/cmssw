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
    BinService::ObjectType otype;
    BinService::BinningType btype;
    MonitorElement::Kind kind;
    BinService::AxisSpecs *xaxis;
    BinService::AxisSpecs *yaxis;
    BinService::AxisSpecs *zaxis;
    MEData() :
      pathName(""), otype(BinService::nObjType), btype(BinService::nBinType), kind(MonitorElement::DQM_KIND_INVALID),
      xaxis(0), yaxis(0), zaxis(0)
    {}
    MEData(std::string const& _pathName, BinService::ObjectType _otype, BinService::BinningType _btype, MonitorElement::Kind _kind,
	   BinService::AxisSpecs const* _xaxis = 0, BinService::AxisSpecs const* _yaxis = 0, BinService::AxisSpecs const* _zaxis = 0) :
      pathName(_pathName), otype(_otype), btype(_btype), kind(_kind),
      xaxis(_xaxis ? new BinService::AxisSpecs(*_xaxis) : 0),
      yaxis(_yaxis ? new BinService::AxisSpecs(*_yaxis) : 0),
      zaxis(_zaxis ? new BinService::AxisSpecs(*_zaxis) : 0)
    {}
    MEData(MEData const& _orig) :
      pathName(_orig.pathName), otype(_orig.otype), btype(_orig.btype), kind(_orig.kind),
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
      otype = _rhs.otype;
      btype = _rhs.btype;
      kind = _rhs.kind;
      xaxis = _rhs.xaxis ? new BinService::AxisSpecs(*_rhs.xaxis) : 0;
      yaxis = _rhs.yaxis ? new BinService::AxisSpecs(*_rhs.yaxis) : 0;
      zaxis = _rhs.zaxis ? new BinService::AxisSpecs(*_rhs.zaxis) : 0;
      return *this;
    }
  };

  class MESet {
  public :
    MESet(std::string const&, MEData const&, bool _readOnly = false);
    virtual ~MESet();

    virtual void book();
    virtual bool retrieve() const;
    virtual void clear() const;

    // default values are necessary (otherwise fill(DetId) will be interpreted as fill(uint32_t)!!)
    virtual void fill(DetId const&, double _wx = 1., double _wy = 1., double _w = 1.);
    virtual void fill(EcalElectronicsId const&, double _wx = 1., double _wy = 1., double _w = 1.);
    virtual void fill(unsigned, double _wx = 1., double _wy = 1., double _w = 1.);
    virtual void fill(double, double _wy = 1., double _w = 1.);

    virtual void setBinContent(DetId const&, double, double _err = 0.);
    virtual void setBinContent(EcalElectronicsId const&, double, double _err = 0.);
    virtual void setBinContent(unsigned, double, double _err = 0.);

    virtual void setBinEntries(DetId const&, double);
    virtual void setBinEntries(EcalElectronicsId const&, double);
    virtual void setBinEntries(unsigned, double);

    virtual double getBinContent(DetId const&, int _bin = 0) const;
    virtual double getBinContent(EcalElectronicsId const&, int _bin = 0) const;
    virtual double getBinContent(unsigned, int _bin = 0) const;

    virtual double getBinError(DetId const&, int _bin = 0) const;
    virtual double getBinError(EcalElectronicsId const&, int _bin = 0) const;
    virtual double getBinError(unsigned, int _bin = 0) const;

    virtual double getBinEntries(DetId const&, int _bin = 0) const;
    virtual double getBinEntries(EcalElectronicsId const&, int _bin = 0) const;
    virtual double getBinEntries(unsigned, int _bin = 0) const;

    virtual void setAxisTitle(std::string const&, int _axis = 1);
    virtual void setBinLabel(unsigned, int, std::string const&, int _axis = 1);

    virtual void reset(double _content = 0., double _err = 0., double _entries = 0.);
    virtual void resetAll(double _content = 0., double _err = 0., double _entries = 0.);

    std::string const& getDir() const { return dir_; }
    void setDir(std::string const& _dir) { dir_ = _dir; }
    std::string const& getName() const { return name_; }
    void setName(std::string const& _name) { name_ = _name; }
    void name(std::map<std::string, std::string> const&) const;
    BinService::ObjectType getObjType() const { return data_->otype; }
    BinService::BinningType getBinType() const { return data_->btype; }
    bool isActive() const { return active_; }

    virtual MonitorElement const* getME(unsigned _offset) const { return (_offset < mes_.size() ? mes_[_offset] : 0); }

    struct const_iterator {
      const_iterator() :
        index_(-1),
        offset_(-1),
        otype_(BinService::nObjType),
        bin_(0),
        meSet_(0)
      {}
      const_iterator(MESet const* _meSet, unsigned _offset, unsigned _index) :
        index_(_index),
        offset_(_offset),
        otype_(BinService::nObjType),
        bin_(0),
        meSet_(_meSet)
      {
        if(!meSet_){
          index_ = unsigned(-1);
          offset_ = unsigned(-1);
        }
        if(index_ == unsigned(-1) || offset_ == unsigned(-1)) return;
        otype_ = binService_->objectFromOffset(meSet_->getObjType(), offset_);
        bin_ = binService_->getBin(otype_, meSet_->getBinType(), index_);
      }
      const_iterator& operator=(const_iterator const& _rhs)
      {
        if(!meSet_) meSet_ = _rhs.meSet_;
        else if(meSet_->getObjType() != _rhs.meSet_->getObjType() ||
                meSet_->getBinType() != _rhs.meSet_->getBinType())
          throw cms::Exception("IncompatibleAssignment")
            << "Iterator of otype " << _rhs.meSet_->getObjType() << " and btype " << _rhs.meSet_->getBinType()
            << " to otype " << meSet_->getObjType() << " and btype " << meSet_->getBinType();

        index_ = _rhs.index_;
        offset_ = _rhs.offset_;
        otype_ = _rhs.otype_;
        bin_ = _rhs.bin_;
        return *this;
      }
      const_iterator& operator++()
      {
        if(!meSet_ || bin_ == 0) return *this;
        index_ += 1;
        bin_ = binService_->getBin(otype_, meSet_->getBinType(), index_);
        if(bin_ == 0){
          index_ = 0;
          offset_ += 1;
          otype_ = binService_->objectFromOffset(meSet_->getObjType(), offset_);
          if(otype_ == BinService::nObjType){
            index_ = unsigned(-1);
            offset_ = unsigned(-1);
            otype_ = BinService::nObjType;
            return *this;
          }
          else{
            bin_ = binService_->getBin(otype_, meSet_->getBinType(), index_);
          }
        }
        return *this;
      }
      bool operator==(const_iterator const& _rhs) const
      {
        return
          meSet_ != 0 &&
          meSet_ == _rhs.meSet_ &&
          index_ == _rhs.index_ &&
          offset_ == _rhs.offset_;
      }
      bool operator!=(const_iterator const& _rhs) const
      {
        return !operator==(_rhs);
      }

      double
      getBinContent() const
      {
        if(bin_ > 0) return meSet_->getBinContent_(offset_, bin_);
        else return 0.;
      }
      double
      getBinError() const
      {
        if(bin_ > 0) return meSet_->getBinError_(offset_, bin_);
        else return 0.;
      }
      double
      getBinEntries() const
      {
        if(bin_ > 0) return meSet_->getBinEntries_(offset_, bin_);
        else return 0.;
      }

      protected:
      unsigned index_;
      unsigned offset_;
      BinService::ObjectType otype_;
      int bin_;
      MESet const* meSet_;
    };

    struct iterator : public const_iterator {
      iterator() :
        const_iterator(),
        meSet_(const_cast<MESet*>(const_iterator::meSet_))
      {}
      iterator(MESet const* _meSet, unsigned _offset, unsigned _index) :
        const_iterator(_meSet, _offset, _index),
        meSet_(const_cast<MESet*>(const_iterator::meSet_))
      {}
      iterator(const_iterator const& _src) :
        const_iterator(_src),
        meSet_(const_cast<MESet*>(const_iterator::meSet_))
      {}
      iterator& operator++()
      {
        const_iterator::operator++();
        return *this;
      }
      iterator& operator=(const_iterator const& _rhs)
      {
        const_iterator::operator=(_rhs);
        meSet_ = const_cast<MESet*>(const_iterator::meSet_);
        return *this;
      }
      bool operator==(iterator const& _rhs) const
      {
        return const_iterator::operator==(_rhs);
      }
      bool operator!=(iterator const& _rhs) const
      {
        return const_iterator::operator!=(_rhs);
      }

      void fill(double _w = 1.)
      {
        if(bin_ > 0) meSet_->fill_(offset_, bin_, _w);
      }
      void setBinContent(double _content, double _err)
      {
        if(bin_ > 0) meSet_->setBinContent_(offset_, bin_, _content, _err);
      }
      void setBinEntries(double _entries)
      {
        if(bin_ > 0) meSet_->setBinEntries_(offset_, bin_, _entries);
      }

      protected:
      MESet* meSet_;
    };

    const_iterator begin() const
    {
      return const_iterator(this, 0, 0);
    }

    const_iterator end() const
    {
      return const_iterator(this, -1, -1);
    }
  
  protected:
    virtual void fill_(unsigned, int, double);
    virtual void fill_(unsigned, double, double, double);
    virtual void setBinContent_(unsigned, int, double, double);
    virtual void setBinEntries_(unsigned, int, double);
    virtual double getBinContent_(unsigned, int) const;
    virtual double getBinError_(unsigned, int) const;
    virtual double getBinEntries_(unsigned, int) const;

    static BinService const* binService_;
    static DQMStore* dqmStore_;

    mutable std::vector<MonitorElement*> mes_;

    mutable std::string dir_;
    mutable std::string name_;
    MEData const* data_;

    mutable bool active_;
    bool readOnly_;
  };

}

#endif

#ifndef DQMSERVICES_CORE_MONITOR_ELEMENT_H
#define DQMSERVICES_CORE_MONITOR_ELEMENT_H

#if __GNUC__ && !defined DQM_DEPRECATED
//#define DQM_DEPRECATED __attribute__((deprecated))
#define DQM_DEPRECATED
#endif

#include "DQMServices/Core/interface/DQMNet.h"

#include "DataFormats/Histograms/interface/MonitorElementCollection.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "TF1.h"
#include "TH1F.h"
#include "TH1S.h"
#include "TH1D.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TH2S.h"
#include "TH2I.h"
#include "TH2D.h"
#include "TH2Poly.h"
#include "TH3F.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TObjString.h"
#include "TAxis.h"
#include "TGraph.h"

#include <mutex>
#include <memory>
#include <string>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstdint>
#include <sys/time.h>
#include <oneapi/tbb/spin_mutex.h>

// TODO: cleaup the usages and remove.
using QReport = MonitorElementData::QReport;
using DQMChannel = MonitorElementData::QReport::DQMChannel;

// TODO: move to a better location (changing all usages)
namespace dqm {
  /** Numeric constants for quality test results.  The smaller the
      number, the less severe the message.  */
  namespace qstatus {
    static const int OTHER = 30;        //< Anything but 'ok','warning' or 'error'.
    static const int DISABLED = 50;     //< Test has been disabled.
    static const int INVALID = 60;      //< Problem preventing test from running.
    static const int INSUF_STAT = 70;   //< Insufficient statistics.
    static const int DID_NOT_RUN = 90;  //< Algorithm did not run.
    static const int STATUS_OK = 100;   //< Test was succesful.
    static const int WARNING = 200;     //< Test had some problems.
    static const int ERROR = 300;       //< Test has failed.
  }                                     // namespace qstatus

  namespace me_util {
    using Channel = DQMChannel;
  }
}  // namespace dqm

// forward declarations for all our friends
namespace dqm::implementation {
  class DQMStore;
  class IBooker;
}  // namespace dqm::implementation
struct DQMTTreeIO;
namespace dqm {
  class DQMFileSaverPB;
}
class DQMService;
class QualityTester;

namespace dqm::impl {

  using dqmmutex = tbb::spin_mutex;

  struct Access {
    std::unique_lock<dqmmutex> guard_;
    MonitorElementData::Key const &key;
    MonitorElementData::Value const &value;
  };
  // TODO: can this be the same type, just const?
  struct AccessMut {
    std::unique_lock<dqmmutex> guard_;
    MonitorElementData::Key const &key;
    MonitorElementData::Value &value;
  };

  struct MutableMonitorElementData {
    MonitorElementData data_;
    dqmmutex lock_;
    Access access() { return Access{std::unique_lock<dqmmutex>(lock_), data_.key_, data_.value_}; }
    AccessMut accessMut() { return AccessMut{std::unique_lock<dqmmutex>(lock_), data_.key_, data_.value_}; }
  };

  /** The base class for all MonitorElements (ME) */
  class MonitorElement {
    // these need to create and destroy MEs.
    friend dqm::implementation::DQMStore;
    friend dqm::implementation::IBooker;
    // these need to access some of the IO related methods.
    friend ::DQMTTreeIO;  // declared in DQMRootSource
    friend ::dqm::DQMFileSaverPB;
    friend ::DQMService;
    // this one only needs syncCoreObject.
    friend ::QualityTester;

  public:
    using Scalar = MonitorElementData::Scalar;
    using Kind = MonitorElementData::Kind;

    // Comparison helper used in DQMStore to insert into sets. This needs deep
    // private access to the MEData, that is why it lives here.
    struct MEComparison {
      using is_transparent = int;  // magic marker to allow C++14 heterogeneous set lookup.

      auto make_tuple(MonitorElement *me) const {
        return std::make_tuple(std::reference_wrapper(me->getPathname()), std::reference_wrapper(me->getName()));
      }
      auto make_tuple(MonitorElementData::Path const &path) const {
        return std::make_tuple(std::reference_wrapper(path.getDirname()), std::reference_wrapper(path.getObjectname()));
      }
      bool operator()(MonitorElement *left, MonitorElement *right) const {
        return make_tuple(left) < make_tuple(right);
      }
      bool operator()(MonitorElement *left, MonitorElementData::Path const &right) const {
        return make_tuple(left) < make_tuple(right);
      }
      bool operator()(MonitorElementData::Path const &left, MonitorElement *right) const {
        return make_tuple(left) < make_tuple(right);
      }
      bool operator()(MonitorElementData::Path const &left, MonitorElementData::Path const &right) const {
        return make_tuple(left) < make_tuple(right);
      }
    };

  private:
    std::shared_ptr<MutableMonitorElementData> mutable_;  // only set if this is a mutable copy of this ME
    // there are no immutable MEs at this time, but we might need them in the future.
    /** 
     * To do anything to the MEs data, one needs to obtain an access object.
     * This object will contain the lock guard if one is needed. We differentiate
     * access for reading and access for mutation (denoted by `Access` or
     * `AccessMut`, however, also read-only access may need to take a lock
     * if it is to a mutable object. 
     * We want all of this inlined and redundant operations any copies/refs
     * optimized away.
     */
    const Access access() const {
      // First, check if there is a mutable object
      if (mutable_) {
        // if there is a mutable object, that is the truth, and we take a lock.
        return mutable_->access();
      }  // else
      throw cms::Exception("LogicError") << "MonitorElement " << getName() << " not backed by any data!";
    }

    AccessMut accessMut() {
      // For completeness, set the legacy `updated` marker.
      this->update();

      // First, check if there is a mutable object
      if (mutable_) {
        // if there is a mutable object, that is the truth, and we take a lock.
        return mutable_->accessMut();
      }  // else
      throw cms::Exception("LogicError") << "MonitorElement " << getName() << " not backed by any data!";
    }

  private:
    // but internal -- only for DQMStore etc.

    // Create ME using this data. A ROOT object pointer may be moved into the
    // new ME. The new ME will own this data.
    MonitorElement(MonitorElementData &&data);
    // Create new ME and take ownership of this data.
    MonitorElement(std::shared_ptr<MutableMonitorElementData> data);
    // Create a new ME sharing data with this existing ME.
    MonitorElement(MonitorElement *me);

    // return a new clone of the data of this ME. Calls ->Clone(), new object
    // is owned by the returned value.
    MonitorElementData cloneMEData();

    // Remove access to the data.
    std::shared_ptr<MutableMonitorElementData> release();

    // re-initialize this ME as a shared copy of the other.
    void switchData(MonitorElement *other);
    // re-initialize taking ownership of this data.
    void switchData(std::shared_ptr<MutableMonitorElementData> data);

    // Replace the ROOT object in this ME's data with the new object, taking
    // ownership. The old object is deleted.
    void switchObject(std::unique_ptr<TH1> &&newobject);

    // copy applicable fileds into the DQMNet core object for compatibility.
    // In a few places these flags are also still used by the ME.
    void syncCoreObject();
    void syncCoreObject(AccessMut &access);

    // check if the ME is currently backed by MEData; if false (almost) any
    // access will throw.
    bool isValid() const { return mutable_ != nullptr; }

    // used to implement getQErrors et. al.
    template <typename FILTER>
    std::vector<MonitorElementData::QReport *> filterQReports(FILTER filter) const;

    // legacy interfaces, there are no alternatives but they should not be used

    /// Compare monitor elements, for ordering in sets.
    bool operator<(const MonitorElement &x) const { return DQMNet::setOrder(data_, x.data_); }
    /// Check the consistency of the axis labels
    static bool CheckBinLabels(const TAxis *a1, const TAxis *a2);
    /// Get the object flags.
    uint32_t flags() const { return data_.flags; }
    /// Mark the object updated.
    void update() { data_.flags |= DQMNet::DQM_PROP_NEW; }

    // mostly used for IO, should be private.
    std::string valueString() const;
    std::string tagString() const;
    std::string tagLabelString() const;
    std::string effLabelString() const;
    std::string qualityTagString(const DQMNet::QValue &qv) const;

    // kept for DQMService. data_ is also used for MEComparison, without it
    // we'd need to keep a copy od the name somewhere else.
    /// true if ME was updated in last monitoring cycle
    bool wasUpdated() const { return data_.flags & DQMNet::DQM_PROP_NEW; }
    void packScalarData(std::string &into, const char *prefix) const;
    void packQualityData(std::string &into) const;
    DQMNet::CoreObject data_;  //< Core object information.

  public:
    MonitorElement &operator=(const MonitorElement &) = delete;
    MonitorElement &operator=(MonitorElement &&) = delete;
    virtual ~MonitorElement();

  public:
    // good to be used in subsystem code

    /// Get the type of the monitor element.
    Kind kind() const { return Kind(data_.flags & DQMNet::DQM_PROP_TYPE_MASK); }

    /// get name of ME
    const std::string &getName() const { return this->data_.objname; }

    /// get pathname of parent folder
    const std::string &getPathname() const { return this->data_.dirname; }

    /// get full name of ME including Pathname
    std::string getFullname() const { return access().key.path_.getFullname(); }

    edm::LuminosityBlockID getRunLumi() { return access().key.id_; }

    MonitorElementData::Scope getScope() { return access().key.scope_; }

    /// true if ME is meant to be stored for each luminosity section
    bool getLumiFlag() const { return access().key.scope_ == MonitorElementData::Scope::LUMI; }

    /// this ME is meant to be an efficiency plot that must not be
    /// normalized when drawn in the DQM GUI.
    void setEfficiencyFlag() {
      auto access = this->accessMut();
      if (access.value.object_)
        access.value.object_->SetBit(TH1::kIsAverage);
    }
    bool getEfficiencyFlag() {
      auto access = this->access();
      return access.value.object_ && access.value.object_->TestBit(TH1::kIsAverage);
    }

  private:
    // A static assert to check that T actually fits in
    // int64_t.
    template <typename T>
    struct fits_in_int64_t {
      int checkArray[sizeof(int64_t) - sizeof(T) + 1];
    };

    void doFill(int64_t x);

  public:
    // filling API.

    void Fill(long long x) {
      fits_in_int64_t<long long>();
      doFill(static_cast<int64_t>(x));
    }
    void Fill(unsigned long long x) {
      fits_in_int64_t<unsigned long long>();
      doFill(static_cast<int64_t>(x));
    }
    void Fill(unsigned long x) {
      fits_in_int64_t<unsigned long>();
      doFill(static_cast<int64_t>(x));
    }
    void Fill(long x) {
      fits_in_int64_t<long>();
      doFill(static_cast<int64_t>(x));
    }
    void Fill(unsigned int x) {
      fits_in_int64_t<unsigned int>();
      doFill(static_cast<int64_t>(x));
    }
    void Fill(int x) {
      fits_in_int64_t<int>();
      doFill(static_cast<int64_t>(x));
    }
    void Fill(short x) {
      fits_in_int64_t<short>();
      doFill(static_cast<int64_t>(x));
    }
    void Fill(unsigned short x) {
      fits_in_int64_t<unsigned short>();
      doFill(static_cast<int64_t>(x));
    }
    void Fill(char x) {
      fits_in_int64_t<char>();
      doFill(static_cast<int64_t>(x));
    }
    void Fill(unsigned char x) {
      fits_in_int64_t<unsigned char>();
      doFill(static_cast<int64_t>(x));
    }

    void Fill(float x) { Fill(static_cast<double>(x)); }
    void Fill(double x);
    void Fill(std::string &value);

    void Fill(double x, double yw);
    void Fill(double x, double y, double zw);
    void Fill(double x, double y, double z, double w);
    DQM_DEPRECATED
    void ShiftFillLast(double y, double ye = 0., int32_t xscale = 1);

  public:
    // additional APIs, mainly for harvesting.

    /// Remove all data from the ME, keept the empty histogram with all its settings.
    virtual void Reset();

    /// true if at least of one of the quality tests returned an error
    bool hasError() const { return data_.flags & DQMNet::DQM_PROP_REPORT_ERROR; }

    /// true if at least of one of the quality tests returned a warning
    bool hasWarning() const { return data_.flags & DQMNet::DQM_PROP_REPORT_WARN; }

    /// true if at least of one of the tests returned some other (non-ok) status
    bool hasOtherReport() const { return data_.flags & DQMNet::DQM_PROP_REPORT_OTHER; }

    /// get QReport corresponding to <qtname> (null pointer if QReport does not exist)
    const MonitorElementData::QReport *getQReport(const std::string &qtname) const;
    /// get map of QReports
    std::vector<MonitorElementData::QReport *> getQReports() const;
    /// access QReport, potentially adding it.
    void getQReport(bool create, const std::string &qtname, MonitorElementData::QReport *&qr, DQMNet::QValue *&qv);

    /// get warnings from last set of quality tests
    std::vector<MonitorElementData::QReport *> getQWarnings() const;
    /// get errors from last set of quality tests
    std::vector<MonitorElementData::QReport *> getQErrors() const;
    /// from last set of quality tests
    std::vector<MonitorElementData::QReport *> getQOthers() const;

    // const and data-independent -- safe
    virtual int getNbinsX() const;
    virtual int getNbinsY() const;
    virtual int getNbinsZ() const;
    virtual int getBin(int binx, int biny) const;
    virtual std::string getAxisTitle(int axis = 1) const;
    virtual std::string getTitle() const;

    // const but data-dependent -- semantically unsafe in RECO
    virtual double getMean(int axis = 1) const;
    virtual double getMeanError(int axis = 1) const;
    virtual double getRMS(int axis = 1) const;
    virtual double getRMSError(int axis = 1) const;
    virtual double getBinContent(int binx) const;
    virtual double getBinContent(int binx, int biny) const;
    virtual double getBinContent(int binx, int biny, int binz) const;
    virtual double getBinError(int binx) const;
    virtual double getBinError(int binx, int biny) const;
    virtual double getBinError(int binx, int biny, int binz) const;
    virtual double getEntries() const;
    virtual double getBinEntries(int bin) const;
    virtual double getBinEntries(int binx, int biny) const;
    virtual double integral() const;

    virtual int64_t getIntValue() const;
    virtual double getFloatValue() const;
    virtual const std::string &getStringValue() const;

    // non-const -- thread safety and semantical issues
    virtual void addBin(TGraph *graph);
    virtual void setBinContent(int binx, double content);
    virtual void setBinContent(int binx, int biny, double content);
    virtual void setBinContent(int binx, int biny, int binz, double content);
    virtual void setBinError(int binx, double error);
    virtual void setBinError(int binx, int biny, double error);
    virtual void setBinError(int binx, int biny, int binz, double error);
    virtual void setBinEntries(int bin, double nentries);
    virtual void setEntries(double nentries);
    virtual void divide(const MonitorElement *, const MonitorElement *, double, double, const char *);
    virtual void setBinLabel(int bin, const std::string &label, int axis = 1);
    virtual void setAxisRange(double xmin, double xmax, int axis = 1);
    virtual void setAxisTitle(const std::string &title, int axis = 1);
    virtual void setAxisTimeDisplay(int value, int axis = 1);
    virtual void setAxisTimeFormat(const char *format = "", int axis = 1);
    virtual void setTitle(const std::string &title);

    // additional operations mainly for booking
    virtual void setXTitle(std::string const &title);
    virtual void setYTitle(std::string const &title);
    virtual void enableSumw2();
    virtual void disableAlphanumeric();
    virtual void setOption(const char *option);
    virtual double getAxisMin(int axis = 1) const;
    virtual double getAxisMax(int axis = 1) const;
    // We should avoid extending histograms in general, and if the behaviour
    // is actually needed, provide a more specific interface rather than
    // relying on the ROOT behaviour.
    DQM_DEPRECATED
    virtual void setCanExtend(unsigned int value);
    // We should decide if we support this (or make it default)
    DQM_DEPRECATED
    virtual void setStatOverflows(bool value);
    virtual bool getStatOverflows();

    // these should be non-const, since they are potentially not thread-safe
    virtual TObject const *getRootObject() const;
    virtual TH1 *getTH1();
    virtual TH1F *getTH1F();
    virtual TH1S *getTH1S();
    virtual TH1D *getTH1D();
    virtual TH1I *getTH1I();
    virtual TH2F *getTH2F();
    virtual TH2S *getTH2S();
    virtual TH2I *getTH2I();
    virtual TH2D *getTH2D();
    virtual TH2Poly *getTH2Poly();
    virtual TH3F *getTH3F();
    virtual TProfile *getTProfile();
    virtual TProfile2D *getTProfile2D();

  private:
    void incompatible(const char *func) const;
    TH1 const *accessRootObject(Access const &access, const char *func, int reqdim) const;
    TH1 *accessRootObject(AccessMut const &, const char *func, int reqdim) const;

    TAxis const *getAxis(Access const &access, const char *func, int axis) const;
    TAxis *getAxis(AccessMut const &access, const char *func, int axis) const;
  };

}  // namespace dqm::impl

// These may become distinct classes in the future.
namespace dqm::reco {
  using MonitorElement = dqm::impl::MonitorElement;
}
namespace dqm::legacy {
  class MonitorElement : public dqm::reco::MonitorElement {
  public:
    // import constructors
    using dqm::reco::MonitorElement::MonitorElement;

    // Add ROOT object accessors without cost here so that harvesting code can
    // still freely use getTH1() and friends.
    using dqm::reco::MonitorElement::getRootObject;
    TObject *getRootObject() const override {
      return const_cast<TObject *>(
          const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getRootObject());
    };
    using dqm::reco::MonitorElement::getTH1;
    virtual TH1 *getTH1() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH1();
    };
    using dqm::reco::MonitorElement::getTH1F;
    virtual TH1F *getTH1F() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH1F();
    };
    using dqm::reco::MonitorElement::getTH1S;
    virtual TH1S *getTH1S() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH1S();
    };
    using dqm::reco::MonitorElement::getTH1D;
    virtual TH1D *getTH1D() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH1D();
    };
    using dqm::reco::MonitorElement::getTH1I;
    virtual TH1I *getTH1I() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH1I();
    };
    using dqm::reco::MonitorElement::getTH2F;
    virtual TH2F *getTH2F() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH2F();
    };
    using dqm::reco::MonitorElement::getTH2S;
    virtual TH2S *getTH2S() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH2S();
    };
    using dqm::reco::MonitorElement::getTH2I;
    virtual TH2I *getTH2I() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH2I();
    };
    using dqm::reco::MonitorElement::getTH2D;
    virtual TH2D *getTH2D() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH2D();
    };
    using dqm::reco::MonitorElement::getTH2Poly;
    virtual TH2Poly *getTH2Poly() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH2Poly();
    };
    using dqm::reco::MonitorElement::getTH3F;
    virtual TH3F *getTH3F() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH3F();
    };
    using dqm::reco::MonitorElement::getTProfile;
    virtual TProfile *getTProfile() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTProfile();
    };
    using dqm::reco::MonitorElement::getTProfile2D;
    virtual TProfile2D *getTProfile2D() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTProfile2D();
    };
  };
}  // namespace dqm::legacy
namespace dqm::harvesting {
  using MonitorElement = dqm::legacy::MonitorElement;
}

#endif  // DQMSERVICES_CORE_MONITOR_ELEMENT_H

#ifndef DQMSERVICES_CORE_MONITOR_ELEMENT_H
#define DQMSERVICES_CORE_MONITOR_ELEMENT_H

#if __GNUC__ && !defined DQM_DEPRECATED
#define DQM_DEPRECATED __attribute__((deprecated))
#endif

#include "DQMServices/Core/interface/DQMNet.h"
#include "DQMServices/Core/interface/QReport.h"

#include "DataFormats/Histograms/interface/MonitorElementCollection.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "TF1.h"
#include "TH1F.h"
#include "TH1S.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TH2S.h"
#include "TH2D.h"
#include "TH3F.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TObjString.h"
#include "TAxis.h"

#include <mutex>
#include <string>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstdint>
#include <sys/time.h>
#include <tbb/spin_mutex.h>

class QCriterion;
class DQMService;
namespace dqm::dqmstoreimpl {
  class DQMStore;
}

// tag for a special constructor, see below
struct MonitorElementNoCloneTag {};

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
    friend dqm::dqmstoreimpl::DQMStore;
    friend DQMService;

  public:
    using Scalar = MonitorElementData::Scalar;
    using Kind = MonitorElementData::Kind;

  protected:
    DQMNet::CoreObject data_;  //< Core object information.
    // TODO: we only use the ::Value part so far.
    // Still using the full thing to remain compatible with the new ME implementation.

    std::atomic<MonitorElementData const *> frozen_;    // only set if this ME is in a product already
    std::atomic<MutableMonitorElementData *> mutable_;  // only set if there is a mutable copy of this ME
    /** 
     * To do anything to the MEs data, one needs to obtain an access object.
     * This object will contain the lock guard if one is needed. We differentiate
     * access for reading and access for mutation (denoted by `Access` or
     * `const Access`, however, also read-only access may need to take a lock
     * if it is to a mutable object. Obtaining mutable access may involve
     * creating a clone of the backing data. In this case, the pointers are
     * updated using atomic operations. It can happen that reads go to the old
     * object while a new object exists already, but this is fine; concurrent
     * reads and writes can happen in arbitrary order. However, we need to
     * protect against the case where clones happen concurrently and avoid
     * leaking memory or loosing updates in this case, using atomics.
     * We want all of this inlined and redundant operations any copies/refs
     * optimized away.
     */
    const Access access() const {
      // First, check if there is a mutable object
      auto mut = mutable_.load();
      if (mut) {
        // if there is a mutable object, that is the truth, and we take a lock.
        return mut->access();
      }  // else
      auto frozen = frozen_.load();
      if (frozen) {
        // in case of an immutable object read from edm products, create an
        // access object without lock.
        return Access{std::unique_lock<dqmmutex>(), frozen->key_, frozen->value_};
      }
      // else
      throw cms::Exception("LogicError") << "MonitorElement not backed by any data!";
    }

    AccessMut accessMut() {
      // For completeness, set the legacy `updated` marker.
      this->update();

      // First, check if there is a mutable object
      auto mut = mutable_.load();
      if (mut) {
        // if there is a mutable object, that is the truth, and we take a lock.
        return mut->accessMut();
      }  // else
      auto frozen = frozen_.load();
      if (!frozen) {
        throw cms::Exception("LogicError") << "MonitorElement not backed by any data!";
      }
      // in case of an immutable object read from edm products, attempt to
      // make a clone.
      MutableMonitorElementData *clone = new MutableMonitorElementData();
      clone->data_.key_ = frozen->key_;
      clone->data_.value_.scalar_ = frozen->value_.scalar_;
      if (frozen->value_.object_) {
        // Clone() the TH1
        clone->data_.value_.object_ = std::unique_ptr<TH1>(static_cast<TH1 *>(frozen->value_.object_->Clone()));
      }

      // now try to set our clone, and see if it was still needed (sb. else
      // might have made a clone already!)
      MutableMonitorElementData *existing = nullptr;
      bool ok = mutable_.compare_exchange_strong(existing, clone);
      if (!ok) {
        // somebody else made a clone already, it is now in existing
        delete clone;
        return existing->accessMut();
      } else {
        // we won the race, and our clone is the real one now.
        return clone->accessMut();
      }
      // in either case, if somebody destroyed the mutable object between us
      // getting the pointer and us locking it, we are screwed. We have to rely
      // on edm and the DQM code to make sure we only turn mutable objects into
      // products once all processing is done (logically, this is safe).
    }

    //TODO:  to be dropped.
    TH1 *reference_;                 //< Current ROOT reference object.
    TH1 *refvalue_;                  //< Soft reference if any.
    std::vector<QReport> qreports_;  //< QReports associated to this object.

    MonitorElement *initialise(Kind kind);
    MonitorElement *initialise(Kind kind, TH1 *rootobj);
    MonitorElement *initialise(Kind kind, const std::string &value);
    void globalize() { data_.moduleId = 0; }
    void setLumi(uint32_t ls) { data_.lumi = ls; }

  public:
    MonitorElement();
    MonitorElement(const std::string *path, const std::string &name);
    MonitorElement(const std::string *path, const std::string &name, uint32_t run, uint32_t moduleId);
    MonitorElement(const MonitorElement &, MonitorElementNoCloneTag);
    MonitorElement(const MonitorElement &);
    MonitorElement &operator=(const MonitorElement &) = delete;
    MonitorElement &operator=(MonitorElement &&) = delete;
    virtual ~MonitorElement();

    /// Compare monitor elements, for ordering in sets.
    bool operator<(const MonitorElement &x) const { return DQMNet::setOrder(data_, x.data_); }

    /// Check the consistency of the axis labels
    static bool CheckBinLabels(const TAxis *a1, const TAxis *a2);

    /// Get the type of the monitor element.
    Kind kind() const { return Kind(data_.flags & DQMNet::DQM_PROP_TYPE_MASK); }

    /// Get the object flags.
    uint32_t flags() const { return data_.flags; }

    /// get name of ME
    const std::string &getName() const { return data_.objname; }

    /// get pathname of parent folder
    const std::string &getPathname() const { return *data_.dirname; }

    /// get full name of ME including Pathname
    const std::string getFullname() const {
      std::string path;
      path.reserve(data_.dirname->size() + data_.objname.size() + 2);
      path += *data_.dirname;
      if (!data_.dirname->empty())
        path += '/';
      path += data_.objname;
      return path;
    }

    /// true if ME was updated in last monitoring cycle
    bool wasUpdated() const { return data_.flags & DQMNet::DQM_PROP_NEW; }

    /// Mark the object updated.
    void update() { data_.flags |= DQMNet::DQM_PROP_NEW; }

    /// specify whether ME should be reset at end of monitoring cycle (default:false);
    /// (typically called by Sources that control the original ME)
    void setResetMe(bool /* flag */) { data_.flags |= DQMNet::DQM_PROP_RESET; }

    /// true if ME is meant to be stored for each luminosity section
    bool getLumiFlag() const { return data_.flags & DQMNet::DQM_PROP_LUMI; }

    /// this ME is meant to be stored for each luminosity section
    void setLumiFlag() { data_.flags |= DQMNet::DQM_PROP_LUMI; }

    /// this ME is meant to be an efficiency plot that must not be
    /// normalized when drawn in the DQM GUI.
    void setEfficiencyFlag() { data_.flags |= DQMNet::DQM_PROP_EFFICIENCY_PLOT; }

    // A static assert to check that T actually fits in
    // int64_t.
    template <typename T>
    struct fits_in_int64_t {
      int checkArray[sizeof(int64_t) - sizeof(T) + 1];
    };

  protected:
    void doFill(int64_t x);

  public:
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

    virtual void Reset();
    DQM_DEPRECATED
    virtual void softReset();

    // mostly used for IO, should be private.
    std::string valueString() const;
    std::string tagString() const;
    std::string tagLabelString() const;
    std::string effLabelString() const;
    std::string qualityTagString(const DQMNet::QValue &qv) const;

    /// true if at least of one of the quality tests returned an error
    bool hasError() const { return data_.flags & DQMNet::DQM_PROP_REPORT_ERROR; }

    /// true if at least of one of the quality tests returned a warning
    bool hasWarning() const { return data_.flags & DQMNet::DQM_PROP_REPORT_WARN; }

    /// true if at least of one of the tests returned some other (non-ok) status
    bool hasOtherReport() const { return data_.flags & DQMNet::DQM_PROP_REPORT_OTHER; }

    /// true if the plot has been marked as an efficiency plot, which
    /// will not be normalized when rendered within the DQM GUI.
    bool isEfficiency() const { return data_.flags & DQMNet::DQM_PROP_EFFICIENCY_PLOT; }

    /// get QReport corresponding to <qtname> (null pointer if QReport does not exist)
    const QReport *getQReport(const std::string &qtname) const;
    /// get map of QReports
    std::vector<QReport *> getQReports() const;
    /// get warnings from last set of quality tests
    std::vector<QReport *> getQWarnings() const;
    /// get errors from last set of quality tests
    std::vector<QReport *> getQErrors() const;
    /// from last set of quality tests
    std::vector<QReport *> getQOthers() const;

    /// run all quality tests
    void runQTests();

    // const and data-independent -- safe
    virtual int getNbinsX() const;
    virtual int getNbinsY() const;
    virtual int getNbinsZ() const;
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

    // non-const -- thread safety and semantical issues
    virtual void setBinContent(int binx, double content);
    virtual void setBinContent(int binx, int biny, double content);
    virtual void setBinContent(int binx, int biny, int binz, double content);
    virtual void setBinError(int binx, double error);
    virtual void setBinError(int binx, int biny, double error);
    virtual void setBinError(int binx, int biny, int binz, double error);
    virtual void setBinEntries(int bin, double nentries);
    virtual void setEntries(double nentries);
    virtual void setBinLabel(int bin, const std::string &label, int axis = 1);
    virtual void setAxisRange(double xmin, double xmax, int axis = 1);
    virtual void setAxisTitle(const std::string &title, int axis = 1);
    virtual void setAxisTimeDisplay(int value, int axis = 1);
    virtual void setAxisTimeFormat(const char *format = "", int axis = 1);
    virtual void setTitle(const std::string &title);
    // --- Operations that origianted in ConcurrentME ---
    virtual void setXTitle(std::string const &title);
    virtual void setYTitle(std::string const &title);
    virtual void enableSumw2();
    virtual void disableAlphanumeric();
    virtual void setOption(const char *option);

    // new operations to reduce usage of getTH*
    virtual double getAxisMin(int axis = 1) const;
    virtual double getAxisMax(int axis = 1) const;
    // We should avoid extending histograms in general, and if the behaviour
    // is actually needed, provide a more specific interface rather than
    // relying on the ROOT behaviour.
    DQM_DEPRECATED
    virtual void setCanExtend(unsigned int value);
    // We should decide if we support this (or make it default)
    DQM_DEPRECATED
    virtual void setStatOverflows(unsigned int value);

    // these should be non-const, since they are potentially not thread-safe
    virtual TObject const *getRootObject() const;
    virtual TH1 *getTH1();
    virtual TH1F *getTH1F();
    virtual TH1S *getTH1S();
    virtual TH1D *getTH1D();
    virtual TH2F *getTH2F();
    virtual TH2S *getTH2S();
    virtual TH2D *getTH2D();
    virtual TH3F *getTH3F();
    virtual TProfile *getTProfile();
    virtual TProfile2D *getTProfile2D();

  public:
    virtual int64_t getIntValue() const;
    virtual double getFloatValue() const;
    virtual const std::string &getStringValue() const;
    void packScalarData(std::string &into, const char *prefix) const;
    void packQualityData(std::string &into) const;

  protected:
    void incompatible(const char *func) const;
    TH1 const *accessRootObject(Access const &access, const char *func, int reqdim) const;
    TH1 *accessRootObject(AccessMut const &, const char *func, int reqdim) const;

    void setAxisTimeOffset(double toffset, const char *option = "local", int axis = 1);

    /// whether soft-reset is enabled; default is false
    bool isSoftResetEnabled() const { return refvalue_ != nullptr; }

    /// whether ME contents should be accumulated over multiple monitoring periods; default: false
    bool isAccumulateEnabled() const { return data_.flags & DQMNet::DQM_PROP_ACCUMULATE; }

    /// true if ME is marked for deletion
    bool markedToDelete() const { return data_.flags & DQMNet::DQM_PROP_MARKTODELETE; }

    /// Mark the object for deletion.
    /// NB: make sure that the following method is not called simultaneously for the same ME
    void markToDelete() { data_.flags |= DQMNet::DQM_PROP_MARKTODELETE; }

    /// reset "was updated" flag
    void resetUpdate() { data_.flags &= ~DQMNet::DQM_PROP_NEW; }

    /// true if ME should be reset at end of monitoring cycle
    bool resetMe() const { return data_.flags & DQMNet::DQM_PROP_RESET; }

    /// if true, will accumulate ME contents (over many periods)
    /// until method is called with flag = false again
    void setAccumulate(bool /* flag */) { data_.flags |= DQMNet::DQM_PROP_ACCUMULATE; }

    TAxis const *getAxis(Access const &access, const char *func, int axis) const;
    TAxis *getAxis(AccessMut const &access, const char *func, int axis) const;

    // ------------ Operations for MEs that are normally never reset ---------
  protected:
    void disableSoftReset();
    void addProfiles(TProfile *h1, TProfile *h2, TProfile *sum, float c1, float c2);
    void addProfiles(TProfile2D *h1, TProfile2D *h2, TProfile2D *sum, float c1, float c2);
    void copyFunctions(TH1 *from, TH1 *to);
    void copyFrom(TH1 *from);

    // --- Operations on MEs that are normally reset at end of monitoring cycle ---
    void getQReport(bool create, const std::string &qtname, QReport *&qr, DQMNet::QValue *&qv);
    void addQReport(const DQMNet::QValue &desc, QCriterion *qc);
    void addQReport(QCriterion *qc);
    void updateQReportStats();

  public:
    const uint32_t run() const { return data_.run; }
    const uint32_t lumi() const { return data_.lumi; }
    const uint32_t moduleId() const { return data_.moduleId; }
  };

}  // namespace dqm::impl

// These will become distinct classes in the future.
namespace dqm::reco {
  typedef dqm::impl::MonitorElement MonitorElement;
}
namespace dqm::legacy {
  class MonitorElement : public dqm::reco::MonitorElement {
  public:
    // import constructors
    using dqm::reco::MonitorElement::MonitorElement;

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
    using dqm::reco::MonitorElement::getTH2F;
    virtual TH2F *getTH2F() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH2F();
    };
    using dqm::reco::MonitorElement::getTH2S;
    virtual TH2S *getTH2S() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH2S();
    };
    using dqm::reco::MonitorElement::getTH2D;
    virtual TH2D *getTH2D() const {
      return const_cast<dqm::legacy::MonitorElement *>(this)->dqm::reco::MonitorElement::getTH2D();
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
    void runQTests();
  };
}  // namespace dqm::legacy
namespace dqm::harvesting {
  typedef dqm::legacy::MonitorElement MonitorElement;
}

#endif  // DQMSERVICES_CORE_MONITOR_ELEMENT_H

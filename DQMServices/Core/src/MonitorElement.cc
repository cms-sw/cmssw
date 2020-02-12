#define __STDC_FORMAT_MACROS 1
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/src/DQMError.h"
#include "TClass.h"
#include "TMath.h"
#include "TList.h"
#include "THashList.h"
#include <iostream>
#include <cassert>
#include <cfloat>
#include <cinttypes>

#if !WITHOUT_CMS_FRAMEWORK
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif

namespace dqm::impl {

  static TH1 *checkRootObject(const std::string &name, TObject *tobj, const char *func, int reqdim) {
    if (!tobj)
      raiseDQMError("MonitorElement",
                    "Method '%s' cannot be invoked on monitor"
                    " element '%s' because it is not a ROOT object.",
                    func,
                    name.c_str());

    auto *h = static_cast<TH1 *>(tobj);
    int ndim = h->GetDimension();
    if (reqdim < 0 || reqdim > ndim)
      raiseDQMError("MonitorElement",
                    "Method '%s' cannot be invoked on monitor"
                    " element '%s' because it requires %d dimensions; this"
                    " object of type '%s' has %d dimensions",
                    func,
                    name.c_str(),
                    reqdim,
                    typeid(*h).name(),
                    ndim);

    return h;
  }

  MonitorElement::MonitorElement(MonitorElementData &&data) {
    this->mutable_ = new MutableMonitorElementData();
    this->mutable_->data_ = std::move(data);
    this->is_owned_ = true;
    syncCoreObject();
  }
  MonitorElement::MonitorElement(MutableMonitorElementData *data) { switchData(data); }
  MonitorElement::MonitorElement(MonitorElement *me) { switchData(me); }

  MonitorElementData MonitorElement::cloneMEData() {
    MonitorElementData out;
    auto access = this->access();
    out.key_ = access.key;
    out.value_.scalar_ = access.value.scalar_;
    if (access.value.object_) {
      out.value_.object_ = std::unique_ptr<TH1>(static_cast<TH1 *>(access.value.object_->Clone()));
    }
    return out;
  }

  MutableMonitorElementData *MonitorElement::release(bool expectOwned) {
    assert(this->is_owned_ == expectOwned);
    MutableMonitorElementData *data = this->mutable_;
    this->mutable_ = nullptr;
    this->is_owned_ = false;
    assert(!expectOwned || data);
    return data;
  }

  void MonitorElement::switchData(MonitorElement *other) {
    assert(other);
    this->mutable_ = other->mutable_;
    this->is_owned_ = false;
    syncCoreObject();
  }

  void MonitorElement::switchData(MutableMonitorElementData *data) {
    this->mutable_ = data;
    this->is_owned_ = true;
    syncCoreObject();
  }

  void MonitorElement::switchObject(std::unique_ptr<TH1> &&newobject) {
    auto access = this->accessMut();
    // Assume kind etc. matches.
    // This should free the old object.
    access.value.object_ = std::move(newobject);
  }

  void MonitorElement::syncCoreObject() {
    auto access = this->accessMut();
    syncCoreObject(access);
  }

  void MonitorElement::syncCoreObject(AccessMut &access) {
    data_.flags &= ~DQMNet::DQM_PROP_TYPE_MASK;
    data_.flags |= (int)access.key.kind_;

    // mark as updated.
    data_.flags |= DQMNet::DQM_PROP_NEW;

    // lumi flag is approximately equivalent to Scope::LUMI.
    data_.flags &= ~DQMNet::DQM_PROP_LUMI;
    if (access.key.scope_ == MonitorElementData::Scope::LUMI) {
      data_.flags |= DQMNet::DQM_PROP_LUMI;
    }

    // these are unsupported and always off.
    data_.flags &= ~DQMNet::DQM_PROP_HAS_REFERENCE;
    data_.flags &= ~DQMNet::DQM_PROP_TAGGED;
    data_.flags &= ~DQMNet::DQM_PROP_RESET;
    data_.flags &= ~DQMNet::DQM_PROP_ACCUMULATE;

    // we use ROOT's internal efficiency flag as the truth
    data_.flags &= ~DQMNet::DQM_PROP_EFFICIENCY_PLOT;
    if (access.value.object_ && access.value.object_->TestBit(TH1::kIsAverage)) {
      data_.flags |= DQMNet::DQM_PROP_EFFICIENCY_PLOT;
    }

    data_.tag = 0;

    // don't touch version (a timestamp).

    // we could set proper values here, but nobody should use them.
    data_.run = 0;
    data_.lumi = 0;

    // these are relics from the threaded migration and should not be used anywhere.
    data_.streamId = 0;
    data_.moduleId = 0;

    // leaking a pointer here, but that should be fine.
    data_.dirname = access.key.path_.getDirname();

    data_.objname = access.key.path_.getObjectname();

    data_.flags &= ~DQMNet::DQM_PROP_REPORT_ALARM;
    data_.qreports.clear();
    for (QReport const &qr : access.value.qreports_) {
      data_.qreports.push_back(qr.getValue());
      switch (qr.getStatus()) {
        case dqm::qstatus::STATUS_OK:
          break;
        case dqm::qstatus::WARNING:
          data_.flags |= DQMNet::DQM_PROP_REPORT_WARN;
          break;
        case dqm::qstatus::ERROR:
          data_.flags |= DQMNet::DQM_PROP_REPORT_ERROR;
          break;
        default:
          data_.flags |= DQMNet::DQM_PROP_REPORT_OTHER;
          break;
      }
    }
  }

  MonitorElement::~MonitorElement() {
    if (is_owned_)
      delete mutable_;
  }

  //utility function to check the consistency of the axis labels
  //taken from TH1::CheckBinLabels which is not public
  bool MonitorElement::CheckBinLabels(const TAxis *a1, const TAxis *a2) {
    // check that axis have same labels
    THashList *l1 = (const_cast<TAxis *>(a1))->GetLabels();
    THashList *l2 = (const_cast<TAxis *>(a2))->GetLabels();

    if (!l1 && !l2)
      return true;
    if (!l1 || !l2) {
      return false;
    }
    // check now labels sizes  are the same
    if (l1->GetSize() != l2->GetSize()) {
      return false;
    }
    for (int i = 1; i <= a1->GetNbins(); ++i) {
      TString label1 = a1->GetBinLabel(i);
      TString label2 = a2->GetBinLabel(i);
      if (label1 != label2) {
        return false;
      }
    }
    return true;
  }

  /// "Fill" ME methods for string
  void MonitorElement::Fill(std::string &value) {
    auto access = this->accessMut();
    update();
    if (kind() == Kind::STRING) {
      access.value.scalar_.str = value;
    } else {
      incompatible(__PRETTY_FUNCTION__);
    }
  }

  /// "Fill" ME methods for double
  void MonitorElement::Fill(double x) {
    auto access = this->accessMut();
    update();
    if (kind() == Kind::INT)
      access.value.scalar_.num = static_cast<int64_t>(x);
    else if (kind() == Kind::REAL)
      access.value.scalar_.real = x;
    else if (kind() == Kind::TH1F)
      accessRootObject(access, __PRETTY_FUNCTION__, 1)->Fill(x, 1);
    else if (kind() == Kind::TH1S)
      accessRootObject(access, __PRETTY_FUNCTION__, 1)->Fill(x, 1);
    else if (kind() == Kind::TH1D)
      accessRootObject(access, __PRETTY_FUNCTION__, 1)->Fill(x, 1);
    else
      incompatible(__PRETTY_FUNCTION__);
  }

  /// "Fill" ME method for int64_t
  void MonitorElement::doFill(int64_t x) {
    auto access = this->accessMut();
    update();
    if (kind() == Kind::INT)
      access.value.scalar_.num = static_cast<int64_t>(x);
    else if (kind() == Kind::REAL)
      access.value.scalar_.real = static_cast<double>(x);
    else if (kind() == Kind::TH1F)
      accessRootObject(access, __PRETTY_FUNCTION__, 1)->Fill(static_cast<double>(x), 1);
    else if (kind() == Kind::TH1S)
      accessRootObject(access, __PRETTY_FUNCTION__, 1)->Fill(static_cast<double>(x), 1);
    else if (kind() == Kind::TH1D)
      accessRootObject(access, __PRETTY_FUNCTION__, 1)->Fill(static_cast<double>(x), 1);
    else
      incompatible(__PRETTY_FUNCTION__);
  }

  /// can be used with 2D (x,y) or 1D (x, w) histograms
  void MonitorElement::Fill(double x, double yw) {
    auto access = this->accessMut();
    update();
    if (kind() == Kind::TH1F)
      accessRootObject(access, __PRETTY_FUNCTION__, 1)->Fill(x, yw);
    else if (kind() == Kind::TH1S)
      accessRootObject(access, __PRETTY_FUNCTION__, 1)->Fill(x, yw);
    else if (kind() == Kind::TH1D)
      accessRootObject(access, __PRETTY_FUNCTION__, 1)->Fill(x, yw);
    else if (kind() == Kind::TH2F)
      static_cast<TH2F *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, yw, 1);
    else if (kind() == Kind::TH2S)
      static_cast<TH2S *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, yw, 1);
    else if (kind() == Kind::TH2D)
      static_cast<TH2D *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, yw, 1);
    else if (kind() == Kind::TPROFILE)
      static_cast<TProfile *>(accessRootObject(access, __PRETTY_FUNCTION__, 1))->Fill(x, yw, 1);
    else
      incompatible(__PRETTY_FUNCTION__);
  }

  /// shift bin to the left and fill last bin with new entry
  /// 1st argument is y value, 2nd argument is y error (default 0)
  /// can be used with 1D or profile histograms only
  void MonitorElement::ShiftFillLast(double y, double ye, int xscale) {
    // TODO: this should take the lock only once to be actually safe.
    // But since it is not const, we don't even claim it is thread-safe.
    update();
    if (kind() == Kind::TH1F || kind() == Kind::TH1S || kind() == Kind::TH1D) {
      int nbins = getNbinsX();
      auto entries = (int)getEntries();
      // first fill bins from left to right
      int index = entries + 1;
      int xlow = 2;
      int xup = nbins;
      // if more entries than bins then start shifting
      if (entries >= nbins) {
        index = nbins;
        xlow = entries - nbins + 3;
        xup = entries + 1;
        // average first bin
        double y1 = getBinContent(1);
        double y2 = getBinContent(2);
        double y1err = getBinError(1);
        double y2err = getBinError(2);
        double N = entries - nbins + 1.;
        if (ye == 0. || y1err == 0. || y2err == 0.) {
          // for errors zero calculate unweighted mean and its error
          double sum = N * y1 + y2;
          y1 = sum / (N + 1.);
          // FIXME check if correct
          double s = (N + 1.) * (N * y1 * y1 + y2 * y2) - sum * sum;
          if (s >= 0.)
            y1err = sqrt(s) / (N + 1.);
          else
            y1err = 0.;
        } else {
          // for errors non-zero calculate weighted mean and its error
          double denom = (1. / y1err + 1. / y2err);
          double mean = (y1 / y1err + y2 / y2err) / denom;
          // FIXME check if correct
          y1err = sqrt(((y1 - mean) * (y1 - mean) / y1err + (y2 - mean) * (y2 - mean) / y2err) / denom / 2.);
          y1 = mean;  // set y1 to mean for filling below
        }
        setBinContent(1, y1);
        setBinError(1, y1err);
        // shift remaining bins to the left
        for (int i = 3; i <= nbins; i++) {
          setBinContent(i - 1, getBinContent(i));
          setBinError(i - 1, getBinError(i));
        }
      }
      // fill last bin with new values
      setBinContent(index, y);
      setBinError(index, ye);
      // set entries
      setEntries(entries + 1);
      // set axis labels and reset drawing option
      char buffer[10];
      sprintf(buffer, "%d", xlow * xscale);
      std::string a(buffer);
      setBinLabel(2, a);
      sprintf(buffer, "%d", xup * xscale);
      std::string b(buffer);
      setBinLabel(nbins, b);
      setBinLabel(1, "av.");
    } else
      incompatible(__PRETTY_FUNCTION__);
  }
  /// can be used with 3D (x, y, z) or 2D (x, y, w) histograms
  void MonitorElement::Fill(double x, double y, double zw) {
    auto access = this->accessMut();
    update();
    if (kind() == Kind::TH2F)
      static_cast<TH2F *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, y, zw);
    else if (kind() == Kind::TH2S)
      static_cast<TH2S *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, y, zw);
    else if (kind() == Kind::TH2D)
      static_cast<TH2D *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, y, zw);
    else if (kind() == Kind::TH3F)
      static_cast<TH3F *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, y, zw, 1);
    else if (kind() == Kind::TPROFILE)
      static_cast<TProfile *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, y, zw);
    else if (kind() == Kind::TPROFILE2D)
      static_cast<TProfile2D *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, y, zw, 1);
    else
      incompatible(__PRETTY_FUNCTION__);
  }

  /// can be used with 3D (x, y, z, w) histograms
  void MonitorElement::Fill(double x, double y, double z, double w) {
    auto access = this->accessMut();
    update();
    if (kind() == Kind::TH3F)
      static_cast<TH3F *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, y, z, w);
    else if (kind() == Kind::TPROFILE2D)
      static_cast<TProfile2D *>(accessRootObject(access, __PRETTY_FUNCTION__, 2))->Fill(x, y, z, w);
    else
      incompatible(__PRETTY_FUNCTION__);
  }

  /// reset ME (ie. contents, errors, etc)
  void MonitorElement::Reset() {
    auto access = this->accessMut();
    update();
    if (kind() == Kind::INT)
      access.value.scalar_.num = 0;
    else if (kind() == Kind::REAL)
      access.value.scalar_.real = 0;
    else if (kind() == Kind::STRING)
      access.value.scalar_.str.clear();
    else
      return accessRootObject(access, __PRETTY_FUNCTION__, 1)->Reset();
  }

  /// convert scalar data into a string.
  void MonitorElement::packScalarData(std::string &into, const char *prefix) const {
    auto access = this->access();
    char buf[64];
    if (kind() == Kind::INT) {
      snprintf(buf, sizeof(buf), "%s%" PRId64, prefix, access.value.scalar_.num);
      into = buf;
    } else if (kind() == Kind::REAL) {
      snprintf(buf, sizeof(buf), "%s%.*g", prefix, DBL_DIG + 2, access.value.scalar_.real);
      into = buf;
    } else if (kind() == Kind::STRING) {
      into.reserve(strlen(prefix) + access.value.scalar_.str.size());
      into += prefix;
      into += access.value.scalar_.str;
    } else
      incompatible(__PRETTY_FUNCTION__);
  }

  /// serialise quality report information into a string.
  void MonitorElement::packQualityData(std::string &into) const { DQMNet::packQualityData(into, data_.qreports); }

  /// returns value of ME in string format (eg. "f = 3.14151926" for double numbers);
  /// relevant only for scalar or string MEs
  std::string MonitorElement::valueString() const {
    std::string result;
    if (kind() == Kind::INT)
      packScalarData(result, "i=");
    else if (kind() == Kind::REAL)
      packScalarData(result, "f=");
    else if (kind() == Kind::STRING)
      packScalarData(result, "s=");
    else
      incompatible(__PRETTY_FUNCTION__);

    return result;
  }

  /// return tagged value of ME in string format
  /// (eg. <name>f=3.14151926</name> for double numbers);
  /// relevant only for sending scalar or string MEs over TSocket
  std::string MonitorElement::tagString() const {
    std::string result;
    std::string val(valueString());
    result.reserve(6 + 2 * data_.objname.size() + val.size());
    result += '<';
    result += data_.objname;
    result += '>';
    result += val;
    result += '<';
    result += '/';
    result += data_.objname;
    result += '>';
    return result;
  }

  /// return label string for the monitor element tag (eg. <name>t=12345</name>)
  std::string MonitorElement::tagLabelString() const {
    char buf[32];
    std::string result;
    size_t len = sprintf(buf, "t=%" PRIu32, data_.tag);

    result.reserve(6 + 2 * data_.objname.size() + len);
    result += '<';
    result += data_.objname;
    result += '>';
    result += buf;
    result += '<';
    result += '/';
    result += data_.objname;
    result += '>';
    return result;
  }

  /// return label string for the monitor element tag (eg. <name>t=12345</name>)
  std::string MonitorElement::effLabelString() const {
    std::string result;

    result.reserve(6 + 2 * data_.objname.size() + 3);
    result += '<';
    result += data_.objname;
    result += '>';
    result += "e=1";
    result += '<';
    result += '/';
    result += data_.objname;
    result += '>';
    return result;
  }

  std::string MonitorElement::qualityTagString(const DQMNet::QValue &qv) const {
    char buf[64];
    std::string result;
    size_t titlelen = data_.objname.size() + qv.qtname.size() + 1;
    size_t buflen = sprintf(buf, "qr=st:%d:%.*g:", qv.code, DBL_DIG + 2, qv.qtresult);

    result.reserve(7 + 2 * titlelen + buflen + qv.algorithm.size() + qv.message.size());
    result += '<';
    result += data_.objname;
    result += '.';
    result += qv.qtname;
    result += '>';
    result += buf;
    result += qv.algorithm;
    result += ':';
    result += qv.message;
    result += '<';
    result += '/';
    result += data_.objname;
    result += '.';
    result += qv.qtname;
    result += '>';
    return result;
  }

  const MonitorElementData::QReport *MonitorElement::getQReport(const std::string &qtname) const {
    MonitorElementData::MonitorElementData::QReport *qr;
    DQMNet::QValue *qv;
    const_cast<MonitorElement *>(this)->getQReport(false, qtname, qr, qv);
    return qr;
  }

  template <typename FILTER>
  std::vector<MonitorElementData::QReport *> MonitorElement::filterQReports(FILTER filter) const {
    auto access = this->access();
    std::vector<MonitorElementData::QReport *> result;
    for (MonitorElementData::QReport const &qr : access.value.qreports_) {
      if (filter(qr)) {
        // const_cast here because this API always violated cons'ness. Should
        // make the result type const and fix all usages.
        result.push_back(const_cast<MonitorElementData::QReport *>(&qr));
      }
    }
    return result;
  }

  std::vector<MonitorElementData::QReport *> MonitorElement::getQReports() const {
    return filterQReports([](MonitorElementData::QReport const &qr) { return true; });
  }

  std::vector<MonitorElementData::QReport *> MonitorElement::getQWarnings() const {
    return filterQReports(
        [](MonitorElementData::QReport const &qr) { return qr.getStatus() == dqm::qstatus::WARNING; });
  }

  std::vector<MonitorElementData::QReport *> MonitorElement::getQErrors() const {
    return filterQReports([](MonitorElementData::QReport const &qr) { return qr.getStatus() == dqm::qstatus::ERROR; });
  }

  std::vector<MonitorElementData::QReport *> MonitorElement::getQOthers() const {
    return filterQReports([](MonitorElementData::QReport const &qr) {
      return qr.getStatus() != dqm::qstatus::STATUS_OK && qr.getStatus() != dqm::qstatus::WARNING &&
             qr.getStatus() != dqm::qstatus::ERROR;
    });
  }

  void MonitorElement::incompatible(const char *func) const {
    raiseDQMError("MonitorElement",
                  "Method '%s' cannot be invoked on monitor"
                  " element '%s'",
                  func,
                  data_.objname.c_str());
  }

  TH1 const *MonitorElement::accessRootObject(Access const &access, const char *func, int reqdim) const {
    if (kind() < Kind::TH1F)
      raiseDQMError("MonitorElement",
                    "Method '%s' cannot be invoked on monitor"
                    " element '%s' because it is not a root object",
                    func,
                    data_.objname.c_str());
    return access.value.object_.get();
  }
  TH1 *MonitorElement::accessRootObject(AccessMut const &access, const char *func, int reqdim) const {
    if (kind() < Kind::TH1F)
      raiseDQMError("MonitorElement",
                    "Method '%s' cannot be invoked on monitor"
                    " element '%s' because it is not a root object",
                    func,
                    data_.objname.c_str());
    return checkRootObject(data_.objname, access.value.object_.get(), func, reqdim);
  }

  /*** getter methods (wrapper around ROOT methods) ****/
  //
  /// get mean value of histogram along x, y or z axis (axis=1, 2, 3 respectively)
  double MonitorElement::getMean(int axis /* = 1 */) const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, axis - 1)->GetMean(axis);
  }

  /// get mean value uncertainty of histogram along x, y or z axis
  /// (axis=1, 2, 3 respectively)
  double MonitorElement::getMeanError(int axis /* = 1 */) const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, axis - 1)->GetMeanError(axis);
  }

  /// get RMS of histogram along x, y or z axis (axis=1, 2, 3 respectively)
  double MonitorElement::getRMS(int axis /* = 1 */) const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, axis - 1)->GetRMS(axis);
  }

  /// get RMS uncertainty of histogram along x, y or z axis(axis=1,2,3 respectively)
  double MonitorElement::getRMSError(int axis /* = 1 */) const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, axis - 1)->GetRMSError(axis);
  }

  /// get # of bins in X-axis
  int MonitorElement::getNbinsX() const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 1)->GetNbinsX();
  }

  /// get # of bins in Y-axis
  int MonitorElement::getNbinsY() const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 2)->GetNbinsY();
  }

  /// get # of bins in Z-axis
  int MonitorElement::getNbinsZ() const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 3)->GetNbinsZ();
  }

  /// get content of bin (1-D)
  double MonitorElement::getBinContent(int binx) const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 1)->GetBinContent(binx);
  }

  /// get content of bin (2-D)
  double MonitorElement::getBinContent(int binx, int biny) const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 2)->GetBinContent(binx, biny);
  }

  /// get content of bin (3-D)
  double MonitorElement::getBinContent(int binx, int biny, int binz) const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 3)->GetBinContent(binx, biny, binz);
  }

  /// get uncertainty on content of bin (1-D) - See TH1::GetBinError for details
  double MonitorElement::getBinError(int binx) const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 1)->GetBinError(binx);
  }

  /// get uncertainty on content of bin (2-D) - See TH1::GetBinError for details
  double MonitorElement::getBinError(int binx, int biny) const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 2)->GetBinError(binx, biny);
  }

  /// get uncertainty on content of bin (3-D) - See TH1::GetBinError for details
  double MonitorElement::getBinError(int binx, int biny, int binz) const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 3)->GetBinError(binx, biny, binz);
  }

  /// get # of entries
  double MonitorElement::getEntries() const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 1)->GetEntries();
  }

  /// get # of bin entries (for profiles)
  double MonitorElement::getBinEntries(int bin) const {
    auto access = this->access();
    if (kind() == Kind::TPROFILE)
      return static_cast<TProfile const *>(accessRootObject(access, __PRETTY_FUNCTION__, 1))->GetBinEntries(bin);
    else if (kind() == Kind::TPROFILE2D)
      return static_cast<TProfile2D const *>(accessRootObject(access, __PRETTY_FUNCTION__, 1))->GetBinEntries(bin);
    else {
      incompatible(__PRETTY_FUNCTION__);
      return 0;
    }
  }

  /// get x-, y- or z-axis title (axis=1, 2, 3 respectively)
  std::string MonitorElement::getAxisTitle(int axis /* = 1 */) const {
    auto access = this->access();
    return getAxis(access, __PRETTY_FUNCTION__, axis)->GetTitle();
  }

  /// get MonitorElement title
  std::string MonitorElement::getTitle() const {
    auto access = this->access();
    return accessRootObject(access, __PRETTY_FUNCTION__, 1)->GetTitle();
  }

  /*** setter methods (wrapper around ROOT methods) ****/
  //
  /// set content of bin (1-D)
  void MonitorElement::setBinContent(int binx, double content) {
    auto access = this->accessMut();
    accessRootObject(access, __PRETTY_FUNCTION__, 1)->SetBinContent(binx, content);
  }

  /// set content of bin (2-D)
  void MonitorElement::setBinContent(int binx, int biny, double content) {
    auto access = this->accessMut();
    accessRootObject(access, __PRETTY_FUNCTION__, 2)->SetBinContent(binx, biny, content);
  }

  /// set content of bin (3-D)
  void MonitorElement::setBinContent(int binx, int biny, int binz, double content) {
    auto access = this->accessMut();
    accessRootObject(access, __PRETTY_FUNCTION__, 3)->SetBinContent(binx, biny, binz, content);
  }

  /// set uncertainty on content of bin (1-D)
  void MonitorElement::setBinError(int binx, double error) {
    auto access = this->accessMut();
    accessRootObject(access, __PRETTY_FUNCTION__, 1)->SetBinError(binx, error);
  }

  /// set uncertainty on content of bin (2-D)
  void MonitorElement::setBinError(int binx, int biny, double error) {
    auto access = this->accessMut();
    accessRootObject(access, __PRETTY_FUNCTION__, 2)->SetBinError(binx, biny, error);
  }

  /// set uncertainty on content of bin (3-D)
  void MonitorElement::setBinError(int binx, int biny, int binz, double error) {
    auto access = this->accessMut();
    accessRootObject(access, __PRETTY_FUNCTION__, 3)->SetBinError(binx, biny, binz, error);
  }

  /// set # of bin entries (to be used for profiles)
  void MonitorElement::setBinEntries(int bin, double nentries) {
    auto access = this->accessMut();
    if (kind() == Kind::TPROFILE)
      static_cast<TProfile *>(accessRootObject(access, __PRETTY_FUNCTION__, 1))->SetBinEntries(bin, nentries);
    else if (kind() == Kind::TPROFILE2D)
      static_cast<TProfile2D *>(accessRootObject(access, __PRETTY_FUNCTION__, 1))->SetBinEntries(bin, nentries);
    else
      incompatible(__PRETTY_FUNCTION__);
  }

  /// set # of entries
  void MonitorElement::setEntries(double nentries) {
    auto access = this->accessMut();
    accessRootObject(access, __PRETTY_FUNCTION__, 1)->SetEntries(nentries);
  }

  /// set bin label for x, y or z axis (axis=1, 2, 3 respectively)
  void MonitorElement::setBinLabel(int bin, const std::string &label, int axis /* = 1 */) {
    bool fail = false;
    {
      auto access = this->accessMut();
      update();
      if (getAxis(access, __PRETTY_FUNCTION__, axis)->GetNbins() >= bin) {
        getAxis(access, __PRETTY_FUNCTION__, axis)->SetBinLabel(bin, label.c_str());
      } else {
        fail = true;
      }
    }
    // do this with the ME lock released to prevent a deadlock
    if (fail) {
      // this also takes the lock, make sure to release it before going to edm
      // (which might take more locks)
      auto name = getFullname();
      edm::LogWarning("MonitorElement") << "*** MonitorElement: WARNING:"
                                        << "setBinLabel: attempting to set label of non-existent bin number for ME: "
                                        << name << " \n";
    }
  }

  /// set x-, y- or z-axis range (axis=1, 2, 3 respectively)
  void MonitorElement::setAxisRange(double xmin, double xmax, int axis /* = 1 */) {
    auto access = this->accessMut();
    getAxis(access, __PRETTY_FUNCTION__, axis)->SetRangeUser(xmin, xmax);
  }

  /// set x-, y- or z-axis title (axis=1, 2, 3 respectively)
  void MonitorElement::setAxisTitle(const std::string &title, int axis /* = 1 */) {
    auto access = this->accessMut();
    getAxis(access, __PRETTY_FUNCTION__, axis)->SetTitle(title.c_str());
  }

  /// set x-, y-, or z-axis to display time values
  void MonitorElement::setAxisTimeDisplay(int value, int axis /* = 1 */) {
    auto access = this->accessMut();
    getAxis(access, __PRETTY_FUNCTION__, axis)->SetTimeDisplay(value);
  }

  /// set the format of the time values that are displayed on an axis
  void MonitorElement::setAxisTimeFormat(const char *format /* = "" */, int axis /* = 1 */) {
    auto access = this->accessMut();
    getAxis(access, __PRETTY_FUNCTION__, axis)->SetTimeFormat(format);
  }

  /// set (ie. change) histogram/profile title
  void MonitorElement::setTitle(const std::string &title) {
    auto access = this->accessMut();
    accessRootObject(access, __PRETTY_FUNCTION__, 1)->SetTitle(title.c_str());
  }

  TAxis *MonitorElement::getAxis(AccessMut const &access, const char *func, int axis) const {
    TH1 *h = accessRootObject(access, func, axis - 1);
    TAxis *a = nullptr;
    if (axis == 1)
      a = h->GetXaxis();
    else if (axis == 2)
      a = h->GetYaxis();
    else if (axis == 3)
      a = h->GetZaxis();

    if (!a)
      raiseDQMError("MonitorElement",
                    "No such axis %d in monitor element"
                    " '%s' of type '%s'",
                    axis,
                    data_.objname.c_str(),
                    typeid(*h).name());

    return a;
  }

  TAxis const *MonitorElement::getAxis(Access const &access, const char *func, int axis) const {
    TH1 const *h = accessRootObject(access, func, axis - 1);
    TAxis const *a = nullptr;
    if (axis == 1)
      a = h->GetXaxis();
    else if (axis == 2)
      a = h->GetYaxis();
    else if (axis == 3)
      a = h->GetZaxis();

    if (!a)
      raiseDQMError("MonitorElement",
                    "No such axis %d in monitor element"
                    " '%s' of type '%s'",
                    axis,
                    data_.objname.c_str(),
                    typeid(*h).name());

    return a;
  }

  void MonitorElement::setXTitle(std::string const &title) {
    auto access = this->accessMut();
    update();
    access.value.object_->SetXTitle(title.c_str());
  }
  void MonitorElement::setYTitle(std::string const &title) {
    auto access = this->accessMut();
    update();
    access.value.object_->SetYTitle(title.c_str());
  }

  void MonitorElement::enableSumw2() {
    auto access = this->accessMut();
    update();
    if (access.value.object_->GetSumw2() == nullptr) {
      access.value.object_->Sumw2();
    }
  }

  void MonitorElement::disableAlphanumeric() {
    auto access = this->accessMut();
    update();
    access.value.object_->GetXaxis()->SetNoAlphanumeric(false);
    access.value.object_->GetYaxis()->SetNoAlphanumeric(false);
  }

  void MonitorElement::setOption(const char *option) {
    auto access = this->accessMut();
    update();
    access.value.object_->SetOption(option);
  }
  double MonitorElement::getAxisMin(int axis) const {
    auto access = this->access();
    return getAxis(access, __PRETTY_FUNCTION__, axis)->GetXmin();
  }

  double MonitorElement::getAxisMax(int axis) const {
    auto access = this->access();
    return getAxis(access, __PRETTY_FUNCTION__, axis)->GetXmax();
  }

  void MonitorElement::setCanExtend(unsigned int value) {
    auto access = this->accessMut();
    access.value.object_->SetCanExtend(value);
  }

  void MonitorElement::setStatOverflows(unsigned int value) {
    auto access = this->accessMut();
    access.value.object_->StatOverflows(value);
  }

  int64_t MonitorElement::getIntValue() const {
    assert(kind() == Kind::INT);
    auto access = this->access();
    return access.value.scalar_.num;
  }
  double MonitorElement::getFloatValue() const {
    assert(kind() == Kind::REAL);
    auto access = this->access();
    return access.value.scalar_.real;
  }
  const std::string &MonitorElement::getStringValue() const {
    assert(kind() == Kind::STRING);
    auto access = this->access();
    return access.value.scalar_.str;
  }

  void MonitorElement::getQReport(bool create,
                                  const std::string &qtname,
                                  MonitorElementData::QReport *&qr,
                                  DQMNet::QValue *&qv) {
    auto access = this->accessMut();
    assert(access.value.qreports_.size() == data_.qreports.size());

    qr = nullptr;
    qv = nullptr;

    size_t pos = 0, end = access.value.qreports_.size();
    while (pos < end && data_.qreports[pos].qtname != qtname)
      ++pos;

    if (pos == end && !create)
      return;
    else if (pos == end) {
      DQMNet::QValue q;
      q.code = dqm::qstatus::DID_NOT_RUN;
      q.qtresult = 0;
      q.qtname = qtname;
      q.message = "NO_MESSAGE_ASSIGNED";
      q.algorithm = "UNKNOWN_ALGORITHM";
      access.value.qreports_.push_back(MonitorElementData::QReport(q));
      syncCoreObject(access);
    }

    qr = &access.value.qreports_[pos];
    qv = &(qr->getValue());
  }

  // -------------------------------------------------------------------
  // TODO: all of these are UNSAFE and have to be NON-const.
  TObject const *MonitorElement::getRootObject() const {
    auto access = this->access();
    return access.value.object_.get();
  }

  TH1 *MonitorElement::getTH1() {
    auto access = this->accessMut();
    return accessRootObject(access, __PRETTY_FUNCTION__, 0);
  }

  TH1F *MonitorElement::getTH1F() {
    auto access = this->accessMut();
    assert(kind() == Kind::TH1F);
    return static_cast<TH1F *>(accessRootObject(access, __PRETTY_FUNCTION__, 1));
  }

  TH1S *MonitorElement::getTH1S() {
    auto access = this->accessMut();
    assert(kind() == Kind::TH1S);
    return static_cast<TH1S *>(accessRootObject(access, __PRETTY_FUNCTION__, 1));
  }

  TH1D *MonitorElement::getTH1D() {
    auto access = this->accessMut();
    assert(kind() == Kind::TH1D);
    return static_cast<TH1D *>(accessRootObject(access, __PRETTY_FUNCTION__, 1));
  }

  TH2F *MonitorElement::getTH2F() {
    auto access = this->accessMut();
    assert(kind() == Kind::TH2F);
    return static_cast<TH2F *>(accessRootObject(access, __PRETTY_FUNCTION__, 2));
  }

  TH2S *MonitorElement::getTH2S() {
    auto access = this->accessMut();
    assert(kind() == Kind::TH2S);
    return static_cast<TH2S *>(accessRootObject(access, __PRETTY_FUNCTION__, 2));
  }

  TH2D *MonitorElement::getTH2D() {
    auto access = this->accessMut();
    assert(kind() == Kind::TH2D);
    return static_cast<TH2D *>(accessRootObject(access, __PRETTY_FUNCTION__, 2));
  }

  TH3F *MonitorElement::getTH3F() {
    auto access = this->accessMut();
    assert(kind() == Kind::TH3F);
    return static_cast<TH3F *>(accessRootObject(access, __PRETTY_FUNCTION__, 3));
  }

  TProfile *MonitorElement::getTProfile() {
    auto access = this->accessMut();
    assert(kind() == Kind::TPROFILE);
    return static_cast<TProfile *>(accessRootObject(access, __PRETTY_FUNCTION__, 1));
  }

  TProfile2D *MonitorElement::getTProfile2D() {
    auto access = this->accessMut();
    assert(kind() == Kind::TPROFILE2D);
    return static_cast<TProfile2D *>(accessRootObject(access, __PRETTY_FUNCTION__, 2));
  }

}  // namespace dqm::impl

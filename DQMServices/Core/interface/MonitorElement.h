#ifndef DQMSERVICES_CORE_MONITOR_ELEMENT_H
# define DQMSERVICES_CORE_MONITOR_ELEMENT_H

# include "DQMServices/Core/interface/DQMNet.h"
# include "DQMServices/Core/interface/QReport.h"
# include "TF1.h"
# include "TH1F.h"
# include "TH1S.h"
# include "TH1D.h"
# include "TH2F.h"
# include "TH2S.h"
# include "TH2D.h"
# include "TH3F.h"
# include "TProfile.h"
# include "TProfile2D.h"
# include "TObjString.h"
# include "TAxis.h"
# include <sys/time.h>
# include <string>
# include <set>
# include <map>
# include <sstream>
# include <iomanip>
# include <cassert>
# include <stdint.h>

# ifndef DQM_ROOT_METHODS
#  define DQM_ROOT_METHODS 1
# endif

class QCriterion;

// tag for a special constructor, see below
struct MonitorElementNoCloneTag {};

/** The base class for all MonitorElements (ME) */
class MonitorElement
{
  friend class DQMStore;
  friend class DQMService;
public:
  struct Scalar
  {
    int64_t             num;
    double              real;
    std::string         str;
  };

  enum Kind
  {
    DQM_KIND_INVALID    = DQMNet::DQM_PROP_TYPE_INVALID,
    DQM_KIND_INT        = DQMNet::DQM_PROP_TYPE_INT,
    DQM_KIND_REAL       = DQMNet::DQM_PROP_TYPE_REAL,
    DQM_KIND_STRING     = DQMNet::DQM_PROP_TYPE_STRING,
    DQM_KIND_TH1F       = DQMNet::DQM_PROP_TYPE_TH1F,
    DQM_KIND_TH1S       = DQMNet::DQM_PROP_TYPE_TH1S,
    DQM_KIND_TH1D       = DQMNet::DQM_PROP_TYPE_TH1D,
    DQM_KIND_TH2F       = DQMNet::DQM_PROP_TYPE_TH2F,
    DQM_KIND_TH2S       = DQMNet::DQM_PROP_TYPE_TH2S,
    DQM_KIND_TH2D       = DQMNet::DQM_PROP_TYPE_TH2D,
    DQM_KIND_TH3F       = DQMNet::DQM_PROP_TYPE_TH3F,
    DQM_KIND_TPROFILE   = DQMNet::DQM_PROP_TYPE_TPROF,
    DQM_KIND_TPROFILE2D = DQMNet::DQM_PROP_TYPE_TPROF2D
  };

  typedef std::vector<QReport>::const_iterator QReportIterator;

private:
  DQMNet::CoreObject    data_;       //< Core object information.
  Scalar                scalar_;     //< Current scalar value.
  TH1                   *object_;    //< Current ROOT object value.
  TH1                   *reference_; //< Current ROOT reference object.
  TH1                   *refvalue_;  //< Soft reference if any.
  std::vector<QReport>  qreports_;   //< QReports associated to this object.

  MonitorElement *initialise(Kind kind);
  MonitorElement *initialise(Kind kind, TH1 *rootobj);
  MonitorElement *initialise(Kind kind, const std::string &value);
  void globalize() {
    data_.streamId = 0;
    data_.moduleId = 0;
  }
  void setLumi(uint32_t ls) {data_.lumi = ls;}

public:
  MonitorElement(void);
  MonitorElement(const std::string *path,
                 const std::string &name,
                 uint32_t run = 0,
                 uint32_t streamId = 0,
                 uint32_t moduleId = 0);
  MonitorElement(const MonitorElement &, MonitorElementNoCloneTag);
  MonitorElement(const MonitorElement &);
  MonitorElement(MonitorElement &&);
  MonitorElement &operator=(const MonitorElement &) = delete;
  MonitorElement &operator=(MonitorElement &&) = delete;
  ~MonitorElement(void);

  /// Compare monitor elements, for ordering in sets.
  bool operator<(const MonitorElement &x) const
    {
      return DQMNet::setOrder(data_, x.data_);
    }

  /// Check the consistency of the axis labels
  static bool CheckBinLabels(const TAxis* a1, const TAxis * a2);

  /// Get the type of the monitor element.
  Kind kind(void) const
    { return Kind(data_.flags & DQMNet::DQM_PROP_TYPE_MASK); }

  /// Get the object flags.
  uint32_t flags(void) const
    { return data_.flags; }

  /// get name of ME
  const std::string &getName(void) const
    { return data_.objname; }

  /// get pathname of parent folder
  const std::string &getPathname(void) const
    { return *data_.dirname; }

  /// get full name of ME including Pathname
  const std::string getFullname(void) const
    {
      std::string path;
      path.reserve(data_.dirname->size() + data_.objname.size() + 2);
      path += *data_.dirname;
      if (! data_.dirname->empty())
        path += '/';
      path += data_.objname;
      return path;
    }

  /// true if ME was updated in last monitoring cycle
  bool wasUpdated(void) const
    { return data_.flags & DQMNet::DQM_PROP_NEW; }

  /// Mark the object updated.
  void update(void)
    { data_.flags |= DQMNet::DQM_PROP_NEW; }

  /// specify whether ME should be reset at end of monitoring cycle (default:false);
  /// (typically called by Sources that control the original ME)
  void setResetMe(bool /* flag */)
    { data_.flags |= DQMNet::DQM_PROP_RESET; }

  /// true if ME is meant to be stored for each luminosity section
  bool getLumiFlag(void) const
    { return data_.flags & DQMNet::DQM_PROP_LUMI; }

  /// this ME is meant to be stored for each luminosity section
  void setLumiFlag(void)
    { data_.flags |= DQMNet::DQM_PROP_LUMI; }

  /// this ME is meant to be an efficiency plot that must not be
  /// normalized when drawn in the DQM GUI.
  void setEfficiencyFlag(void)
    { data_.flags |= DQMNet::DQM_PROP_EFFICIENCY_PLOT; }

  // A static assert to check that T actually fits in
  // int64_t.
  template <typename T>
  struct fits_in_int64_t
  {
    int checkArray[sizeof(int64_t) - sizeof(T) + 1];
  };

  void Fill(long long x) { fits_in_int64_t<long long>(); doFill(static_cast<int64_t>(x)); }
  void Fill(unsigned long long x) { fits_in_int64_t<unsigned long long>(); doFill(static_cast<int64_t>(x)); }
  void Fill(unsigned long x) { fits_in_int64_t<unsigned long>(); doFill(static_cast<int64_t>(x)); }
  void Fill(long x) { fits_in_int64_t<long>(); doFill(static_cast<int64_t>(x)); }
  void Fill(unsigned int x) { fits_in_int64_t<unsigned int>(); doFill(static_cast<int64_t>(x)); }
  void Fill(int x) { fits_in_int64_t<int>(); doFill(static_cast<int64_t>(x)); }
  void Fill(short x) { fits_in_int64_t<short>(); doFill(static_cast<int64_t>(x)); }
  void Fill(unsigned short x) { fits_in_int64_t<unsigned short>(); doFill(static_cast<int64_t>(x)); }
  void Fill(char x) { fits_in_int64_t<char>(); doFill(static_cast<int64_t>(x)); }
  void Fill(unsigned char x) { fits_in_int64_t<unsigned char>(); doFill(static_cast<int64_t>(x)); }

  void Fill(float x)    { Fill(static_cast<double>(x)); }
  void Fill(double x);
  void Fill(std::string &value);

  void Fill(double x, double yw);
  void Fill(double x, double y, double zw);
  void Fill(double x, double y, double z, double w);
  void ShiftFillLast(double y, double ye = 0., int32_t xscale = 1);
  void Reset(void);

  std::string valueString(void) const;
  std::string tagString(void) const;
  std::string tagLabelString(void) const;
  std::string effLabelString(void) const;
  std::string qualityTagString(const DQMNet::QValue &qv) const;
  void packScalarData(std::string &into, const char *prefix) const;
  void packQualityData(std::string &into) const;

  /// true if at least of one of the quality tests returned an error
  bool hasError(void) const
    { return data_.flags & DQMNet::DQM_PROP_REPORT_ERROR; }

  /// true if at least of one of the quality tests returned a warning
  bool hasWarning(void) const
    { return data_.flags & DQMNet::DQM_PROP_REPORT_WARN; }

  /// true if at least of one of the tests returned some other (non-ok) status
  bool hasOtherReport(void) const
    { return data_.flags & DQMNet::DQM_PROP_REPORT_OTHER; }

    /// true if the plot has been marked as an efficiency plot, which
    /// will not be normalized when rendered within the DQM GUI.
  bool isEfficiency(void) const
    { return data_.flags & DQMNet::DQM_PROP_EFFICIENCY_PLOT; }

  /// get QReport corresponding to <qtname> (null pointer if QReport does not exist)
  const QReport *getQReport(const std::string &qtname) const;

  /// get map of QReports
  std::vector<QReport *> getQReports(void) const;

  /// get warnings from last set of quality tests
  std::vector<QReport *> getQWarnings(void) const;

  /// get errors from last set of quality tests
  std::vector<QReport *> getQErrors(void) const;

  /// get "other" (i.e. non-error, non-warning, non-"ok") QReports
  /// from last set of quality tests
  std::vector<QReport *> getQOthers(void) const;

  /// run all quality tests
  void runQTests(void);

private:
  void doFill(int64_t x);
  void incompatible(const char *func) const;
  TH1 *accessRootObject(const char *func, int reqdim) const;

public:
#if DQM_ROOT_METHODS
  double getMean(int axis = 1) const;
  double getMeanError(int axis = 1) const;
  double getRMS(int axis = 1) const;
  double getRMSError(int axis = 1) const;
  int getNbinsX(void) const;
  int getNbinsY(void) const;
  int getNbinsZ(void) const;
  double getBinContent(int binx) const;
  double getBinContent(int binx, int biny) const;
  double getBinContent(int binx, int biny, int binz) const;
  double getBinError(int binx) const;
  double getBinError(int binx, int biny) const;
  double getBinError(int binx, int biny, int binz) const;
  double getEntries(void) const;
  double getBinEntries(int bin) const;

private:
  double getYmin(void) const;
  double getYmax(void) const;

public:
  std::string getAxisTitle(int axis = 1) const;
  std::string getTitle(void) const;
  void setBinContent(int binx, double content);
  void setBinContent(int binx, int biny, double content);
  void setBinContent(int binx, int biny, int binz, double content);
  void setBinError(int binx, double error);
  void setBinError(int binx, int biny, double error);
  void setBinError(int binx, int biny, int binz, double error);
  void setBinEntries(int bin, double nentries);
  void setEntries(double nentries);
  void setBinLabel(int bin, const std::string &label, int axis = 1);
  void setAxisRange(double xmin, double xmax, int axis = 1);
  void setAxisTitle(const std::string &title, int axis = 1);
  void setAxisTimeDisplay(int value, int axis = 1);
  void setAxisTimeFormat(const char *format = "", int axis = 1);

private:
  void setAxisTimeOffset(double toffset, const char *option="local", int axis = 1);

public:
  void setTitle(const std::string &title);
#endif // DQM_ROOT_METHODS

private:
  /// whether soft-reset is enabled; default is false
  bool isSoftResetEnabled(void) const
    { return refvalue_ != 0; }

  /// whether ME contents should be accumulated over multiple monitoring periods; default: false
  bool isAccumulateEnabled(void) const
    { return data_.flags & DQMNet::DQM_PROP_ACCUMULATE; }

  /// true if ME is marked for deletion
  bool markedToDelete(void) const
    { return data_.flags & DQMNet::DQM_PROP_MARKTODELETE; }

  /// Mark the object for deletion.
  /// NB: make sure that the following method is not called simultaneously for the same ME
  void markToDelete(void)
    { data_.flags |= DQMNet::DQM_PROP_MARKTODELETE; }

private:
  /// reset "was updated" flag
  void resetUpdate(void)
    { data_.flags &= ~DQMNet::DQM_PROP_NEW; }

  /// true if ME should be reset at end of monitoring cycle
  bool resetMe(void) const
    { return data_.flags & DQMNet::DQM_PROP_RESET; }

  /// if true, will accumulate ME contents (over many periods)
  /// until method is called with flag = false again
  void setAccumulate(bool /* flag */)
    { data_.flags |= DQMNet::DQM_PROP_ACCUMULATE; }

  TAxis *getAxis(const char *func, int axis) const;

  // ------------ Operations for MEs that are normally never reset ---------
public:
  void softReset(void);
private:
  void disableSoftReset(void);
  void addProfiles(TProfile *h1, TProfile *h2, TProfile *sum, float c1, float c2);
  void addProfiles(TProfile2D *h1, TProfile2D *h2, TProfile2D *sum, float c1, float c2);
  void copyFunctions(TH1 *from, TH1 *to);
  void copyFrom(TH1 *from);


  // --- Operations on MEs that are normally reset at end of monitoring cycle ---
  void getQReport(bool create, const std::string &qtname, QReport *&qr, DQMNet::QValue *&qv);
  void addQReport(const DQMNet::QValue &desc, QCriterion *qc);
  void addQReport(QCriterion *qc);
  void updateQReportStats(void);

public:
  TObject *getRootObject(void) const;
  TH1 *getTH1(void) const;
  TH1F *getTH1F(void) const;
  TH1S *getTH1S(void) const;
  TH1D *getTH1D(void) const;
  TH2F *getTH2F(void) const;
  TH2S *getTH2S(void) const;
  TH2D *getTH2D(void) const;
  TH3F *getTH3F(void) const;
  TProfile *getTProfile(void) const;
  TProfile2D *getTProfile2D(void) const;

  TObject *getRefRootObject(void) const;
  TH1 *getRefTH1(void) const;
  TH1F *getRefTH1F(void) const;
  TH1S *getRefTH1S(void) const;
  TH1D *getRefTH1D(void) const;
  TH2F *getRefTH2F(void) const;
  TH2S *getRefTH2S(void) const;
  TH2D *getRefTH2D(void) const;
  TH3F *getRefTH3F(void) const;
  TProfile *getRefTProfile(void) const;
  TProfile2D *getRefTProfile2D(void) const;

  int64_t getIntValue(void) const
    {
      assert(kind() == DQM_KIND_INT);
      return scalar_.num;
    }

  double getFloatValue(void) const
    {
      assert(kind() == DQM_KIND_REAL);
      return scalar_.real;
    }

  const std::string &getStringValue(void) const
    {
      assert(kind() == DQM_KIND_STRING);
      return scalar_.str;
    }

  DQMNet::TagList getTags(void) const // DEPRECATED
    {
      DQMNet::TagList tags;
      if (data_.flags & DQMNet::DQM_PROP_TAGGED)
        tags.push_back(data_.tag);
      return tags;
    }

  const uint32_t getTag(void) const
    { return data_.tag; }

  const uint32_t run(void) const {return data_.run;}
  const uint32_t lumi(void) const {return data_.lumi;}
  const uint32_t streamId(void) const {return data_.streamId;}
  const uint32_t moduleId(void) const {return data_.moduleId;}
};

#endif // DQMSERVICES_CORE_MONITOR_ELEMENT_H

#ifndef DQMSERVICES_CORE_MONITOR_ELEMENT_H
# define DQMSERVICES_CORE_MONITOR_ELEMENT_H

# include "DQMServices/Core/interface/DQMNet.h"
# include "DQMServices/Core/interface/QReport.h"
# include "FWCore/Utilities/interface/EDMException.h"
# include "TF1.h"
# include "TH1F.h"
# include "TH2F.h"
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

# ifndef DQM_ROOT_METHODS
#  define DQM_ROOT_METHODS 1
# endif

class QCriterion;

/** The base class for all MonitorElements (ME) */
class MonitorElement
{
public:
  struct Value
  {
    int			num;
    double		real;
    std::string		str;
    TObject		*tobj;
  };

  enum Kind
  {
    DQM_KIND_INVALID,
    DQM_KIND_INT,
    DQM_KIND_REAL,
    DQM_KIND_STRING,
    DQM_KIND_TH1F,
    DQM_KIND_TH2F,
    DQM_KIND_TH3F,
    DQM_KIND_TPROFILE,
    DQM_KIND_TPROFILE2D
  };

  typedef std::vector<QReport>::const_iterator QReportIterator;

  static const uint32_t DQM_FLAG_RESET      = 0x01000000;
  static const uint32_t DQM_FLAG_ACCUMULATE = 0x02000000;

private:
  Kind			kind_;
  DQMNet::CoreObject	data_;
  std::string		name_;
  std::string		path_;
  std::vector<QReport>	qreports_;
  size_t		nqerror_;    //< errors from last set of quality tests
  size_t		nqwarning_;  //< warnings from last set of quality tests
  size_t		nqother_;    //< other quality reports (not an error, warning or ok)

  Value			curvalue_;
  TH1			*refvalue_;
  
  MonitorElement *initialise(Kind kind, const std::string &path);
  MonitorElement *initialise(Kind kind, const std::string &path, TH1 *rootobj);
  MonitorElement *initialise(Kind kind, const std::string &path, const std::string &value);

public:
  MonitorElement(void);
  ~MonitorElement(void);

  /// Get the type of the monitor element.
  Kind kind(void) const
    { return kind_; }

  /// Get the object flags.
  uint32_t flags(void) const
    { return data_.flags; }

  /// get name of ME
  const std::string &getName(void) const
    { return name_; }

  /// get pathname of parent folder
  const std::string &getPathname(void) const
    { return path_; }

  /// get full name of ME including Pathname
  const std::string &getFullname(void) const
    { return data_.name; }

  /// true if ME was updated in last monitoring cycle
  bool wasUpdated(void) const
    { return data_.flags & DQMNet::DQM_FLAG_NEW; }

  /// Mark the object updated.
  void update(void)
    { data_.flags |= DQMNet::DQM_FLAG_NEW; }

  /// specify whether ME should be reset at end of monitoring cycle (default:false);
  /// (typically called by Sources that control the original ME)
  void setResetMe(bool flag)
    { data_.flags |= DQM_FLAG_RESET; }

  void Fill(float x);
  void Fill(float x, float yw);
  void Fill(float x, float y, float zw);
  void Fill(float x, float y, float z, float w);
  void Reset(void);

  std::string valueString(void) const;
  std::string tagString(void) const;
  std::string qualityTagString(const DQMNet::QValue &qv) const;

  /// true if at least of one of the quality tests returned an error
  bool hasError(void) const
    { return nqerror_ > 0; }

  /// true if at least of one of the quality tests returned a warning
  bool hasWarning(void) const
    { return nqwarning_ > 0; }

  /// true if at least of one of the tests returned some other (non-ok) status
  bool hasOtherReport(void) const
    { return nqother_ > 0; }

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
  void incompatible(const char *func) const;
  TH1 *accessRootObject(const char *func, int reqdim) const;

public:
#if DQM_ROOT_METHODS
  float getMean(int axis = 1) const;
  float getMeanError(int axis = 1) const;
  float getRMS(int axis = 1) const;
  float getRMSError(int axis = 1) const;
  int getNbinsX(void) const;
  int getNbinsY(void) const;
  int getNbinsZ(void) const;
  float getBinContent(int binx) const;
  float getBinContent(int binx, int biny) const;
  float getBinContent(int binx, int biny, int binz) const;
  float getBinError(int binx) const;
  float getBinError(int binx, int biny) const;
  float getBinError(int binx, int biny, int binz) const;
  float getEntries(void) const;
  float getBinEntries(int bin) const;

private:
  float getYmin(void) const;
  float getYmax(void) const;

public:
  std::string getAxisTitle(int axis = 1) const;
  std::string getTitle(void) const;
  void setBinContent(int binx, float content);
  void setBinContent(int binx, int biny, float content);
  void setBinContent(int binx, int biny, int binz, float content);
  void setBinError(int binx, float error);
  void setBinError(int binx, int biny, float error);
  void setBinError(int binx, int biny, int binz, float error);
  void setBinEntries(int bin, float nentries);
  void setEntries(float nentries);
  void setBinLabel(int bin, const std::string &label, int axis = 1);
  void setAxisRange(float xmin, float xmax, int axis = 1);
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
    { return data_.flags & DQM_FLAG_ACCUMULATE; }

private:
  /// reset "was updated" flag
  void resetUpdate(void)
    { data_.flags &= ~DQMNet::DQM_FLAG_NEW; }

  /// true if ME should be reset at end of monitoring cycle
  bool resetMe(void) const
    { return data_.flags & DQM_FLAG_RESET; }

  /// if true, will accumulate ME contents (over many periods)
  /// until method is called with flag = false again
  void setAccumulate(bool flag)
    { data_.flags |= DQM_FLAG_ACCUMULATE; }

  TAxis *getAxis(const char *func, int axis) const;

  // ------------ Operations for MEs that are normally never reset ---------
  void softReset(void);
  void disableSoftReset(void);
  void addProfiles(TProfile *h1, TProfile *h2, TProfile *sum, float c1, float c2);
  void addProfiles(TProfile2D *h1, TProfile2D *h2, TProfile2D *sum, float c1, float c2);
  void copyFunctions(TH1 *from, TH1 *to);
  void copyFrom(TH1 *from);
    

  // --- Operations on MEs that are normally reset at end of monitoring cycle ---
  void getQReport(bool create, const std::string &qtname, QReport *&qr, DQMNet::QValue *&qv);
  void addQReport(const DQMNet::QValue &desc, QCriterion *qc);
  void addQReport(QCriterion *qc);
  void clear(void);

  void setReference(MonitorElement *other);

public:
  TObject *getRootObject(void) const;
  TH1 *getTH1(void) const;
  TH1F *getTH1F(void) const;
  TH2F *getTH2F(void) const;
  TH3F *getTH3F(void) const;
  TProfile *getTProfile(void) const;
  TProfile2D *getTProfile2D(void) const;

  TObject *getRefRootObject(void) const;
  TH1 *getRefTH1(void) const;
  TH1F *getRefTH1F(void) const;
  TH2F *getRefTH2F(void) const;
  TH3F *getRefTH3F(void) const;
  TProfile *getRefTProfile(void) const;
  TProfile2D *getRefTProfile2D(void) const;

  const int &getIntValue(void) const
    {
      assert(kind_ == DQM_KIND_INT);
      return curvalue_.num;
    }

  const double &getFloatValue(void) const
    {
      assert(kind_ == DQM_KIND_REAL);
      return curvalue_.real;
    }

  const std::string &getStringValue(void) const
    {
      assert(kind_ == DQM_KIND_STRING);
      return curvalue_.str;
    }

  const DQMNet::TagList &getTags(void) const
    { return data_.tags; }

private:
  friend class DQMStore;
  friend class DQMService;

  //MonitorElement(const MonitorElement &);
  //MonitorElement &operator=(const MonitorElement &);
};

#endif // DQMSERVICES_CORE_MONITOR_ELEMENT_H

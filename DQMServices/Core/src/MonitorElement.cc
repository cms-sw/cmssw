#define DQM_ROOT_METHODS 1
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/QTest.h"
#include "DQMServices/Core/src/DQMError.h"
#include "TClass.h"
#include "TMath.h"
#include "TList.h"
#include <iostream>
#include <cassert>

static TH1 *
checkRootObject(const std::string &name, TObject *tobj, const char *func, int reqdim)
{
  if (! tobj)
    raiseDQMError("MonitorElement", "Method '%s' cannot be invoked on monitor"
		  " element '%s' because it is not a ROOT object.",
		  func, name.c_str());

  TH1 *h = dynamic_cast<TH1 *>(tobj);
  if (! h)
    raiseDQMError("MonitorElement", "Method '%s' cannot be invoked on monitor"
		  " element '%s' because it is not a ROOT histogram; it is of"
		  " type '%s'", func, name.c_str(), typeid(*tobj).name());

  int ndim = h->GetDimension();
  if (reqdim < 0 || reqdim > ndim)
    raiseDQMError("MonitorElement", "Method '%s' cannot be invoked on monitor"
		  " element '%s' because it requires %d dimensions; this"
		  " object of type '%s' has %d dimensions",
		  func, name.c_str(), reqdim, typeid(*h).name(), ndim);

  return h;
}

MonitorElement *
MonitorElement::initialise(Kind kind, const std::string &path)
{
  const char *slash = strrchr(path.c_str(), '/');
  data_.name = path;
  kind_ = kind;
  if (slash)
  {
    name_ = slash+1;
    path_ = std::string(path.c_str(), slash);
  }
  else
  {
    name_ = path;
    path_.clear();
  }

  switch (kind)
  {
  case DQM_KIND_INT:
  case DQM_KIND_REAL:
  case DQM_KIND_STRING:
    data_.object = new TObjString;
    static_cast<TObjString *>(data_.object)
      ->SetString(tagString().c_str());
    data_.flags |= DQMNet::DQM_FLAG_SCALAR;
    break;

  case DQM_KIND_TH1F:
  case DQM_KIND_TH1S:
  case DQM_KIND_TH2F:
  case DQM_KIND_TH2S:
  case DQM_KIND_TH3F:
  case DQM_KIND_TPROFILE:
  case DQM_KIND_TPROFILE2D:
    break;

  default:
    raiseDQMError("MonitorElement", "cannot initialise monitor element '%s'"
		  " to invalid type %d", path.c_str(), (int) kind);
  }

  return this;
}

MonitorElement *
MonitorElement::initialise(Kind kind, const std::string &path, TH1 *rootobj)
{
  initialise(kind, path);
  switch (kind)
  {
  case DQM_KIND_TH1F:
    assert(dynamic_cast<TH1F *>(rootobj));
    curvalue_.tobj = data_.object = rootobj;
    break;

  case DQM_KIND_TH1S:
    assert(dynamic_cast<TH1S *>(rootobj));
    curvalue_.tobj = data_.object = rootobj;
    break;

  case DQM_KIND_TH2F:
    assert(dynamic_cast<TH2F *>(rootobj));
    curvalue_.tobj = data_.object = rootobj;
    break;

  case DQM_KIND_TH2S:
    assert(dynamic_cast<TH2S *>(rootobj));
    curvalue_.tobj = data_.object = rootobj;
    break;

  case DQM_KIND_TH3F:
    assert(dynamic_cast<TH3F *>(rootobj));
    curvalue_.tobj = data_.object = rootobj;
    break;

  case DQM_KIND_TPROFILE:
    assert(dynamic_cast<TProfile *>(rootobj));
    curvalue_.tobj = data_.object = rootobj;
    break;

  case DQM_KIND_TPROFILE2D:
    assert(dynamic_cast<TProfile2D *>(rootobj));
    curvalue_.tobj = data_.object = rootobj;
    break;

  default:
    raiseDQMError("MonitorElement", "cannot initialise monitor element '%s'"
		  " as a root object with type %d", path.c_str(), (int) kind);
  }

  return this;
}

MonitorElement *
MonitorElement::initialise(Kind kind, const std::string &path, const std::string &value)
{
  initialise(kind, path);
  if (kind ==  DQM_KIND_STRING)
  {
    curvalue_.str = value;
    static_cast<TObjString *>(data_.object)
      ->SetString(tagString().c_str());
  }
  else
    raiseDQMError("MonitorElement", "cannot initialise monitor element '%s'"
		  " as a string with type %d", path.c_str(), (int) kind);

  return this;
}

MonitorElement::MonitorElement(void)
  : kind_ (DQM_KIND_INVALID),
    nqerror_ (0),
    nqwarning_ (0),
    nqother_ (0),
    refvalue_ (0)
{
  data_.version = 0;
  data_.object = 0;
  data_.reference = 0;
  data_.flags = DQMNet::DQM_FLAG_NEW;

  curvalue_.num = 0;
  curvalue_.real = 0;
  curvalue_.tobj = 0;
}

MonitorElement::~MonitorElement(void)
{
  delete data_.object;
  delete refvalue_;
}

/// "Fill" ME methods:
/// can be used with 1D histograms or scalars
void
MonitorElement::Fill(float x)
{
  update();
  if (kind_ == DQM_KIND_INT)
  {
    curvalue_.num = int(x);
    static_cast<TObjString *>(data_.object)
      ->SetString(tagString().c_str());
  }
  else if (kind_ == DQM_KIND_REAL)
  {
    curvalue_.real = x;
    static_cast<TObjString *>(data_.object)
      ->SetString(tagString().c_str());
  }
  else if (kind_ == DQM_KIND_TH1F)
    accessRootObject(__PRETTY_FUNCTION__, 1)
      ->Fill(x, 1);
  else if (kind_ == DQM_KIND_TH1S)
    accessRootObject(__PRETTY_FUNCTION__, 1)
      ->Fill(x, 1);
  else
    incompatible(__PRETTY_FUNCTION__);
}

/// can be used with 2D (x,y) or 1D (x, w) histograms
void
MonitorElement::Fill(float x, float yw)
{
  update();
  if (kind_ == DQM_KIND_TH1F)
    accessRootObject(__PRETTY_FUNCTION__, 1)
      ->Fill(x, yw);
  else if (kind_ == DQM_KIND_TH1S)
    accessRootObject(__PRETTY_FUNCTION__, 1)
      ->Fill(x, yw);
  else if (kind_ == DQM_KIND_TH2F)
    static_cast<TH2F *>(accessRootObject(__PRETTY_FUNCTION__, 2))
      ->Fill(x, yw, 1);
  else if (kind_ == DQM_KIND_TH2S)
    static_cast<TH2S *>(accessRootObject(__PRETTY_FUNCTION__, 2))
      ->Fill(x, yw, 1);
  else if (kind_ == DQM_KIND_TPROFILE)
    static_cast<TProfile *>(accessRootObject(__PRETTY_FUNCTION__, 1))
      ->Fill(x, yw, 1);
  else
    incompatible(__PRETTY_FUNCTION__);
}

/// shift bin to the left and fill last bin with new entry
/// 1st argument is y value, 2nd argument is y error (default 0)
/// can be used with 1D or profile histograms only
void
MonitorElement::ShiftFillLast(float y, float ye, int xscale)
{
  update();
  if (kind_ == DQM_KIND_TH1F || kind_ == DQM_KIND_TH1S ) 
  {
    int nbins = getNbinsX();
    int entries = (int)getEntries();
    // first fill bins from left to right
    int index = entries + 1 ;
    int xlow = 2 ; int xup = nbins ;
    // if more entries than bins then start shifting
    if ( entries >= nbins ) 
    {
      index = nbins;
      xlow = entries - nbins + 2 ; xup = entries ; 
      // average first bin
      float y1 = getBinContent(1);
      float y2 = getBinContent(2);
      float y1err = getBinError(1);
      float y2err = getBinError(2);
      float N = entries - nbins + 1.;
      if ( ye == 0. || y1err == 0. || y2err == 0.) 
      {
        // for errors zero calculate unweighted mean and its error
	float sum = N*y1 + y2;
        y1 = sum/(N+1.) ;
	// FIXME check if correct
        y1err = sqrt((N+1.)*(N*y1*y1 + y2*y2) - sum*sum)/(N+1.);  
      }
      else 
      {
        // for errors non-zero calculate weighted mean and its error
        float denom = (1./y1err + 1./y2err);
        float mean = (y1/y1err + y2/y2err)/denom;
	// FIXME check if correct
	y1err = sqrt(((y1-mean)*(y1-mean)/y1err +
                      (y2-mean)*(y2-mean)/y2err)/denom/2.);
	y1 = mean; // set y1 to mean for filling below
      }
      setBinContent(1,y1);
      setBinError(1,y1err);
      // shift remaining bins to the left
      for ( int i = 3; i <= nbins ; i++) 
      {
        setBinContent(i-1,getBinContent(i));
        setBinError(i-1,getBinError(i));
      }
    }
    // fill last bin with new values
    setBinContent(index,y);
    setBinError(index,ye); 
    // set entries
    setEntries(entries+1);
    // set axis labels and reset drawing option
    char buffer [10];
    sprintf (buffer, "%d", xlow*xscale); 
    std::string a(buffer); setBinLabel(2,a);
    sprintf (buffer, "%d", xup*xscale); 
    std::string b(buffer); setBinLabel(nbins,b);
    setBinLabel(1,"av.");
    static_cast<TH1*>(getRootObject())->SetOption("HIST");
  }
  else
    incompatible(__PRETTY_FUNCTION__);
}
/// can be used with 3D (x, y, z) or 2D (x, y, w) histograms
void
MonitorElement::Fill(float x, float y, float zw)
{
  update();
  if (kind_ == DQM_KIND_TH2F)
    static_cast<TH2F *>(accessRootObject(__PRETTY_FUNCTION__, 2))
      ->Fill(x, y, zw);
  else if (kind_ == DQM_KIND_TH2S)
    static_cast<TH2S *>(accessRootObject(__PRETTY_FUNCTION__, 2))
      ->Fill(x, y, zw);
  else if (kind_ == DQM_KIND_TH3F)
    static_cast<TH3F *>(accessRootObject(__PRETTY_FUNCTION__, 2))
      ->Fill(x, y, zw, 1);
  else if (kind_ == DQM_KIND_TPROFILE)
    static_cast<TProfile *>(accessRootObject(__PRETTY_FUNCTION__, 2))
      ->Fill(x, y, zw);
  else if (kind_ == DQM_KIND_TPROFILE2D)
    static_cast<TProfile2D *>(accessRootObject(__PRETTY_FUNCTION__, 2))
      ->Fill(x, y, zw, 1);
  else
    incompatible(__PRETTY_FUNCTION__);
}

/// can be used with 3D (x, y, z, w) histograms
void
MonitorElement::Fill(float x, float y, float z, float w)
{
  update();
  if (kind_ == DQM_KIND_TH3F)
    static_cast<TH3F *>(accessRootObject(__PRETTY_FUNCTION__, 2))
      ->Fill(x, y, z, w);
  else if (kind_ == DQM_KIND_TPROFILE2D)
    static_cast<TProfile2D *>(accessRootObject(__PRETTY_FUNCTION__, 2))
      ->Fill(x, y, z, w);
  else
    incompatible(__PRETTY_FUNCTION__);
}

/// reset ME (ie. contents, errors, etc)
void
MonitorElement::Reset(void)
{
  update();
  if (kind_ == DQM_KIND_INT)
    curvalue_.num = 0;
  else if (kind_ == DQM_KIND_REAL)
    curvalue_.real = 0;
  else if (kind_ == DQM_KIND_STRING)
    curvalue_.str.clear();
  else
    return accessRootObject(__PRETTY_FUNCTION__, 1)
      ->Reset();
}

/// returns value of ME in string format (eg. "f = 3.14151926" for float numbers);
/// relevant only for scalar or string MEs
std::string
MonitorElement::valueString(void) const
{
  std::ostringstream buf;
  if (kind_ == DQM_KIND_INT)
    buf << "i=" << curvalue_.num;
  else if (kind_ == DQM_KIND_REAL)
    buf << "f=" << std::setprecision(16) << curvalue_.real;
  else if (kind_ == DQM_KIND_STRING)
    buf << "s=" << curvalue_.str;
  else
    incompatible(__PRETTY_FUNCTION__);
  return buf.str();
}

/// return tagged value of ME in string format 
/// (eg. <name>f=3.14151926</name> for float numbers);
/// relevant only for sending scalar or string MEs over TSocket
std::string
MonitorElement::tagString(void) const
{ return "<" + getName() + ">" + valueString() + "</" + getName() + ">"; }

std::string
MonitorElement::qualityTagString(const DQMNet::QValue &qv) const
{
  std::string title;
  title.reserve(name_.size() + qv.qtname.size() + 2);
  title += name_;
  title += '.';
  title += qv.qtname;

  std::ostringstream retval;
  retval << "<" << title << ">"
	 << "qr=st." << qv.code << "." << qv.message
	 << "</" << title << ">";
  return retval.str();
}

const QReport *
MonitorElement::getQReport(const std::string &qtname) const
{
  QReport *qr;
  DQMNet::QValue *qv;
  const_cast<MonitorElement *>(this)->getQReport(false, qtname, qr, qv);
  return qr;
}

std::vector<QReport *>
MonitorElement::getQReports(void) const
{
  std::vector<QReport *> result;
  result.reserve(qreports_.size());
  for (size_t i = 0, e = qreports_.size(); i != e; ++i)
    result.push_back(const_cast<QReport *>(&qreports_[i]));
  return result;
}

std::vector<QReport *>
MonitorElement::getQWarnings(void) const
{
  std::vector<QReport *> result;
  result.reserve(qreports_.size());
  for (size_t i = 0, e = qreports_.size(); i != e; ++i)
    if (data_.qreports[i].code == dqm::qstatus::WARNING)
      result.push_back(const_cast<QReport *>(&qreports_[i]));
  return result;
}

std::vector<QReport *>
MonitorElement::getQErrors(void) const
{
  std::vector<QReport *> result;
  result.reserve(qreports_.size());
  for (size_t i = 0, e = qreports_.size(); i != e; ++i)
    if (data_.qreports[i].code == dqm::qstatus::ERROR)
      result.push_back(const_cast<QReport *>(&qreports_[i]));
  return result;
}

std::vector<QReport *>
MonitorElement::getQOthers(void) const
{
  std::vector<QReport *> result;
  result.reserve(qreports_.size());
  for (size_t i = 0, e = qreports_.size(); i != e; ++i)
    if (data_.qreports[i].code != dqm::qstatus::STATUS_OK
	&& data_.qreports[i].code != dqm::qstatus::WARNING
	&& data_.qreports[i].code != dqm::qstatus::ERROR)
      result.push_back(const_cast<QReport *>(&qreports_[i]));
  return result;
}

/// run all quality tests
void
MonitorElement::runQTests(void)
{
  assert(qreports_.size() == data_.qreports.size());

  // Rerun quality tests where the ME or the quality algorithm was modified.
  bool dirty = wasUpdated();
  for (size_t i = 0, e = data_.qreports.size(); i < e; ++i)
  {
    DQMNet::QValue &qv = data_.qreports[i];
    QReport &qr = qreports_[i];
    QCriterion *qc = qr.qcriterion_;
    qr.qvalue_ = &qv;

    // if (qc && (dirty || qc->wasModified()))  // removed for new QTest (abm-090503)
    if (qc && dirty ) 
    {
      assert(qc->getName() == qv.qtname);
      std::string oldMessage = qv.message;
      int oldStatus = qv.code;

      qc->runTest(this, qr, qv);

      if (oldStatus != qv.code || oldMessage != qv.message)
	update();
    }
  }

  // Update QReport statistics.
  updateQReportStats();
}

void
MonitorElement::incompatible(const char *func) const
{
  raiseDQMError("MonitorElement", "Method '%s' cannot be invoked on monitor"
		" element '%s'", func, data_.name.c_str());
}

TH1 *
MonitorElement::accessRootObject(const char *func, int reqdim) const
{ return checkRootObject(data_.name, curvalue_.tobj, func, reqdim); }

/*** getter methods (wrapper around ROOT methods) ****/
// 
/// get mean value of histogram along x, y or z axis (axis=1, 2, 3 respectively)
float
MonitorElement::getMean(int axis /* = 1 */) const
{ return accessRootObject(__PRETTY_FUNCTION__, axis-1)
    ->GetMean(axis); }

/// get mean value uncertainty of histogram along x, y or z axis 
/// (axis=1, 2, 3 respectively)
float
MonitorElement::getMeanError(int axis /* = 1 */) const
{ return accessRootObject(__PRETTY_FUNCTION__, axis-1)
    ->GetMeanError(axis); }

/// get RMS of histogram along x, y or z axis (axis=1, 2, 3 respectively)
float
MonitorElement::getRMS(int axis /* = 1 */) const
{ return accessRootObject(__PRETTY_FUNCTION__, axis-1)
    ->GetRMS(axis); }

/// get RMS uncertainty of histogram along x, y or z axis(axis=1,2,3 respectively)
float
MonitorElement::getRMSError(int axis /* = 1 */) const
{ return accessRootObject(__PRETTY_FUNCTION__, axis-1)
    ->GetRMSError(axis); }

/// get # of bins in X-axis
int
MonitorElement::getNbinsX(void) const
{ return accessRootObject(__PRETTY_FUNCTION__, 1)
    ->GetNbinsX(); }

/// get # of bins in Y-axis
int
MonitorElement::getNbinsY(void) const
{ return accessRootObject(__PRETTY_FUNCTION__, 2)
    ->GetNbinsY(); }

/// get # of bins in Z-axis
int
MonitorElement::getNbinsZ(void) const
{ return accessRootObject(__PRETTY_FUNCTION__, 3)
    ->GetNbinsZ(); }

/// get content of bin (1-D)
float
MonitorElement::getBinContent(int binx) const
{ return accessRootObject(__PRETTY_FUNCTION__, 1)
    ->GetBinContent(binx); }

/// get content of bin (2-D)
float
MonitorElement::getBinContent(int binx, int biny) const
{ return accessRootObject(__PRETTY_FUNCTION__, 2)
    ->GetBinContent(binx, biny); }

/// get content of bin (3-D)
float
MonitorElement::getBinContent(int binx, int biny, int binz) const
{ return accessRootObject(__PRETTY_FUNCTION__, 3)
    ->GetBinContent(binx, biny, binz); }

/// get uncertainty on content of bin (1-D) - See TH1::GetBinError for details
float
MonitorElement::getBinError(int binx) const
{ return accessRootObject(__PRETTY_FUNCTION__, 1)
    ->GetBinError(binx); }

/// get uncertainty on content of bin (2-D) - See TH1::GetBinError for details
float
MonitorElement::getBinError(int binx, int biny) const
{ return accessRootObject(__PRETTY_FUNCTION__, 2)
    ->GetBinError(binx, biny); }

/// get uncertainty on content of bin (3-D) - See TH1::GetBinError for details
float
MonitorElement::getBinError(int binx, int biny, int binz) const
{ return accessRootObject(__PRETTY_FUNCTION__, 3)
    ->GetBinError(binx, biny, binz); }

/// get # of entries
float
MonitorElement::getEntries(void) const
{ return accessRootObject(__PRETTY_FUNCTION__, 1)
    ->GetEntries(); }

/// get # of bin entries (for profiles)
float
MonitorElement::getBinEntries(int bin) const
{
  if (kind_ == DQM_KIND_TPROFILE)
    return static_cast<TProfile *>(accessRootObject(__PRETTY_FUNCTION__, 1))
      ->GetBinEntries(bin);
  else if (kind_ == DQM_KIND_TPROFILE2D)
    return static_cast<TProfile2D *>(accessRootObject(__PRETTY_FUNCTION__, 1))
      ->GetBinEntries(bin);
  else
  {
    incompatible(__PRETTY_FUNCTION__);
    return 0;
  }
}

/// get min Y value (for profiles)
float
MonitorElement::getYmin(void) const
{
  if (kind_ == DQM_KIND_TPROFILE)
    return static_cast<TProfile *>(accessRootObject(__PRETTY_FUNCTION__, 1))
      ->GetYmin();
  else
  {
    incompatible(__PRETTY_FUNCTION__);
    return 0;
  }
}

/// get max Y value (for profiles)
float
MonitorElement::getYmax(void) const
{
  if (kind_ == DQM_KIND_TPROFILE)
    return static_cast<TProfile *>(accessRootObject(__PRETTY_FUNCTION__, 1))
      ->GetYmax();
  else
  {
    incompatible(__PRETTY_FUNCTION__);
    return 0;
  }
}

/// get x-, y- or z-axis title (axis=1, 2, 3 respectively)
std::string
MonitorElement::getAxisTitle(int axis /* = 1 */) const
{ return getAxis(__PRETTY_FUNCTION__, axis)
    ->GetTitle(); }

/// get MonitorElement title
std::string
MonitorElement::getTitle(void) const
{ return accessRootObject(__PRETTY_FUNCTION__, 1)
    ->GetTitle(); }

/*** setter methods (wrapper around ROOT methods) ****/
// 
/// set content of bin (1-D)
void
MonitorElement::setBinContent(int binx, float content)
{
  update();
  accessRootObject(__PRETTY_FUNCTION__, 1)
    ->SetBinContent(binx, content);
}

/// set content of bin (2-D)
void
MonitorElement::setBinContent(int binx, int biny, float content)
{
  update();
  accessRootObject(__PRETTY_FUNCTION__, 2)
    ->SetBinContent(binx, biny, content); }

/// set content of bin (3-D)
void
MonitorElement::setBinContent(int binx, int biny, int binz, float content)
{
  update();
  accessRootObject(__PRETTY_FUNCTION__, 3)
    ->SetBinContent(binx, biny, binz, content); }

/// set uncertainty on content of bin (1-D)
void
MonitorElement::setBinError(int binx, float error)
{
  update();
  accessRootObject(__PRETTY_FUNCTION__, 1)
    ->SetBinError(binx, error);
}

/// set uncertainty on content of bin (2-D)
void
MonitorElement::setBinError(int binx, int biny, float error)
{
  update();
  accessRootObject(__PRETTY_FUNCTION__, 2)
    ->SetBinError(binx, biny, error);
}

/// set uncertainty on content of bin (3-D)
void
MonitorElement::setBinError(int binx, int biny, int binz, float error)
{
  update();
  accessRootObject(__PRETTY_FUNCTION__, 3)
    ->SetBinError(binx, biny, binz, error);
}

/// set # of bin entries (to be used for profiles)
void
MonitorElement::setBinEntries(int bin, float nentries)
{
  update();
  if (kind_ == DQM_KIND_TPROFILE)
    static_cast<TProfile *>(accessRootObject(__PRETTY_FUNCTION__, 1))
      ->SetBinEntries(bin, nentries);
  else if (kind_ == DQM_KIND_TPROFILE2D)
    static_cast<TProfile2D *>(accessRootObject(__PRETTY_FUNCTION__, 1))
      ->SetBinEntries(bin, nentries);
  else
    incompatible(__PRETTY_FUNCTION__);
}

/// set # of entries
void
MonitorElement::setEntries(float nentries)
{
  update();
  accessRootObject(__PRETTY_FUNCTION__, 1)
    ->SetEntries(nentries);
}

/// set bin label for x, y or z axis (axis=1, 2, 3 respectively)
void
MonitorElement::setBinLabel(int bin, const std::string &label, int axis /* = 1 */)
{
  update();
  getAxis(__PRETTY_FUNCTION__, axis)
    ->SetBinLabel(bin, label.c_str());
}

/// set x-, y- or z-axis range (axis=1, 2, 3 respectively)
void
MonitorElement::setAxisRange(float xmin, float xmax, int axis /* = 1 */)
{
  update();
  getAxis(__PRETTY_FUNCTION__, axis)
    ->SetRangeUser(xmin, xmax);
}

/// set x-, y- or z-axis title (axis=1, 2, 3 respectively)
void
MonitorElement::setAxisTitle(const std::string &title, int axis /* = 1 */)
{
  update();
  getAxis(__PRETTY_FUNCTION__, axis)
    ->SetTitle(title.c_str());
}

/// set x-, y-, or z-axis to display time values
void
MonitorElement::setAxisTimeDisplay(int value, int axis /* = 1 */)
{
  update();
  getAxis(__PRETTY_FUNCTION__, axis)
    ->SetTimeDisplay(value);
}

/// set the format of the time values that are displayed on an axis
void
MonitorElement::setAxisTimeFormat(const char *format /* = "" */, int axis /* = 1 */)
{
  update();
  getAxis(__PRETTY_FUNCTION__, axis)
    ->SetTimeFormat(format);
}

/// set the time offset, if option = "gmt" then the offset is treated as a GMT time
void
MonitorElement::setAxisTimeOffset(double toffset, const char *option /* ="local" */, int axis /* = 1 */)
{
  update();
  getAxis(__PRETTY_FUNCTION__, axis)
    ->SetTimeOffset(toffset, option);
}

/// set (ie. change) histogram/profile title
void
MonitorElement::setTitle(const std::string &title)
{
  update();
  accessRootObject(__PRETTY_FUNCTION__, 1)
    ->SetTitle(title.c_str());
}

TAxis *
MonitorElement::getAxis(const char *func, int axis) const
{
  TH1 *h = accessRootObject(func, axis-1);
  TAxis *a = 0;
  if (axis == 1)
    a = h->GetXaxis();
  else if (axis == 2)
    a = h->GetYaxis();
  else if (axis == 3)
    a = h->GetZaxis();

  if (! a)
    raiseDQMError("MonitorElement", "No such axis %d in monitor element"
		  " '%s' of type '%s'", axis, data_.name.c_str(),
		  typeid(*h).name());

  return a;
}

// ------------ Operations for MEs that are normally never reset ---------

/// reset contents (does not erase contents permanently)
/// (makes copy of current contents; will be subtracted from future contents)
void
MonitorElement::softReset(void)
{
  update();

  // Create the reference object the first time this is called.
  // On subsequent calls accumulate the current value to the
  // reference, and then reset the current value.  This way the
  // future contents will have the reference "subtracted".
  if (kind_ == DQM_KIND_TH1F)
  {
    TH1F *orig = static_cast<TH1F *>(curvalue_.tobj);
    TH1F *r = static_cast<TH1F *>(refvalue_);
    if (! r)
    {
      refvalue_ = r = new TH1F((std::string(orig->GetName()) + "_ref").c_str(),
			       orig->GetTitle(),
			       orig->GetNbinsX(),
			       orig->GetXaxis()->GetXmin(),
			       orig->GetXaxis()->GetXmax());
      r->SetDirectory(0);
      r->Reset();
    }

    r->Add(orig);
    orig->Reset();
  }
  else if (kind_ == DQM_KIND_TH1S)
  {
    TH1S *orig = static_cast<TH1S *>(curvalue_.tobj);
    TH1S *r = static_cast<TH1S *>(refvalue_);
    if (! r)
    {
      refvalue_ = r = new TH1S((std::string(orig->GetName()) + "_ref").c_str(),
			       orig->GetTitle(),
			       orig->GetNbinsX(),
			       orig->GetXaxis()->GetXmin(),
			       orig->GetXaxis()->GetXmax());
      r->SetDirectory(0);
      r->Reset();
    }

    r->Add(orig);
    orig->Reset();
  }
  else if (kind_ == DQM_KIND_TH2F)
  {
    TH2F *orig = static_cast<TH2F *>(curvalue_.tobj);
    TH2F *r = static_cast<TH2F *>(refvalue_);
    if (! r)
    {
      refvalue_ = r = new TH2F((std::string(orig->GetName()) + "_ref").c_str(),
			       orig->GetTitle(),
			       orig->GetNbinsX(),
			       orig->GetXaxis()->GetXmin(),
			       orig->GetXaxis()->GetXmax(),
			       orig->GetNbinsY(),
			       orig->GetYaxis()->GetXmin(),
			       orig->GetYaxis()->GetXmax());
      r->SetDirectory(0);
      r->Reset();
    }

    r->Add(orig);
    orig->Reset();
  }
  else if (kind_ == DQM_KIND_TH2S)
  {
    TH2S *orig = static_cast<TH2S *>(curvalue_.tobj);
    TH2S *r = static_cast<TH2S *>(refvalue_);
    if (! r)
    {
      refvalue_ = r = new TH2S((std::string(orig->GetName()) + "_ref").c_str(),
			       orig->GetTitle(),
			       orig->GetNbinsX(),
			       orig->GetXaxis()->GetXmin(),
			       orig->GetXaxis()->GetXmax(),
			       orig->GetNbinsY(),
			       orig->GetYaxis()->GetXmin(),
			       orig->GetYaxis()->GetXmax());
      r->SetDirectory(0);
      r->Reset();
    }

    r->Add(orig);
    orig->Reset();
  }
  else if (kind_ == DQM_KIND_TH3F)
  {
    TH3F *orig = static_cast<TH3F *>(curvalue_.tobj);
    TH3F *r = static_cast<TH3F *>(refvalue_);
    if (! r)
    {
      refvalue_ = r = new TH3F((std::string(orig->GetName()) + "_ref").c_str(),
			       orig->GetTitle(),
			       orig->GetNbinsX(),
			       orig->GetXaxis()->GetXmin(),
			       orig->GetXaxis()->GetXmax(),
			       orig->GetNbinsY(),
			       orig->GetYaxis()->GetXmin(),
			       orig->GetYaxis()->GetXmax(),
			       orig->GetNbinsZ(),
			       orig->GetZaxis()->GetXmin(),
			       orig->GetZaxis()->GetXmax());
      r->SetDirectory(0);
      r->Reset();
    }

    r->Add(orig);
    orig->Reset();
  }
  else if (kind_ == DQM_KIND_TPROFILE)
  {
    TProfile *orig = static_cast<TProfile *>(curvalue_.tobj);
    TProfile *r = static_cast<TProfile *>(refvalue_);
    if (! r)
    {
      refvalue_ = r = new TProfile((std::string(orig->GetName()) + "_ref").c_str(),
				   orig->GetTitle(),
				   orig->GetNbinsX(),
				   orig->GetXaxis()->GetXmin(),
				   orig->GetXaxis()->GetXmax(),
				   orig->GetYaxis()->GetXmin(),
				   orig->GetYaxis()->GetXmax(),
				   orig->GetErrorOption());
      r->SetDirectory(0);
      r->Reset();
    }

    addProfiles(r, orig, r, 1, 1);
    orig->Reset();
  }
  else if (kind_ == DQM_KIND_TPROFILE2D)
  {
    TProfile2D *orig = static_cast<TProfile2D *>(curvalue_.tobj);
    TProfile2D *r = static_cast<TProfile2D *>(refvalue_);
    if (! r)
    {
      refvalue_ = r = new TProfile2D((std::string(orig->GetName()) + "_ref").c_str(),
				     orig->GetTitle(),
				     orig->GetNbinsX(),
				     orig->GetXaxis()->GetXmin(),
				     orig->GetXaxis()->GetXmax(),
				     orig->GetNbinsY(),
				     orig->GetYaxis()->GetXmin(),
				     orig->GetYaxis()->GetXmax(),
				     orig->GetZaxis()->GetXmin(),
				     orig->GetZaxis()->GetXmax(),
				     orig->GetErrorOption());
      r->SetDirectory(0);
      r->Reset();
    }

    addProfiles(r, orig, r, 1, 1);
    orig->Reset();
  }
  else
    incompatible(__PRETTY_FUNCTION__);
}

/// reverts action of softReset
void
MonitorElement::disableSoftReset(void)
{
  if (refvalue_)
  {
    if (kind_ == DQM_KIND_TH1F
	|| kind_ == DQM_KIND_TH1S
	|| kind_ == DQM_KIND_TH2F
	|| kind_ == DQM_KIND_TH2S
	|| kind_ == DQM_KIND_TH3F)
    {
      TH1 *orig = static_cast<TH1 *>(curvalue_.tobj);
      orig->Add(refvalue_);
    }
    else if (kind_ == DQM_KIND_TPROFILE)
    {
      TProfile *orig = static_cast<TProfile *>(curvalue_.tobj);
      TProfile *r = static_cast<TProfile *>(refvalue_);
      addProfiles(orig, r, orig, 1, 1);
    }
    else if (kind_ == DQM_KIND_TPROFILE2D)
    {
      TProfile2D *orig = static_cast<TProfile2D *>(curvalue_.tobj);
      TProfile2D *r = static_cast<TProfile2D *>(refvalue_);
      addProfiles(orig, r, orig, 1, 1);
    }
    else
      incompatible(__PRETTY_FUNCTION__);

    delete refvalue_;
    refvalue_ = 0;
  }
}
  
// implementation: Giuseppe.Della-Ricca@ts.infn.it
// Can be called with sum = h1 or sum = h2
void
MonitorElement::addProfiles(TProfile *h1, TProfile *h2, TProfile *sum, float c1, float c2)
{
  assert(h1);
  assert(h2);
  assert(sum);

  static const Int_t NUM_STAT = 6;
  Double_t stats1[NUM_STAT];
  Double_t stats2[NUM_STAT];
  Double_t stats3[NUM_STAT];

  for (Int_t i = 0; i < NUM_STAT; ++i)
    stats1[i] = stats2[i] = stats3[i] = 0;

  h1->GetStats(stats1);
  h2->GetStats(stats2);

  for (Int_t i = 0; i < NUM_STAT; ++i)
    stats3[i] = c1*stats1[i] + c2*stats2[i];

  stats3[1] = c1*TMath::Abs(c1)*stats1[1]
	      + c2*TMath::Abs(c2)*stats2[1];

  Double_t entries = c1*h1->GetEntries() + c2* h2->GetEntries();
  TArrayD* h1sumw2 = h1->GetSumw2();
  TArrayD* h2sumw2 = h2->GetSumw2();
  for (Int_t bin = 0, nbin = sum->GetNbinsX()+1; bin <= nbin; ++bin) 
  {
    Double_t entries = c1*h1->GetBinEntries(bin)
		       + c2*h2->GetBinEntries(bin);
    Double_t content = c1*h1->GetBinEntries(bin)*h1->GetBinContent(bin)
		       + c2*h2->GetBinEntries(bin)*h2->GetBinContent(bin);
    Double_t error = TMath::Sqrt(c1*TMath::Abs(c1)*h1sumw2->fArray[bin]
				 + c2*TMath::Abs(c2)*h2sumw2->fArray[bin]);
    sum->SetBinContent(bin, content);
    sum->SetBinError(bin, error);
    sum->SetBinEntries(bin, entries);
  }
  
  sum->SetEntries(entries);
  sum->PutStats(stats3);
}

// implementation: Giuseppe.Della-Ricca@ts.infn.it
// Can be called with sum = h1 or sum = h2
void
MonitorElement::addProfiles(TProfile2D *h1, TProfile2D *h2, TProfile2D *sum, float c1, float c2)
{
  assert(h1);
  assert(h2);
  assert(sum);

  static const Int_t NUM_STAT = 9;
  Double_t stats1[NUM_STAT];
  Double_t stats2[NUM_STAT];
  Double_t stats3[NUM_STAT];
  for (Int_t i = 0; i < NUM_STAT; ++i)
    stats1[i] = stats2[i] = stats3[i] = 0;

  h1->GetStats(stats1);
  h2->GetStats(stats2);

  for (Int_t i = 0; i < NUM_STAT; i++)
    stats3[i] = c1*stats1[i] + c2*stats2[i];

  stats3[1] = c1*TMath::Abs(c1)*stats1[1]
	      + c2*TMath::Abs(c2)*stats2[1];

  Double_t entries = c1*h1->GetEntries() + c2*h2->GetEntries();
  TArrayD *h1sumw2 = h1->GetSumw2();
  TArrayD *h2sumw2 = h2->GetSumw2();
  for (Int_t xbin = 0, nxbin = sum->GetNbinsX()+1; xbin <= nxbin; ++xbin)
    for (Int_t ybin = 0, nybin = sum->GetNbinsY()+1; ybin <= nybin; ++ybin)
    {
      Int_t bin = sum->GetBin(xbin, ybin);
      Double_t entries = c1*h1->GetBinEntries(bin)
			 + c2*h2->GetBinEntries(bin);
      Double_t content = c1*h1->GetBinEntries(bin)*h1->GetBinContent(bin)
			 + c2*h2->GetBinEntries(bin)*h2->GetBinContent(bin);
      Double_t error = TMath::Sqrt(c1*TMath::Abs(c1)*h1sumw2->fArray[bin]
				   + c2*TMath::Abs(c2)*h2sumw2->fArray[bin]);

      sum->SetBinContent(bin, content);
      sum->SetBinError(bin, error);
      sum->SetBinEntries(bin, entries);
    }
  sum->SetEntries(entries);
  sum->PutStats(stats3);
}

void
MonitorElement::copyFunctions(TH1 *from, TH1 *to)
{
  // will copy functions only if local-copy and original-object are equal
  // (ie. no soft-resetting or accumulating is enabled)
  if (isSoftResetEnabled() || isAccumulateEnabled())
    return;

  update();
  TList *fromf = from->GetListOfFunctions();
  TList *tof   = to->GetListOfFunctions();
  for (int i = 0, nfuncs = fromf ? fromf->GetSize() : 0; i < nfuncs; ++i)
  {
    TObject *obj = fromf->At(i);
    // not interested in statistics
    if (obj->IsA()->GetName() == "TPaveStats")
      continue;

    if(TF1 *fn = dynamic_cast<TF1 *>(obj))
      tof->Add(new TF1(*fn));
    //else if (dynamic_cast<TPaveStats *>(obj))
    //  ; // FIXME? tof->Add(new TPaveStats(*stats));
    else
      raiseDQMError("MonitorElement", "Cannot extract function '%s' of type"
		    " '%s' from monitor element '%s' for a copy",
		    obj->GetName(), obj->IsA()->GetName(), data_.name.c_str());
  }
}

void
MonitorElement::copyFrom(TH1 *from)
{
  TH1 *orig = accessRootObject(__PRETTY_FUNCTION__, 1);
  if (orig->GetTitle() != from->GetTitle())
    orig->SetTitle(from->GetTitle());

  if (!isAccumulateEnabled())
    orig->Reset();

  if (isSoftResetEnabled())
  {
    if (kind_ == DQM_KIND_TH1F
	|| kind_ == DQM_KIND_TH1S
	|| kind_ == DQM_KIND_TH2F
	|| kind_ == DQM_KIND_TH2S
	|| kind_ == DQM_KIND_TH3F)
      // subtract "reference"
      orig->Add(from, refvalue_, 1, -1);
    else if (kind_ == DQM_KIND_TPROFILE)
      // subtract "reference"
      addProfiles(static_cast<TProfile *>(from),
		  static_cast<TProfile *>(refvalue_),
		  static_cast<TProfile *>(orig),
		  1, -1);
    else if (kind_ == DQM_KIND_TPROFILE2D)
      // subtract "reference"
      addProfiles(static_cast<TProfile2D *>(from),
		  static_cast<TProfile2D *>(refvalue_),
		  static_cast<TProfile2D *>(orig),
		  1, -1);
    else
      incompatible(__PRETTY_FUNCTION__);
  }
  else
    orig->Add(from);

  copyFunctions(from, orig);
}

// --- Operations on MEs that are normally reset at end of monitoring cycle ---
void
MonitorElement::getQReport(bool create, const std::string &qtname, QReport *&qr, DQMNet::QValue *&qv)
{
  assert(qreports_.size() == data_.qreports.size());

  qr = 0;
  qv = 0;

  size_t pos = 0, end = qreports_.size();
  while (pos < end && data_.qreports[pos].qtname != qtname)
    ++pos;

  if (pos == end && ! create)
    return;
  else if (pos == end)
  {
    data_.qreports.push_back(DQMNet::QValue());
    qreports_.push_back(QReport(0, 0));

    DQMNet::QValue &q = data_.qreports.back();
    q.code = dqm::qstatus::DID_NOT_RUN;
    q.qtname = qtname;
    q.message = "NO_MESSAGE_ASSIGNED";
    q.algorithm = "UNKNOWN_ALGORITHM";
  }

  qr = &qreports_[pos];
  qv = &data_.qreports[pos];
}
  
/// Add quality report, from DQMStore.
void
MonitorElement::addQReport(const DQMNet::QValue &desc, QCriterion *qc)
{
  QReport *qr;
  DQMNet::QValue *qv;
  getQReport(true, desc.qtname, qr, qv);
  qr->qcriterion_ = qc;
  *qv = desc;
  update();
}

void
MonitorElement::addQReport(QCriterion *qc)
{
  QReport *qr;
  DQMNet::QValue *qv;
  getQReport(true, qc->getName(), qr, qv);
  qv->code = dqm::qstatus::DID_NOT_RUN;
  qv->message = "NO_MESSAGE_ASSIGNED";
  qr->qcriterion_ = qc;
  update();
}

/// Refresh QReport stats, usually after MEs were read in from a file.
void
MonitorElement::updateQReportStats(void)
{
  nqerror_ = nqwarning_ = nqother_ = 0;
  for (size_t i = 0, e = data_.qreports.size(); i < e; ++i)
    switch (data_.qreports[i].code)
    {
    case dqm::qstatus::STATUS_OK: break;
    case dqm::qstatus::WARNING:   ++nqwarning_; break;
    case dqm::qstatus::ERROR:     ++nqerror_; break;
    default:                      ++nqother_; break;
    }

  data_.flags &= ~(DQMNet::DQM_FLAG_REPORT_ERROR
		   | DQMNet::DQM_FLAG_REPORT_WARNING
		   | DQMNet::DQM_FLAG_REPORT_OTHER);
  if (nqerror_)
    data_.flags |= DQMNet::DQM_FLAG_REPORT_ERROR;
  if (nqwarning_)
    data_.flags |= DQMNet::DQM_FLAG_REPORT_WARNING;
  if (nqother_)
    data_.flags |= DQMNet::DQM_FLAG_REPORT_OTHER;
}

// -------------------------------------------------------------------
TObject *
MonitorElement::getRootObject(void) const
{
  const_cast<MonitorElement *>(this)->update();
  return data_.object;
}

TH1 *
MonitorElement::getTH1(void) const
{
  const_cast<MonitorElement *>(this)->update();
  return accessRootObject(__PRETTY_FUNCTION__, 0);
}

TH1F *
MonitorElement::getTH1F(void) const
{
  assert(kind_ == DQM_KIND_TH1F);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TH1F *>(accessRootObject(__PRETTY_FUNCTION__, 1));
}

TH1S *
MonitorElement::getTH1S(void) const
{
  assert(kind_ == DQM_KIND_TH1S);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TH1S *>(accessRootObject(__PRETTY_FUNCTION__, 1));
}

TH2F *
MonitorElement::getTH2F(void) const
{
  assert(kind_ == DQM_KIND_TH2F);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TH2F *>(accessRootObject(__PRETTY_FUNCTION__, 2));
}

TH2S *
MonitorElement::getTH2S(void) const
{
  assert(kind_ == DQM_KIND_TH2S);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TH2S *>(accessRootObject(__PRETTY_FUNCTION__, 2));
}

TH3F *
MonitorElement::getTH3F(void) const
{
  assert(kind_ == DQM_KIND_TH3F);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TH3F *>(accessRootObject(__PRETTY_FUNCTION__, 3));
}

TProfile *
MonitorElement::getTProfile(void) const
{
  assert(kind_ == DQM_KIND_TPROFILE);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TProfile *>(accessRootObject(__PRETTY_FUNCTION__, 1));
}

TProfile2D *
MonitorElement::getTProfile2D(void) const
{
  assert(kind_ == DQM_KIND_TPROFILE2D);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TProfile2D *>(accessRootObject(__PRETTY_FUNCTION__, 2));
}

// -------------------------------------------------------------------
TObject *
MonitorElement::getRefRootObject(void) const
{
  const_cast<MonitorElement *>(this)->update();
  return data_.reference;
}

TH1 *
MonitorElement::getRefTH1(void) const
{
  const_cast<MonitorElement *>(this)->update();
  return checkRootObject(data_.name, data_.reference, __PRETTY_FUNCTION__, 0);
}

TH1F *
MonitorElement::getRefTH1F(void) const
{
  assert(kind_ == DQM_KIND_TH1F);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TH1F *>
    (checkRootObject(data_.name, data_.reference, __PRETTY_FUNCTION__, 1));
}

TH1S *
MonitorElement::getRefTH1S(void) const
{
  assert(kind_ == DQM_KIND_TH1S);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TH1S *>
    (checkRootObject(data_.name, data_.reference, __PRETTY_FUNCTION__, 1));
}

TH2F *
MonitorElement::getRefTH2F(void) const
{
  assert(kind_ == DQM_KIND_TH2F);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TH2F *>
    (checkRootObject(data_.name, data_.reference, __PRETTY_FUNCTION__, 2));
}

TH2S *
MonitorElement::getRefTH2S(void) const
{
  assert(kind_ == DQM_KIND_TH2S);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TH2S *>
    (checkRootObject(data_.name, data_.reference, __PRETTY_FUNCTION__, 2));
}

TH3F *
MonitorElement::getRefTH3F(void) const
{
  assert(kind_ == DQM_KIND_TH3F);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TH3F *>
    (checkRootObject(data_.name, data_.reference, __PRETTY_FUNCTION__, 3));
}

TProfile *
MonitorElement::getRefTProfile(void) const
{
  assert(kind_ == DQM_KIND_TPROFILE);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TProfile *>
    (checkRootObject(data_.name, data_.reference, __PRETTY_FUNCTION__, 1));
}

TProfile2D *
MonitorElement::getRefTProfile2D(void) const
{
  assert(kind_ == DQM_KIND_TPROFILE2D);
  const_cast<MonitorElement *>(this)->update();
  return dynamic_cast<TProfile2D *>
    (checkRootObject(data_.name, data_.reference, __PRETTY_FUNCTION__, 2));
}

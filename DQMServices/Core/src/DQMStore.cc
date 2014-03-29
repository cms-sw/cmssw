#include "DQMServices/Core/interface/Standalone.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/QTest.h"
#include "DQMServices/Core/src/DQMError.h"
#include "classlib/utils/RegexpMatch.h"
#include "classlib/utils/Regexp.h"
#include "classlib/utils/StringOps.h"
#include "TFile.h"
#include "TROOT.h"
#include "TKey.h"
#include "TClass.h"
#include "TSystem.h"
#include <iterator>
#include <cerrno>
#include <boost/algorithm/string.hpp>
#include <fstream>

/** @var DQMStore::verbose_
    Universal verbose flag for DQM. */

/** @var DQMStore::verboseQT_
    Verbose flag for xml-based QTests. */

/** @var DQMStore::reset_

Flag used to print out a warning when calling quality tests.
twice without having called reset() in between; to be reset in
DQMOldReceiver::runQualityTests.  */

/** @var DQMStore::collateHistograms_ */

/** @var DQMStore::readSelectedDirectory_
    If non-empty, read from file only selected directory. */

/** @var DQMStore::pwd_
    Current directory. */

/** @var DQMStore::qtests_.
    All the quality tests.  */

/** @var DQMStore::qalgos_
    Set of all the available quality test algorithms. */

//////////////////////////////////////////////////////////////////////
/// name of global monitoring folder (containing all sources subdirectories)
static const std::string s_monitorDirName = "DQMData";
static const std::string s_referenceDirName = "Reference";
static const std::string s_collateDirName = "Collate";
static const std::string s_safe = "/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+=_()# ";

static const lat::Regexp s_rxmeval ("^<(.*)>(i|f|s|e|t|qr)=(.*)</\\1>$");
static const lat::Regexp s_rxmeqr1 ("^st:(\\d+):([-+e.\\d]+):([^:]*):(.*)$");
static const lat::Regexp s_rxmeqr2 ("^st\\.(\\d+)\\.(.*)$");
static const lat::Regexp s_rxtrace ("(.*)\\((.*)\\+0x.*\\).*");

//////////////////////////////////////////////////////////////////////
/// Check whether the @a path is a subdirectory of @a ofdir.  Returns
/// true both for an exact match and any nested subdirectory.
static bool
isSubdirectory(const std::string &ofdir, const std::string &path)
{
  return (ofdir.empty()
          || (path.size() >= ofdir.size()
              && path.compare(0, ofdir.size(), ofdir) == 0
              && (path.size() == ofdir.size()
                  || path[ofdir.size()] == '/')));
}

static void
cleanTrailingSlashes(const std::string &path, std::string &clean, const std::string *&cleaned)
{
  clean.clear();
  cleaned = &path;

  size_t len = path.size();
  for ( ; len > 0 && path[len-1] == '/'; --len)
    ;

  if (len != path.size())
  {
    clean = path.substr(0, len);
    cleaned = &clean;
  }
}

static void
splitPath(std::string &dir, std::string &name, const std::string &path)
{
  size_t slash = path.rfind('/');
  if (slash != std::string::npos)
  {
    dir.append(path, 0, slash);
    name.append(path, slash+1, std::string::npos);
  }
  else
    name = path;
}

static void
mergePath(std::string &path, const std::string &dir, const std::string &name)
{
  path.reserve(dir.size() + name.size() + 2);
  path += dir;
  if (! path.empty())
    path += '/';
  path += name;
}

template <class T>
QCriterion *
makeQCriterion(const std::string &qtname)
{ return new T(qtname); }

template <class T>
void
initQCriterion(std::map<std::string, QCriterion *(*)(const std::string &)> &m)
{ m[T::getAlgoName()] = &makeQCriterion<T>; }


/////////////////////////////////////////////////////////////
fastmatch::fastmatch (std::string const& _fastString) :
  fastString_ (_fastString),  matching_ (UseFull)
{
  try
  {
    regexp_ = NULL;
    regexp_ = new lat::Regexp(fastString_, 0, lat::Regexp::Wildcard);
    regexp_->study();
  }
  catch (lat::Error &e)
  {
    delete regexp_;
    raiseDQMError("DQMStore", "Invalid wildcard pattern '%s' in quality"
                  " test specification", fastString_.c_str());
  }

  // count stars ( "*" )
  size_t starCount = 0;
  int pos = -1;
  while (true)
  {
    pos = fastString_.find("*", pos + 1 );
    if ((size_t)pos == std::string::npos)
      break;
    starCount ++;
  }

  // investigate for heuristics
  if ((fastString_.find('"') != std::string::npos)  ||
      (fastString_.find(']') != std::string::npos)  ||
      (fastString_.find('?') != std::string::npos)  ||
      (fastString_.find('\\') != std::string::npos) ||
      (starCount > 2))
  {
    // no fast version can be used
    return;
  }

  // match for pattern "*MyString" and "MyString*"
  if (starCount == 1)
  {
    if (boost::algorithm::starts_with(fastString_, "*"))
    {
      matching_ = OneStarStart;
      fastString_.erase(0,1);
      return;
    }

    if (boost::algorithm::ends_with(fastString_, "*"))
    {
      matching_ = OneStarEnd;
      fastString_.erase(fastString_.length()-1,1);
      return;
    }
  }

  // match for pattern "*MyString*"
  if (starCount == 2)
  {
    if (boost::algorithm::starts_with(fastString_, "*") &&
        boost::algorithm::ends_with(fastString_, "*"))
    {
      matching_ = TwoStar;
      fastString_.erase(0,1);
      fastString_.erase(fastString_.size() - 1, 1);
      return;
    }
  }
}

fastmatch::~fastmatch()
{
  if (regexp_ != NULL)
    delete regexp_;
}

bool fastmatch::compare_strings_reverse(std::string const& pattern,
                                        std::string const& input) const
{
  if (input.size() < pattern.size())
    return false;

  // compare the two strings character by character for equalness:
  // this does not create uneeded copies of std::string. The
  // boost::algorithm implementation does
  std::string::const_reverse_iterator rit_pattern = pattern.rbegin();
  std::string::const_reverse_iterator rit_input = input.rbegin();

  for (; rit_pattern < pattern.rend(); rit_pattern++, rit_input++)
  {
    if (*rit_pattern != *rit_input)
      // found a difference, fail
      return false;
  }
  return true;
}

bool fastmatch::compare_strings(std::string const& pattern,
                                std::string const& input) const
{
  if (input.size() < pattern.size())
    return false;

  // compare the two strings character by character for equalness:
  // this does not create uneeded copies of std::string. The
  // boost::algorithm implementation does.
  std::string::const_iterator rit_pattern = pattern.begin();
  std::string::const_iterator rit_input = input.begin();

  for (; rit_pattern < pattern.end(); rit_pattern++, rit_input++)
  {
    if (*rit_pattern != *rit_input)
      // found a difference, fail
      return false;
  }
  return true;
}

bool fastmatch::match(std::string const& s) const
{
  switch (matching_)
  {
  case OneStarStart:
    return compare_strings_reverse(fastString_, s);

  case OneStarEnd:
    return compare_strings(fastString_, s);

  case TwoStar:
    return (s.find(fastString_) != std::string::npos);

  default:
    return regexp_->match(s);
  }
}

void DQMStore::IBooker::cd(void) {
  owner_->cd();
}

void DQMStore::IBooker::cd(const std::string &dir) {
  owner_->cd(dir);
}

void DQMStore::IBooker::setCurrentFolder(const std::string &fullpath) {
  owner_->setCurrentFolder(fullpath);
}

void DQMStore::IBooker::tag(MonitorElement *me, unsigned int tag) {
  owner_->tag(me, tag);
}

/** Function to transfer the local copies of histograms from each
    stream into the global ROOT Object. Since this involves de-facto a
    booking action in the case in which the global object is not yet
    there, the function requires the acquisition of the central lock
    into the DQMStore.
    In case we book the global object for the first time, no Add action is
    needed since the ROOT histograms is cloned starting from the local
    one. */

void DQMStore::mergeAndResetMEsRunSummaryCache(uint32_t run,
                                               uint32_t streamId,
                                               uint32_t moduleId) {
  if (verbose_ > 1)
    std::cout << "Merging objects from run: "
              << run
              << ", stream: " << streamId
              << " module: " << moduleId << std::endl;
  std::string null_str("");
  MonitorElement proto(&null_str, null_str, run, streamId, moduleId);
  std::set<MonitorElement>::const_iterator e = data_.end();
  std::set<MonitorElement>::const_iterator i = data_.lower_bound(proto);
  while (i != e) {
    if (i->data_.run != run
        || i->data_.streamId != streamId
        || i->data_.moduleId != moduleId)
      break;

    // Handle Run-based histograms only.
    if (i->getLumiFlag()) {
      ++i;
      continue;
    }

    MonitorElement global_me(*i);
    global_me.globalize();
    // Since this accesses the data, the operation must be
    // be locked.
    std::lock_guard<std::mutex> guard(book_mutex_);
    std::set<MonitorElement>::const_iterator me = data_.find(global_me);
    if (me != data_.end()) {
      if (verbose_ > 1)
        std::cout << "Found global Object, using it. ";
      me->getTH1()->Add(i->getTH1());
    } else {
      if (verbose_ > 1)
        std::cout << "No global Object found. ";
      std::pair<std::set<MonitorElement>::const_iterator, bool> gme;
      gme = data_.insert(global_me);
      assert(gme.second);
    }
    // TODO(rovere): eventually reset the local object and mark it as reusable??
    ++i;
  }
}

void DQMStore::mergeAndResetMEsLuminositySummaryCache(uint32_t run,
						      uint32_t lumi,
						      uint32_t streamId,
						      uint32_t moduleId) {
  if (verbose_ > 1)
    std::cout << "Merging objects from run: "
              << run << 	" lumi: " << lumi
              << ", stream: " << streamId
              << " module: " << moduleId << std::endl;
  std::string null_str("");
  MonitorElement proto(&null_str, null_str, run, streamId, moduleId);
  std::set<MonitorElement>::const_iterator e = data_.end();
  std::set<MonitorElement>::const_iterator i = data_.lower_bound(proto);
  while (i != e) {
    if (i->data_.run != run
        || i->data_.streamId != streamId
        || i->data_.moduleId != moduleId)
      break;

    // Handle LS-based histograms only.
    if (not i->getLumiFlag()) {
      ++i;
      continue;
    }

    MonitorElement global_me(*i);
    global_me.globalize();
    global_me.setLumi(lumi);
    // Since this accesses the data, the operation must be
    // be locked.
    std::lock_guard<std::mutex> guard(book_mutex_);
    std::set<MonitorElement>::const_iterator me = data_.find(global_me);
    if (me != data_.end()) {
      if (verbose_ > 1)
        std::cout << "Found global Object, using it --> ";
      me->getTH1()->Add(i->getTH1());
    } else {
      if (verbose_ > 1)
        std::cout << "No global Object found. ";
      std::pair<std::set<MonitorElement>::const_iterator, bool> gme;
      gme = data_.insert(global_me);
      assert(gme.second);
    }
    const_cast<MonitorElement*>(&*i)->Reset();
    // TODO(rovere): eventually reset the local object and mark it as reusable??
    ++i;
  }
}

//////////////////////////////////////////////////////////////////////
DQMStore::DQMStore(const edm::ParameterSet &pset, edm::ActivityRegistry& ar)
  : verbose_ (1),
    verboseQT_ (1),
    reset_ (false),
    collateHistograms_ (false),
    enableMultiThread_(false),
    readSelectedDirectory_ (""),
    run_(0),
    streamId_(0),
    moduleId_(0),
    pwd_ (""),
    ibooker_(0)
{
  if (!ibooker_)
    ibooker_ = new DQMStore::IBooker(this);
  initializeFrom(pset);
  if(pset.getUntrackedParameter<bool>("forceResetOnBeginRun",false)) {
    ar.watchPostSourceRun(this,&DQMStore::forceReset);
  }
}

DQMStore::DQMStore(const edm::ParameterSet &pset)
  : verbose_ (1),
    verboseQT_ (1),
    reset_ (false),
    collateHistograms_ (false),
    enableMultiThread_(false),
    readSelectedDirectory_ (""),
    run_(0),
    streamId_(0),
    moduleId_(0),
    pwd_ (""),
    ibooker_(0)
{
  if (!ibooker_)
    ibooker_ = new DQMStore::IBooker(this);
  initializeFrom(pset);
}

DQMStore::~DQMStore(void)
{
  for (QCMap::iterator i = qtests_.begin(), e = qtests_.end(); i != e; ++i)
    delete i->second;

  for (QTestSpecs::iterator i = qtestspecs_.begin(), e = qtestspecs_.end(); i != e; ++i)
    delete i->first;

}

void
DQMStore::initializeFrom(const edm::ParameterSet& pset) {
  makeDirectory("");
  reset();

  // set steerable parameters
  verbose_ = pset.getUntrackedParameter<int>("verbose", 0);
  if (verbose_ > 0)
    std::cout << "DQMStore: verbosity set to " << verbose_ << std::endl;

  verboseQT_ = pset.getUntrackedParameter<int>("verboseQT", 0);
  if (verbose_ > 0)
    std::cout << "DQMStore: QTest verbosity set to " << verboseQT_ << std::endl;

  collateHistograms_ = pset.getUntrackedParameter<bool>("collateHistograms", false);
  if (collateHistograms_)
    std::cout << "DQMStore: histogram collation is enabled\n";

  enableMultiThread_ = pset.getUntrackedParameter<bool>("enableMultiThread", false);
  if (enableMultiThread_)
    std::cout << "DQMStore: MultiThread option is enabled\n";

  std::string ref = pset.getUntrackedParameter<std::string>("referenceFileName", "");
  if (! ref.empty())
  {
    std::cout << "DQMStore: using reference file '" << ref << "'\n";
    readFile(ref, true, "", s_referenceDirName, StripRunDirs, false);
  }

  initQCriterion<Comp2RefChi2>(qalgos_);
  initQCriterion<Comp2RefKolmogorov>(qalgos_);
  initQCriterion<ContentsXRange>(qalgos_);
  initQCriterion<ContentsYRange>(qalgos_);
  initQCriterion<MeanWithinExpected>(qalgos_);
  initQCriterion<Comp2RefEqualH>(qalgos_);
  initQCriterion<DeadChannel>(qalgos_);
  initQCriterion<NoisyChannel>(qalgos_);
  initQCriterion<ContentsWithinExpected>(qalgos_);
  initQCriterion<CompareToMedian>(qalgos_);
  initQCriterion<CompareLastFilledBin>(qalgos_);
  initQCriterion<CheckVariance>(qalgos_);

  scaleFlag_ = pset.getUntrackedParameter<double>("ScalingFlag", 0.0);
  if (verbose_ > 0)
    std::cout << "DQMStore: Scaling Flag set to " << scaleFlag_ << std::endl;
}

/* Generic method to do a backtrace and print it to stdout. It is
 customised to properly get the routine that called the booking of the
 histograms, which, following the usual stack, is at position 4. The
 name of the calling function is properly demangled and the original
 shared library including this function is also printed. For a more
 detailed explanation of the routines involved, see here:
 http://www.gnu.org/software/libc/manual/html_node/Backtraces.html
 http://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_demangling.html.*/

void
DQMStore::print_trace (const std::string &dir, const std::string &name)
{
  static std::ofstream stream("histogramBookingBT.log");
  void *array[10];
  size_t size;
  char **strings;
  int r=0;
  lat::RegexpMatch m;
  m.reset();

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);

  if ((size > 4)
      &&s_rxtrace.match(strings[4], 0, 0, &m))
  {
    char * demangled = abi::__cxa_demangle(m.matchString(strings[4], 2).c_str(), 0, 0, &r);
    stream << "\"" << dir << "/"
           << name << "\" "
           << (r ? m.matchString(strings[4], 2) : demangled) << " "
           << m.matchString(strings[4], 1) << "\n";
    free(demangled);
  }
  else
    stream << "Skipping "<< dir << "/" << name
           << " with stack size " << size << "\n";
  /* In this case print the full stack trace, up to main or to the
   * maximum stack size, i.e. 10. */
  if (verbose_ > 4)
  {
    size_t i;
    m.reset();

    for (i = 0; i < size; i++)
      if (s_rxtrace.match(strings[i], 0, 0, &m))
      {
        char * demangled = abi::__cxa_demangle(m.matchString(strings[i], 2).c_str(), 0, 0, &r);
        stream << "\t\t" << i << "/" << size << " "
               << (r ? m.matchString(strings[i], 2) : demangled) << " "
               << m.matchString(strings[i], 1) << std::endl;
        free (demangled);
      }
  }
  free (strings);
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// set verbose level (0 turns all non-error messages off)
void
DQMStore::setVerbose(unsigned /* level */)
{ return; }

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// return pathname of current directory
const std::string &
DQMStore::pwd(void) const
{ return pwd_; }

/// go to top directory (ie. root)
void
DQMStore::cd(void)
{ setCurrentFolder(""); }

/// cd to subdirectory (if there)
void
DQMStore::cd(const std::string &subdir)
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(subdir, clean, cleaned);

  if (! dirExists(*cleaned))
    raiseDQMError("DQMStore", "Cannot 'cd' into non-existent directory '%s'",
                  cleaned->c_str());

  setCurrentFolder(*cleaned);
}

/// set the last directory in fullpath as the current directory(create if needed);
/// to be invoked by user to specify directories for monitoring objects
/// before booking;
/// commands book1D (etc) & removeElement(name) imply elements in this directory!;
void
DQMStore::setCurrentFolder(const std::string &fullpath)
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(fullpath, clean, cleaned);
  makeDirectory(*cleaned);
  pwd_ = *cleaned;
}

/// equivalent to "cd .."
void
DQMStore::goUp(void)
{
  size_t pos = pwd_.rfind('/');
  if (pos == std::string::npos)
    setCurrentFolder("");
  else
    setCurrentFolder(pwd_.substr(0, pos));
}

// -------------------------------------------------------------------
/// get folder corresponding to inpath wrt to root (create subdirs if necessary)
void
DQMStore::makeDirectory(const std::string &path)
{
  std::string prev;
  std::string subdir;
  std::string name;
  prev.reserve(path.size());
  subdir.reserve(path.size());
  name.reserve(path.size());
  size_t prevname = 0;
  size_t slash = 0;

  while (true)
  {
    // Create this subdirectory component.
    subdir.clear();
    subdir.append(path, 0, slash);
    name.clear();
    name.append(subdir, prevname, std::string::npos);
    if (! prev.empty() && findObject(prev, name))
      raiseDQMError("DQMStore", "Attempt to create subdirectory '%s'"
                    " which already exists as a monitor element",
                    subdir.c_str());

    if (! dirs_.count(subdir))
      dirs_.insert(subdir);

    // Stop if we've reached the end (including possibly a trailing slash).
    if (slash+1 >= path.size())
      break;

    // Find the next slash, making sure we progress.  If reach the end,
    // process the last path component; the next loop round will terminate.
    prevname = slash ? slash+1 : slash;
    prev = subdir;
    if ((slash = path.find('/', ++slash)) == std::string::npos)
      slash = path.size();
  }
}

/// true if directory exists
bool
DQMStore::dirExists(const std::string &path) const
{ return dirs_.count(path) > 0; }

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
template <class HISTO, class COLLATE>
MonitorElement *
DQMStore::book(const std::string &dir, const std::string &name,
               const char *context, int kind,
               HISTO *h, COLLATE collate)
{
  assert(name.find('/') == std::string::npos);
  if (verbose_ > 3)
    print_trace(dir, name);
  std::string path;
  mergePath(path, dir, name);

  // Put us in charge of h.
  h->SetDirectory(0);

  // Check if the request monitor element already exists.
  MonitorElement *me = findObject(dir, name, run_, 0, streamId_, moduleId_);
  if (me)
  {
    if (collateHistograms_)
    {
      collate(me, h);
      delete h;
      return me;
    }
    else
    {
      if (verbose_ > 1)
        std::cout << "DQMStore: "
                  << context << ": monitor element '"
                  << path << "' already exists, collating" << std::endl;
      me->Reset();
      collate(me, h);
      delete h;
      return me;
    }
  }
  else
  {
    // Create and initialise core object.
    assert(dirs_.count(dir));
    MonitorElement proto(&*dirs_.find(dir), name, run_, streamId_, moduleId_);
    me = const_cast<MonitorElement &>(*data_.insert(proto).first)
      .initialise((MonitorElement::Kind)kind, h);

    // Initialise quality test information.
    QTestSpecs::iterator qi = qtestspecs_.begin();
    QTestSpecs::iterator qe = qtestspecs_.end();
    for ( ; qi != qe; ++qi)
    {
        if ( qi->first->match(path) )
                me->addQReport(qi->second);
    }

    // Assign reference if we have one.
    std::string refdir;
    refdir.reserve(s_referenceDirName.size() + dir.size() + 2);
    refdir += s_referenceDirName;
    refdir += '/';
    refdir += dir;

    if (MonitorElement *refme = findObject(refdir, name))
    {
      me->data_.flags |= DQMNet::DQM_PROP_HAS_REFERENCE;
      me->reference_ = refme->object_;
    }

    // Return the monitor element.
    return me;
  }
}

MonitorElement *
DQMStore::book(const std::string &dir,
               const std::string &name,
               const char *context)
{
  assert(name.find('/') == std::string::npos);
  if (verbose_ > 3)
    print_trace(dir, name);

  // Check if the request monitor element already exists.
  if (MonitorElement *me = findObject(dir, name))
  {
    if (verbose_ > 1)
    {
      std::string path;
      mergePath(path, dir, name);

      std::cout << "DQMStore: "
                << context << ": monitor element '"
                << path << "' already exists, resetting" << std::endl;
    }
    me->Reset();
    return me;
  }
  else
  {
    // Create it and return for initialisation.
    assert(dirs_.count(dir));
    MonitorElement nme(&*dirs_.find(dir), name);
    return &const_cast<MonitorElement &>(*data_.insert(nme).first);
  }
}

// -------------------------------------------------------------------
/// Book int.
MonitorElement *
DQMStore::bookInt(const std::string &dir, const std::string &name)
{
  if (collateHistograms_)
  {
    if (MonitorElement *me = findObject(dir, name))
    {
      me->Fill(0);
      return me;
    }
  }

  return book(dir, name, "bookInt")
    ->initialise(MonitorElement::DQM_KIND_INT);
}

/// Book int.
MonitorElement *
DQMStore::bookInt(const char *name)
{ return bookInt(pwd_, name); }

/// Book int.
MonitorElement *
DQMStore::bookInt(const std::string &name)
{
  return bookInt(pwd_, name);
}

// -------------------------------------------------------------------
/// Book float.
MonitorElement *
DQMStore::bookFloat(const std::string &dir, const std::string &name)
{
  if (collateHistograms_)
  {
    if (MonitorElement *me = findObject(dir, name))
    {
      me->Fill(0.);
      return me;
    }
  }

  return book(dir, name, "bookFloat")
    ->initialise(MonitorElement::DQM_KIND_REAL);
}

/// Book float.
MonitorElement *
DQMStore::bookFloat(const char *name)
{ return bookFloat(pwd_, name); }

/// Book float.
MonitorElement *
DQMStore::bookFloat(const std::string &name)
{
  return bookFloat(pwd_, name);
}

// -------------------------------------------------------------------
/// Book string.
MonitorElement *
DQMStore::bookString(const std::string &dir,
                     const std::string &name,
                     const std::string &value)
{
  if (collateHistograms_)
  {
    if (MonitorElement *me = findObject(dir, name))
      return me;
  }

  return book(dir, name, "bookString")
    ->initialise(MonitorElement::DQM_KIND_STRING, value);
}

/// Book string.
MonitorElement *
DQMStore::bookString(const char *name, const char *value)
{ return bookString(pwd_, name, value); }

/// Book string.
MonitorElement *
DQMStore::bookString(const std::string &name, const std::string &value)
{
  return bookString(pwd_, name, value);
}

// -------------------------------------------------------------------
/// Book 1D histogram based on TH1F.
MonitorElement *
DQMStore::book1D(const std::string &dir, const std::string &name, TH1F *h)
{
  return book(dir, name, "book1D", MonitorElement::DQM_KIND_TH1F, h, collate1D);
}

/// Book 1D histogram based on TH1S.
MonitorElement *
DQMStore::book1S(const std::string &dir, const std::string &name, TH1S *h)
{
  return book(dir, name, "book1S", MonitorElement::DQM_KIND_TH1S, h, collate1S);
}

/// Book 1D histogram based on TH1D.
MonitorElement *
DQMStore::book1DD(const std::string &dir, const std::string &name, TH1D *h)
{
  return book(dir, name, "book1DD", MonitorElement::DQM_KIND_TH1D, h, collate1DD);
}

/// Book 1D histogram.
MonitorElement *
DQMStore::book1D(const char *name, const char *title,
                 int nchX, double lowX, double highX)
{
  return book1D(pwd_, name, new TH1F(name, title, nchX, lowX, highX));
}

/// Book 1D histogram.
MonitorElement *
DQMStore::book1D(const std::string &name, const std::string &title,
                 int nchX, double lowX, double highX)
{
  return book1D(pwd_, name, new TH1F(name.c_str(), title.c_str(), nchX, lowX, highX));
}

/// Book 1S histogram.
MonitorElement *
DQMStore::book1S(const char *name, const char *title,
                 int nchX, double lowX, double highX)
{
  return book1S(pwd_, name, new TH1S(name, title, nchX, lowX, highX));
}

/// Book 1S histogram.
MonitorElement *
DQMStore::book1S(const std::string &name, const std::string &title,
                 int nchX, double lowX, double highX)
{
  return book1S(pwd_, name, new TH1S(name.c_str(), title.c_str(), nchX, lowX, highX));
}

/// Book 1S histogram.
MonitorElement *
DQMStore::book1DD(const char *name, const char *title,
                  int nchX, double lowX, double highX)
{
  return book1DD(pwd_, name, new TH1D(name, title, nchX, lowX, highX));
}

/// Book 1S histogram.
MonitorElement *
DQMStore::book1DD(const std::string &name, const std::string &title,
                  int nchX, double lowX, double highX)
{
  return book1DD(pwd_, name, new TH1D(name.c_str(), title.c_str(), nchX, lowX, highX));
}

/// Book 1D variable bin histogram.
MonitorElement *
DQMStore::book1D(const char *name, const char *title,
                 int nchX, float *xbinsize)
{
  return book1D(pwd_, name, new TH1F(name, title, nchX, xbinsize));
}

/// Book 1D variable bin histogram.
MonitorElement *
DQMStore::book1D(const std::string &name, const std::string &title,
                 int nchX, float *xbinsize)
{
  return book1D(pwd_, name, new TH1F(name.c_str(), title.c_str(), nchX, xbinsize));
}

/// Book 1D histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book1D(const char *name, TH1F *source)
{
  return book1D(pwd_, name, static_cast<TH1F *>(source->Clone(name)));
}

/// Book 1D histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book1D(const std::string &name, TH1F *source)
{
  return book1D(pwd_, name, static_cast<TH1F *>(source->Clone(name.c_str())));
}

/// Book 1S histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book1S(const char *name, TH1S *source)
{
  return book1S(pwd_, name, static_cast<TH1S *>(source->Clone(name)));
}

/// Book 1S histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book1S(const std::string &name, TH1S *source)
{
  return book1S(pwd_, name, static_cast<TH1S *>(source->Clone(name.c_str())));
}

/// Book 1D double histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book1DD(const char *name, TH1D *source)
{
  return book1DD(pwd_, name, static_cast<TH1D *>(source->Clone(name)));
}

/// Book 1D double histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book1DD(const std::string &name, TH1D *source)
{
  return book1DD(pwd_, name, static_cast<TH1D *>(source->Clone(name.c_str())));
}

// -------------------------------------------------------------------
/// Book 2D histogram based on TH2F.
MonitorElement *
DQMStore::book2D(const std::string &dir, const std::string &name, TH2F *h)
{
  return book(dir, name, "book2D", MonitorElement::DQM_KIND_TH2F, h, collate2D);
}

/// Book 2D histogram based on TH2S.
MonitorElement *
DQMStore::book2S(const std::string &dir, const std::string &name, TH2S *h)
{
  return book(dir, name, "book2S", MonitorElement::DQM_KIND_TH2S, h, collate2S);
}

/// Book 2D histogram based on TH2D.
MonitorElement *
DQMStore::book2DD(const std::string &dir, const std::string &name, TH2D *h)
{
  return book(dir, name, "book2DD", MonitorElement::DQM_KIND_TH2D, h, collate2DD);
}

/// Book 2D histogram.
MonitorElement *
DQMStore::book2D(const char *name, const char *title,
                 int nchX, double lowX, double highX,
                 int nchY, double lowY, double highY)
{
  return book2D(pwd_, name, new TH2F(name, title,
                                     nchX, lowX, highX,
                                     nchY, lowY, highY));
}

/// Book 2D histogram.
MonitorElement *
DQMStore::book2D(const std::string &name, const std::string &title,
                 int nchX, double lowX, double highX,
                 int nchY, double lowY, double highY)
{
  return book2D(pwd_, name, new TH2F(name.c_str(), title.c_str(),
                                     nchX, lowX, highX,
                                     nchY, lowY, highY));
}

/// Book 2S histogram.
MonitorElement *
DQMStore::book2S(const char *name, const char *title,
                 int nchX, double lowX, double highX,
                 int nchY, double lowY, double highY)
{
  return book2S(pwd_, name, new TH2S(name, title,
                                     nchX, lowX, highX,
                                     nchY, lowY, highY));
}

/// Book 2S histogram.
MonitorElement *
DQMStore::book2S(const std::string &name, const std::string &title,
                 int nchX, double lowX, double highX,
                 int nchY, double lowY, double highY)
{
  return book2S(pwd_, name, new TH2S(name.c_str(), title.c_str(),
                                     nchX, lowX, highX,
                                     nchY, lowY, highY));
}

/// Book 2D double histogram.
MonitorElement *
DQMStore::book2DD(const char *name, const char *title,
                  int nchX, double lowX, double highX,
                  int nchY, double lowY, double highY)
{
  return book2DD(pwd_, name, new TH2D(name, title,
                                      nchX, lowX, highX,
                                      nchY, lowY, highY));
}

/// Book 2S histogram.
MonitorElement *
DQMStore::book2DD(const std::string &name, const std::string &title,
                  int nchX, double lowX, double highX,
                  int nchY, double lowY, double highY)
{
  return book2DD(pwd_, name, new TH2D(name.c_str(), title.c_str(),
                                      nchX, lowX, highX,
                                      nchY, lowY, highY));
}

/// Book 2D variable bin histogram.
MonitorElement *
DQMStore::book2D(const char *name, const char *title,
                 int nchX, float *xbinsize, int nchY, float *ybinsize)
{
  return book2D(pwd_, name, new TH2F(name, title,
                                     nchX, xbinsize, nchY, ybinsize));
}

/// Book 2D variable bin histogram.
MonitorElement *
DQMStore::book2D(const std::string &name, const std::string &title,
                 int nchX, float *xbinsize, int nchY, float *ybinsize)
{
  return book2D(pwd_, name, new TH2F(name.c_str(), title.c_str(),
                                     nchX, xbinsize, nchY, ybinsize));
}

/// Book 2D histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book2D(const char *name, TH2F *source)
{
  return book2D(pwd_, name, static_cast<TH2F *>(source->Clone(name)));
}

/// Book 2D histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book2D(const std::string &name, TH2F *source)
{
  return book2D(pwd_, name, static_cast<TH2F *>(source->Clone(name.c_str())));
}

/// Book 2DS histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book2S(const char *name, TH2S *source)
{
  return book2S(pwd_, name, static_cast<TH2S *>(source->Clone(name)));
}

/// Book 2DS histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book2S(const std::string &name, TH2S *source)
{
  return book2S(pwd_, name, static_cast<TH2S *>(source->Clone(name.c_str())));
}

/// Book 2DS histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book2DD(const char *name, TH2D *source)
{
  return book2DD(pwd_, name, static_cast<TH2D *>(source->Clone(name)));
}

/// Book 2DS histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book2DD(const std::string &name, TH2D *source)
{
  return book2DD(pwd_, name, static_cast<TH2D *>(source->Clone(name.c_str())));
}

// -------------------------------------------------------------------
/// Book 3D histogram based on TH3F.
MonitorElement *
DQMStore::book3D(const std::string &dir, const std::string &name, TH3F *h)
{
  return book(dir, name, "book3D", MonitorElement::DQM_KIND_TH3F, h, collate3D);
}

/// Book 3D histogram.
MonitorElement *
DQMStore::book3D(const char *name, const char *title,
                 int nchX, double lowX, double highX,
                 int nchY, double lowY, double highY,
                 int nchZ, double lowZ, double highZ)
{
  return book3D(pwd_, name, new TH3F(name, title,
                                     nchX, lowX, highX,
                                     nchY, lowY, highY,
                                     nchZ, lowZ, highZ));
}

/// Book 3D histogram.
MonitorElement *
DQMStore::book3D(const std::string &name, const std::string &title,
                 int nchX, double lowX, double highX,
                 int nchY, double lowY, double highY,
                 int nchZ, double lowZ, double highZ)
{
  return book3D(pwd_, name, new TH3F(name.c_str(), title.c_str(),
                                     nchX, lowX, highX,
                                     nchY, lowY, highY,
                                     nchZ, lowZ, highZ));
}

/// Book 3D histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book3D(const char *name, TH3F *source)
{
  return book3D(pwd_, name, static_cast<TH3F *>(source->Clone(name)));
}

/// Book 3D histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book3D(const std::string &name, TH3F *source)
{
  return book3D(pwd_, name, static_cast<TH3F *>(source->Clone(name.c_str())));
}

// -------------------------------------------------------------------
/// Book profile histogram based on TProfile.
MonitorElement *
DQMStore::bookProfile(const std::string &dir, const std::string &name, TProfile *h)
{
  return book(dir, name, "bookProfile",
              MonitorElement::DQM_KIND_TPROFILE,
              h, collateProfile);
}

/// Book profile.  Option is one of: " ", "s" (default), "i", "G" (see
/// TProfile::BuildOptions).  The number of channels in Y is
/// disregarded in a profile plot.
MonitorElement *
DQMStore::bookProfile(const char *name, const char *title,
                      int nchX, double lowX, double highX,
                      int /* nchY */, double lowY, double highY,
                      const char *option /* = "s" */)
{
  return bookProfile(pwd_, name, new TProfile(name, title,
                                              nchX, lowX, highX,
                                              lowY, highY,
                                              option));
}

/// Book profile.  Option is one of: " ", "s" (default), "i", "G" (see
/// TProfile::BuildOptions).  The number of channels in Y is
/// disregarded in a profile plot.
MonitorElement *
DQMStore::bookProfile(const std::string &name, const std::string &title,
                      int nchX, double lowX, double highX,
                      int /* nchY */, double lowY, double highY,
                      const char *option /* = "s" */)
{
  return bookProfile(pwd_, name, new TProfile(name.c_str(), title.c_str(),
                                              nchX, lowX, highX,
                                              lowY, highY,
                                              option));
}

/// Book profile.  Option is one of: " ", "s" (default), "i", "G" (see
/// TProfile::BuildOptions).  The number of channels in Y is
/// disregarded in a profile plot.
MonitorElement *
DQMStore::bookProfile(const char *name, const char *title,
                      int nchX, double lowX, double highX,
                      double lowY, double highY,
                      const char *option /* = "s" */)
{
  return bookProfile(pwd_, name, new TProfile(name, title,
                                              nchX, lowX, highX,
                                              lowY, highY,
                                              option));
}

/// Book profile.  Option is one of: " ", "s" (default), "i", "G" (see
/// TProfile::BuildOptions).  The number of channels in Y is
/// disregarded in a profile plot.
MonitorElement *
DQMStore::bookProfile(const std::string &name, const std::string &title,
                      int nchX, double lowX, double highX,
                      double lowY, double highY,
                      const char *option /* = "s" */)
{
  return bookProfile(pwd_, name, new TProfile(name.c_str(), title.c_str(),
                                              nchX, lowX, highX,
                                              lowY, highY,
                                              option));
}

/// Book variable bin profile.  Option is one of: " ", "s" (default), "i", "G" (see
/// TProfile::BuildOptions).  The number of channels in Y is
/// disregarded in a profile plot.
MonitorElement *
DQMStore::bookProfile(const char *name, const char *title,
                      int nchX, double *xbinsize,
                      int /* nchY */, double lowY, double highY,
                      const char *option /* = "s" */)
{
  return bookProfile(pwd_, name, new TProfile(name, title,
                                              nchX, xbinsize,
                                              lowY, highY,
                                              option));
}

/// Book variable bin profile.  Option is one of: " ", "s" (default), "i", "G" (see
/// TProfile::BuildOptions).  The number of channels in Y is
/// disregarded in a profile plot.
MonitorElement *
DQMStore::bookProfile(const std::string &name, const std::string &title,
                      int nchX, double *xbinsize,
                      int /* nchY */, double lowY, double highY,
                      const char *option /* = "s" */)
{
  return bookProfile(pwd_, name, new TProfile(name.c_str(), title.c_str(),
                                              nchX, xbinsize,
                                              lowY, highY,
                                              option));
}

/// Book variable bin profile.  Option is one of: " ", "s" (default), "i", "G" (see
/// TProfile::BuildOptions).  The number of channels in Y is
/// disregarded in a profile plot.
MonitorElement *
DQMStore::bookProfile(const char *name, const char *title,
                      int nchX, double *xbinsize,
                      double lowY, double highY,
                      const char *option /* = "s" */)
{
  return bookProfile(pwd_, name, new TProfile(name, title,
                                              nchX, xbinsize,
                                              lowY, highY,
                                              option));
}

/// Book variable bin profile.  Option is one of: " ", "s" (default), "i", "G" (see
/// TProfile::BuildOptions).  The number of channels in Y is
/// disregarded in a profile plot.
MonitorElement *
DQMStore::bookProfile(const std::string &name, const std::string &title,
                      int nchX, double *xbinsize,
                      double lowY, double highY,
                      const char *option /* = "s" */)
{
  return bookProfile(pwd_, name, new TProfile(name.c_str(), title.c_str(),
                                              nchX, xbinsize,
                                              lowY, highY,
                                              option));
}

/// Book TProfile by cloning an existing profile.
MonitorElement *
DQMStore::bookProfile(const char *name, TProfile *source)
{
  return bookProfile(pwd_, name, static_cast<TProfile *>(source->Clone(name)));
}

/// Book TProfile by cloning an existing profile.
MonitorElement *
DQMStore::bookProfile(const std::string &name, TProfile *source)
{
  return bookProfile(pwd_, name, static_cast<TProfile *>(source->Clone(name.c_str())));
}

// -------------------------------------------------------------------
/// Book 2D profile histogram based on TProfile2D.
MonitorElement *
DQMStore::bookProfile2D(const std::string &dir, const std::string &name, TProfile2D *h)
{
  return book(dir, name, "bookProfile2D",
              MonitorElement::DQM_KIND_TPROFILE2D,
              h, collateProfile2D);
}

/// Book 2-D profile.  Option is one of: " ", "s" (default), "i", "G"
/// (see TProfile2D::BuildOptions).  The number of channels in Z is
/// disregarded in a 2-D profile.
MonitorElement *
DQMStore::bookProfile2D(const char *name, const char *title,
                        int nchX, double lowX, double highX,
                        int nchY, double lowY, double highY,
                        int /* nchZ */, double lowZ, double highZ,
                        const char *option /* = "s" */)
{
  return bookProfile2D(pwd_, name, new TProfile2D(name, title,
                                                  nchX, lowX, highX,
                                                  nchY, lowY, highY,
                                                  lowZ, highZ,
                                                  option));
}

/// Book 2-D profile.  Option is one of: " ", "s" (default), "i", "G"
/// (see TProfile2D::BuildOptions).  The number of channels in Z is
/// disregarded in a 2-D profile.
MonitorElement *
DQMStore::bookProfile2D(const std::string &name, const std::string &title,
                        int nchX, double lowX, double highX,
                        int nchY, double lowY, double highY,
                        int /* nchZ */, double lowZ, double highZ,
                        const char *option /* = "s" */)
{
  return bookProfile2D(pwd_, name, new TProfile2D(name.c_str(), title.c_str(),
                                                  nchX, lowX, highX,
                                                  nchY, lowY, highY,
                                                  lowZ, highZ,
                                                  option));
}

/// Book 2-D profile.  Option is one of: " ", "s" (default), "i", "G"
/// (see TProfile2D::BuildOptions).  The number of channels in Z is
/// disregarded in a 2-D profile.
MonitorElement *
DQMStore::bookProfile2D(const char *name, const char *title,
                        int nchX, double lowX, double highX,
                        int nchY, double lowY, double highY,
                        double lowZ, double highZ,
                        const char *option /* = "s" */)
{
  return bookProfile2D(pwd_, name, new TProfile2D(name, title,
                                                  nchX, lowX, highX,
                                                  nchY, lowY, highY,
                                                  lowZ, highZ,
                                                  option));
}

/// Book 2-D profile.  Option is one of: " ", "s" (default), "i", "G"
/// (see TProfile2D::BuildOptions).  The number of channels in Z is
/// disregarded in a 2-D profile.
MonitorElement *
DQMStore::bookProfile2D(const std::string &name, const std::string &title,
                        int nchX, double lowX, double highX,
                        int nchY, double lowY, double highY,
                        double lowZ, double highZ,
                        const char *option /* = "s" */)
{
  return bookProfile2D(pwd_, name, new TProfile2D(name.c_str(), title.c_str(),
                                                  nchX, lowX, highX,
                                                  nchY, lowY, highY,
                                                  lowZ, highZ,
                                                  option));
}

/// Book TProfile2D by cloning an existing profile.
MonitorElement *
DQMStore::bookProfile2D(const char *name, TProfile2D *source)
{
  return bookProfile2D(pwd_, name, static_cast<TProfile2D *>(source->Clone(name)));
}

/// Book TProfile2D by cloning an existing profile.
MonitorElement *
DQMStore::bookProfile2D(const std::string &name, TProfile2D *source)
{
  return bookProfile2D(pwd_, name, static_cast<TProfile2D *>(source->Clone(name.c_str())));
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
bool
DQMStore::checkBinningMatches(MonitorElement *me, TH1 *h)
{
  if (me->getTH1()->GetNbinsX() != h->GetNbinsX()
      || me->getTH1()->GetNbinsY() != h->GetNbinsY()
      || me->getTH1()->GetNbinsZ() != h->GetNbinsZ()
      || me->getTH1()->GetXaxis()->GetXmin() != h->GetXaxis()->GetXmin()
      || me->getTH1()->GetYaxis()->GetXmin() != h->GetYaxis()->GetXmin()
      || me->getTH1()->GetZaxis()->GetXmin() != h->GetZaxis()->GetXmin()
      || me->getTH1()->GetXaxis()->GetXmax() != h->GetXaxis()->GetXmax()
      || me->getTH1()->GetYaxis()->GetXmax() != h->GetYaxis()->GetXmax()
      || me->getTH1()->GetZaxis()->GetXmax() != h->GetZaxis()->GetXmax())
  {
    //  edm::LogWarning ("DQMStore")
    std::cout << "*** DQMStore: WARNING:"
              << "checkBinningMatches: different binning - cannot add object '"
              << h->GetName() << "' of type "
              << h->IsA()->GetName() << " to existing ME: '"
              << me->getFullname() << "'\n";
    return false;
  }
  return true;
}

void
DQMStore::collate1D(MonitorElement *me, TH1F *h)
{
  if (checkBinningMatches(me,h))
    me->getTH1F()->Add(h);
}

void
DQMStore::collate1S(MonitorElement *me, TH1S *h)
{
  if (checkBinningMatches(me,h))
    me->getTH1S()->Add(h);
}

void
DQMStore::collate1DD(MonitorElement *me, TH1D *h)
{
  if (checkBinningMatches(me,h))
    me->getTH1D()->Add(h);
}

void
DQMStore::collate2D(MonitorElement *me, TH2F *h)
{
  if (checkBinningMatches(me,h))
    me->getTH2F()->Add(h);
}

void
DQMStore::collate2S(MonitorElement *me, TH2S *h)
{
  if (checkBinningMatches(me,h))
    me->getTH2S()->Add(h);
}

void
DQMStore::collate2DD(MonitorElement *me, TH2D *h)
{
  if (checkBinningMatches(me,h))
    me->getTH2D()->Add(h);
}

void
DQMStore::collate3D(MonitorElement *me, TH3F *h)
{
  if (checkBinningMatches(me,h))
    me->getTH3F()->Add(h);
}

void
DQMStore::collateProfile(MonitorElement *me, TProfile *h)
{
  if (checkBinningMatches(me,h))
  {
    TProfile *meh = me->getTProfile();
    me->addProfiles(h, meh, meh, 1, 1);
  }
}

void
DQMStore::collateProfile2D(MonitorElement *me, TProfile2D *h)
{
  if (checkBinningMatches(me,h))
  {
    TProfile2D *meh = me->getTProfile2D();
    me->addProfiles(h, meh, meh, 1, 1);
  }
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// tag ME as <myTag> (myTag > 0)
void
DQMStore::tag(MonitorElement *me, unsigned int myTag)
{
  if (! myTag)
    raiseDQMError("DQMStore", "Attempt to tag monitor element '%s'"
                  " with a zero tag", me->getFullname().c_str());
  if ((me->data_.flags & DQMNet::DQM_PROP_TAGGED) && myTag != me->data_.tag)
    raiseDQMError("DQMStore", "Attempt to tag monitor element '%s'"
                  " twice with multiple tags", me->getFullname().c_str());

  me->data_.tag = myTag;
  me->data_.flags |= DQMNet::DQM_PROP_TAGGED;
}

/// tag ME specified by full pathname (e.g. "my/long/dir/my_histo")
void
DQMStore::tag(const std::string &path, unsigned int myTag)
{
  std::string dir;
  std::string name;
  splitPath(dir, name, path);

  if (MonitorElement *me = findObject(dir, name))
    tag(me, myTag);
  else
    raiseDQMError("DQMStore", "Attempt to tag non-existent monitor element"
                  " '%s' with tag %u", path.c_str(), myTag);

}

/// tag all children of folder (does NOT include subfolders)
void
DQMStore::tagContents(const std::string &path, unsigned int myTag)
{
  MonitorElement proto(&path, std::string());
  MEMap::iterator e = data_.end();
  MEMap::iterator i = data_.lower_bound(proto);
  for ( ; i != e && path == *i->data_.dirname; ++i)
    tag(const_cast<MonitorElement *>(&*i), myTag);
}

/// tag all children of folder, including all subfolders and their children;
/// path must be an exact path name
void
DQMStore::tagAllContents(const std::string &path, unsigned int myTag)
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(path, clean, cleaned);
  MonitorElement proto(cleaned, std::string());

  // FIXME: WILDCARDS? Old one supported them, but nobody seemed to use them.
  MEMap::iterator e = data_.end();
  MEMap::iterator i = data_.lower_bound(proto);
  while (i != e && isSubdirectory(*cleaned, *i->data_.dirname))
  {
    tag(const_cast<MonitorElement *>(&*i), myTag);
    ++i;
  }
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// get list of subdirectories of current directory
std::vector<std::string>
DQMStore::getSubdirs(void) const
{
  std::vector<std::string> result;
  std::set<std::string>::const_iterator e = dirs_.end();
  std::set<std::string>::const_iterator i = dirs_.find(pwd_);

  // If we didn't find current directory, the tree is empty, so quit.
  if (i == e)
    return result;

  // Skip the current directory and then start looking for immediate
  // subdirectories in the dirs_ list.  Stop when we are no longer in
  // (direct or indirect) subdirectories of pwd_.  Note that we don't
  // "know" which order the set will sort A/B, A/B/C and A/D.
  while (++i != e && isSubdirectory(pwd_, *i))
    if (i->find('/', pwd_.size()+1) == std::string::npos)
      result.push_back(*i);

  return result;
}

/// get list of (non-dir) MEs of current directory
std::vector<std::string>
DQMStore::getMEs(void) const
{
  MonitorElement proto(&pwd_, std::string());
  std::vector<std::string> result;
  MEMap::const_iterator e = data_.end();
  MEMap::const_iterator i = data_.lower_bound(proto);
  for ( ; i != e && isSubdirectory(pwd_, *i->data_.dirname); ++i)
    if (pwd_ == *i->data_.dirname)
      result.push_back(i->getName());

  return result;
}

/// true if directory (or any subfolder at any level below it) contains
/// at least one monitorable element
bool
DQMStore::containsAnyMonitorable(const std::string &path) const
{
  MonitorElement proto(&path, std::string());
  MEMap::const_iterator e = data_.end();
  MEMap::const_iterator i = data_.lower_bound(proto);
  return (i != e && isSubdirectory(path, *i->data_.dirname));
}

/// get ME from full pathname (e.g. "my/long/dir/my_histo")
MonitorElement *
DQMStore::get(const std::string &path) const
{
  std::string dir;
  std::string name;
  splitPath(dir, name, path);
  MonitorElement proto(&dir, name);
  MEMap::const_iterator mepos = data_.find(proto);
  return (mepos == data_.end() ? 0
          : const_cast<MonitorElement *>(&*mepos));
}

/// get all MonitorElements tagged as <tag>
std::vector<MonitorElement *>
DQMStore::get(unsigned int tag) const
{
  // FIXME: Use reverse map [tag -> path] / [tag -> dir]?
  std::vector<MonitorElement *> result;
  for (MEMap::const_iterator i = data_.begin(), e = data_.end(); i != e; ++i)
  {
    const MonitorElement &me = *i;
    if ((me.data_.flags & DQMNet::DQM_PROP_TAGGED) && me.data_.tag == tag)
      result.push_back(const_cast<MonitorElement *>(&me));
  }
  return result;
}

/// get vector with all children of folder
/// (does NOT include contents of subfolders)
std::vector<MonitorElement *>
DQMStore::getContents(const std::string &path) const
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(path, clean, cleaned);
  MonitorElement proto(cleaned, std::string());

  std::vector<MonitorElement *> result;
  MEMap::const_iterator e = data_.end();
  MEMap::const_iterator i = data_.lower_bound(proto);
  for ( ; i != e && isSubdirectory(*cleaned, *i->data_.dirname); ++i)
    if (*cleaned == *i->data_.dirname)
      result.push_back(const_cast<MonitorElement *>(&*i));

  return result;
}

/// same as above for tagged MonitorElements
std::vector<MonitorElement *>
DQMStore::getContents(const std::string &path, unsigned int tag) const
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(path, clean, cleaned);
  MonitorElement proto(cleaned, std::string());

  std::vector<MonitorElement *> result;
  MEMap::const_iterator e = data_.end();
  MEMap::const_iterator i = data_.lower_bound(proto);
  for ( ; i != e && isSubdirectory(*cleaned, *i->data_.dirname); ++i)
    if (*cleaned == *i->data_.dirname
        && (i->data_.flags & DQMNet::DQM_PROP_TAGGED)
        && i->data_.tag == tag)
      result.push_back(const_cast<MonitorElement *>(&*i));

  return result;
}

/// get contents;
/// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
/// if showContents = false, change form to <dir pathname>:
/// (useful for subscription requests; meant to imply "all contents")
void
DQMStore::getContents(std::vector<std::string> &into, bool showContents /* = true */) const
{
  into.clear();
  into.reserve(dirs_.size());

  MEMap::const_iterator me = data_.end();
  std::set<std::string>::const_iterator di = dirs_.begin();
  std::set<std::string>::const_iterator de = dirs_.end();
  for ( ; di != de; ++di)
  {
    MonitorElement proto(&*di, std::string());
    MEMap::const_iterator mi = data_.lower_bound(proto);
    MEMap::const_iterator m = mi;
    size_t sz = di->size() + 2;
    size_t nfound = 0;
    for ( ; m != me && isSubdirectory(*di, *m->data_.dirname); ++m)
      if (*di == *m->data_.dirname)
      {
        sz += m->data_.objname.size() + 1;
        ++nfound;
      }

    if (! nfound)
      continue;

    std::vector<std::string>::iterator istr
      = into.insert(into.end(), std::string());

    if (showContents)
    {
      istr->reserve(sz);

      *istr += *di;
      *istr += ':';
      for (sz = 0; mi != m; ++mi)
      {
        if (*di != *mi->data_.dirname)
          continue;

        if (sz > 0)
          *istr += ',';

        *istr += mi->data_.objname;
        ++sz;
      }
    }
    else
    {
      istr->reserve(di->size() + 2);
      *istr += *di;
      *istr += ':';
    }
  }
}

/// get MonitorElement <name> in directory <dir>
/// (null if MonitorElement does not exist)
MonitorElement *
DQMStore::findObject(const std::string &dir,
                     const std::string &name,
                     const uint32_t run /* = 0 */,
                     const uint32_t lumi /* = 0 */,
                     const uint32_t streamId /* = 0 */,
                     const uint32_t moduleId /* = 0 */) const
{
  if (dir.find_first_not_of(s_safe) != std::string::npos)
    raiseDQMError("DQMStore", "Monitor element path name '%s' uses"
                  " unacceptable characters", dir.c_str());
  if (name.find_first_not_of(s_safe) != std::string::npos)
    raiseDQMError("DQMStore", "Monitor element path name '%s' uses"
                  " unacceptable characters", name.c_str());

  MonitorElement proto;
  proto.data_.dirname  = &dir;
  proto.data_.objname  = name;
  proto.data_.run      = run;
  proto.data_.lumi     = lumi;
  proto.data_.streamId = streamId;
  proto.data_.moduleId = moduleId;

  MEMap::const_iterator mepos = data_.find(proto);
  return (mepos == data_.end() ? 0
          : const_cast<MonitorElement *>(&*mepos));
}

/** get tags for various maps, return vector with strings of the form
    <dir pathname>:<obj1>/<tag1>/<tag2>,<obj2>/<tag1>/<tag3>, etc. */
void
DQMStore::getAllTags(std::vector<std::string> &into) const
{
  into.clear();
  into.reserve(dirs_.size());

  MEMap::const_iterator me = data_.end();
  std::set<std::string>::const_iterator di = dirs_.begin();
  std::set<std::string>::const_iterator de = dirs_.end();
  char tagbuf[32]; // more than enough for '/' and up to 10 digits

  for ( ; di != de; ++di)
  {
    MonitorElement proto(&*di, std::string());
    MEMap::const_iterator mi = data_.lower_bound(proto);
    MEMap::const_iterator m = mi;
    size_t sz = di->size() + 2;
    size_t nfound = 0;
    for ( ; m != me && isSubdirectory(*di, *m->data_.dirname); ++m)
      if (*di == *m->data_.dirname && (m->data_.flags & DQMNet::DQM_PROP_TAGGED))
      {
        // the tags count for '/' + up to 10 digits, otherwise ',' + ME name
        sz += 1 + m->data_.objname.size() + 11;
        ++nfound;
      }

    if (! nfound)
      continue;

    std::vector<std::string>::iterator istr
      = into.insert(into.end(), std::string());

    istr->reserve(sz);

    *istr += *di;
    *istr += ':';
    for (sz = 0; mi != m; ++mi)
    {
      if (*di == *m->data_.dirname && (m->data_.flags & DQMNet::DQM_PROP_TAGGED))
      {
        sprintf(tagbuf, "/%u", mi->data_.tag);
        if (sz > 0)
          *istr += ',';
        *istr += m->data_.objname;
        *istr += tagbuf;
        ++sz;
      }
    }
  }
}

/// get vector with children of folder, including all subfolders + their children;
/// must use an exact pathname
std::vector<MonitorElement*>
DQMStore::getAllContents(const std::string &path,
                         uint32_t runNumber /* = 0 */,
                         uint32_t lumi /* = 0 */) const
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(path, clean, cleaned);
  MonitorElement proto(cleaned, std::string(), runNumber);
  proto.setLumi(lumi);

  std::vector<MonitorElement *> result;
  MEMap::const_iterator e = data_.end();
  MEMap::const_iterator i = data_.lower_bound(proto);
  for ( ; i != e && isSubdirectory(*cleaned, *i->data_.dirname); ++i) {
    if (runNumber != 0) {
      if (i->data_.run > runNumber // TODO[rovere]: pleonastic? first we encounter local ME of the same run ...
          || i->data_.streamId != 0
          || i->data_.moduleId != 0)
        break;
    }
    if (lumi != 0) {
      if (i->data_.lumi > lumi
          || i->data_.streamId != 0
          || i->data_.moduleId != 0)
        break;
    }
    if (runNumber != 0 or lumi !=0) {
      assert(i->data_.streamId == 0);
      assert(i->data_.moduleId == 0);
    }
    result.push_back(const_cast<MonitorElement *>(&*i));
  }
  return result;
}

/// get vector with children of folder, including all subfolders + their children;
/// matches names against a wildcard pattern matched against the full ME path
std::vector<MonitorElement*>
DQMStore::getMatchingContents(const std::string &pattern, lat::Regexp::Syntax syntaxType /* = Wildcard */) const
{
  lat::Regexp rx;
  try
  {
    rx = lat::Regexp(pattern, 0, syntaxType);
    rx.study();
  }
  catch (lat::Error &e)
  {
    raiseDQMError("DQMStore", "Invalid regular expression '%s': %s",
                  pattern.c_str(), e.explain().c_str());
  }

  std::string path;
  std::vector<MonitorElement *> result;
  MEMap::const_iterator i = data_.begin();
  MEMap::const_iterator e = data_.end();
  for ( ; i != e; ++i)
  {
    path.clear();
    mergePath(path, *i->data_.dirname, i->data_.objname);
    if (rx.match(path))
      result.push_back(const_cast<MonitorElement *>(&*i));
  }

  return result;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** Invoke this method after flushing all recently changed monitoring.
    Clears updated flag on all recently updated MEs and calls their
    Reset() method for those that have resetMe = true. */
void
DQMStore::reset(void)
{
  MEMap::iterator mi = data_.begin();
  MEMap::iterator me = data_.end();
  for ( ; mi != me; ++mi)
  {
    MonitorElement &me = const_cast<MonitorElement &>(*mi);
    if (mi->wasUpdated())
    {
      if (me.resetMe())
        me.Reset();
      me.resetUpdate();
    }
  }

  reset_ = true;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** Invoke this method after flushing all recently changed monitoring.
    Clears updated flag on all MEs and calls their Reset() method. */
void
DQMStore::forceReset(void)
{
  MEMap::iterator mi = data_.begin();
  MEMap::iterator me = data_.end();
  for ( ; mi != me; ++mi)
  {
    MonitorElement &me = const_cast<MonitorElement &>(*mi);
    me.Reset();
    me.resetUpdate();
  }

  reset_ = true;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// extract object (TH1F, TH2F, ...) from <to>; return success flag
/// flag fromRemoteNode indicating if ME arrived from different node
bool
DQMStore::extract(TObject *obj, const std::string &dir, bool overwrite)
{
  // NB: Profile histograms inherit from TH*D, checking order matters.
  MonitorElement *refcheck = 0;
  if (TProfile *h = dynamic_cast<TProfile *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName());
    if (! me)
      me = bookProfile(dir, h->GetName(), (TProfile *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collateProfile(me, h);
    refcheck = me;
  }
  else if (TProfile2D *h = dynamic_cast<TProfile2D *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName());
    if (! me)
      me = bookProfile2D(dir, h->GetName(), (TProfile2D *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collateProfile2D(me, h);
    refcheck = me;
  }
  else if (TH1F *h = dynamic_cast<TH1F *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName());
    if (! me)
      me = book1D(dir, h->GetName(), (TH1F *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate1D(me, h);
    refcheck = me;
  }
  else if (TH1S *h = dynamic_cast<TH1S *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName());
    if (! me)
      me = book1S(dir, h->GetName(), (TH1S *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate1S(me, h);
    refcheck = me;
  }
  else if (TH1D *h = dynamic_cast<TH1D *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName());
    if (! me)
      me = book1DD(dir, h->GetName(), (TH1D *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate1DD(me, h);
    refcheck = me;
  }
  else if (TH2F *h = dynamic_cast<TH2F *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName());
    if (! me)
      me = book2D(dir, h->GetName(), (TH2F *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate2D(me, h);
    refcheck = me;
  }
  else if (TH2S *h = dynamic_cast<TH2S *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName());
    if (! me)
      me = book2S(dir, h->GetName(), (TH2S *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate2S(me, h);
    refcheck = me;
  }
  else if (TH2D *h = dynamic_cast<TH2D *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName());
    if (! me)
      me = book2DD(dir, h->GetName(), (TH2D *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate2DD(me, h);
    refcheck = me;
  }
  else if (TH3F *h = dynamic_cast<TH3F *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName());
    if (! me)
      me = book3D(dir, h->GetName(), (TH3F *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate3D(me, h);
    refcheck = me;
  }
  else if (dynamic_cast<TObjString *>(obj))
  {
    lat::RegexpMatch m;
    if (! s_rxmeval.match(obj->GetName(), 0, 0, &m))
    {
      if (strstr(obj->GetName(), "CMSSW"))
      {
        if (verbose_)
          std::cout << "Input file version: " << obj->GetName() << std::endl;
        return true;
      }
      else if (strstr(obj->GetName(), "DQMPATCH"))
      {
        if (verbose_)
          std::cout << "DQM patch version: " << obj->GetName() << std::endl;
        return true;
      }
      else
      {
        std::cout << "*** DQMStore: WARNING: cannot extract object '"
                  << obj->GetName() << "' of type '"
                  << obj->IsA()->GetName() << "'\n";
        return false;
      }
    }

    std::string label = m.matchString(obj->GetName(), 1);
    std::string kind = m.matchString(obj->GetName(), 2);
    std::string value = m.matchString(obj->GetName(), 3);

    if (kind == "i")
    {
      MonitorElement *me = findObject(dir, label);
      if (! me || overwrite)
      {
        if (! me) me = bookInt(dir, label);
        me->Fill(atoll(value.c_str()));
      }
    }
    else if (kind == "f")
    {
      MonitorElement *me = findObject(dir, label);
      if (! me || overwrite)
      {
        if (! me) me = bookFloat(dir, label);
        me->Fill(atof(value.c_str()));
      }
    }
    else if (kind == "s")
    {
      MonitorElement *me = findObject(dir, label);
      if (! me)
        me = bookString(dir, label, value);
      else if (overwrite)
        me->Fill(value);
    }
    else if (kind == "e")
    {
      MonitorElement *me = findObject(dir, label);
      if (! me)
      {
        std::cout << "*** DQMStore: WARNING: no monitor element '"
                  << label << "' in directory '"
                  << dir << "' to be marked as efficiency plot.\n";
        return false;
      }
      me->setEfficiencyFlag();
    }
    else if (kind == "t")
    {
      MonitorElement *me = findObject(dir, label);
      if (! me)
      {
        std::cout << "*** DQMStore: WARNING: no monitor element '"
                  << label << "' in directory '"
                  << dir << "' for a tag\n";
        return false;
      }
      errno = 0;
      char *endp = 0;
      unsigned long val = strtoul(value.c_str(), &endp, 10);
      if ((val == 0 && errno) || *endp || val > ~uint32_t(0))
      {
        std::cout << "*** DQMStore: WARNING: cannot restore tag '"
                  << value << "' for monitor element '"
                  << label << "' in directory '"
                  << dir << "' - invalid value\n";
        return false;
      }
      tag(me, val);
    }
    else if (kind == "qr")
    {
      // Handle qreports, but skip them while reading in references.
      if (! isSubdirectory(s_referenceDirName, dir))
      {
        size_t dot = label.find('.');
        if (dot == std::string::npos)
        {
          std::cout << "*** DQMStore: WARNING: quality report label in '" << label
                    << "' is missing a '.' and cannot be extracted\n";
          return false;
        }

        std::string mename (label, 0, dot);
        std::string qrname (label, dot+1, std::string::npos);

        m.reset();
        DQMNet::QValue qv;
        if (s_rxmeqr1.match(value, 0, 0, &m))
        {
          qv.code = atoi(m.matchString(value, 1).c_str());
          qv.qtresult = strtod(m.matchString(value, 2).c_str(), 0);
          qv.message = m.matchString(value, 4);
          qv.qtname = qrname;
          qv.algorithm = m.matchString(value, 3);
        }
        else if (s_rxmeqr2.match(value, 0, 0, &m))
        {
          qv.code = atoi(m.matchString(value, 1).c_str());
          qv.qtresult = 0; // unavailable in old format
          qv.message = m.matchString(value, 2);
          qv.qtname = qrname;
          // qv.algorithm unavailable in old format
        }
        else
        {
          std::cout << "*** DQMStore: WARNING: quality test value '"
                    << value << "' is incorrectly formatted\n";
          return false;
        }

        MonitorElement *me = findObject(dir, mename);
        if (! me)
        {
          std::cout << "*** DQMStore: WARNING: no monitor element '"
                    << mename << "' in directory '"
                    << dir << "' for quality test '"
                    << label << "'\n";
          return false;
        }

        me->addQReport(qv, /* FIXME: getQTest(qv.qtname)? */ 0);
      }
    }
    else
    {
      std::cout << "*** DQMStore: WARNING: cannot extract object '"
                << obj->GetName() << "' of type '"
                << obj->IsA()->GetName() << "'\n";
      return false;
    }
  }
  else if (TNamed *n = dynamic_cast<TNamed *>(obj))
  {
    // For old DQM data.
    std::string s;
    s.reserve(6 + strlen(n->GetTitle()) + 2*strlen(n->GetName()));
    s += '<'; s += n->GetName(); s += '>';
    s += n->GetTitle();
    s += '<'; s += '/'; s += n->GetName(); s += '>';
    TObjString os(s.c_str());
    return extract(&os, dir, overwrite);
  }
  else
  {
    std::cout << "*** DQMStore: WARNING: cannot extract object '"
              << obj->GetName() << "' of type '" << obj->IsA()->GetName()
              << "' and with title '" << obj->GetTitle() << "'\n";
    return false;
  }

  // If we just read in a reference monitor element, and there is a
  // monitor element with the same name, link the two together. The
  // other direction is handled by the initialise() method.
  if (refcheck && isSubdirectory(s_referenceDirName, dir))
  {
    std::string mdir(dir, s_referenceDirName.size()+1, std::string::npos);
    if (MonitorElement *master = findObject(mdir, obj->GetName()))
    {
      master->data_.flags |= DQMNet::DQM_PROP_HAS_REFERENCE;
      master->reference_ = refcheck->object_;
    }
  }

  return true;
}

/// Use this for saving monitoring objects in ROOT files with dir structure;
/// cd into directory (create first if it doesn't exist);
/// returns success flag
bool
DQMStore::cdInto(const std::string &path) const
{
  assert(! path.empty());

  // Find the first path component.
  size_t start = 0;
  size_t end = path.find('/', start);
  if (end == std::string::npos)
    end = path.size();

  while (true)
  {
    // Check if this subdirectory component exists.  If yes, make sure
    // it is actually a subdirectory.  Otherwise create or cd into it.
    std::string part(path, start, end-start);
    TObject *o = gDirectory->Get(part.c_str());
    if (o && ! dynamic_cast<TDirectory *>(o))
      raiseDQMError("DQMStore", "Attempt to create directory '%s' in a file"
                    " fails because the part '%s' already exists and is not"
                    " directory", path.c_str(), part.c_str());
    else if (! o)
      gDirectory->mkdir(part.c_str());

    if (! gDirectory->cd(part.c_str()))
      raiseDQMError("DQMStore", "Attempt to create directory '%s' in a file"
                    " fails because could not cd into subdirectory '%s'",
                    path.c_str(), part.c_str());

    // Stop if we reached the end, ignoring any trailing '/'.
    if (end+1 >= path.size())
      break;

    // Find the next path component.
    start = end+1;
    end = path.find('/', start);
    if (end == std::string::npos)
      end = path.size();
  }

  return true;
}

/// save directory with monitoring objects into root file <filename>;
/// include quality test results with status >= minimum_status
/// (defined in Core/interface/QTestStatus.h);
/// if directory="", save full monitoring structure
void
DQMStore::save(const std::string &filename,
               const std::string &path /* = "" */,
               const std::string &pattern /* = "" */,
               const std::string &rewrite /* = "" */,
               const uint32_t run /* = 0 */,
               SaveReferenceTag ref /* = SaveWithReference */,
               int minStatus /* = dqm::qstatus::STATUS_OK */,
               const std::string &fileupdate /* = RECREATE */)
{
  std::set<std::string>::iterator di, de;
  MEMap::iterator mi, me = data_.end();
  DQMNet::QReports::const_iterator qi, qe;
  int nme=0;

  // TFile flushes to disk with fsync() on every TDirectory written to the
  // file.  This makes DQM file saving painfully slow, and ironically makes
  // it _more_ likely the file saving gets interrupted and corrupts the file.
  // The utility class below simply ignores the flush synchronisation.
  class TFileNoSync : public TFile
  {
  public:
    TFileNoSync(const char *file, const char *opt) : TFile(file, opt) {}
    virtual Int_t SysSync(Int_t) override { return 0; }
  };

  // open output file, on 1st save recreate, later update
  if (verbose_)
    std::cout << "\n DQMStore: Opening TFile '" << filename
              << "' with option '" << fileupdate <<"'\n";

  TFileNoSync f(filename.c_str(), fileupdate.c_str()); // open file
  if(f.IsZombie())
    raiseDQMError("DQMStore", "Failed to create/update file '%s'", filename.c_str());
  f.cd();

  // Construct a regular expression from the pattern string.
  std::auto_ptr<lat::Regexp> rxpat;
  if (! pattern.empty())
    rxpat.reset(new lat::Regexp(pattern.c_str()));

  // Prepare a path for the reference object selection.
  std::string refpath;
  refpath.reserve(s_referenceDirName.size() + path.size() + 2);
  refpath += s_referenceDirName;
  if (! path.empty())
  {
    refpath += '/';
    refpath += path;
  }

  // Loop over the directory structure.
  for (di = dirs_.begin(), de = dirs_.end(); di != de; ++di)
  {
    // Check if we should process this directory.  We process the
    // requested part of the object tree, including references.
    if (! path.empty()
        && ! isSubdirectory(path, *di)
        && ! isSubdirectory(refpath, *di))
      continue;

    // Loop over monitor elements in this directory.
    MonitorElement proto(&*di, std::string(), run, 0, 0);
    mi = data_.lower_bound(proto);
    for ( ; mi != me && isSubdirectory(*di, *mi->data_.dirname); ++mi)
    {
      if (verbose_ > 1)
        std::cout << "Run: " << (*mi).run()
                  << " Lumi: " << (*mi).lumi()
                  << " LumiFlag: " << (*mi).getLumiFlag()
                  << " streamId: " << (*mi).streamId()
                  << " moduleId: " << (*mi).moduleId()
                  << " fullpathname: " << (*mi).getPathname() << std::endl;
      // Skip if it isn't a direct child.
      if (*di != *mi->data_.dirname)
        continue;

      // Keep backward compatibility with the old way of
      // booking/handlind MonitorElements into the DQMStore. If run is
      // 0 it means that a booking happened w/ the old non-threadsafe
      // style, and we have to ignore the streamId and moduleId as a
      // consequence.

      if (run != 0 && (mi->data_.streamId !=0 || mi->data_.moduleId !=0))
        continue;

      // Handle reference histograms, with three distinct cases:
      // 1) Skip all references entirely on saving.
      // 2) Blanket saving of all references.
      // 3) Save only references for monitor elements with qtests.
      // The latter two are affected by "path" sub-tree selection,
      // i.e. references are saved only in the selected tree part.
      if (isSubdirectory(refpath, *mi->data_.dirname))
      {
        if (ref == SaveWithoutReference)
          // Skip the reference entirely.
          continue;
        else if (ref == SaveWithReference)
          // Save all references regardless of qtests.
          ;
        else if (ref == SaveWithReferenceForQTest)
        {
          // Save only references for monitor elements with qtests
          // with an optional cut on minimum quality test result.
          int status = -1;
          std::string mname(mi->getFullname(), s_referenceDirName.size()+1, std::string::npos);
          MonitorElement *master = get(mname);
          if (master)
            for (size_t i = 0, e = master->data_.qreports.size(); i != e; ++i)
              status = std::max(status, master->data_.qreports[i].code);

          if (! master || status < minStatus)
          {
            if (verbose_ > 1)
              std::cout << "DQMStore::save: skipping monitor element '"
                        << mi->data_.objname << "' while saving, status is "
                        << status << ", required minimum status is "
                        << minStatus << std::endl;
            continue;
          }
        }
      }

      if (verbose_ > 1)
        std::cout << "DQMStore::save: saving monitor element '"
                  << mi->data_.objname << "'\n";
      nme++; // count saved histograms

      // Create the directory.
      gDirectory->cd("/");
      if (di->empty())
        cdInto(s_monitorDirName);
      else if (rxpat.get())
        cdInto(s_monitorDirName + '/' + lat::StringOps::replace(*di, *rxpat, rewrite));
      else
        cdInto(s_monitorDirName + '/' + *di);

      // Save the object.
      switch (mi->kind())
      {
      case MonitorElement::DQM_KIND_INT:
      case MonitorElement::DQM_KIND_REAL:
      case MonitorElement::DQM_KIND_STRING:
        TObjString(mi->tagString().c_str()).Write();
        break;

      default:
        mi->object_->Write();
        break;
      }

      // Save quality reports if this is not in reference section.
      if (! isSubdirectory(s_referenceDirName, *mi->data_.dirname))
      {
        qi = mi->data_.qreports.begin();
        qe = mi->data_.qreports.end();
        for ( ; qi != qe; ++qi)
          TObjString(mi->qualityTagString(*qi).c_str()).Write();
      }

      // Save efficiency tag, if any
      if (mi->data_.flags & DQMNet::DQM_PROP_EFFICIENCY_PLOT)
        TObjString(mi->effLabelString().c_str()).Write();

      // Save tag if any
      if (mi->data_.flags & DQMNet::DQM_PROP_TAGGED)
        TObjString(mi->tagLabelString().c_str()).Write();
    }
  }

  f.Close();

  // Maybe make some noise.
  if (verbose_)
    std::cout << "DQMStore::save: successfully wrote " << nme
              << " objects from path '" << path
              << "' into DQM file '" << filename << "'\n";
}

/// read ROOT objects from file <file> in directory <onlypath>;
/// return total # of ROOT objects read
unsigned int
DQMStore::readDirectory(TFile *file,
                        bool overwrite,
                        const std::string &onlypath,
                        const std::string &prepend,
                        const std::string &curdir,
                        OpenRunDirs stripdirs)
{
  unsigned int ntot = 0;
  unsigned int count = 0;

  if (! file->cd(curdir.c_str()))
    raiseDQMError("DQMStore", "Failed to process directory '%s' while"
                  " reading file '%s'", curdir.c_str(), file->GetName());

  // Figure out current directory name, but strip out the top
  // directory into which we dump everything.
  std::string dirpart = curdir;
  if (dirpart.compare(0, s_monitorDirName.size(), s_monitorDirName) == 0)
  {
    if (dirpart.size() == s_monitorDirName.size())
      dirpart.clear();
    else if (dirpart[s_monitorDirName.size()] == '/')
      dirpart.erase(0, s_monitorDirName.size()+1);
  }

  // See if we are going to skip this directory.
  bool skip = (! onlypath.empty() && ! isSubdirectory(onlypath, dirpart));

  if (prepend == s_collateDirName ||
      prepend == s_referenceDirName ||
      stripdirs == StripRunDirs )
  {
    // Remove Run # and RunSummary dirs
    // first look for Run summary,
    // if that is found and erased, also erase Run dir
    size_t slash = dirpart.find('/');
    size_t pos = dirpart.find("/Run summary");
    if (slash != std::string::npos && pos !=std::string::npos)
    {
      dirpart.erase(pos,12);

      pos = dirpart.find("Run ");
      size_t length = dirpart.find('/',pos+1)-pos+1;
      if (pos !=std::string::npos)
        dirpart.erase(pos,length);
    }
  }

  // If we are prepending, add it to the directory name,
  // and suppress reading of already existing reference histograms
  if (prepend == s_collateDirName ||
      prepend == s_referenceDirName)
  {
    size_t slash = dirpart.find('/');
    // If we are reading reference, skip previous reference.
    if (slash == std::string::npos   // skip if Reference is toplevel folder, i.e. no slash
        && slash+1+s_referenceDirName.size() == dirpart.size()
        && dirpart.compare(slash+1, s_referenceDirName.size(), s_referenceDirName) == 0)
      return 0;

    slash = dirpart.find('/');
    // Skip reading of EventInfo subdirectory.
    if (slash != std::string::npos
        && slash + 10 == dirpart.size()
        && dirpart.compare( slash+1 , 9 , "EventInfo") == 0) {
      if (verbose_)
        std::cout << "DQMStore::readDirectory: skipping '" << dirpart << "'\n";
      return 0;
    }

    // Add prefix.
    if (dirpart.empty())
      dirpart = prepend;
    else
      dirpart = prepend + '/' + dirpart;
  }
  else if (! prepend.empty())
  {
    if (dirpart.empty())
      dirpart = prepend;
    else
      dirpart = prepend + '/' + dirpart;
  }

  // Loop over the contents of this directory in the file.
  // Post-pone string object handling to happen after other
  // objects have been read in so we are guaranteed to have
  // histograms by the time we read in quality tests and tags.
  TKey *key;
  TIter next (gDirectory->GetListOfKeys());
  std::list<TObject *> delayed;
  while ((key = (TKey *) next()))
  {
    std::auto_ptr<TObject> obj(key->ReadObj());
    if (dynamic_cast<TDirectory *>(obj.get()))
    {
      std::string subdir;
      subdir.reserve(curdir.size() + strlen(obj->GetName()) + 2);
      subdir += curdir;
      if (! curdir.empty())
        subdir += '/';
      subdir += obj->GetName();

      ntot += readDirectory(file, overwrite, onlypath, prepend, subdir, stripdirs);
    }
    else if (skip)
      ;
    else if (dynamic_cast<TObjString *>(obj.get()))
    {
      delayed.push_back(obj.release());
    }
    else
    {
      if (verbose_ > 2)
        std::cout << "DQMStore: reading object '" << obj->GetName()
                  << "' of type '" << obj->IsA()->GetName()
                  << "' from '" << file->GetName()
                  << "' into '" << dirpart << "'\n";

      makeDirectory(dirpart);
      if (extract(obj.get(), dirpart, overwrite))
        ++count;
    }
  }

  while (! delayed.empty())
  {
    if (verbose_ > 2)
      std::cout << "DQMStore: reading object '" << delayed.front()->GetName()
                << "' of type '" << delayed.front()->IsA()->GetName()
                << "' from '" << file->GetName()
                << "' into '" << dirpart << "'\n";

    makeDirectory(dirpart);
    if (extract(delayed.front(), dirpart, overwrite))
      ++count;

    delete delayed.front();
    delayed.pop_front();
  }

  if (verbose_ > 1)
    std::cout << "DQMStore: read " << count << '/' << ntot
              << " objects from directory '" << dirpart << "'\n";

  return ntot + count;
}

/// public open/read root file <filename>, and copy MonitorElements;
/// if flag=true, overwrite identical MonitorElements (default: false);
/// if onlypath != "", read only selected directory
/// if prepend !="", prepend string to path
/// note: by default this method keeps the dir structure as in file
/// and does not update monitor element references!
bool
DQMStore::open(const std::string &filename,
               bool overwrite /* = false */,
               const std::string &onlypath /* ="" */,
               const std::string &prepend /* ="" */,
               OpenRunDirs stripdirs /* =KeepRunDirs */,
               bool fileMustExist /* =true */)
{
  return readFile(filename,overwrite,onlypath,prepend,stripdirs,fileMustExist);
}

/// public load root file <filename>, and copy MonitorElements;
/// overwrite identical MonitorElements (default: true);
/// set DQMStore.collateHistograms to true to sum several files
/// note: by default this method strips off run dir structure
bool
DQMStore::load(const std::string &filename,
               OpenRunDirs stripdirs /* =StripRunDirs */,
               bool fileMustExist /* =true */)
{
  bool overwrite = true;
  if (collateHistograms_) overwrite = false;
  if (verbose_)
  {
    std::cout << "DQMStore::load: reading from file '" << filename << "'\n";
    if (collateHistograms_)
      std::cout << "DQMStore::load: in collate mode   " << "\n";
    else
      std::cout << "DQMStore::load: in overwrite mode   " << "\n";
  }

  return readFile(filename,overwrite,"","",stripdirs,fileMustExist);

}

/// private readFile <filename>, and copy MonitorElements;
/// if flag=true, overwrite identical MonitorElements (default: false);
/// if onlypath != "", read only selected directory
/// if prepend !="", prepend string to path
/// if StripRunDirs is set the run and run summary folders are erased.
bool
DQMStore::readFile(const std::string &filename,
                   bool overwrite /* = false */,
                   const std::string &onlypath /* ="" */,
                   const std::string &prepend /* ="" */,
                   OpenRunDirs stripdirs /* =StripRunDirs */,
                   bool fileMustExist /* =true */)
{

  if (verbose_)
    std::cout << "DQMStore::readFile: reading from file '" << filename << "'\n";

  std::auto_ptr<TFile> f;

  try
  {
    f.reset(TFile::Open(filename.c_str()));
    if (! f.get() || f->IsZombie())
      raiseDQMError("DQMStore", "Failed to open file '%s'", filename.c_str());
  }
  catch (std::exception &)
  {
    if (fileMustExist)
      throw;
    else
    {
    if (verbose_)
      std::cout << "DQMStore::readFile: file '" << filename << "' does not exist, continuing\n";
    return false;
    }
  }

  unsigned n = readDirectory(f.get(), overwrite, onlypath, prepend, "", stripdirs);
  f->Close();

  MEMap::iterator mi = data_.begin();
  MEMap::iterator me = data_.end();
  for ( ; mi != me; ++mi)
    const_cast<MonitorElement &>(*mi).updateQReportStats();

  if (verbose_)
  {
    std::cout << "DQMStore::open: successfully read " << n
              << " objects from file '" << filename << "'";
    if (! onlypath.empty())
      std::cout << " from directory '" << onlypath << "'";
    if (! prepend.empty())
      std::cout << " into directory '" << prepend << "'";
    std::cout << std::endl;
  }
  return true;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// delete directory and all contents;
/// delete directory (all contents + subfolders);
void
DQMStore::rmdir(const std::string &path)
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(path, clean, cleaned);
  MonitorElement proto(cleaned, std::string());

  MEMap::iterator e = data_.end();
  MEMap::iterator i = data_.lower_bound(proto);
  while (i != e && isSubdirectory(*cleaned, *i->data_.dirname))
    data_.erase(i++);

  std::set<std::string>::iterator de = dirs_.end();
  std::set<std::string>::iterator di = dirs_.lower_bound(*cleaned);
  while (di != de && isSubdirectory(*cleaned, *di))
    dirs_.erase(di++);
}

/// remove all monitoring elements from directory;
void
DQMStore::removeContents(const std::string &dir)
{
  MonitorElement proto(&dir, std::string());
  MEMap::iterator e = data_.end();
  MEMap::iterator i = data_.lower_bound(proto);
  while (i != e && isSubdirectory(dir, *i->data_.dirname))
    if (dir == *i->data_.dirname)
      data_.erase(i++);
    else
      ++i;
}

/// erase all monitoring elements in current directory (not including subfolders);
void
DQMStore::removeContents(void)
{
  removeContents(pwd_);
}

/// erase monitoring element in current directory
/// (opposite of book1D,2D,etc. action);
void
DQMStore::removeElement(const std::string &name)
{
  removeElement(pwd_, name);
}

/// remove monitoring element from directory;
/// if warning = true, print message if element does not exist
void
DQMStore::removeElement(const std::string &dir, const std::string &name, bool warning /* = true */)
{
  MonitorElement proto(&dir, name);
  MEMap::iterator pos = data_.find(proto);
  if (pos == data_.end() && warning)
    std::cout << "DQMStore: WARNING: attempt to remove non-existent"
              << " monitor element '" << name << "' in '" << dir << "'\n";
  else
    data_.erase(pos);
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// get QCriterion corresponding to <qtname>
/// (null pointer if QCriterion does not exist)
QCriterion *
DQMStore::getQCriterion(const std::string &qtname) const
{
  QCMap::const_iterator i = qtests_.find(qtname);
  QCMap::const_iterator e = qtests_.end();
  return (i == e ? 0 : i->second);
}

/// create quality test with unique name <qtname> (analogous to ME name);
/// quality test can then be attached to ME with useQTest method
/// (<algo_name> must match one of known algorithms)
QCriterion *
DQMStore::createQTest(const std::string &algoname, const std::string &qtname)
{
  if (qtests_.count(qtname))
    raiseDQMError("DQMStore", "Attempt to create duplicate quality test '%s'",
                  qtname.c_str());

  QAMap::iterator i = qalgos_.find(algoname);
  if (i == qalgos_.end())
    raiseDQMError("DQMStore", "Cannot create a quality test using unknown"
                  " algorithm '%s'", algoname.c_str());

  QCriterion *qc = i->second(qtname);
  qc->setVerbose(verboseQT_);

  qtests_[qtname] = qc;
  return qc;
}

/// attach quality test <qtname> to directory contents
/// (need exact pathname without wildcards, e.g. A/B/C);
void
DQMStore::useQTest(const std::string &dir, const std::string &qtname)
{
  // Clean the path
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(dir, clean, cleaned);

  // Validate the path.
  if (cleaned->find_first_not_of(s_safe) != std::string::npos)
    raiseDQMError("DQMStore", "Monitor element path name '%s'"
                  " uses unacceptable characters", cleaned->c_str());

  // Redirect to the pattern match version.
  useQTestByMatch(*cleaned + "/*", qtname);
}

/// attach quality test <qc> to monitor elements matching <pattern>.
int
DQMStore::useQTestByMatch(const std::string &pattern, const std::string &qtname)
{
  QCriterion *qc = getQCriterion(qtname);
  if (! qc)
    raiseDQMError("DQMStore", "Cannot apply non-existent quality test '%s'",
                  qtname.c_str());

  fastmatch * fm = new fastmatch( pattern );

  // Record the test for future reference.
  QTestSpec qts(fm, qc);
  qtestspecs_.push_back(qts);

  // Apply the quality test.
  MEMap::iterator mi = data_.begin();
  MEMap::iterator me = data_.end();
  std::string path;
  int cases = 0;
  for ( ; mi != me; ++mi)
  {
    path.clear();
    mergePath(path, *mi->data_.dirname, mi->data_.objname);
    if (fm->match(path))
    {
      ++cases;
      const_cast<MonitorElement &>(*mi).addQReport(qts.second);
    }
  }

  //return the number of matched cases
  return cases;
}
/// run quality tests (also finds updated contents in last monitoring cycle,
/// including newly added content)
void
DQMStore::runQTests(void)
{

  if (verbose_ > 0)
    std::cout << "DQMStore: running runQTests() with reset = "
              << ( reset_ ? "true" : "false" ) << std::endl;

  // Apply quality tests to each monitor element, skipping references.
  MEMap::iterator mi = data_.begin();
  MEMap::iterator me = data_.end();
  for ( ; mi != me; ++mi)
    if (! isSubdirectory(s_referenceDirName, *mi->data_.dirname))
      const_cast<MonitorElement &>(*mi).runQTests();

  reset_ = false;
}

/// get "global" folder <path> status (one of:STATUS_OK, WARNING, ERROR, OTHER);
/// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
/// see Core/interface/QTestStatus.h for details on "OTHER"
int
DQMStore::getStatus(const std::string &path /* = "" */) const
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(path, clean, cleaned);

  int status = dqm::qstatus::STATUS_OK;
  MEMap::const_iterator mi = data_.begin();
  MEMap::const_iterator me = data_.end();
  for ( ; mi != me; ++mi)
  {
    if (! cleaned->empty() && ! isSubdirectory(*cleaned, *mi->data_.dirname))
      continue;

    if (mi->hasError())
      return dqm::qstatus::ERROR;
    else if (mi->hasWarning())
      status = dqm::qstatus::WARNING;
    else if (status < dqm::qstatus::WARNING
             && mi->hasOtherReport())
      status = dqm::qstatus::OTHER;
  }
  return status;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// reset contents (does not erase contents permanently)
/// (makes copy of current contents; will be subtracted from future contents)
void
DQMStore::softReset(MonitorElement *me)
{
  if (me)
    me->softReset();
}

// reverts action of softReset
void
DQMStore::disableSoftReset(MonitorElement *me)
{
  if (me)
    me->disableSoftReset();
}

/// if true, will accumulate ME contents (over many periods)
/// until method is called with flag = false again
void
DQMStore::setAccumulate(MonitorElement *me, bool flag)
{
  if (me)
    me->setAccumulate(flag);
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void
DQMStore::showDirStructure(void) const
{
  std::vector<std::string> contents;
  getContents(contents);

  std::cout << " ------------------------------------------------------------\n"
            << "                    Directory structure:                     \n"
            << " ------------------------------------------------------------\n";

  std::copy(contents.begin(), contents.end(),
            std::ostream_iterator<std::string>(std::cout, "\n"));

  std::cout << " ------------------------------------------------------------\n";
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// check if the collate option is active on the DQMStore
bool
DQMStore::isCollate(void) const
{
  return collateHistograms_;
}
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// check if the monitor element is in auto-collation folder
bool
DQMStore::isCollateME(MonitorElement *me) const
{ return me && isSubdirectory(s_collateDirName, *me->data_.dirname); }
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** Invoke this method after flushing all recently changed monitoring.
    Clears updated flag on all MEs and calls their Reset() method. */
void
DQMStore::scaleElements(void)
{
  if (scaleFlag_ == 0.0) return;
  if (verbose_ > 0)
    std::cout << " =========== " << " ScaleFlag " << scaleFlag_ << std::endl;
  double factor = scaleFlag_;
  int events = 1;
  if (dirExists("Info/EventInfo")) {
    if ( scaleFlag_ == -1.0) {
      MonitorElement * scale_me = get("Info/EventInfo/ScaleFactor");
      if (scale_me && scale_me->kind()==MonitorElement::DQM_KIND_REAL) factor = scale_me->getFloatValue();
    }
    MonitorElement * event_me = get("Info/EventInfo/processedEvents");
    if (event_me && event_me->kind()==MonitorElement::DQM_KIND_INT) events = event_me->getIntValue();
  }
  factor = factor/(events*1.0);

  MEMap::iterator mi = data_.begin();
  MEMap::iterator me = data_.end();
  for ( ; mi != me; ++mi)
  {
    MonitorElement &me = const_cast<MonitorElement &>(*mi);
    switch (me.kind())
      {
      case MonitorElement::DQM_KIND_TH1F:
        {
          me.getTH1F()->Scale(factor);
          break;
        }
      case MonitorElement::DQM_KIND_TH1S:
        {
          me.getTH1S()->Scale(factor);
          break;
        }
      case MonitorElement::DQM_KIND_TH1D:
        {
          me.getTH1D()->Scale(factor);
          break;
        }
      case MonitorElement::DQM_KIND_TH2F:
        {
          me.getTH2F()->Scale(factor);
          break;
        }
      case MonitorElement::DQM_KIND_TH2S:
        {
          me.getTH2S()->Scale(factor);
          break;
        }
      case MonitorElement::DQM_KIND_TH2D:
        {
          me.getTH2D()->Scale(factor);
          break;
        }
      case MonitorElement::DQM_KIND_TH3F:
        {
          me.getTH3F()->Scale(factor);
          break;
        }
      case MonitorElement::DQM_KIND_TPROFILE:
        {
          me.getTProfile()->Scale(factor);
          break;
        }
      case MonitorElement::DQM_KIND_TPROFILE2D:
        {
          me.getTProfile2D()->Scale(factor);
          break;
        }
      default:
        if (verbose_ > 0)
          std::cout << " The DQM object '" << me.getFullname() << "' is not scalable object " << std::endl;
        continue;
      }
  }
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:

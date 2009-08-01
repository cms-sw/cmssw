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
#include <iterator>

/** @var DQMStore::verbose_
    Universal verbose flag for DQM. */

/** @var DQMStore::verboseQT_
    Verbose flag for xml-based QTests. */

/** @var DQMStore::reset_

    Flag used to print out a warning when calling quality tests.
    twice without having called reset() in between; to be reset in
    DQMOldReceiver::runQualityTests.  */

/** @var DQMStore::collateHistograms_ */

/** @var DQMStore::outputFileRecreate_ */

/** @var DQMStore::readSelectedDirectory_
    If non-empty, read from file only selected directory. */

/** @var DQMStore::pwd_
    Current directory. */

/** @var DQMStore::own_
    All the monitor elements. */

/** @var DQMStore::removed_
    Removed contents since last cycle.  */

/** @var DQMStore::qtests_.
    All the quality tests.  */

/** @var DQMStore::qalgos_
    Set of all the available quality test algorithms. */

//////////////////////////////////////////////////////////////////////
/// pathname for root folder
static std::string ROOT_PATHNAME = ".";

/// name of global monitoring folder (containing all sources subdirectories)
static std::string s_monitorDirName = "DQMData";
static std::string s_referenceDirName = "Reference";
static std::string s_collateDirName = "Collate";
static std::string s_dqmPatchVersion = "0";
static std::string s_safe = "/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+=_()# ";
static DQMStore *s_instance = 0;

static const lat::Regexp s_rxmeval ("^<(.*)>(i|f|s|qr)=(.*)</\\1>$");
static const lat::Regexp s_rxmeqr1 ("^st:(\\d+):([-+e.\\d]+):(.*)$");
static const lat::Regexp s_rxmeqr2 ("^st\\.(\\d+)\\.(.*)$");

//////////////////////////////////////////////////////////////////////
/// Check whether @a path is a subdirectory of @a ofdir.  Returns
/// true also if ofdir == path.
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

template <class T>
QCriterion *
makeQCriterion(const std::string &qtname)
{ return new T(qtname); }

template <class T>
void
initQCriterion(std::map<std::string, QCriterion *(*)(const std::string &)> &m)
{ m[T::getAlgoName()] = &makeQCriterion<T>; }


//////////////////////////////////////////////////////////////////////
DQMStore *
DQMStore::instance(void)
{
  if (! s_instance)
  {
    try
    {
      DQMStore *bei = new DQMStore(edm::ParameterSet());
      assert (s_instance == bei);
    }
    catch(DQMError &e)
    {
      std::cout << e.what() << std::endl;
      exit(255);
    }
  }

  return s_instance;
}

DQMStore::DQMStore(const edm::ParameterSet &pset)
  : verbose_ (1),
    verboseQT_ (1),
    reset_ (false),
    collateHistograms_ (false),
    readSelectedDirectory_ (""),
    outputFileRecreate_ (true),
    pwd_ ("")
{
  assert(! s_instance);
  s_instance = this;
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

  std::string ref = pset.getUntrackedParameter<std::string>("referenceFileName", "");
  if (! ref.empty())
  {
    std::cout << "DQMStore: using reference file '" << ref << "'\n";
    readFile(ref, true, "", s_referenceDirName, StripRunDirs);
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
}

DQMStore::~DQMStore(void)
{
  for (QCMap::iterator i = qtests_.begin(), e = qtests_.end(); i != e; ++i)
    delete i->second;

  for (QTestSpecs::iterator i = qtestspecs_.begin(), e = qtestspecs_.end(); i != e; ++i)
    delete i->first;

  if (s_instance == this)
    s_instance = 0;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// set verbose level (0 turns all non-error messages off)
void
DQMStore::setVerbose(unsigned level) 
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
  size_t slash = 0;
  while (true)
  {
    // Create this subdirectory component.
    std::string subdir(path, 0, slash);
    if (data_.count(subdir))
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
MonitorElement *
DQMStore::initialise(MonitorElement *me, const std::string &path)
{
  // Initialise quality test information.
  QTestSpecs::iterator qi = qtestspecs_.begin();
  QTestSpecs::iterator qe = qtestspecs_.end();
  for ( ; qi != qe; ++qi)
    if (qi->first->match(path))
      me->addQReport(qi->second);

  // If we have a reference, assign it now.  Note that we can't use
  // getReferenceME() because "me" hasn't been initialised yet.
  std::string refpath;
  refpath.reserve(s_referenceDirName.size() + path.size() + 2);
  refpath += s_referenceDirName;
  refpath += '/';
  refpath += path;

  MEMap::const_iterator refpos = data_.find(refpath);
  if (refpos != data_.end())
    me->data_.reference = refpos->second.data_.object;

  // Return the monitor element.
  return me;
}

template <class HISTO, class COLLATE>
MonitorElement *
DQMStore::book(const std::string &dir, const std::string &name,
	       const char *context, int kind,
	       HISTO *h, COLLATE collate)
{
  // Put us in charge of h.
  h->SetDirectory(0);

  // Check if the request monitor element already exists.
  std::string path;
  if (MonitorElement *me = findObject(dir, name, path))
  {
    if (collateHistograms_)
    {
      collate(me, h);
      delete h;
      return me;
    }
    else
    {    
      if (verbose_>1)
        std::cout << "DQMStore: "
                  << context << ": monitor element '"
                  << path << "' already exists, collating" << std::endl;
      me->Reset();
      collate(me, h);
      delete h;
      return me;
    }
  }

  // Create and initialise it.
  return initialise(&data_[path], path)
    ->initialise((MonitorElement::Kind) kind, path, h);
}

MonitorElement *
DQMStore::book(const std::string &dir, const std::string &name,
	       std::string &path, const char *context)
{
  // Check if the request monitor element already exists.
  if (MonitorElement *me = findObject(dir, name, path)) {
      if (verbose_>1)
        std::cout << "DQMStore: "
                  << context << ": monitor element '"
                  << path << "' already exists, resetting" << std::endl;
      me->Reset();
      return me;
  }

  // Create it and return for initialisation.
  return initialise(&data_[path], path);
}

// -------------------------------------------------------------------
/// Book int.
MonitorElement *
DQMStore::bookInt(const std::string &dir, const std::string &name)
{
  std::string path;
  if (collateHistograms_)
  {
    if (MonitorElement *me = findObject(dir, name, path)) {
      me->Fill(0);
      return me;
    }
  }
  return book(dir, name, path, "bookInt")
    ->initialise(MonitorElement::DQM_KIND_INT, path);
}

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
  std::string path;
  if (collateHistograms_)
  {
    if (MonitorElement *me = findObject(dir, name, path)) {
      me->Fill(0.);
      return me;
    }
  }
  return book(dir, name, path, "bookFloat")
    ->initialise(MonitorElement::DQM_KIND_REAL, path);
}

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
  std::string path;
  if (collateHistograms_)
  {
    if (MonitorElement *me = findObject(dir, name, path)) {
      return me;
    }
  }
  return book(dir, name, path, "bookString")
    ->initialise(MonitorElement::DQM_KIND_STRING, path, value);
}

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

/// Book 1D histogram.
MonitorElement *
DQMStore::book1D(const std::string &name, const std::string &title,
		 int nchX, double lowX, double highX)
{
  return book1D(pwd_, name, new TH1F(name.c_str(), title.c_str(), nchX, lowX, highX));
}

/// Book 1S histogram.
MonitorElement *
DQMStore::book1S(const std::string &name, const std::string &title,
		 int nchX, double lowX, double highX)
{
  return book1S(pwd_, name, new TH1S(name.c_str(), title.c_str(), nchX, lowX, highX));
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
DQMStore::book1D(const std::string &name, TH1F *source)
{
  return book1D(pwd_, name, static_cast<TH1F *>(source->Clone(name.c_str())));
}

/// Book 1S histogram by cloning an existing histogram.
MonitorElement *
DQMStore::book1S(const std::string &name, TH1S *source)
{
  return book1S(pwd_, name, static_cast<TH1S *>(source->Clone(name.c_str())));
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
  return book(dir, name, "book2", MonitorElement::DQM_KIND_TH2S, h, collate2S);
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
DQMStore::book2S(const std::string &name, const std::string &title,
		 int nchX, double lowX, double highX,
		 int nchY, double lowY, double highY)
{
  return book2S(pwd_, name, new TH2S(name.c_str(), title.c_str(),
				     nchX, lowX, highX,
				     nchY, lowY, highY));
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
DQMStore::book2D(const std::string &name, TH2F *source)
{
  return book2D(pwd_, name, static_cast<TH2F *>(source->Clone(name.c_str())));
}

/// Book 2DShistogram by cloning an existing histogram.
MonitorElement *
DQMStore::book2S(const std::string &name, TH2S *source)
{
  return book2S(pwd_, name, static_cast<TH2S *>(source->Clone(name.c_str())));
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
DQMStore::bookProfile(const std::string &name, const std::string &title,
		      int nchX, double lowX, double highX,
		      int nchY, double lowY, double highY,
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
DQMStore::bookProfile2D(const std::string &name, const std::string &title,
			int nchX, double lowX, double highX,
			int nchY, double lowY, double highY,
			int nchZ, double lowZ, double highZ,
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
DQMStore::bookProfile2D(const std::string &name, TProfile2D *source)
{
  return bookProfile2D(pwd_, name, static_cast<TProfile2D *>(source->Clone(name.c_str())));
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void
DQMStore::collate1D(MonitorElement *me, TH1F *h)
{ me->getTH1F()->Add(h); }

void
DQMStore::collate1S(MonitorElement *me, TH1S *h)
{ me->getTH1S()->Add(h); }

void
DQMStore::collate2D(MonitorElement *me, TH2F *h)
{ me->getTH2F()->Add(h); }

void
DQMStore::collate2S(MonitorElement *me, TH2S *h)
{ me->getTH2S()->Add(h); }

void
DQMStore::collate3D(MonitorElement *me, TH3F *h)
{ me->getTH3F()->Add(h); }

void
DQMStore::collateProfile(MonitorElement *me, TProfile *h)
{
  TProfile *meh = me->getTProfile();
  me->addProfiles(h, meh, meh, 1, 1);
}

void
DQMStore::collateProfile2D(MonitorElement *me, TProfile2D *h)
{
  TProfile2D *meh = me->getTProfile2D();
  me->addProfiles(h, meh, meh, 1, 1);
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

  DQMNet::TagList::iterator pos
    = std::lower_bound(me->data_.tags.begin(), me->data_.tags.end(), myTag);

  if (pos == me->data_.tags.end() || *pos > myTag)
    me->data_.tags.insert(pos, myTag);
}

/// tag ME specified by full pathname (e.g. "my/long/dir/my_histo")
void
DQMStore::tag(const std::string &path, unsigned int myTag)
{
  MEMap::iterator mepos = data_.find(path);
  if (mepos == data_.end())
    raiseDQMError("DQMStore", "Attempt to tag non-existent monitor element"
		  " '%s' with tag %u", path.c_str(), myTag);

  tag(&mepos->second, myTag);
}

/// tag all children of folder (does NOT include subfolders)
void
DQMStore::tagContents(const std::string &path, unsigned int myTag)
{
  MEMap::iterator e = data_.end();
  MEMap::iterator i = data_.lower_bound(path);
  for ( ; i != e && i->second.path_ == path; ++i)
    tag(&i->second, myTag);
}

/// tag all children of folder, including all subfolders and their children;
/// path must be an exact path name
void
DQMStore::tagAllContents(const std::string &path, unsigned int myTag)
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(path, clean, cleaned);

  // FIXME: WILDCARDS? Old one supported them, but nobody seemed to use them.
  MEMap::iterator e = data_.end();
  MEMap::iterator i = data_.lower_bound(*cleaned);
  while (i != e && isSubdirectory(*cleaned, i->first))
  {
    tag(&i->second, myTag);
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
  std::vector<std::string> result;
  MEMap::const_iterator e = data_.end();
  MEMap::const_iterator i = data_.lower_bound(pwd_);
  for ( ; i != e && isSubdirectory(pwd_, i->second.path_); ++i)
    if (i->second.path_ == pwd_)
      result.push_back(i->second.name_);

  return result;
}

/// true if directory (or any subfolder at any level below it) contains
/// at least one monitorable element
bool
DQMStore::containsAnyMonitorable(const std::string &path) const
{
  MEMap::const_iterator e = data_.end();
  MEMap::const_iterator i = data_.lower_bound(path);
  return (i != e
	  && i->first.compare(0, path.size(), path) == 0
	  && i->first[path.size()] == '/');
}

/// get ME from full pathname (e.g. "my/long/dir/my_histo")
MonitorElement *
DQMStore::get(const std::string &path) const
{
  MEMap::const_iterator mepos = data_.find(path);
  return (mepos == data_.end() ? 0
	  : const_cast<MonitorElement *>(&mepos->second));
}

/// get all MonitorElements tagged as <tag>
std::vector<MonitorElement *>
DQMStore::get(unsigned int tag) const
{
  // FIXME: Use reverse map [tag -> path] / [tag -> dir]?
  std::vector<MonitorElement *> result;
  for (MEMap::const_iterator i = data_.begin(), e = data_.end(); i != e; ++i)
  {
    const MonitorElement &me = i->second;
    DQMNet::TagList::const_iterator te = me.data_.tags.end();
    DQMNet::TagList::const_iterator ti = me.data_.tags.begin();
    ti = std::lower_bound(ti, te, tag);
    if (ti != te && *ti == tag)
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

  std::vector<MonitorElement *> result;
  MEMap::const_iterator e = data_.end();
  MEMap::const_iterator i = data_.lower_bound(*cleaned);
  for ( ; i != e && isSubdirectory(*cleaned, i->second.path_); ++i)
    if (i->second.path_ == *cleaned)
      result.push_back(const_cast<MonitorElement *>(&i->second));

  return result;
}

/// same as above for tagged MonitorElements
std::vector<MonitorElement *>
DQMStore::getContents(const std::string &path, unsigned int tag) const
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(path, clean, cleaned);

  std::vector<MonitorElement *> result;
  MEMap::const_iterator e = data_.end();
  MEMap::const_iterator i = data_.lower_bound(*cleaned);
  for ( ; i != e && isSubdirectory(*cleaned, i->second.path_); ++i)
  {
    const MonitorElement &me = i->second;
    if (me.path_ != *cleaned)
      continue;
    DQMNet::TagList::const_iterator te = me.data_.tags.end();
    DQMNet::TagList::const_iterator ti = me.data_.tags.begin();
    ti = std::lower_bound(ti, te, tag);
    if (ti != te && *ti == tag)
      result.push_back(const_cast<MonitorElement *>(&me));
  }

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
    MEMap::const_iterator mi = data_.lower_bound(*di);
    MEMap::const_iterator m = mi;
    size_t sz = di->size() + 2;
    size_t nfound = 0;
    for ( ; m != me && isSubdirectory(*di, m->second.path_); ++m)
      if (*di == m->second.path_)
      {
	sz += m->second.name_.size() + 1;
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
	if (*di != mi->second.path_)
	  continue;

	if (sz > 0)
	  *istr += ',';

	*istr += mi->second.name_;
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
DQMStore::findObject(const std::string &dir, const std::string &name, std::string &path) const
{
  path.clear();
  path.reserve(dir.size() + name.size() + 2);
  if (! dir.empty())
  {
    path += dir;
    path += '/';
  }
  path += name;

  if (path.find_first_not_of(s_safe) != std::string::npos)
    raiseDQMError("DQMStore", "Monitor element path name '%s' uses"
		  " unacceptable characters", path.c_str());

  MEMap::const_iterator mepos = data_.find(path);
  return (mepos == data_.end() ? 0
	  : const_cast<MonitorElement *>(&mepos->second));
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
  for ( ; di != de; ++di)
  {
    MEMap::const_iterator mi = data_.lower_bound(*di);
    MEMap::const_iterator m = mi;
    size_t sz = di->size() + 2;
    size_t nfound = 0;
    for ( ; m != me && isSubdirectory(*di, m->second.path_); ++m)
      if (*di == m->second.path_ && ! m->second.data_.tags.empty())
      {
        // the tags count for '/' + up to 10 digits, otherwise ',' + ME name
	sz += 1 + m->second.name_.size() + 11*m->second.data_.tags.size();
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
      if (*di != mi->second.path_ || mi->second.data_.tags.empty())
	continue;

      if (sz > 0)
	*istr += ',';

      *istr += mi->second.name_;

      for (size_t ti = 0, te = mi->second.data_.tags.size(); ti < te; ++ti)
      {
	char tagbuf[32]; // more than enough for '/' and up to 10 digits
	sprintf(tagbuf, "/%u", mi->second.data_.tags[ti]);
	*istr += tagbuf;
      }

      ++sz;
    }
  }
}

/// get vector with children of folder, including all subfolders + their children;
/// must use an exact pathname
std::vector<MonitorElement*>
DQMStore::getAllContents(const std::string &path) const
{
  std::string clean;
  const std::string *cleaned = 0;
  cleanTrailingSlashes(path, clean, cleaned);

  std::vector<MonitorElement *> result;
  MEMap::const_iterator e = data_.end();
  MEMap::const_iterator i = data_.lower_bound(*cleaned);
  for ( ; i != e && isSubdirectory(*cleaned, i->second.path_); ++i)
    result.push_back(const_cast<MonitorElement *>(&i->second));

  return result;
}

/// get vector with children of folder, including all subfolders + their children;
/// matches names against a wildcard pattern matched against the full ME path
std::vector<MonitorElement*>
DQMStore::getMatchingContents(const std::string &pattern) const
{
  lat::Regexp rx;
  try
  {
    rx = lat::Regexp(pattern, 0, lat::Regexp::Wildcard);
    rx.study();
  }
  catch (lat::Error &e)
  {
    raiseDQMError("DQMStore", "Invalid regular expression '%s': %s",
		  pattern.c_str(), e.explain().c_str());
  }

  std::vector<MonitorElement *> result;
  MEMap::const_iterator i = data_.begin();
  MEMap::const_iterator e = data_.end();
  for ( ; i != e; ++i)
    if (rx.match(i->first))
      result.push_back(const_cast<MonitorElement *>(&i->second));

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
    if (mi->second.wasUpdated())
    {
      if (mi->second.resetMe())
	mi->second.Reset();
      mi->second.resetUpdate();
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
  std::string path;
  if (TH1F *h = dynamic_cast<TH1F *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName(), path);
    if (! me)
      me = book1D(dir, h->GetName(), (TH1F *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate1D(me, h);
  }
  else if (TH1S *h = dynamic_cast<TH1S *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName(), path);
    if (! me)
      me = book1S(dir, h->GetName(), (TH1S *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate1S(me, h);
  }
  else if (TH2F *h = dynamic_cast<TH2F *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName(), path);
    if (! me)
      me = book2D(dir, h->GetName(), (TH2F *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate2D(me, h);
  }
  else if (TH2S *h = dynamic_cast<TH2S *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName(), path);
    if (! me)
      me = book2S(dir, h->GetName(), (TH2S *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate2S(me, h);
  }
  else if (TH3F *h = dynamic_cast<TH3F *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName(), path);
    if (! me)
      me = book3D(dir, h->GetName(), (TH3F *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collate3D(me, h);
  }
  else if (TProfile *h = dynamic_cast<TProfile *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName(), path);
    if (! me)
      me = bookProfile(dir, h->GetName(), (TProfile *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collateProfile(me, h);
  }
  else if (TProfile2D *h = dynamic_cast<TProfile2D *>(obj))
  {
    MonitorElement *me = findObject(dir, h->GetName(), path);
    if (! me)
      me = bookProfile2D(dir, h->GetName(), (TProfile2D *) h->Clone());
    else if (overwrite)
      me->copyFrom(h);
    else if (isCollateME(me) || collateHistograms_)
      collateProfile2D(me, h);
  }
  else if (dynamic_cast<TObjString *>(obj))
  {
    std::string path;
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
      MonitorElement *me = findObject(dir, label, path);
      if (! me || overwrite)
      {
	if (! me) me = bookInt(dir, label);
	me->Fill(atoi(value.c_str()));
      }
    }
    else if (kind == "f")
    {
      MonitorElement *me = findObject(dir, label, path);
      if (! me || overwrite)
      {
	if (! me) me = bookFloat(dir, label);
	me->Fill(atof(value.c_str()));
      }
    }
    else if (kind == "s")
    {
      MonitorElement *me = findObject(dir, label, path);
      if (! me)
	me = bookString(dir, label, value);
      else if (overwrite)
      {
	me->curvalue_.str = value;
	static_cast<TObjString *>(me->data_.object)
	  ->SetString(me->tagString().c_str());
      }
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

        std::string path;
        std::string mename (label, 0, dot);
        std::string qrname (label, dot+1, std::string::npos);

        m.reset();
        DQMNet::QValue qv;
	if (s_rxmeqr1.match(value, 0, 0, &m))
	{
	  qv.code = atoi(m.matchString(value, 1).c_str());
	  qv.qtresult = strtod(m.matchString(value, 2).c_str(), 0);
	  qv.message = m.matchString(value, 3);
	  qv.qtname = qrname;
	}
	else if (s_rxmeqr2.match(value, 0, 0, &m))
	{
	  qv.code = atoi(m.matchString(value, 1).c_str());
	  qv.qtresult = 0; // unavailable in old format
	  qv.message = m.matchString(value, 2);
	  qv.qtname = qrname;
	}
	else
        {
	  std::cout << "*** DQMStore: WARNING: quality test value '"
		    << value << "' is incorrectly formatted\n";
	  return false;
        }

        MonitorElement *me = findObject(dir, mename, path);
        if (! me)
        {
	  std::cout << "*** DQMStore: WARNING: no monitor element '"
		    << mename << "' for quality test '"
		    << label << "' \n";
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
	       SaveReferenceTag ref /* = SaveWithReferenceForQTest */,
	       int minStatus /* = dqm::qstatus::STATUS_OK */)
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
    virtual Int_t SysSync(Int_t) { return 0; }
  };

  // open output file, on 1st save recreate, later update
  std::string opt = "UPDATE";
  if (outputFileRecreate_) 
    opt = "RECREATE";
  if (verbose_>0)
    std::cout << "\n DQMStore: Opening TFile '" << filename 
              << "' with option '" << opt <<"'\n";

  TFileNoSync f(filename.c_str(), opt.c_str()); // open file
  if(f.IsZombie())
    raiseDQMError("DQMStore", "Failed to create/update file '%s'", filename.c_str());
  f.cd();

  if (outputFileRecreate_) // write versions once upon opening
  {
    TObjString(edm::getReleaseVersion().c_str()).Write(); // Save CMSSW version
    TObjString(getDQMPatchVersion().c_str()).Write(); // Save DQM patch version
    outputFileRecreate_=false;
  }
  
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
    mi = data_.lower_bound(*di);
    for ( ; mi != me && isSubdirectory(*di, mi->second.path_); ++mi)
    {
      // Skip if it isn't a direct child.
      if (mi->second.path_ != *di)
	continue;

      // Handle reference histograms, with three distinct cases:
      // 1) Skip all references entirely on saving.
      // 2) Blanket saving of all references.
      // 3) Save only references for monitor elements with qtests.
      // The latter two are affected by "path" sub-tree selection,
      // i.e. references are saved only in the selected tree part.
      if (isSubdirectory(refpath, mi->first))
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
	  std::string mname(mi->first, s_referenceDirName.size()+1, std::string::npos);
	  MonitorElement *master = get(mname);
	  if (master)
	    for (size_t i = 0, e = master->data_.qreports.size(); i != e; ++i)
	      status = std::max(status, master->data_.qreports[i].code);

	  if (! master || status < minStatus)
	  {
	    if (verbose_>1)
	      std::cout << "DQMStore::save: skipping monitor element '"
		        << mi->first << "' while saving, status is "
			<< status << ", required minimum status is "
			<< minStatus << std::endl;
	    continue;
	  }
        }
      }

      if (verbose_>1)
	std::cout << "DQMStore::save: saving monitor element '"
		  << mi->first << "'\n";
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
      mi->second.data_.object->Write();

      // Save quality reports if this is not in reference section.
      if (! isSubdirectory(s_referenceDirName, mi->first))
      {
	qi = mi->second.data_.qreports.begin();
	qe = mi->second.data_.qreports.end();
	for ( ; qi != qe; ++qi)
	  TObjString(mi->second.qualityTagString(*qi).c_str()).Write();
      }
    }
  }

  f.Close();

#if !WITHOUT_CMS_FRAMEWORK
  // Report the file to job report service.
  edm::Service<edm::JobReport> jr;
  if (jr.isAvailable())
  {
    std::map<std::string, std::string> info;
    info["Source"] = "DQMStore";
    info["FileClass"] = "DQM";
    jr->reportAnalysisFile(filename, info);
  }
#endif

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
    if (dirpart.size() == s_monitorDirName.size())
      dirpart.clear();
    else if (dirpart[s_monitorDirName.size()] == '/')
      dirpart.erase(0, s_monitorDirName.size()+1);
      
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
  TKey *key;
  TIter next (gDirectory->GetListOfKeys());
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
    else
    {
      if (verbose_>2)
	std::cout << "DQMStore: reading object '" << obj->GetName()
		  << "' of type '" << obj->IsA()->GetName()
		  << "' from '" << file->GetName()
		  << "' into '" << dirpart << "'\n";

      makeDirectory(dirpart);
      if (extract(obj.get(), dirpart, overwrite))
	++count;
    }
  }

  if (verbose_>1)
    std::cout << "DQMStore: read " << count << '/' << ntot
	      << " objects from directory '" << dirpart << "'\n";

  return ntot + count;
}

/// public open/read root file <filename>, and copy MonitorElements;
/// if flag=true, overwrite identical MonitorElements (default: false);
/// if onlypath != "", read only selected directory
/// if prepend !="", prepend string to path
/// note: this method keeps the dir structure as in file
/// and does not update monitor element references!
void
DQMStore::open(const std::string &filename,
	       bool overwrite /* = false */,
	       const std::string &onlypath /* ="" */,
	       const std::string &prepend /* ="" */)
{
   readFile(filename,overwrite,onlypath,prepend,KeepRunDirs);
}

/// public load root file <filename>, and copy MonitorElements;
/// overwrite identical MonitorElements (default: true);
/// set DQMStore.collateHistograms to true to sum several files
/// note: this method strips off run dir structure
void 
DQMStore::load(const std::string &filename,
               OpenRunDirs stripdirs /* =StripRunDirs */)
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
    
   readFile(filename,overwrite,"","",stripdirs);
     
}

/// private readFile <filename>, and copy MonitorElements;
/// if flag=true, overwrite identical MonitorElements (default: false);
/// if onlypath != "", read only selected directory
/// if prepend !="", prepend string to path
/// if StripRunDirs is set the run and run summary folders are erased.
void
DQMStore::readFile(const std::string &filename,
	       bool overwrite /* = false */,
	       const std::string &onlypath /* ="" */,
	       const std::string &prepend /* ="" */,
	       OpenRunDirs stripdirs /* =StripRunDirs */)
{

  if (verbose_)
    std::cout << "DQMStore::open: reading from file '" << filename << "'\n";

  std::auto_ptr<TFile> f(TFile::Open(filename.c_str()));
  if (! f.get() || f->IsZombie())
    raiseDQMError("DQMStore", "Failed to open file '%s'", filename.c_str());

  unsigned n = readDirectory(f.get(), overwrite, onlypath, prepend, "", stripdirs);
  f->Close();

  MEMap::iterator mi = data_.begin();
  MEMap::iterator me = data_.end();
  for ( ; mi != me; ++mi)
    mi->second.updateQReportStats();

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
}

/// version info
std::string
DQMStore::getFileReleaseVersion(const std::string &filename)
{
  TFile f(filename.c_str());
  if (f.IsZombie())
    raiseDQMError("DQMStore", "Failed to open file '%s'", filename.c_str());

  // Loop over the contents of this directory in the file.
  TKey *key;
  TIter next (gDirectory->GetListOfKeys());
  std::string name;

  while ((key = (TKey *) next()))
  {
    name = key->ReadObj()->GetName();
    if (name.compare(0, 6, "\"CMSSW") == 0
	|| name.compare(0, 5, "CMSSW") == 0)
      break;
  }

  f.Close();
  return name;
}

std::string
DQMStore::getFileDQMPatchVersion(const std::string &filename)
{
  TFile f(filename.c_str());
  if (f.IsZombie())
    raiseDQMError("DQMStore", "Failed to open file '%s'", filename.c_str());

  // Loop over the contents of this directory in the file.
  TKey *key;
  TIter next (gDirectory->GetListOfKeys());
  std::string name;

  while ((key = (TKey *) next()))
  {
    name = key->ReadObj()->GetName();
    if (name.compare(0, 8, "DQMPATCH") == 0)
      break;
  }

  f.Close();
  return name;
}

std::string
DQMStore::getDQMPatchVersion(void)
{ return "DQMPATCH:" + s_dqmPatchVersion; } 

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

  MEMap::iterator e = data_.end();
  MEMap::iterator i = data_.lower_bound(*cleaned);
  while (i != e && isSubdirectory(*cleaned, i->second.path_))
  {
    removed_.push_back(i->second.data_.name);
    data_.erase(i++);
  }

  std::set<std::string>::iterator de = dirs_.end();
  std::set<std::string>::iterator di = dirs_.lower_bound(*cleaned);
  while (di != de && isSubdirectory(*cleaned, *di))
    dirs_.erase(di++);
}

/// remove all monitoring elements from directory; 
void
DQMStore::removeContents(const std::string &dir)
{
  MEMap::iterator e = data_.end();
  MEMap::iterator i = data_.lower_bound(dir);
  while (i != e && isSubdirectory(dir, i->second.path_))
    if (i->second.path_ == dir)
    {
      removed_.push_back(i->second.data_.name);
      data_.erase(i++);
    }
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
  std::string path;
  path.reserve(dir.size() + name.size() + 2);
  if (! dir.empty())
  {
    path += dir;
    path += '/';
  }
  path += name;

  MEMap::iterator pos = data_.find(path);
  if (pos == data_.end() && warning)
    std::cout << "DQMStore: WARNING: attempt to remove non-existent"
	      << " monitor element '" << path << "'\n";
  else
  {
    data_.erase(pos);
    removed_.push_back(path);
  }
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

  // Clean the path
  lat::Regexp *rx = 0;
  try
  {
    rx = new lat::Regexp(pattern, 0, lat::Regexp::Wildcard);
    rx->study();
  }
  catch (lat::Error &e)
  {
    delete rx;
    raiseDQMError("DQMStore", "Invalid wildcard pattern '%s' in quality"
		  " test specification", pattern.c_str());
  }

  // Record the test for future reference.
  QTestSpec qts(rx, qc);
  qtestspecs_.push_back(qts);

  // Apply the quality test.
  MEMap::iterator mi = data_.begin();
  MEMap::iterator me = data_.end();

  int cases = 0;
  for ( ; mi != me; ++mi)
    if (rx->match(mi->first))
    {
      ++cases;
      mi->second.addQReport(qts.second);
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
    if (mi->first.compare(0, s_referenceDirName.size(), s_referenceDirName))
      mi->second.runQTests();

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
    if (! cleaned->empty() && ! isSubdirectory(*cleaned, mi->second.path_))
      continue;

    if (mi->second.hasError())
      return dqm::qstatus::ERROR;
    else if (mi->second.hasWarning())
      status = dqm::qstatus::WARNING;
    else if (status < dqm::qstatus::WARNING
	     && mi->second.hasOtherReport())
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
/// reference histogram (from file) 

// copy into Referencedir
bool
DQMStore::makeReferenceME(MonitorElement *me)
{
  std::string refpath;
  refpath.reserve(s_referenceDirName.size() + me->path_.size() + 2);
  refpath += s_referenceDirName;
  refpath += '/';
  refpath += me->path_;

  makeDirectory(refpath);
  if (extract(me->data_.object, refpath, false))
  {
    MonitorElement *refme = getReferenceME(me);
    me->data_.reference = refme->data_.object;
    return true;
  }
  else
    return false;
}

// refme for given me
MonitorElement *
DQMStore::getReferenceME(MonitorElement *me) const
{
  std::string refdir;
  refdir.reserve(s_referenceDirName.size() + me->path_.size() + 2);
  refdir += s_referenceDirName;
  refdir += '/';
  refdir += me->path_;

  std::string dummy;
  return findObject(refdir, me->name_, dummy);
}

// check for existing
bool
DQMStore::isReferenceME(MonitorElement *me) const
{ return me && isSubdirectory(s_referenceDirName, me->path_); }

// check for existing
bool
DQMStore::isCollateME(MonitorElement *me) const
{ return me && isSubdirectory(s_collateDirName, me->path_); }

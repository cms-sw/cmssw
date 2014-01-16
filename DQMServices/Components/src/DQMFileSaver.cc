#include "DQMServices/Components/src/DQMFileSaver.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>
#include <TString.h>
#include <TSystem.h>

//--------------------------------------------------------
static void
getAnInt(const edm::ParameterSet &ps, int &value, const std::string &name)
{
  value = ps.getUntrackedParameter<int>(name, value);
  if (value < 1 && value != -1)
    throw cms::Exception("DQMFileSaver")
      << "Invalid '" << name << "' parameter '" << value
      << "'.  Must be -1 or >= 1.";
}

void
DQMFileSaver::saveForOffline(const std::string &workflow, int run, int lumi)
{

  char suffix[64];
  sprintf(suffix, "R%09d", run);

  char rewrite[128];
  if (lumi == 0) // save for run
    sprintf(rewrite, "\\1Run %d/\\2/Run summary", run);
  else
    sprintf(rewrite, "\\1Run %d/\\2/By Lumi Section %d-%d", irun_, ilumi_, ilumi_);

  size_t pos = 0;
  std::string wflow;
  wflow.reserve(workflow.size() + 3);
  wflow = workflow;
  while ((pos = wflow.find('/', pos)) != std::string::npos)
    wflow.replace(pos++, 1, "__");

  std::string filename = fileBaseName_ + suffix + wflow + child_ + ".root";

  if (lumi == 0) // save for run
  {
    // set run end flag
    dbe_->cd();
    dbe_->setCurrentFolder("Info/ProvInfo");

    // do this, because ProvInfo is not yet run in offline DQM
    MonitorElement* me = dbe_->get("Info/ProvInfo/CMSSW"); 
    if (!me) me = dbe_->bookString("CMSSW",edm::getReleaseVersion().c_str() );
    
    me = dbe_->get("Info/ProvInfo/runIsComplete");
    if (!me) me = dbe_->bookFloat("runIsComplete");

    if (me)
    { 
      if (runIsComplete_)
        me->Fill(1.);
      else
        me->Fill(0.);
    }

    dbe_->save(filename,
             "", 
	     "^(Reference/)?([^/]+)", 
	     rewrite,
	     (DQMStore::SaveReferenceTag) saveReference_,
	     saveReferenceQMin_,
	     fileUpdate_);
  }
  else // save EventInfo folders for luminosity sections
  {
    std::vector<std::string> systems = (dbe_->cd(), dbe_->getSubdirs());

    std::cout << " DQMFileSaver: storing EventInfo folders for Run: " 
              << irun_ << ", Lumi Section: " << ilumi_ << ", Subsystems: " ;
	      
    for (size_t i = 0, e = systems.size(); i != e; ++i) {
      if (systems[i] != "Reference") {
        dbe_->cd();
	std::cout << systems[i] << "  " ;
        dbe_->save(filename,
	   systems[i]+"/EventInfo", "^(Reference/)?([^/]+)", rewrite,
	   DQMStore::SaveWithoutReference,
	   dqm::qstatus::STATUS_OK,
	   fileUpdate_);
	// from now on update newly created file
	if (fileUpdate_=="RECREATE") fileUpdate_="UPDATE";
      }
    }
    std::cout << "\n";
  }

  if (pastSavedFiles_.size() == 0)
  {
    // save JobReport upon creation of file (once per job)
    saveJobReport(filename);
    pastSavedFiles_.push_back(filename);
  }
  
}

static void
doSaveForOnline(std::list<std::string> &pastSavedFiles,
		size_t numKeepSavedFiles,
		DQMStore *store,
		const std::string &filename,
		const std::string &directory,
		const std::string &rxpat,
		const std::string &rewrite,
		DQMStore::SaveReferenceTag saveref,
		int saveRefQMin)
{
  store->save(filename, directory , rxpat, 
         rewrite, saveref, saveRefQMin);
  pastSavedFiles.push_back(filename);
  if (pastSavedFiles.size() > numKeepSavedFiles)
  {
    remove(pastSavedFiles.front().c_str());
    pastSavedFiles.pop_front();
  }
}

void
DQMFileSaver::saveForOnline(const std::string &suffix, const std::string &rewrite)
{
  std::vector<std::string> systems = (dbe_->cd(), dbe_->getSubdirs());

  for (size_t i = 0, e = systems.size(); i != e; ++i)
  {
    if (systems[i] != "Reference")
    {
      dbe_->cd();
      if (MonitorElement* me = dbe_->get(systems[i] + "/EventInfo/processName"))
      {
	doSaveForOnline(pastSavedFiles_, numKeepSavedFiles_, dbe_,
			fileBaseName_ + me->getStringValue() + suffix + child_ + ".root",
			"", "^(Reference/)?([^/]+)", rewrite,
	                (DQMStore::SaveReferenceTag) saveReference_,
	                saveReferenceQMin_);
        return;
      }
    }
  }
  
  // look for EventInfo folder in an unorthodox location
  for (size_t i = 0, e = systems.size(); i != e; ++i)
    if (systems[i] != "Reference")
    { 
      dbe_->cd();
      std::vector<MonitorElement*> pNamesVector = dbe_->getMatchingContents("^" + systems[i] + "/.*/EventInfo/processName",lat::Regexp::Perl);
      if (pNamesVector.size() > 0){
        doSaveForOnline(pastSavedFiles_, numKeepSavedFiles_, dbe_,
                        fileBaseName_ + systems[i] + suffix + child_ + ".root",
                        "", "^(Reference/)?([^/]+)", rewrite,
                        (DQMStore::SaveReferenceTag) saveReference_,
                        saveReferenceQMin_);
        pNamesVector.clear();
        return;
      }
    }

  // if no EventInfo Folder is found, then store subsystem wise
  for (size_t i = 0, e = systems.size(); i != e; ++i)
    if (systems[i] != "Reference")
      doSaveForOnline(pastSavedFiles_, numKeepSavedFiles_, dbe_,
                      fileBaseName_ + systems[i] + suffix + child_ + ".root",
	              systems[i], "^(Reference/)?([^/]+)", rewrite,
	              (DQMStore::SaveReferenceTag) saveReference_,
	              saveReferenceQMin_);
}

void
DQMFileSaver::saveJobReport(const std::string &filename)
{

  // Report the file to job report service.
  edm::Service<edm::JobReport> jr;
  if (jr.isAvailable())
  {
    std::map<std::string, std::string> info;
    info["Source"] = "DQMStore";
    info["FileClass"] = "DQM";
    jr->reportAnalysisFile(filename, info);
  }

}

//--------------------------------------------------------
DQMFileSaver::DQMFileSaver(const edm::ParameterSet &ps)
  : convention_ (Offline),
    workflow_ (""),
    producer_ ("DQM"),
    dirName_ ("."),
    child_ (""),
    version_ (1),
    runIsComplete_ (false),
    saveByLumiSection_ (-1),
    saveByEvent_ (-1),
    saveByMinute_ (-1),
    saveByTime_ (-1),
    saveByRun_ (1),
    saveAtJobEnd_ (false),
    saveReference_ (DQMStore::SaveWithReference),
    saveReferenceQMin_ (dqm::qstatus::STATUS_OK),
    forceRunNumber_ (-1),
    fileBaseName_ (""),
    fileUpdate_ ("RECREATE"),
    dbe_ (&*edm::Service<DQMStore>()),
    irun_ (-1),
    ilumi_ (-1),
    ilumiprev_ (-1),
    ievent_ (-1),
    nrun_ (0),
    nlumi_ (0),
    nevent_ (0),
    numKeepSavedFiles_ (5)
{
  // Determine the file saving convention, and adjust defaults accordingly.
  std::string convention = ps.getUntrackedParameter<std::string>("convention", "Offline");
  if (convention == "Offline")
    convention_ = Offline;
  else if (convention == "Online")
    convention_ = Online;
  else
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'convention' parameter '" << convention << "'."
      << "  Expected one of 'Online' or 'Offline'.";

  // If this isn't online convention, check workflow.
  if (convention_ != Online)
  {
    workflow_ = ps.getUntrackedParameter<std::string>("workflow", workflow_);
    if (workflow_.empty()
	|| workflow_[0] != '/'
	|| *workflow_.rbegin() == '/'
	|| std::count(workflow_.begin(), workflow_.end(), '/') != 3
        || workflow_.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
				       "abcdefghijklmnopqrstuvwxyz"
				       "0123456789"
				       "-_/") != std::string::npos)
      throw cms::Exception("DQMFileSaver")
	<< "Invalid 'workflow' parameter '" << workflow_
	<< "'.  Expected '/A/B/C'.";
  }
  else if (! ps.getUntrackedParameter<std::string>("workflow", "").empty())
    throw cms::Exception("DQMFileSaver")
      << "The 'workflow' parameter must be empty in 'Online' convention.";
  else // for online set parameters
  {
    workflow_="/Global/Online/P5";
  }
    
  // Allow file producer to be set to specific values in certain conditions.
  producer_ = ps.getUntrackedParameter<std::string>("producer", producer_);
  if (convention_ == Online
      && producer_ != "DQM"
      && producer_ != "HLTDQM"
      && producer_ != "Playback")
  {
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'producer' parameter '" << producer_
      << "'.  Expected 'DQM', 'HLTDQM' or 'Playback'.";
  }
  else if (convention_ != Online && producer_ != "DQM")
  {
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'producer' parameter '" << producer_
      << "'.  Expected 'DQM'.";
  }

  // version number to be used in filename
  version_ = ps.getUntrackedParameter<int>("version", version_);
  // flag to signal that file contains data from complete run
  runIsComplete_ = ps.getUntrackedParameter<bool>("runIsComplete", runIsComplete_);

  // Check how we should save the references.
  std::string refsave = ps.getUntrackedParameter<std::string>("referenceHandling", "default");
  if (refsave == "default")
    ;
  else if (refsave == "skip") 
  {
    saveReference_ = DQMStore::SaveWithoutReference;
  //  std::cout << "skip saving all references" << std::endl;
  }
  else if (refsave == "all")
  {
    saveReference_ = DQMStore::SaveWithReference;
  //  std::cout << "saving all references" << std::endl;
  }
  else if (refsave == "qtests")
  {
    saveReference_ = DQMStore::SaveWithReferenceForQTest;
  //  std::cout << "saving qtest references" << std::endl;
  }
  else
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'referenceHandling' parameter '" << refsave
      << "'.  Expected 'default', 'skip', 'all' or 'qtests'.";

  // Check minimum required quality test result for which reference is saved.
  saveReferenceQMin_ = ps.getUntrackedParameter<int>("referenceRequireStatus", saveReferenceQMin_);

  // Get and check the output directory.
  struct stat s;
  dirName_ = ps.getUntrackedParameter<std::string>("dirName", dirName_);
  if (dirName_.empty() || stat(dirName_.c_str(), &s) == -1)
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'dirName' parameter '" << dirName_ << "'.";

  // Find out when and how to save files.  The following contraints apply:
  // - For online, allow files to be saved at event and time intervals.
  // - For online and offline, allow files to be saved per run, lumi and job end
  // - For offline allow run number to be overridden (for mc data).
  if (convention_ == Online)
  {
    getAnInt(ps, saveByEvent_, "saveByEvent");
    getAnInt(ps, saveByMinute_, "saveByMinute");
    getAnInt(ps, saveByTime_, "saveByTime");
    getAnInt(ps, numKeepSavedFiles_, "maxSavedFilesCount");
  }

  if (convention_ == Online || convention_ == Offline)
  {
    getAnInt(ps, saveByRun_, "saveByRun");
    getAnInt(ps, saveByLumiSection_, "saveByLumiSection");
  }

  if (convention_ != Online)
  {
    getAnInt(ps, forceRunNumber_, "forceRunNumber");
    saveAtJobEnd_ = ps.getUntrackedParameter<bool>("saveAtJobEnd", saveAtJobEnd_);
  }

  if (saveAtJobEnd_ && forceRunNumber_ < 1)
    throw cms::Exception("DQMFileSaver")
      << "If saving at the end of the job, the run number must be"
      << " overridden to a specific value using 'forceRunNumber'.";

  
  // Set up base file name and determine the start time.
  char version[8];
  sprintf(version, "_V%04d_", int(version_));
  version[7]='\0';
  fileBaseName_ = dirName_ + "/" + producer_ + version;
  gettimeofday(&start_, 0);
  saved_ = start_;

  // Log some information what we will do.
  edm::LogInfo("DQMFileSaver")
    << "DQM file saving settings:\n"
    << " using base file name '" << fileBaseName_ << "'\n"
    << " forcing run number " << forceRunNumber_ << "\n"
    << " saving every " << saveByLumiSection_ << " lumi section(s)\n"
    << " saving every " << saveByEvent_ << " event(s)\n"
    << " saving every " << saveByMinute_ << " minute(s)\n"
    << " saving every 2^n*" << saveByTime_ << " minutes \n"
    << " saving every " << saveByRun_ << " run(s)\n"
    << " saving at job end: " << (saveAtJobEnd_ ? "yes" : "no") << "\n"
    << " keeping at most " << numKeepSavedFiles_ << " files\n";
}

//--------------------------------------------------------
void
DQMFileSaver::beginJob()
{
  irun_ = ilumi_ = ilumiprev_ = ievent_ = -1;
  nrun_ = nlumi_ = nevent_ = 0;
}

void
DQMFileSaver::beginRun(const edm::Run &r, const edm::EventSetup &)
{
  irun_     = (forceRunNumber_ == -1 ? r.id().run() : forceRunNumber_);
  ++nrun_;
}

void
DQMFileSaver::beginLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup &)
{
  ilumi_    = l.id().luminosityBlock();
  if (ilumiprev_ == -1) ilumiprev_ = ilumi_;
  ++nlumi_;
}

void DQMFileSaver::analyze(const edm::Event &e, const edm::EventSetup &)
{
  ++nevent_;

  ievent_   = e.id().event();

  // Check if we should save for this event.
  char suffix[64];
  if (ievent_ > 0 && saveByEvent_ > 0 && nevent_ == saveByEvent_)
  {
    if (convention_ != Online)
      throw cms::Exception("DQMFileSaver")
	<< "Internal error, can save files by event"
	<< " only in Online mode.";

    sprintf(suffix, "_R%09d_E%08d", irun_, ievent_);
    saveForOnline(suffix, "\\1\\2");
    nevent_ = 0;
  }

  // Check if we should save due to elapsed time.
  if ( ievent_ > 0 && ( saveByMinute_ > 0 || saveByTime_ > 0 ) )
  {
    if (convention_ != Online)
      throw cms::Exception("DQMFileSaver")
	<< "Internal error, can save files by time"
	<< " only in Online mode.";

    // Compute elapsed time in minutes.
    struct timeval tv;
    gettimeofday(&tv, 0);

    double totalelapsed = ((tv.tv_sec + tv.tv_usec*1e-6)
		 - (start_.tv_sec + start_.tv_usec*1e-6)) / 60;
    double elapsed = ((tv.tv_sec + tv.tv_usec*1e-6)
		      - (saved_.tv_sec + saved_.tv_usec*1e-6)) / 60;

    // Save if enough time has elapsed since the last save.
    if ( (saveByMinute_ > 0 && elapsed > saveByMinute_ ) ||
         (saveByTime_ > 0   && totalelapsed > saveByTime_ ) )
    {
      if ( saveByTime_ > 0 ) saveByTime_ *= 2;
      saved_ = tv;
      sprintf(suffix, "_R%09d_T%08d", irun_, int(totalelapsed));
      char rewrite[64]; sprintf(rewrite, "\\1Run %d/\\2/Run summary", irun_);
      saveForOnline(suffix, rewrite);
    }
  }
}

void
DQMFileSaver::endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
{

  if (ilumi_ > 0 && saveByLumiSection_ > 0 )
  {
    if (convention_ != Online && convention_ != Offline )
      throw cms::Exception("DQMFileSaver")
	<< "Internal error, can save files at end of lumi block"
	<< " only in Online or Offline mode.";

    if (convention_ == Online && nlumi_ == saveByLumiSection_) // insist on lumi section ordering
    {
      char suffix[64];
      char rewrite[128];
      sprintf(suffix, "_R%09d_L%06d", irun_, ilumi_);
      sprintf(rewrite, "\\1Run %d/\\2/By Lumi Section %d-%d", irun_, ilumiprev_, ilumi_);
      saveForOnline(suffix, rewrite);
      ilumiprev_ = -1;
      nlumi_ = 0;
    }
    if (convention_ == Offline)
      saveForOffline(workflow_, irun_, ilumi_);
  }
}

void
DQMFileSaver::endRun(const edm::Run &, const edm::EventSetup &)
{
  if (irun_ > 0 && saveByRun_ > 0 && nrun_ == saveByRun_)
  {
    if (convention_ == Online)
    {
      char suffix[64]; sprintf(suffix, "_R%09d", irun_);
      char rewrite[64]; sprintf(rewrite, "\\1Run %d/\\2/Run summary", irun_);
      saveForOnline(suffix, rewrite);
    }
    else if (convention_ == Offline)
      saveForOffline(workflow_, irun_, 0);
    else
      throw cms::Exception("DQMFileSaver")
	<< "Internal error.  Can only save files in endRun()"
	<< " in Online and Offline modes.";

    nrun_ = 0;
  }
}

void
DQMFileSaver::endJob(void)
{ 
  if (saveAtJobEnd_)
  {
    if (convention_ == Offline && forceRunNumber_ > 0)
      saveForOffline(workflow_, forceRunNumber_, 0);
    else
      throw cms::Exception("DQMFileSaver")
	<< "Internal error.  Can only save files at the end of the"
	<< " job in Offline mode with run number overridden.";
  }
    
}

void
DQMFileSaver::postForkReacquireResources(unsigned int childIndex, unsigned int numberOfChildren)
{
  // this is copied from IOPool/Output/src/PoolOutputModule.cc, for consistency
  unsigned int digits = 0;
  while (numberOfChildren != 0) {
    ++digits;
    numberOfChildren /= 10;
  }
  // protect against zero numberOfChildren
  if (digits == 0) {
    digits = 3;
  }

  char buffer[digits + 2];
  snprintf(buffer, digits + 2, "_%0*d", digits, childIndex);
  child_ = std::string(buffer);
}

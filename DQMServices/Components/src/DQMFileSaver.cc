#include "DQMServices/Components/src/DQMFileSaver.h"
#include "DQMServices/Core/interface/DQMStore.h"
// #include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
// #include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <vector>

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

static void
saveForOffline(DQMStore *dbe,
	       const std::string &fileBaseName,
	       const std::string &workflow,
	       int run)
{
  char suffix[64];
  sprintf(suffix, "R%09d", run);

  char rewrite[64];
  sprintf(rewrite, "Run %d/\\1/Run summary/", run);

  size_t pos = 0;
  std::string wflow;
  wflow.reserve(workflow.size() + 3);
  wflow = workflow;
  while ((pos = wflow.find('/', pos)) != std::string::npos)
    wflow.replace(pos++, 1, "__");

  dbe->save(fileBaseName + suffix + wflow + ".root",
	     "", "^([^/]+)/", rewrite);
}

static void
saveForOnline(DQMStore *dbe,
	      const std::string &fileBaseName,
	      const std::string &suffix,
	      const std::string &rewrite)
{
  std::vector<std::string> systems = (dbe->cd(), dbe->getSubdirs());
  for (size_t i = 0, e = systems.size(); i != e; ++i)
    dbe->save(fileBaseName + systems[i] + suffix + ".root",
	      systems[i], "^([^/]+)/", rewrite);
}

//--------------------------------------------------------
DQMFileSaver::DQMFileSaver(const edm::ParameterSet &ps)
  : convention_ (Offline),
    workflow_ (""),
    producer_ ("DQM"),
    dirName_ ("."),
    saveByLumiSection_ (-1),
    saveByEvent_ (-1),
    saveByMinute_ (-1),
    saveByRun_ (1),
    saveAtJobEnd_ (false),
    forceRunNumber_ (-1),
    fileBaseName_ (""),
    dbe_ (&*edm::Service<DQMStore>()),
    irun_ (-1),
    ilumi_ (-1),
    ilumiprev_ (-1),
    ievent_ (-1),
    nrun_ (0),
    nlumi_ (0),
    nevent_ (0)
{
  // Determine the file saving convention, and adjust defaults accordingly.
  std::string convention = ps.getUntrackedParameter<std::string>("convention", "Offline");
  if (convention == "Offline")
    convention_ = Offline;
  else if (convention == "Online")
    convention_ = Online;
  else if (convention == "RelVal")
  {
    convention_ = RelVal;
    saveByRun_ = -1;
    saveAtJobEnd_ = true;
    forceRunNumber_ = 1;
  }
  else
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'convention' parameter '" << convention << "'."
      << "  Expected one of 'Online', 'Offline' or 'RelVal'.";

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

  // In RelVal mode use workflow "stream" name instead of producer label.
  if (convention_ == RelVal)
    producer_ = workflow_.substr(1, workflow_.find('/', 1)-1);

  // Get and check the output directory.
  struct stat s;
  dirName_ = ps.getUntrackedParameter<std::string>("dirName", dirName_);
  if (dirName_.empty() || stat(dirName_.c_str(), &s) == -1)
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'dirName' parameter '" << dirName_ << "'.";

  // Find out when and how to save files.  The following contraints apply:
  // - For online, allow files to be saved at lumi, event and time intervals.
  // - For online and offline, allow files to be saved per run.
  // - For offline and relval, allow files to be saved at job end
  //   and run number to be overridden (for mc data).
  if (convention_ == Online)
  {
    getAnInt(ps, saveByLumiSection_, "saveByLumiSection");
    getAnInt(ps, saveByEvent_, "saveByEvent");
    getAnInt(ps, saveByMinute_, "saveByMinute");
  }

  if (convention_ == Online || convention_ == Offline)
    getAnInt(ps, saveByRun_, "saveByRun");

  if (convention_ != Online)
  {
    getAnInt(ps, forceRunNumber_, "forceRunNumber");
    saveAtJobEnd_ = ps.getUntrackedParameter<bool>("saveAtJobEnd", saveAtJobEnd_);
    if (convention_ == RelVal && ! saveAtJobEnd_)
      saveAtJobEnd_ = true;
    if (convention_ == RelVal && forceRunNumber_ == -1)
      forceRunNumber_ = 1;
  }

  if (saveAtJobEnd_ && forceRunNumber_ < 1)
    throw cms::Exception("DQMFileSaver")
      << "If saving at the end of the job, the run number must be"
      << " overridden to a specific value using 'forceRunNumber'.";

  // Set up base file name and determine the start time.
  fileBaseName_ = dirName_ + "/" + producer_ + "_";
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
    << " saving every " << saveByRun_ << " run(s)\n"
    << " saving at job end: " << (saveAtJobEnd_ ? "yes" : "no") << "\n";
}

//--------------------------------------------------------
void
DQMFileSaver::beginJob(const edm::EventSetup &)
{
  irun_ = ilumi_ = ilumiprev_ = ievent_ = -1;
  nrun_ = nlumi_ = nevent_ = 0;
}

void
DQMFileSaver::beginRun(const edm::Run &, const edm::EventSetup &)
{
  ++nrun_;
}

void
DQMFileSaver::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
{
  ++nlumi_;
}

void DQMFileSaver::analyze(const edm::Event &e, const edm::EventSetup &)
{
  ++nevent_;

  // Get event parameters.
  irun_     = (forceRunNumber_ == -1 ? e.id().run() : forceRunNumber_);
  ilumi_    = e.luminosityBlock();
  ievent_   = e.id().event();
  if (ilumiprev_ == -1)
    ilumiprev_ = ilumi_;

  // Check if we should save for this event.
  char suffix[64];
  if (ievent_ > 0 && saveByEvent_ > 0 && nevent_ == saveByEvent_)
  {
    if (convention_ != Online)
      throw cms::Exception("DQMFileSaver")
	<< "Internal error, can save files by event"
	<< " only in Online mode.";

    sprintf(suffix, "_R%09d_E%08d", irun_, ievent_);
    saveForOnline(dbe_, fileBaseName_, suffix, "\\1/");
    nevent_ = 0;
  }

  // Check if we should save due to elapsed time.
  if (ievent_ > 0 && saveByMinute_ > 0)
  {
    if (convention_ != Online)
      throw cms::Exception("DQMFileSaver")
	<< "Internal error, can save files by time"
	<< " only in Online mode.";

    // Compute elapsed time in minutes.
    struct timeval tv;
    gettimeofday(&tv, 0);
    double elapsed = ((tv.tv_sec + tv.tv_usec*1e-6)
		      - (saved_.tv_sec + saved_.tv_usec*1e-6)) / 60;

    // Save if enough time has elapsed since the last save.
    if (elapsed > saveByMinute_)
    {
      saved_ = tv;
      elapsed = ((tv.tv_sec + tv.tv_usec*1e-6)
		 - (start_.tv_sec + start_.tv_usec*1e-6)) / 60;
      sprintf(suffix, "_R%09d_T%08d", irun_, int(elapsed));
      saveForOnline(dbe_, fileBaseName_, suffix, "\\1/");
    }
  }
}

void
DQMFileSaver::endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
{
  if (ilumi_ > 0 && saveByLumiSection_ > 0 && nlumi_ == saveByLumiSection_)
  {
    if (convention_ != Online)
      throw cms::Exception("DQMFileSaver")
	<< "Internal error, can save files at end of lumi block"
	<< " only in Online mode.";

    char suffix[64];
    char rewrite[128];
    sprintf(suffix, "_R%09d_L%06d", irun_, ilumi_);
    sprintf(rewrite, "Run %d/\\1/By Lumi Section %d-%d/", irun_, ilumiprev_, ilumi_);
    saveForOnline(dbe_, fileBaseName_, suffix, rewrite);
    ilumiprev_ = -1;
    nlumi_ = 0;
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
      char rewrite[64]; sprintf(rewrite, "Run %d/\\1/Run summary/", irun_);
      saveForOnline(dbe_, fileBaseName_, suffix, rewrite);
    }
    else if (convention_ == Offline)
      saveForOffline(dbe_, fileBaseName_, workflow_, irun_);
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
    if (convention_ == RelVal)
    {
      size_t pos;
      std::string release = edm::getReleaseVersion();
      while ((pos = release.find('"')) != std::string::npos)
	release.erase(pos, 1);

      pos = fileBaseName_.rfind('/');
      std::string stream = fileBaseName_.substr(pos+1, fileBaseName_.size()-pos-2);
      dbe_->save(fileBaseName_ + release + ".root", "",
		 "^([^/]+)/", stream + "/\\1/");
    }
    else if (convention_ == Offline && forceRunNumber_ > 0)
      saveForOffline(dbe_, fileBaseName_, workflow_, forceRunNumber_);
    else
      throw cms::Exception("DQMFileSaver")
	<< "Internal error.  Can only save files at the end of the"
	<< " job in RelVal and Offline modes with run number overridden.";
  }
}

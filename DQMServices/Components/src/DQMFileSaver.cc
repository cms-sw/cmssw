#include "DQMServices/Components/src/DQMFileSaver.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
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
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <TString.h>
#include <TSystem.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>

//--------------------------------------------------------
const std::string DQMFileSaver::streamPrefix_("stream");
const std::string DQMFileSaver::streamSuffix_("Histograms");

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

static std::string
dataFileExtension(DQMFileSaver::FileFormat fileFormat)
{
  std::string extension;
  if (fileFormat == DQMFileSaver::ROOT)
    extension = ".root";
  else if (fileFormat ==  DQMFileSaver::PB)
    extension = ".pb";
  return extension;
}

static std::string
onlineOfflineFileName(const std::string &fileBaseName,
                      const std::string &suffix,
                      const std::string &workflow,
                      const std::string &child,
                      DQMFileSaver::FileFormat fileFormat)
{
  size_t pos = 0;
  std::string wflow;
  wflow.reserve(workflow.size() + 3);
  wflow = workflow;
  while ((pos = wflow.find('/', pos)) != std::string::npos)
    wflow.replace(pos++, 1, "__");

  std::string filename = fileBaseName + suffix + wflow + child + dataFileExtension(fileFormat);
  return filename;
}

void
DQMFileSaver::saveForOfflinePB(const std::string &workflow,
                               int run)
{
  char suffix[64];
  sprintf(suffix, "R%09d", run);
  std::string filename = onlineOfflineFileName(fileBaseName_, std::string(suffix), workflow, child_, PB);
  dbe_->savePB(filename, filterName_);
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

  std::string filename = onlineOfflineFileName(fileBaseName_, std::string(suffix), workflow, child_, ROOT);

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
               enableMultiThread_ ? run : 0,
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
                   systems[i]+"/EventInfo", "^(Reference/)?([^/]+)",
                   rewrite,
                   enableMultiThread_ ? run : 0,
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
		int saveRefQMin,
                const std::string &filterName,
                DQMFileSaver::FileFormat fileFormat)
{
  // TODO(rovere): fix the online case. so far we simply rely on the
  // fact that we assume we will not run multithreaded in online.
  if (fileFormat == DQMFileSaver::ROOT)
    store->save(filename,
                directory,
                rxpat,
                rewrite,
                0,
                saveref,
                saveRefQMin);
  else if (fileFormat == DQMFileSaver::PB)
    store->savePB(filename, filterName);
  pastSavedFiles.push_back(filename);
  if (pastSavedFiles.size() > numKeepSavedFiles)
  {
    remove(pastSavedFiles.front().c_str());
    pastSavedFiles.pop_front();
  }
}

void
DQMFileSaver::saveForOnlinePB(const std::string &suffix)
{
  // The file name contains the Online workflow name,
  // as we do not want to look inside the DQMStore,
  // and the @a suffix, defined in the run/lumi transitions.
  // TODO(diguida): add the possibility to change the dir structure with rewrite.
  std::string filename = onlineOfflineFileName(fileBaseName_, suffix, workflow_, child_, PB);
  doSaveForOnline(pastSavedFiles_, numKeepSavedFiles_, dbe_,
		  filename,
		  "", "^(Reference/)?([^/]+)", "\\1\\2",
		  (DQMStore::SaveReferenceTag) saveReference_,
		  saveReferenceQMin_,
		  filterName_,
		  PB);
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
	                saveReferenceQMin_,
			"", ROOT);
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
                        saveReferenceQMin_,
			"", ROOT);
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
                      saveReferenceQMin_,
                      "", ROOT);
}

void
DQMFileSaver::fillJson(int run, int lumi, const std::string& dataFilePathName, boost::property_tree::ptree& pt)
{
  namespace bpt = boost::property_tree;
  namespace bfs = boost::filesystem;
  int hostnameReturn;
  char host[32];
  hostnameReturn = gethostname(host ,sizeof(host));
  if (hostnameReturn == -1)
    throw cms::Exception("DQMFileSaver")
          << "Internal error, cannot get host name";
  //TODO(diguida): use a fake pid for testing purposes
  int pid = getpid();
  std::ostringstream oss_pid;
  oss_pid << pid;

  // Stat the data file: if not there, throw
  struct stat dataFileStat;
  if (stat(dataFilePathName.c_str(), &dataFileStat) != 0)
    throw cms::Exception("DQMFileSaver")
          << "Internal error, cannot get data file: "
	  << dataFilePathName;
  // Extract only the data file name from the full path
  std::string dataFileName = bfs::path(dataFilePathName).filename().string();
  // The availability test of the FastMonitoringService was done in the ctor.
  bpt::ptree data;
  bpt::ptree processedEvents, acceptedEvents, errorEvents, bitmask, fileList, fileSize, inputFiles;
  processedEvents.put("", fms_->getEventsProcessedForLumi(lumi)); // Processed events
  acceptedEvents.put("", fms_->getEventsProcessedForLumi(lumi)); // Accepted events, same as processed for our purposes
  errorEvents.put("", 0); // Error events
  bitmask.put("", 0); // Bitmask of abs of CMSSW return code
  fileList.put("", dataFileName); // Data file the information refers to
  fileSize.put("", dataFileStat.st_size); // Size in bytes of the data file
  inputFiles.put("", ""); // We do not care about input files!

  data.push_back(std::make_pair("", processedEvents));
  data.push_back(std::make_pair("", acceptedEvents));
  data.push_back(std::make_pair("", errorEvents));
  data.push_back(std::make_pair("", bitmask));
  data.push_back(std::make_pair("", fileList));
  data.push_back(std::make_pair("", fileSize));
  data.push_back(std::make_pair("", inputFiles));

  pt.add_child("data", data);
  // The availability test of the EvFDaqDirector Service was done in the ctor.
  bfs::path outJsonDefName(edm::Service<evf::EvFDaqDirector>()->baseRunDir()); //we assume this file is written bu the EvF Output module
  outJsonDefName /= (std::string("output_") + oss_pid.str() + std::string(".jsd"));
  pt.put("definition", outJsonDefName.string());
  char sourceInfo[64]; //host and pid information
  sprintf(sourceInfo, "%s_%d", host, pid);
  pt.put("source", sourceInfo);
}

void
DQMFileSaver::saveForFilterUnit(const std::string& rewrite, int run, int lumi, const DQMFileSaver::FileFormat fileFormat)
{
  // get from DAQ2 services where to store the files according to their format
  namespace bpt = boost::property_tree;
  bpt::ptree pt;
  std::string openHistoFilePathName;
  std::string histoFilePathName;
  std::string openJsonFilePathName = edm::Service<evf::EvFDaqDirector>()->getOpenOutputJsonFilePath(lumi, stream_label_);
  std::string jsonFilePathName = edm::Service<evf::EvFDaqDirector>()->getOutputJsonFilePath(lumi, stream_label_);
  if (fileFormat == ROOT)
  {
    openHistoFilePathName = edm::Service<evf::EvFDaqDirector>()->getOpenRootHistogramFilePath(lumi, stream_label_);
    histoFilePathName = edm::Service<evf::EvFDaqDirector>()->getRootHistogramFilePath(lumi, stream_label_);
    // Save the file with the full directory tree,
    // modifying it according to @a rewrite,
    // but not looking for MEs inside the DQMStore, as in the online case,
    // nor filling new MEs, as in the offline case.
    // TODO(diguida): check if this is multithread friendly!
    dbe_->save(openHistoFilePathName,
               "",
               "^(Reference/)?([^/]+)",
               rewrite,
               0,
               (DQMStore::SaveReferenceTag) saveReference_,
               saveReferenceQMin_,
               fileUpdate_);
  }
  else if (fileFormat == PB)
  {
    openHistoFilePathName = edm::Service<evf::EvFDaqDirector>()->getOpenProtocolBufferHistogramFilePath(lumi, stream_label_);
    histoFilePathName = edm::Service<evf::EvFDaqDirector>()->getProtocolBufferHistogramFilePath(lumi, stream_label_);
    // Save the file in the open directory.
    // TODO(diguida): check if this is multithread friendly!
    dbe_->savePB(openHistoFilePathName, filterName_);
  }
  else
    throw cms::Exception("DQMFileSaver")
          << "Internal error, can save files"
          << " only in ROOT or ProtocolBuffer format.";
  this->fillJson(run, lumi, openHistoFilePathName, pt);
  // Write the json file in the open directory.
  write_json(openJsonFilePathName, pt);
  // Now move the the data and json files into the output directory.
  int result = rename(openHistoFilePathName.c_str(), histoFilePathName.c_str());
  // Check that the histogram file is there
  struct stat histoFileStat;
  if ((result != 0) || (stat(histoFilePathName.c_str(), &histoFileStat) != 0))
    throw cms::Exception("DQMFileSaver")
          << "Internal error, cannot get data file: "
	  << histoFilePathName;
  // TODO(diguida): check that the histogram file size after renaming is the same as the one in the json tree
  result = rename(openJsonFilePathName.c_str(), jsonFilePathName.c_str());
  if (result != 0)
    throw cms::Exception("DQMFileSaver")
          << "Internal error, cannot move json file: "
	  << openJsonFilePathName
	  << " into: " << jsonFilePathName;
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
    fileFormat_(ROOT),
    workflow_ (""),
    producer_ ("DQM"),
    stream_label_ (""),
    dirName_ ("."),
    child_ (""),
    filterName_(""),
    version_ (1),
    runIsComplete_ (false),
    enableMultiThread_(ps.getUntrackedParameter<bool>("enableMultiThread", false)),
    saveByLumiSection_ (-1),
    saveByEvent_ (-1),
    saveByMinute_ (-1),
    saveByTime_ (-1),
    saveByRun_ (-1),
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
    numKeepSavedFiles_ (5),
    fms_(nullptr)
{
  // Determine the file saving convention, and adjust defaults accordingly.
  std::string convention = ps.getUntrackedParameter<std::string>("convention", "Offline");
  if (convention == "Offline")
    convention_ = Offline;
  else if (convention == "Online")
    convention_ = Online;
  else if (convention == "FilterUnit")
    convention_ = FilterUnit;
  else
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'convention' parameter '" << convention << "'."
      << "  Expected one of 'Online' or 'Offline' or 'FilterUnit'.";

  // If this is neither online nor FU convention, check workflow.
  // In this way, FU is treated as online, so we cannot specify a workflow. TBC
  if (convention_ != Online && convention_ != FilterUnit)
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
      << "The 'workflow' parameter must be empty in 'Online' and 'FilterUnit' conventions.";
  else // for online set parameters
  {
    workflow_="/Global/Online/P5";
  }

  // Determine the file format, and adjust defaults accordingly.
  std::string fileFormat = ps.getUntrackedParameter<std::string>("fileFormat", "ROOT");
  if (fileFormat == "ROOT")
    fileFormat_ = ROOT;
  else if (fileFormat == "PB")
    fileFormat_ = PB;
  else
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'fileFormat' parameter '" << fileFormat << "'."
      << "  Expected one of 'ROOT' or 'PB'.";

  // Allow file producer to be set to specific values in certain conditions.
  producer_ = ps.getUntrackedParameter<std::string>("producer", producer_);
  // Setting the same constraints on file producer both for online and FilterUnit conventions
  // TODO(diguida): limit the producer for FilterUnit to be 'DQM' or 'HLTDQM'?
  // TODO(diguida): how to handle histograms produced in the playback for the FU case?
  if ((convention_ == Online || convention_ == FilterUnit)
      && producer_ != "DQM"
      && producer_ != "HLTDQM"
      && producer_ != "Playback")
  {
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'producer' parameter '" << producer_
      << "'.  Expected 'DQM', 'HLTDQM' or 'Playback'.";
  }
  else if (convention_ != Online
           && convention != FilterUnit
           && producer_ != "DQM")
  {
    throw cms::Exception("DQMFileSaver")
      << "Invalid 'producer' parameter '" << producer_
      << "'.  Expected 'DQM'.";
  }
  stream_label_ = streamPrefix_ + producer_ + streamSuffix_;

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

  filterName_ = ps.getUntrackedParameter<std::string>("filterName", filterName_);
  // Find out when and how to save files.  The following contraints apply:
  // - For online, filter unit, and offline, allow files to be saved per lumi
  // - For online, allow files to be saved per run, at event and time intervals.
  // - For offline allow files to be saved per run, at job end, and run number to be overridden (for mc data).
  if (convention_ == Online || convention_ == Offline || convention_ == FilterUnit)
  {
    getAnInt(ps, saveByLumiSection_, "saveByLumiSection");
  }

  if (convention_ == Online)
  {
    getAnInt(ps, saveByRun_, "saveByRun");
    getAnInt(ps, saveByEvent_, "saveByEvent");
    getAnInt(ps, saveByMinute_, "saveByMinute");
    getAnInt(ps, saveByTime_, "saveByTime");
    getAnInt(ps, numKeepSavedFiles_, "maxSavedFilesCount");
  }

  if (convention_ == Offline)
  {
    getAnInt(ps, saveByRun_, "saveByRun");
    getAnInt(ps, forceRunNumber_, "forceRunNumber");
    saveAtJobEnd_ = ps.getUntrackedParameter<bool>("saveAtJobEnd", saveAtJobEnd_);
  }

  // Set up base file name:
  // - for online and offline, follow the convention <dirName>/<producer>_V<4digits>_
  // - for FilterUnit, we need to follow the DAQ2 convention, so we need the run and lumisection:
  //   the path is provided by the DAQ director service.
  if (convention_ != FilterUnit)
  {
    char version[8];
    sprintf(version, "_V%04d_", int(version_));
    version[7]='\0';
    fileBaseName_ = dirName_ + "/" + producer_ + version;
  }
  else
  {
    // For FU, dirName_ will not be considered at all
    edm::LogInfo("DQMFileSaver")
      << "The base dir provided in the configuration '" << dirName_ << "'\n"
      << " will not be considered: for FU, the DAQ2 services will handle directories\n";
    //check that DAQ2 services are available: throw if not
    fms_ = (evf::FastMonitoringService *) (edm::Service<evf::MicroStateService>().operator->());
    evf::EvFDaqDirector * daqDirector = (evf::EvFDaqDirector *) (edm::Service<evf::EvFDaqDirector>().operator->());
    if (!(fms_ && daqDirector))
      throw cms::Exception("DQMFileSaver")
              << "Internal error, cannot initialize DAQ services.";
  }
  //Determine the start time.
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
  // For Filter Unit, create an empty ini file:
  // it is needed by the HLT deamon in order to start merging
  // The run number is established in the service
  // TODO(diguida): check that they are the same?
  if (convention_ == FilterUnit)
  {
    evf::EvFDaqDirector * daqDirector = (evf::EvFDaqDirector *) (edm::Service<evf::EvFDaqDirector>().operator->());
    const std::string initFileName = daqDirector->getInitFilePath(stream_label_);
    std::ofstream file(initFileName);
    file.close();
  }
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
    if (fileFormat_ == ROOT)
      saveForOnline(suffix, "\\1\\2");
    else if (fileFormat_ == PB)
      saveForOnlinePB(suffix);
    else
      throw cms::Exception("DQMFileSaver")
        << "Internal error, can save files"
        << " only in ROOT or ProtocolBuffer format.";
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
      char rewrite[64];
      sprintf(rewrite, "\\1Run %d/\\2/Run summary", irun_);
      if (fileFormat_ == ROOT)
        saveForOnline(suffix, rewrite);
      else if (fileFormat_ == PB)
        saveForOnlinePB(suffix);
      else
        throw cms::Exception("DQMFileSaver")
          << "Internal error, can save files"
          << " only in ROOT or ProtocolBuffer format.";
    }
  }
}

void
DQMFileSaver::endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
{

  if (ilumi_ > 0 && saveByLumiSection_ > 0 )
  {
    if (convention_ != Online && convention_ != FilterUnit && convention_ != Offline )
      throw cms::Exception("DQMFileSaver")
	<< "Internal error, can save files at end of lumi block"
	<< " only in Online, FilterUnit or Offline mode.";

    if (convention_ == Online && nlumi_ == saveByLumiSection_) // insist on lumi section ordering
    {
      char suffix[64];
      char rewrite[128];
      sprintf(suffix, "_R%09d_L%06d", irun_, ilumi_);
      sprintf(rewrite, "\\1Run %d/\\2/By Lumi Section %d-%d", irun_, ilumiprev_, ilumi_);
      if (fileFormat_ == ROOT)
        saveForOnline(suffix, rewrite);
      else if (fileFormat_ == PB)
        saveForOnlinePB(suffix);
      else
        throw cms::Exception("DQMFileSaver")
          << "Internal error, can save files"
          << " only in ROOT or ProtocolBuffer format.";
      ilumiprev_ = -1;
      nlumi_ = 0;
    }
    if (convention_ == FilterUnit) // store at every lumi section end
    {
      char rewrite[128];
      sprintf(rewrite, "\\1Run %d/\\2/By Lumi Section %d-%d", irun_, ilumi_, ilumi_);
      saveForFilterUnit(rewrite, irun_, ilumi_, fileFormat_);
    }
    if (convention_ == Offline)
    {
      if (fileFormat_ == ROOT)
        saveForOffline(workflow_, irun_, ilumi_);
      else
      // TODO(diguida): do we need to support lumisection saving in Offline for PB?
      // In this case, for ROOT, we only save EventInfo folders: we can filter them...
        throw cms::Exception("DQMFileSaver")
          << "Internal error, can save files"
          << " only in ROOT format.";
    }
  }
}

void
DQMFileSaver::endRun(const edm::Run &, const edm::EventSetup &)
{
  if (irun_ > 0 && saveByRun_ > 0 && nrun_ == saveByRun_)
  {
    if (convention_ == Online)
    {
      char suffix[64];
      sprintf(suffix, "_R%09d", irun_);
      char rewrite[64];
      sprintf(rewrite, "\\1Run %d/\\2/Run summary", irun_);
      if (fileFormat_ == ROOT)
        saveForOnline(suffix, rewrite);
      else if (fileFormat_ == PB)
        saveForOnlinePB(suffix);
      else
        throw cms::Exception("DQMFileSaver")
          << "Internal error, can save files"
          << " only in ROOT or ProtocolBuffer format.";
    }
    else if (convention_ == Offline && fileFormat_ == ROOT)
      saveForOffline(workflow_, irun_, 0);
    else if (convention_ == Offline && fileFormat_ == PB)
      saveForOfflinePB(workflow_, irun_);
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
    else if (convention_ == Offline)
      saveForOffline(workflow_, irun_, 0);
    else
      throw cms::Exception("DQMFileSaver")
	<< "Internal error.  Can only save files at the end of the"
	<< " job in Offline mode.";
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

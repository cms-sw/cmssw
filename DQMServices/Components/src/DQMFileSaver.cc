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

#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <TString.h>
#include <TSystem.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

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
DQMFileSaver::saveForOfflinePB(const std::string &workflow, int run) const
{
  char suffix[64];
  sprintf(suffix, "R%09d", run);
  std::string filename = onlineOfflineFileName(fileBaseName_, std::string(suffix), workflow, child_, PB);
  dbe_->savePB(filename, filterName_);
}

void
DQMFileSaver::saveForOffline(const std::string &workflow, int run, int lumi) const
{
  char suffix[64];
  sprintf(suffix, "R%09d", run);

  char rewrite[128];
  if (lumi == 0) // save for run
    sprintf(rewrite, "\\1Run %d/\\2/Run summary", run);
  else
    sprintf(rewrite, "\\1Run %d/\\2/By Lumi Section %d-%d", run, lumi, lumi);

  std::string filename = onlineOfflineFileName(fileBaseName_, std::string(suffix), workflow, child_, ROOT);

  if (lumi == 0) // save for run
  {
    // set run end flag
    dbe_->cd();
    dbe_->setCurrentFolder("Info/ProvInfo");
    
    // do this, because ProvInfo is not yet run in offline DQM
    MonitorElement* me = dbe_->get("Info/ProvInfo/CMSSW"); 
    if (!me) me = dbe_->bookString("CMSSW", edm::getReleaseVersion().c_str() );
    
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
               lumi,
               (DQMStore::SaveReferenceTag) saveReference_,
               saveReferenceQMin_,
               fileUpdate_ ? "UPDATE" : "RECREATE");
  }
  else // save EventInfo folders for luminosity sections
  {
    std::vector<std::string> systems = (dbe_->cd(), dbe_->getSubdirs());

    edm::LogAbsolute msg("fileAction");
    msg << "DQMFileSaver: storing EventInfo folders for Run: "
              << run << ", Lumi Section: " << lumi << ", Subsystems: " ;

    for (size_t i = 0, e = systems.size(); i != e; ++i) {
      if (systems[i] != "Reference") {
        dbe_->cd();
        msg << systems[i] << "  " ;

        dbe_->save(filename,
                   systems[i]+"/EventInfo", "^(Reference/)?([^/]+)",
                   rewrite,
                   enableMultiThread_ ? run : 0,
                   lumi,
                   DQMStore::SaveWithoutReference,
                   dqm::qstatus::STATUS_OK,
                   fileUpdate_ ? "UPDATE" : "RECREATE");

        // from now on update newly created file
        if (fileUpdate_.load() == 0) fileUpdate_ = 1;
      }
    }
  }
}

static void
doSaveForOnline(DQMStore *store,
		int run,
		bool enableMultiThread,
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
                enableMultiThread ? run : 0,
                0,
                saveref,
                saveRefQMin);
  else if (fileFormat == DQMFileSaver::PB)
    store->savePB(filename,
		  filterName,
		  enableMultiThread ? run : 0);
}

void
DQMFileSaver::saveForOnlinePB(int run, const std::string &suffix) const
{
  // The file name contains the Online workflow name,
  // as we do not want to look inside the DQMStore,
  // and the @a suffix, defined in the run/lumi transitions.
  // TODO(diguida): add the possibility to change the dir structure with rewrite.
  std::string filename = onlineOfflineFileName(fileBaseName_, suffix, workflow_, child_, PB);
  doSaveForOnline(dbe_, run, enableMultiThread_,
		  filename,
		  "", "^(Reference/)?([^/]+)", "\\1\\2",
		  (DQMStore::SaveReferenceTag) saveReference_,
		  saveReferenceQMin_,
		  filterName_,
		  PB);
}

void
DQMFileSaver::saveForOnline(int run, const std::string &suffix, const std::string &rewrite) const
{
  std::vector<std::string> systems = (dbe_->cd(), dbe_->getSubdirs());

  for (size_t i = 0, e = systems.size(); i != e; ++i)
  {
    if (systems[i] != "Reference")
    {
      dbe_->cd();
      if (MonitorElement* me = dbe_->get(systems[i] + "/EventInfo/processName"))
      {
	doSaveForOnline(dbe_, run, enableMultiThread_,
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
        doSaveForOnline(dbe_, run, enableMultiThread_,
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
      doSaveForOnline(dbe_, run, enableMultiThread_,
                      fileBaseName_ + systems[i] + suffix + child_ + ".root",
	              systems[i], "^(Reference/)?([^/]+)", rewrite,
	              (DQMStore::SaveReferenceTag) saveReference_,
                      saveReferenceQMin_,
                      "", ROOT);
}


boost::property_tree::ptree
DQMFileSaver::fillJson(int run, int lumi, const std::string& dataFilePathName, const std::string transferDestinationStr, evf::FastMonitoringService *fms)
{
  namespace bpt = boost::property_tree;
  namespace bfs = boost::filesystem;
  
  bpt::ptree pt;

  int hostnameReturn;
  char host[32];
  hostnameReturn = gethostname(host ,sizeof(host));
  if (hostnameReturn == -1)
    throw cms::Exception("fillJson")
          << "Internal error, cannot get host name";
  
  int pid = getpid();
  std::ostringstream oss_pid;
  oss_pid << pid;

  int nProcessed = fms ? (fms->getEventsProcessedForLumi(lumi)) : -1;

  // Stat the data file: if not there, throw
  std::string dataFileName;
  struct stat dataFileStat;
  dataFileStat.st_size=0;
  if (nProcessed) {
    if (stat(dataFilePathName.c_str(), &dataFileStat) != 0)
      throw cms::Exception("fillJson")
            << "Internal error, cannot get data file: "
            << dataFilePathName;
    // Extract only the data file name from the full path
    dataFileName = bfs::path(dataFilePathName).filename().string();
  }
  // The availability test of the FastMonitoringService was done in the ctor.
  bpt::ptree data;
  bpt::ptree processedEvents, acceptedEvents, errorEvents, bitmask, fileList, fileSize, inputFiles, fileAdler32, transferDestination;

  processedEvents.put("", nProcessed); // Processed events
  acceptedEvents.put("", nProcessed); // Accepted events, same as processed for our purposes

  errorEvents.put("", 0); // Error events
  bitmask.put("", 0); // Bitmask of abs of CMSSW return code
  fileList.put("", dataFileName); // Data file the information refers to
  fileSize.put("", dataFileStat.st_size); // Size in bytes of the data file
  inputFiles.put("", ""); // We do not care about input files!
  fileAdler32.put("", -1); // placeholder to match output json definition
  transferDestination.put("", transferDestinationStr); // SM Transfer destination field

  data.push_back(std::make_pair("", processedEvents));
  data.push_back(std::make_pair("", acceptedEvents));
  data.push_back(std::make_pair("", errorEvents));
  data.push_back(std::make_pair("", bitmask));
  data.push_back(std::make_pair("", fileList));
  data.push_back(std::make_pair("", fileSize));
  data.push_back(std::make_pair("", inputFiles));
  data.push_back(std::make_pair("", fileAdler32));
  data.push_back(std::make_pair("", transferDestination));

  pt.add_child("data", data);

  if (fms == nullptr) {
    pt.put("definition", "/fakeDefinition.jsn");
  } else {
    // The availability test of the EvFDaqDirector Service was done in the ctor.
    bfs::path outJsonDefName(edm::Service<evf::EvFDaqDirector>()->baseRunDir()); //we assume this file is written bu the EvF Output module
    outJsonDefName /= (std::string("output_") + oss_pid.str() + std::string(".jsd"));
    pt.put("definition", outJsonDefName.string());
  }

  char sourceInfo[64]; //host and pid information
  sprintf(sourceInfo, "%s_%d", host, pid);
  pt.put("source", sourceInfo);

  return pt;
}

void
DQMFileSaver::saveForFilterUnit(const std::string& rewrite, int run, int lumi,  const DQMFileSaver::FileFormat fileFormat) const
{
  // get from DAQ2 services where to store the files according to their format
  namespace bpt = boost::property_tree;

  std::string openJsonFilePathName;
  std::string jsonFilePathName;
  std::string openHistoFilePathName;
  std::string histoFilePathName;

  // create the files names
  if (fakeFilterUnitMode_) {
    std::string runDir = str(boost::format("%s/run%06d") % dirName_ % run);
    std::string baseName = str(boost::format("%s/run%06d_ls%04d_%s") % runDir % run % lumi % stream_label_ );

    boost::filesystem::create_directories(runDir);

    jsonFilePathName = baseName + ".jsn";
    openJsonFilePathName = jsonFilePathName + ".open";

    histoFilePathName = baseName + dataFileExtension(fileFormat);
    openHistoFilePathName = histoFilePathName + ".open";
  } else {
    openJsonFilePathName = edm::Service<evf::EvFDaqDirector>()->getOpenOutputJsonFilePath(lumi, stream_label_);
    jsonFilePathName = edm::Service<evf::EvFDaqDirector>()->getOutputJsonFilePath(lumi, stream_label_);

    if (fileFormat == ROOT) {
      openHistoFilePathName = edm::Service<evf::EvFDaqDirector>()->getOpenRootHistogramFilePath(lumi, stream_label_);
      histoFilePathName = edm::Service<evf::EvFDaqDirector>()->getRootHistogramFilePath(lumi, stream_label_);
    } else if (fileFormat == PB) {
      openHistoFilePathName = edm::Service<evf::EvFDaqDirector>()->getOpenProtocolBufferHistogramFilePath(lumi, stream_label_);
      histoFilePathName = edm::Service<evf::EvFDaqDirector>()->getProtocolBufferHistogramFilePath(lumi, stream_label_);
    }
  }

  if (fms_ ? fms_->getEventsProcessedForLumi(lumi) : true) {
    if (fileFormat == ROOT)
    {
      // Save the file with the full directory tree,
      // modifying it according to @a rewrite,
      // but not looking for MEs inside the DQMStore, as in the online case,
      // nor filling new MEs, as in the offline case.
      dbe_->save(openHistoFilePathName,
             "",
             "^(Reference/)?([^/]+)",
             rewrite,
             enableMultiThread_ ? run : 0,
             lumi,
             (DQMStore::SaveReferenceTag) saveReference_,
             saveReferenceQMin_,
             fileUpdate_ ? "UPDATE" : "RECREATE",
             true);
    }
    else if (fileFormat == PB)
    {
      // Save the file in the open directory.
      dbe_->savePB(openHistoFilePathName,
        filterName_,
        enableMultiThread_ ? run : 0,
        lumi,
        true);
    }
    else
      throw cms::Exception("DQMFileSaver")
        << "Internal error, can save files"
        << " only in ROOT or ProtocolBuffer format.";

    // Now move the the data and json files into the output directory.
    rename(openHistoFilePathName.c_str(), histoFilePathName.c_str());
  }

  // Write the json file in the open directory.
  bpt::ptree pt = fillJson(run, lumi, histoFilePathName, transferDestination_, fms_);
  write_json(openJsonFilePathName, pt);
  rename(openJsonFilePathName.c_str(), jsonFilePathName.c_str());
}

void
DQMFileSaver::saveJobReport(const std::string &filename) const
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
    enableMultiThread_ (false),
    saveByLumiSection_ (-1),
    saveByRun_ (-1),
    saveAtJobEnd_ (false),
    saveReference_ (DQMStore::SaveWithReference),
    saveReferenceQMin_ (dqm::qstatus::STATUS_OK),
    forceRunNumber_ (-1),
    fileBaseName_ (""),
    fileUpdate_ (0),
    dbe_ (&*edm::Service<DQMStore>()),
    nrun_ (0),
    nlumi_ (0),
    irun_ (0),
    fms_(nullptr)
{
  // Determine the file saving convention, and adjust defaults accordingly.
  std::string convention = ps.getUntrackedParameter<std::string>("convention", "Offline");
  fakeFilterUnitMode_ = ps.getUntrackedParameter<bool>("fakeFilterUnitMode", false);

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
  else if (fakeFilterUnitMode_)
  {
    edm::LogInfo("DQMFileSaver")
      << "Fake FU mode, files are saved under <dirname>/runXXXXXX/runXXXXXX_lsXXXX_<stream_Label>.pb.\n";
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

  // Log some information what we will do.
  edm::LogInfo("DQMFileSaver")
    << "DQM file saving settings:\n"
    << " using base file name '" << fileBaseName_ << "'\n"
    << " forcing run number " << forceRunNumber_ << "\n"
    << " saving every " << saveByLumiSection_ << " lumi section(s)\n"
    << " saving every " << saveByRun_ << " run(s)\n"
    << " saving at job end: " << (saveAtJobEnd_ ? "yes" : "no") << "\n";
}

//--------------------------------------------------------
void
DQMFileSaver::beginJob()
{
  nrun_ = nlumi_ = irun_ = 0;
  
  // Determine if we are running multithreading asking to the DQMStore. Not to be moved in the ctor
  enableMultiThread_ = dbe_->enableMultiThread_;

  if ((convention_ == FilterUnit) && (!fakeFilterUnitMode_))
  {
    transferDestination_ = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations(stream_label_);
  } 
}

std::shared_ptr<saverDetails::NoCache>
DQMFileSaver::globalBeginRun(const edm::Run &r, const edm::EventSetup &) const
{
  ++nrun_;

  // For Filter Unit, create an empty ini file:
  // it is needed by the HLT deamon in order to start merging
  // The run number is established in the service
  // TODO(diguida): check that they are the same?
  if ((convention_ == FilterUnit) && (!fakeFilterUnitMode_))
  {
    evf::EvFDaqDirector * daqDirector = (evf::EvFDaqDirector *) (edm::Service<evf::EvFDaqDirector>().operator->());
    const std::string initFileName = daqDirector->getInitFilePath(stream_label_);
    std::ofstream file(initFileName);
    file.close();
  }

  return nullptr;
}

std::shared_ptr<saverDetails::NoCache>
DQMFileSaver::globalBeginLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup &) const
{
  ++nlumi_;
  return nullptr;
}

void DQMFileSaver::analyze(edm::StreamID, const edm::Event &e, const edm::EventSetup &) const
{
  //save by event and save by time are not supported
  //anymore in the threaded framework. please use
  //savebyLumiSection instead.
}

void
DQMFileSaver::globalEndLuminosityBlock(const edm::LuminosityBlock & iLS, const edm::EventSetup &) const
{
  int ilumi    = iLS.id().luminosityBlock();
  int irun     = iLS.id().run();
  if (ilumi > 0 && saveByLumiSection_ > 0 )
  {
    if (convention_ != Online && convention_ != FilterUnit && convention_ != Offline )
      throw cms::Exception("DQMFileSaver")
	<< "Internal error, can save files at end of lumi block"
	<< " only in Online, FilterUnit or Offline mode.";

    if (convention_ == Online && (nlumi_ % saveByLumiSection_) == 0) // insist on lumi section ordering
    {
      char suffix[64];
      char rewrite[128];
      sprintf(suffix, "_R%09d_L%06d", irun, ilumi);
      sprintf(rewrite, "\\1Run %d/\\2/By Lumi Section %d-%d", irun, ilumi-nlumi_, ilumi);
      if (fileFormat_ == ROOT)
        saveForOnline(irun, suffix, rewrite);
      else if (fileFormat_ == PB)
        saveForOnlinePB(irun, suffix);
      else
        throw cms::Exception("DQMFileSaver")
          << "Internal error, can save files"
          << " only in ROOT or ProtocolBuffer format.";
    }

    // Store at every lumi section end only if some events have been processed.
    // Caveat: if faking FilterUnit, i.e. not accessing DAQ2 services,
    // we cannot ask FastMonitoringService the processed events, so we are forced
    // to save the file at every lumisection, even with no statistics.
    // Here, we protect the call to get the processed events in a lumi section
    // by testing the pointer to FastMonitoringService: if not null, i.e. in real FU mode,
    // we check that the events are not 0; otherwise, we skip the test, so we store at every lumi transition. 
    // TODO(diguida): allow fake FU mode to skip file creation at empty lumi sections.
    if (convention_ == FilterUnit && (fms_ ? fms_->shouldWriteFiles(ilumi) : !fms_))
    {
      char rewrite[128];
      sprintf(rewrite, "\\1Run %d/\\2/By Lumi Section %d-%d", irun, ilumi, ilumi);
      saveForFilterUnit(rewrite, irun, ilumi, fileFormat_);
    }
    if (convention_ == Offline)
    {
      if (fileFormat_ == ROOT)
        saveForOffline(workflow_, irun, ilumi);
      else
      // TODO(diguida): do we need to support lumisection saving in Offline for PB?
      // In this case, for ROOT, we only save EventInfo folders: we can filter them...
        throw cms::Exception("DQMFileSaver")
          << "Internal error, can save files"
          << " only in ROOT format.";
    }

    // after saving per LS, delete the old LS global histograms.
    dbe_->deleteUnusedLumiHistograms(enableMultiThread_ ? irun : 0, ilumi);
  }
}

void
DQMFileSaver::globalEndRun(const edm::Run & iRun, const edm::EventSetup &) const
{
  int irun     = iRun.id().run();
  irun_        = irun;
  if (irun > 0 && saveByRun_ > 0 && (nrun_ % saveByRun_) == 0)
    {
      if (convention_ == Online)
	{
	  char suffix[64];
	  sprintf(suffix, "_R%09d", irun);
	  char rewrite[64];
	  sprintf(rewrite, "\\1Run %d/\\2/Run summary", irun);
	  if (fileFormat_ == ROOT)
	    saveForOnline(irun, suffix, rewrite);
	  else if (fileFormat_ == PB)
	    saveForOnlinePB(irun, suffix);
	  else
	    throw cms::Exception("DQMFileSaver")
	      << "Internal error, can save files"
	      << " only in ROOT or ProtocolBuffer format.";
	}
      else if (convention_ == Offline && fileFormat_ == ROOT)
	saveForOffline(workflow_, irun, 0);
      else if (convention_ == Offline && fileFormat_ == PB)
	saveForOfflinePB(workflow_, irun);
      else
	throw cms::Exception("DQMFileSaver")
	  << "Internal error.  Can only save files in endRun()"
	  << " in Online and Offline modes.";
    }

  // create a fake EoR file for testing purposes.
  if (fakeFilterUnitMode_) {
    edm::LogInfo("DQMFileSaver")
      << "Producing fake EoR file.\n";

    std::string runDir = str(boost::format("%s/run%06d") % dirName_ % irun);
    std::string jsonFilePathName = str(boost::format("%s/run%06d_ls0000_EoR.jsn") % runDir % irun);
    std::string openJsonFilePathName = jsonFilePathName + ".open";

    boost::filesystem::create_directories(runDir);

    using namespace boost::property_tree;
    ptree pt;
    ptree data;

    ptree child1, child2, child3;

    child1.put("", -1);    // Processed
    child2.put("", -1);    // Accepted
    child3.put("", nlumi_);  // number of lumi

    data.push_back(std::make_pair("", child1));
    data.push_back(std::make_pair("", child2));
    data.push_back(std::make_pair("", child3));

    pt.add_child("data", data);
    pt.put("definition", "/non-existant/");
    pt.put("source", "--hostname--");

    std::ofstream file(jsonFilePathName);
    write_json(file, pt, true);
    file.close();

    rename(openJsonFilePathName.c_str(), jsonFilePathName.c_str());
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
  
  // save JobReport once per job
  char suffix[64];
  sprintf(suffix, "R%09d", irun_.load());
  std::string filename = onlineOfflineFileName(fileBaseName_, suffix, workflow_, child_, fileFormat_);
  saveJobReport(filename);
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

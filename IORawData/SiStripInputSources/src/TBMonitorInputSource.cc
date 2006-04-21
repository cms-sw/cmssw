
#include "IORawData/SiStripInputSources/interface/TBMonitorInputSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PluginManager/PluginCapabilities.h"

#include <iostream>
#include <memory>
#include <cstdlib>

using namespace edm;
using namespace std;

TBMonitorInputSource::TBMonitorInputSource(const edm::ParameterSet& pset, edm::InputSourceDescription const& desc) : 
  edm::ExternalInputSource(pset,desc),
  m_file(0),
  m_task(SiStripHistoNamingScheme::task(pset.getUntrackedParameter<string>("CommissioningTask","Pedestals")))
{
  //Check Commissioning Task
  if (m_task == SiStripHistoNamingScheme::UNKNOWN_TASK) edm::LogWarning("Commissioning|TBMonitorIS") << "Unknown commissioning task. Value used: " << pset.getUntrackedParameter<string>("CommissioningTask","Pedestals") << "; values accepted: Pedestals, ApvTiming, FedTiming, OptoScan, VpspScan, ApvLatency.";

  produces< edm::DetSetVector< Histo > >();
  m_taskId = taskId(m_task);
}

//-----------------------------------------------------------------------------

TBMonitorInputSource::~TBMonitorInputSource() {;}

//-----------------------------------------------------------------------------

bool TBMonitorInputSource::produce(edm::Event& e) {

  //input source designed to attach all information to the first event...
  if (event() > 1) {
    LogDebug("Commissioning|TBMonitorIS") << "[TBMonitorInputSource::produce]: Warning : All \"TBMonitor\" histos from file list attached to event 1.";
    return false;
  }

  //create DetSetVectors to hold Histograms.
  edm::DetSetVector< Profile >* profile_collection = new edm::DetSetVector< Profile >();
  edm::DetSetVector< Histo >* histo_collection = new edm::DetSetVector< Histo >();

  //loop over files, find tprofiles.
  for (std::vector<std::string>::const_iterator file = fileNames().begin(); file != fileNames().end(); file++) {
  openFile(*file);
  if (!m_file) continue;
  findProfiles(m_file,profile_collection);
  }

  if (profile_collection->empty()) return false;

  //Loop DetSetVector and convert apv numbering scheme for 4-apv modules from 32,33,34,35 to 32,33,36,37. (only relevent for tasks conducted on the apv level).

 for (edm::DetSetVector<Profile>::const_iterator idetset = profile_collection->begin(); idetset != profile_collection->end(); idetset++) {
   
   edm::DetSet<Histo>& hist = histo_collection->find_or_insert(idetset->id);
   hist.data.reserve(idetset->data.size());

   for (edm::DetSet<Profile>::const_iterator prof = idetset->data.begin(); prof != idetset->data.end(); prof++) {
       
     if (idetset->data.size() == 4) {
       string name(prof->get().GetName());
       //unpack name and find channel
       SiStripHistoNamingScheme::HistoTitle h_title = SiStripHistoNamingScheme::histoTitle(name);
       //fix channel numbers
       if ((h_title.channel_ == 34) | (h_title.channel_ == 35)) h_title.channel_ +=2;
     }
   
     string name(prof->get().GetName()); 
     SiStripHistoNamingScheme::HistoTitle h_title = SiStripHistoNamingScheme::histoTitle(name);

  //convert to TH1Fs (temporary)//////////////////////
  TH1F th1f;
  convert(prof->get(), th1f);
  hist.data.push_back(Histo(th1f));

   }
 }

 //clean up
  if (profile_collection) delete profile_collection;
 ///////////////////////////////////////////////////////

  //update the event...
  std::auto_ptr< edm::DetSetVector< Histo > > collection(histo_collection);
  e.put(collection);

  return true;
}

//-----------------------------------------------------------------------------

void TBMonitorInputSource::findProfiles(TDirectory* dir, edm::DetSetVector< Profile >* histos) {

  std::vector< TDirectory* > dirs;
  dirs.push_back(dir);

  //loop through all directories and record tprofiles (matching label m_taskId) contained within them.

    while ( !dirs.empty() ) { 
     dirProfiles(dirs[0], &dirs, histos);
      dirs.erase(dirs.begin());
    }
}
 
//-----------------------------------------------------------------------------


 void TBMonitorInputSource::dirProfiles(TDirectory* dir, std::vector< TDirectory* >* dirs, edm::DetSetVector< Profile >* profs) {

    TList* keylist = dir->GetListOfKeys();

    if (keylist) {
      TObject* obj = keylist->First(); //the object

      if (obj) {
	bool loop = true;
	while (loop) { 
	  if (obj == keylist->Last()) {loop = false;}
 
	  if (dynamic_cast<TDirectory*>(dir->Get(obj->GetName()))) {dirs->reserve(dirs->size() + 1); dirs->push_back((TDirectory*)dir->Get(obj->GetName()));}

	  if (dynamic_cast<TProfile*>(dir->Get(obj->GetName()))) {
	    const TProfile& tprof = *dynamic_cast<TProfile*>(dir->Get(obj->GetName()));

	    //look for "task id" in the histo title (quick check)
	    const string name(tprof.GetName());

	    if (name.find(m_taskId) != std::string::npos) {

	      //extract histogram details from encoded histogram name.
	      SiStripHistoNamingScheme::HistoTitle h_title = histoTitle(name);

	      //create a profile and update the name to the "standard format"
	      Profile tprof_updated(tprof);
	      string newName = SiStripHistoNamingScheme::histoTitle(h_title.task_,h_title.contents_,h_title.keyType_,h_title.keyValue_,h_title.granularity_,h_title.channel_, h_title.extraInfo_);
	      const_cast<TProfile&>(tprof_updated.get()).SetName(newName.c_str());

	      //update DetSetVector with updated profile using histo key (indicating control path) as the index.
	      edm::DetSet<Profile>& prof = profs->find_or_insert(h_title.keyValue_);
	      prof.data.reserve(prof.data.size() + 1); 
	      prof.data.push_back(Profile(tprof_updated));
	    }
}
	  obj = keylist->After(obj);
	}
      }
    }
 }

//-----------------------------------------------------------------------------

void TBMonitorInputSource::openFile(const std::string& filename) {

  if (m_file!=0) {
    m_file->Close();
    m_file=0;
  }

  m_file=TFile::Open(filename.c_str());

  if (m_file==0) {
    LogDebug("Commissioning|TBMonitorIS") << "Unable to open " << filename;
    return;
  } 
}

//-----------------------------------------------------------------------------

void TBMonitorInputSource::setRunAndEventInfo() {

  //Get the run number from each file in list and compare...
  std::string run;
  for (std::vector<std::string>::const_iterator file = fileNames().begin(); file != fileNames().end(); file++) {
    unsigned int ipass = file->find("TBMonitor");
    unsigned int ipath = file->find("_");
    //check run numbers from multiple files are the same...
    if ((file != fileNames().begin()) && run.compare(file->substr(ipass+9,ipath-ipass-9))) {LogDebug("Commissioning|TBMonitorIS") << "Warning: Differing run numbers retrieved from input files. Recording last in file list.";}
  run = ((ipass != string::npos) && (ipath != string::npos)) ? file->substr(ipass+9,ipath-ipass-9) : string("0");
  }

    LogDebug("Commissioning|TBMonitorIS") << "Run number: " << run;
  
  //set run number
    setRunNumber(atoi(run.c_str()));

  // time is a hack
  edm::TimeValue_t present_time = presentTime();
  unsigned long time_between_events = timeBetweenEvents();
  setTime(present_time + time_between_events);
}

//-----------------------------------------------------------------------------

std::string TBMonitorInputSource::taskId(SiStripHistoNamingScheme::Task task) {

  std::string taskId("");

  if (task == SiStripHistoNamingScheme::PEDESTALS) {taskId = "Profile_ped";}
  else if (task == SiStripHistoNamingScheme::FED_CABLING) {/* to be set*/}
  else if (task == SiStripHistoNamingScheme::VPSP_SCAN) {taskId = "vpsp_mean";}
  else if (task == SiStripHistoNamingScheme::OPTO_SCAN) {taskId = "_gain";}
  else if (task == SiStripHistoNamingScheme::APV_TIMING) {taskId= "tick_chip";}
  else if (task == SiStripHistoNamingScheme::FED_TIMING) {taskId = "tickfed_chip";}
  else if (task == SiStripHistoNamingScheme::APV_LATENCY) {/* to be set*/}

  else {edm::LogWarning("Commissioning|TBMonitorIS") << "[TBMonitorInputSource::taskId]: Unknown Commissioning task, filling event with ALL TProfile's found in specified files.";}

  return taskId;

}

//-----------------------------------------------------------------------------

void TBMonitorInputSource::convert( const TProfile& prof, TH1F& th1f) {
  
  th1f.SetTitle(prof.GetTitle());
  th1f.SetName(prof.GetName());
  th1f.SetBins(prof.GetNbinsX(), prof.GetXaxis()->GetXmin(), prof.GetXaxis()->GetXmax() );
  
  for (Int_t ibin = 0; ibin < th1f.GetNbinsX(); ibin++) {
    
    th1f.SetBinContent(ibin + 1, prof.GetBinContent(ibin + 1));
    th1f.SetBinError(ibin + 1, prof.GetBinError(ibin + 1));
  }
}

//-----------------------------------------------------------------------------

SiStripHistoNamingScheme::HistoTitle TBMonitorInputSource::histoTitle(const string& histo_name) {
  
  //initialise SiStripHistoNamingScheme::HistoTitle object
  SiStripHistoNamingScheme::HistoTitle title;
  
  title.task_   = SiStripHistoNamingScheme::UNKNOWN_TASK;
  title.contents_   = SiStripHistoNamingScheme::COMBINED;
  title.keyType_     = SiStripHistoNamingScheme::FEC;
  title.keyValue_    = 0;
  title.granularity_ = SiStripHistoNamingScheme::UNKNOWN_GRAN;
  title.channel_     = 0;
  title.extraInfo_ = "";
  
  //scan histogram name

  unsigned int start = histo_name.find("0x");
  if (start == std::string::npos) {start = histo_name.find("-") - 1;} //due to variations in TBMonitor histo titling
  
  unsigned int stop = histo_name.find("_", start+2);
  if (stop == std::string::npos) stop = histo_name.find("-", start+2);//due to variations in TBMonitor histo titling

  // Set SiStripHistoNamingScheme::HistoTitle::task_

  title.task_ = m_task;
  
  // Set SiStripHistoNamingScheme::HistoTitle::extraInfo_
  
  //extract gain and digital level from histo name if task is BIASGAIN
  if (m_task == SiStripHistoNamingScheme::OPTO_SCAN) {
    if ((histo_name.find("_gain") != std::string::npos) &&
	((histo_name.find("tick") != std::string::npos) | (histo_name.find("base") != std::string::npos))) {
      
      stringstream nm;
      nm << SiStripHistoNamingScheme::gain() << histo_name.substr((histo_name.size() - 1),1);
      
      if (histo_name.find("tick") != std::string::npos) {
	nm << SiStripHistoNamingScheme::digital() << 1;}
      else {nm << SiStripHistoNamingScheme::digital() << 0;}

      title.extraInfo_ = nm.str();
    }
    else {edm::LogError("Commissioning|TBMonitorIS") << "Inconsistency in TBMonitor histogram name for the OPTO_SCAN task. One or more of the strings \"gain\", \"tick\" and \"base\" were not found.";}
  }
  
  // Set SiStripHistoNamingScheme::HistoTitle::channel_
  
  title.channel_ = (stop != std::string::npos) ? atoi(histo_name.substr(stop+1, 3).c_str()) : 0;//apvnum = 0 if  not specified.
  
  if ((histo_name.size() - stop) == 2) {title.channel_ = atoi(histo_name.substr(stop+1, 1).c_str());}//due to variations in TBMonitor histo titling.

  // Set SiStripHistoNamingScheme::HistoTitle::granularity_
  
  if (m_task == SiStripHistoNamingScheme::VPSP_SCAN) {
    title.granularity_ = SiStripHistoNamingScheme::APV;
    title.channel_ += 32; // convert "apv number (0-5) to HW address (32-37).
  }
  
  else if ((m_task == SiStripHistoNamingScheme::OPTO_SCAN) || (m_task == SiStripHistoNamingScheme::APV_TIMING) || (m_task == SiStripHistoNamingScheme::FED_TIMING)) { 
    title.granularity_ = SiStripHistoNamingScheme::LLD_CHAN;}
  
  else if (m_task == SiStripHistoNamingScheme::PEDESTALS) {
    title.granularity_ = SiStripHistoNamingScheme::MODULE;}
  
  else {edm::LogWarning("Commissioning|TBMonitorIS") << "[TBMonitorInputSource::histoName]: Unknown Commissioning task. Setting SiStripHistoNamingScheme::HistoName::granularity_ to \"UNKNOWN GRANULARITY\".";}

  // Set SiStripHistoNamingScheme::HistoTitle::keyValue_
  
  stringstream os(""); 
  if (stop != std::string::npos) {
    os << histo_name.substr(start+2,(stop-start-2));}
  else { os << histo_name.substr(start+2);}
  
  unsigned int idlocal;
  os >> hex >> idlocal;
  title.keyValue_ = (idlocal<<2) | (title.channel_ & 0x3);//updates key to the format defined in DQM/SiStripCommon/interface/SiStripGenerateKey.h

  return title;
}


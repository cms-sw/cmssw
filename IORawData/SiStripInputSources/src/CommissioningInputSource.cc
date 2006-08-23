#include "IORawData/SiStripInputSources/interface/CommissioningInputSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PluginManager/PluginCapabilities.h"
//CondFormats
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"

#include <iostream>
#include <iomanip>
#include <memory>
#include <cstdlib>

using namespace edm;
using namespace std;

CommissioningInputSource::CommissioningInputSource(const edm::ParameterSet& pset, edm::InputSourceDescription const& desc) : 
  edm::ExternalInputSource(pset,desc),

  //initialise private data members
  m_inputFile(0),
  m_outputFile(0),
  m_outputFilename(pset.getUntrackedParameter<string>("outputFilename","Client")),
  m_run(0)
{
  produces< edm::DetSetVector< Profile > >();
}

//-----------------------------------------------------------------------------

CommissioningInputSource::~CommissioningInputSource() {
if (m_outputFile) delete m_outputFile;}

//-----------------------------------------------------------------------------

bool CommissioningInputSource::produce(edm::Event& e) {

  //input source designed to attach all information to the first event...
  if (event() > 1) {
    LogDebug("Commissioning|TBMonitorIS") << "[CommissioningInputSource::produce]: Warning : All \"TBMonitor\" histos from file list attached to event 1.";
    return false;
  }

  //construct and name output file...
  string name = m_outputFilename.substr( 0, m_outputFilename.find(".root",0));
  stringstream ss; ss << name << "_" << setfill('0') << setw(7) << 
		     m_run << ".root";
  m_outputFile = new TFile(ss.str().c_str(), "RECREATE");

  //create DetSetVectors to hold Histograms.
  edm::DetSetVector< Histo >* histo_collection = new edm::DetSetVector< Histo >();
  edm::DetSetVector< Profile >* comm_collection = new edm::DetSetVector< Profile >();

  //loop over files, find TH1Fs
  for (std::vector<std::string>::const_iterator file = fileNames().begin(); file != fileNames().end(); file++) {
  openFile(*file);
  if (!m_inputFile) continue;
  findHistos(m_inputFile,histo_collection);
  }

  if (histo_collection->empty()) return false;
  
  //combine TH1Fs into commissioning histogram
  for (edm::DetSetVector<Histo>::const_iterator idetset = histo_collection->begin(); idetset != histo_collection->end(); idetset++) {
    
    edm::DetSet<Profile>& comm_vec = comm_collection->find_or_insert(idetset->id);
    comm_vec.data.reserve(6);
    
    //Change to relevent directory in output file for storage of "commissioning histogram"
    SiStripControlKey::ControlPath c_path = SiStripControlKey::path(idetset->id);
    string path = SiStripHistoNamingScheme::controlPath(c_path.fecCrate_, c_path.fecSlot_, c_path.fecRing_, c_path.ccuAddr_, c_path.ccuChan_);
    stringstream ss; ss << m_outputFile->GetName() << ":/" << "DQMData/" << path;
    TDirectory* mother = m_outputFile->GetDirectory(ss.str().c_str());
    mother->cd();

    //for sorting module histograms
    map< unsigned int, vector<const Histo*> > histo_map;
    
    for (edm::DetSet<Histo>::const_iterator hist = idetset->data.begin(); hist != idetset->data.end(); hist++) {
     
      //unpack histogram name
      const string name(hist->get().GetName());
      SiStripHistoNamingScheme::HistoTitle h_title = SiStripHistoNamingScheme::histoTitle(name);
      
      //use channel as map key
      unsigned int key = h_title.channel_;
      
      //if HistoTitle::extraInfo_ contains a gain value + digital level, update key.
      if ((h_title.extraInfo_.find(sistrip::gain_) != string::npos) &&
	  (h_title.extraInfo_.find(sistrip::digital_) != string::npos)) {
	
	string::size_type index = h_title.extraInfo_.find(sistrip::gain_);
	unsigned short gain = atoi(h_title.extraInfo_.substr((index + 4),1).c_str());
	index = h_title.extraInfo_.find(sistrip::digital_);
	unsigned short digital = atoi(h_title.extraInfo_.substr((index + 7),1).c_str());
	key = ((key & 3) << 7) | ((gain & 3) << 1) | (digital & 1);
      }

      //if HistoTitle::extraInfo_ contains Pedestal run information, update key (ignore common mode histograms)
      if (h_title.extraInfo_.find(sistrip::pedsAndRawNoise_) != string::npos) {
	key = ((key & 3) << 7) | 0x1;
      }

      if (h_title.extraInfo_.find(sistrip::residualsAndNoise_) != string::npos) {
	key = ((key & 3) << 7) | 0x2;
      }

      if (h_title.extraInfo_.find(sistrip::commonMode_) != string::npos) continue;
      
      //update map
      if (histo_map[key].empty()) {
	histo_map[key].reserve(3); histo_map[key].resize(3,0);}

      if (h_title.contents_ == sistrip::SUM) {
	histo_map[key][0] = &(*hist);}
      else if (h_title.contents_ == sistrip::SUM2) {
	histo_map[key][1] = &(*hist);}
      else if (h_title.contents_ == sistrip::NUM) {
	histo_map[key][2] = &(*hist);}
    }
    
    //loop through map and combine related histograms
    for (map< unsigned int, vector<const Histo*> >::const_iterator it = histo_map.begin(); it != histo_map.end(); it++) {
      
      //find relevent histos and combine them here.
      if (it->second[0] && it->second[1] && it->second[2]) {
	TProfile commHist;
	combine(it->second[0]->get(),it->second[1]->get(),it->second[2]->get(), commHist);

	//unpack "number of entries" histogram name
	SiStripHistoNamingScheme::HistoTitle h_title = SiStripHistoNamingScheme::histoTitle(it->second[2]->get().GetName());
	
	//set name of commissioning histogram using task_, keyType_, keyValue_, granularity_, channel_ and extraInfo_ from "number of entries" histogram 
	string comm_name = SiStripHistoNamingScheme::histoTitle(h_title.task_, sistrip::COMBINED, sistrip::FED, h_title.keyValue_, h_title.granularity_, h_title.channel_, h_title.extraInfo_);
	commHist.SetName(comm_name.c_str());
	
	//add commissioning histogram to DetSetVector
	comm_vec.data.push_back(Profile(commHist));
	
	//add commissioning histogram to output file
	mother->WriteTObject(&commHist);
      }

      else {
	SiStripHistoNamingScheme::HistoTitle h_title;
	stringstream os("");
       if (it->second[0]) {
	 h_title = SiStripHistoNamingScheme::histoTitle(it->second[0]->get().GetName()); os << h_title.keyValue_;}
       else if (it->second[1]) {
	 h_title = SiStripHistoNamingScheme::histoTitle(it->second[1]->get().GetName()); os << h_title.keyValue_;}
       else if (it->second[2]) { 
	 h_title = SiStripHistoNamingScheme::histoTitle(it->second[2]->get().GetName()); os << h_title.keyValue_;}
       else {os << "unknown";}
   
	LogDebug("Commissioning|TBMonitorIS") << "[CommissioningInputSource::produce]: 3 histograms (entries, sum and sum of squares) with otherwise identical names are required at the source for each device. One or more are missing for module key: " << os.str() << ". Module being ignored....";}
    }
  }
  
  //clean up
  if (histo_collection) delete histo_collection;
  
  //update the event...
  std::auto_ptr< edm::DetSetVector< Profile > > collection(comm_collection);
  e.put(collection);
  
  return true;
}

//-----------------------------------------------------------------------------

void CommissioningInputSource::findHistos(TDirectory* dir, edm::DetSetVector< Histo >* histos) {
  
  std::vector< TDirectory* > dirs;
  dirs.reserve(20000);
  dirs.push_back(dir);
  
  //loop through all directories and record th1fs contained within them.

  while ( !dirs.empty() ) { 
    dirHistos(dirs[0], &dirs, histos);
      dirs.erase(dirs.begin());
  }
}

//-----------------------------------------------------------------------------

void CommissioningInputSource::dirHistos(TDirectory* dir, std::vector< TDirectory* >* dirs, edm::DetSetVector< Histo >* histos) {
  
  TList* keylist = dir->GetListOfKeys();

  if (keylist) {
    TObject* obj = keylist->First(); //the object

    if (obj) {
      bool loop = true;
      while (loop) { 
	if (obj == keylist->Last()) {loop = false;}
	
	if (dynamic_cast<TDirectory*>(dir->Get(obj->GetName()))) {

	  TDirectory* child = dynamic_cast<TDirectory*>(dir->Get(obj->GetName()));

	  //update record of directories
	  dirs->push_back(child);
 
	  //update output file to have identical dir/child TDirectory structure to input file.
	  addOutputDir(dir,child);
	}
	
	if (dynamic_cast<TH1F*>(dir->Get(obj->GetName()))) {
	  const TH1F& th1f = *dynamic_cast<TH1F*>(dir->Get(obj->GetName()));

	  //extract histogram details from encoded histogram name.
	  const string name(th1f.GetName());
	  SiStripHistoNamingScheme::HistoTitle h_title = SiStripHistoNamingScheme::histoTitle(name);

	  //update DetSetVector with th1f using fec key as the index. Find fec key.......
	  string path(dir->GetPath());
	  string::size_type index = path.find("ControlView");
	  string control = path.substr(index);

	  SiStripHistoNamingScheme::ControlPath fec_path = SiStripHistoNamingScheme::controlPath(control);
	  unsigned int fec_key = SiStripControlKey::key(fec_path.fecCrate_, fec_path.fecSlot_, fec_path.fecRing_, fec_path.ccuAddr_, fec_path.ccuChan_);
 
	  //check histo key (fed id, channel) corresponds to its control path (directory path) using control cabling.
	  /*
	  pair< unsigned int,unsigned int > fed_channel = SiStripReadoutKey::path(h_name.histoKey_);
	  const vector<unsigned short>& feds = fed_cabling_->feds();
	  const FedChannelConnection& connection = fed_cabling_->connection(fed_channel.first, fed_channel.second);
	  unsigned int fec_key_confirm = SiStripControlKey::key(connection.fecCrate(), connection.fecSlot(), connection.fecRing(),connection.ccuAddr(), connection.ccuChan());

	  if (fec_key_confirm != fec_key) {LogDebug("Commissioning|TBMonitorIS") << "[CommissioningInputSource::dirHistos]: Warning control path of histogram " << path << "does not correspond to fed key provided: " << h_name.histoKey_ << ". The control path is being used.";}
	  */

	  //update DetSetVector...

	  edm::DetSet<Histo>& histo = histos->find_or_insert(fec_key);
	  histo.data.reserve(6); 
	  histo.data.push_back(Histo(th1f));
	}
	obj = keylist->After(obj);
      }
    }
  }
}

//-----------------------------------------------------------------------------

void CommissioningInputSource::addOutputDir(TDirectory* mother, TDirectory* child) {
  
  //find directory path of "dir" (the mother)
  string mother_path(mother->GetPath());
  string::size_type index = mother_path.find(":");
  const char* path = mother_path.substr(index + 2).c_str();
  const char* name = child->GetName(), *title = child->GetTitle();
  
  //open corresponding directory in output file
  stringstream ss;
  ss << m_outputFile->GetName() << ":/" << path;
  m_outputFile->Cd(ss.str().c_str());
  TDirectory* output_mother = m_outputFile->GetDirectory(0);
  
  //create "directory" (the child) equivalent in output file. 
  TDirectory output_child(name, title);
  output_mother->Add(&output_child);

}

//-----------------------------------------------------------------------------

void CommissioningInputSource::openFile(const std::string& filename) {
  
  if (m_inputFile!=0) {
    m_inputFile->Close();
    m_inputFile=0;
  }
  
  m_inputFile=TFile::Open(filename.c_str());
  
  if (m_inputFile==0) {
    edm::LogWarning("Commissioning|TBMonitorIS") << "Unable to open " << filename;
    return;
  } 
}

//-----------------------------------------------------------------------------

void CommissioningInputSource::setRunAndEventInfo() {

  //Get the run number from each file in list and compare.
  std::string run;
  for (std::vector<std::string>::const_iterator file = fileNames().begin(); file != fileNames().end(); file++) {
    unsigned int ipass = file->find("_");
    unsigned int ipath = file->find(".root");
    //check run numbers from multiple files are the same...
    if ((file != fileNames().begin()) && run.compare(file->substr(ipass+1,ipath-ipass-1))) {edm::LogWarning("Commissioning|TBMonitorIS") << "Warning: Differing run numbers retrieved from input files. Recording last in file list.";}
    run = ((ipass != string::npos) && (ipath != string::npos)) ? file->substr((ipass+1),(ipath-ipass-1)) : string("0");
  }

    LogDebug("Commissioning|TBMonitorIS") << "Run number: " << run;

  //set private data member
  m_run = (unsigned short)(atoi(run.c_str()));

  //set run number
  setRunNumber(m_run);

  // time is a hack
  edm::TimeValue_t present_time = presentTime();
  unsigned long time_between_events = timeBetweenEvents();
  setTime(present_time + time_between_events);
}

//-----------------------------------------------------------------------------

void CommissioningInputSource::combine(const TH1& sum, const TH1& sum2, const TH1& entries, TProfile& commHist) {
  
  if ((sum.GetNbinsX() != sum2.GetNbinsX()) | (sum2.GetNbinsX() != entries.GetNbinsX())) { edm::LogError("Commissioning|TBMonitorIS") << "[ClientForCommissioning::generateCommissioingHisto]: Warning: Number of bins for histograms not identical"; return;}
  
  commHist.SetBins(sum.GetNbinsX(), sum.GetXaxis()->GetXmin(),sum.GetXaxis()->GetXmax());

  TH1F meanSum2; //dummy container to calculate errors
  meanSum2.SetBins(sum.GetNbinsX(), sum.GetXaxis()->GetXmin(),sum.GetXaxis()->GetXmax());
  meanSum2.Divide(&sum2,&entries);

  for (unsigned int ii = 0; ii < (unsigned int)commHist.GetNbinsX(); ii++) {
    if (entries.GetBinContent((Int_t)(ii + 1))) {
      Double_t entry = entries.GetBinContent((Int_t)(ii + 1));
      Double_t content = sum.GetBinContent((Int_t)(ii + 1))/entries.GetBinContent((Int_t)(ii + 1));
      Double_t error = sqrt(fabs(meanSum2.GetBinContent((Int_t)(ii + 1)) - content * content));
      setBinStats(commHist,(Int_t)(ii+1),(Int_t)entry,content,error); 
    }
  }
}

//-----------------------------------------------------------------------------

void CommissioningInputSource::setBinStats(TProfile& prof, Int_t bin, Int_t entries, Double_t content, Double_t error) {

  prof.SetBinEntries(bin,entries);
  prof.SetBinContent(bin,content*entries);
  prof.SetBinError(bin,sqrt(entries*entries*error*error + content*content*entries));

}

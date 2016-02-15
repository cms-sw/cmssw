// system include files
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

#include "PhysicsTools/FWLite/interface/CommandLineParser.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TStyle.h"
#include "TPaveStats.h"
#include "TPad.h"
#include "TROOT.h"

#define MAXCHPEREVENT 500
std::string rootInputFileName_;
std::string rootOutputFileName_;
std::string plotsDirName_;
std::string htmlDirName_;
std::string htmlFileName_;
int newRowEvery_;
int thumbnailSize_;
bool outputRawHistograms_;
bool selectDigiBasedOnTubeName_;
int maxEvents_;
TFile* rootInputFile_;
TFile* rootOutputFile_;
TH2F* firstEventHistMeanMaps[3];
TH2F* firstEventHistRMSMaps[3];
TTree* eventTree_;
std::vector<float> evtNumbers_;
std::vector<float> orbitNumberSecs_;
std::vector<float> indexVals_;
std::vector<float> motorCurrentVals_;
std::vector<float> reelVals_;
// tree content
int treeEventNum_;
int treeOrbitNum_;
int treeBx_;
int treeIndex_;
int treeMsgCounter_;
float treeMotorCurrent_;
float treeMotorVoltage_;
int treeReelPos_;
float treeTimestamp1_;
float treeTriggerTimestamp_;
char treeTubeName_[100];
int treeNChInEvent_;
uint32_t treeDriverStatus_;
//uint32_t treeChDenseIndex_[MAXCHPEREVENT];
float treeChHistMean_[MAXCHPEREVENT];
float treeChHistRMS_[MAXCHPEREVENT];

int detieta_[MAXCHPEREVENT];
int detiphi_[MAXCHPEREVENT];
int detidepth_[MAXCHPEREVENT];

uint16_t treeChHistBinContentCap0_[MAXCHPEREVENT][32];
uint16_t treeChHistBinContentCap1_[MAXCHPEREVENT][32];
uint16_t treeChHistBinContentCap2_[MAXCHPEREVENT][32];
uint16_t treeChHistBinContentCap3_[MAXCHPEREVENT][32];
//
// class declaration
//
struct RawHistoData
{
  RawHistoData() { }
  RawHistoData(std::string setTubeName, HcalDetId setDetId, int maxEvents)
  {
    tubeName = setTubeName;
    detId = setDetId;
    eventNumbers.reserve(maxEvents);
    reelPositions.reserve(maxEvents);
    histoAverages.reserve(maxEvents);
    histoRMSs.reserve(maxEvents);
  }

  HcalDetId detId;
  std::string tubeName;
  std::vector<float> eventNumbers;
  std::vector<float> reelPositions;
  std::vector<float> histoAverages;
  std::vector<float> histoRMSs;
};


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
std::string intToString(int num)
{
    using namespace std;
    ostringstream myStream;
    myStream << num << flush;
    return (myStream.str ());
}
//
std::string getRawHistName(int ievent, int ieta, int iphi, int depth, int capId)
{
  std::string histName = "rawHistEvent";
  histName+=intToString(ievent);
  histName+="_Ieta";
  histName+=intToString(ieta);
  histName+="_Iphi";
  histName+=intToString(iphi);
  histName+="_Depth";
  histName+=intToString(depth);
  histName+="_capID";
  histName+=intToString(capId);
  
  return histName;
}
std::string getRawHistName(int ievent, int ieta, int iphi, int depth)
{
  std::string histName = "rawHistEvent";
  histName+=intToString(ievent);
  histName+="_Ieta";
  histName+=intToString(ieta);
  histName+="_Iphi";
  histName+=intToString(iphi);
  histName+="_Depth";
  histName+=intToString(depth);
  
  return histName;
}
//
std::string getGraphName(const HcalDetId& detId, std::string tubeName)
{
  std::string histName=tubeName;
  histName+="_Ieta";
  histName+=intToString(detId.ieta());
  histName+="_Iphi";
  histName+=intToString(detId.iphi());
  histName+="_Depth";
  histName+=intToString(detId.depth());
  
  return histName;
}
//
int getEventFromHistName(std::string histName)
{
  int undIetaPos = histName.find("_Ieta");
  return atoi(histName.substr(9,undIetaPos-9).c_str());
}
//
int getIetaFromHistName(std::string histName)
{
  int undIetaPos = histName.find("_Ieta");
  int undIphiPos = histName.find("_Iphi");
  return atoi(histName.substr(undIetaPos+5,undIphiPos-undIetaPos-5).c_str());
}
//
int getIphiFromHistName(std::string histName)
{
  int undIphiPos = histName.find("_Iphi");
  int undDepthPos = histName.find("_Depth");
  return atoi(histName.substr(undIphiPos+5,undDepthPos-undIphiPos-5).c_str());
}
//
int getDepthFromHistName(std::string histName)
{
  int undDepthPos = histName.find("_Depth");
  return atoi(histName.substr(undDepthPos+6,histName.length()-undDepthPos-6).c_str());
}
//
bool isDigiAssociatedToSourceTube(const HcalDetId& detId, std::string tubeName)
{
  using namespace std;
  int ieta = detId.ieta();
  int iphi = detId.iphi();
  int depth = detId.depth();
  int ietaAbs = detId.ietaAbs();
  //"H2_HB_PHI11_LAYER0_SRCTUBE" // example tube for H2
  //"HFM01_ETA29_PHI55_T1A_SRCTUBE" // example tube for HF/P5
  //"H2_FAKETEST_1_PHI57" // fake H2 tube
  int tubePhi = atof(tubeName.substr(tubeName.find("PHI")+3,tubeName.find("_LAYER")-1).c_str());
  if(tubeName.find("HB") != string::npos)
  {
    if(detId.subdet() != HcalBarrel)
      return false;
    // for HB, tubes go along eta (constant phi)-->keep all eta/depth for specific iphi
    if(tubePhi==iphi)
      return true;
  }
  else if(tubeName.find("HE") != string::npos)
  {
    if(detId.subdet() != HcalEndcap)
      return false;
    // for HE, tubes go along eta
    // for odd and even tube phis, keep all eta/phi/depth at same phi as tube
    if(tubePhi==iphi)
      return true;
    // for even tube phis, also keep lower odd phi channels at high eta
    else if(tubePhi%2==0 && (tubePhi-1)==iphi)
    {
      if(depth==1 || depth==2)
      {
        if(ietaAbs > 20)
          return true;
      }
      else if(depth==3)
      {
        if(ietaAbs==27 || ietaAbs==28)
          return true;
      }
    }
  }
  else if(tubeName.find("HFM") != string::npos)
  {
    // for HF, tubes go into one tower (require same eta,phi)
    int tubeEta = atof(tubeName.substr(tubeName.find("ETA")+3,tubeName.find("_PHI")-1).c_str());
    int tubePhi = atof(tubeName.substr(tubeName.find("PHI")+3,tubeName.find("_T")-1).c_str());
    ieta = fabs(ieta);
    if(tubeEta==ieta && tubePhi==iphi)
      return true;
  }
  else if(tubeName.find("HFP") != string::npos)
  {
    // for HF, tubes go into one tower (require same eta,phi)
    int tubeEta = atof(tubeName.substr(tubeName.find("ETA")+3,tubeName.find("_PHI")-1).c_str());
    int tubePhi = atof(tubeName.substr(tubeName.find("PHI")+3,tubeName.find("_T")-1).c_str());
    if(tubeEta==ieta && tubePhi==iphi)
      return true;
  }
  else if(tubeName.find("FAKE") != string::npos)
  {
    // just keep all the digis for fake
    return true;
  }
  return false;
}


//
// member functions
//
void startHtml()
{
  using namespace std;
  ofstream htmlFile(htmlFileName_);
  if(htmlFile.is_open())
  {
    htmlFile << "<!DOCTYPE html>\n";
    htmlFile << "<html>\n";
    htmlFile << "<head>\n";
    htmlFile << "<title>Histogram Data" << "</title>\n";
    htmlFile << "</head>\n";
    htmlFile << "<body>\n";
    if(selectDigiBasedOnTubeName_)
    {
      htmlFile << "<h2>Histogram Data for all channels matching tubes</h2>\n";
    }
    htmlFile.close();
  }
}
//
void appendHtml(std::string tubeName, std::vector<std::string>& imageNames, std::string htmlFileName)
{
  using namespace std;
  string outputPath = htmlDirName_;
  outputPath+="/";
  outputPath+=htmlFileName;
  ofstream htmlFile(outputPath);
  if(htmlFile.is_open())
  {
    htmlFile << "<!DOCTYPE html>\n";
    htmlFile << "<html>\n";
    htmlFile << "<head>\n";
    htmlFile << "<title>Histogram Data" << "</title>\n";
    htmlFile << "</head>\n";
    htmlFile << "<body>\n";
    htmlFile << "<h2>Histogram Data</h2>\n";
    htmlFile << "<hr>\n";
    htmlFile << "<h3>Tube name: " << tubeName << "</h3>\n";
    htmlFile << "<table>\n";
    htmlFile << "<tr>\n";
    int counter = 0;
    for(std::vector<std::string>::const_iterator imageName = imageNames.begin(); imageName != imageNames.end();
        ++imageName)
    {
      if(counter % newRowEvery_ == 0)
      {
        htmlFile << "</tr>\n";
        htmlFile << "<tr>\n";
      }
      htmlFile << "<td><a href=\"../" << *imageName << "\"><img width=" << thumbnailSize_ << " src=\"../" << *imageName << "\"></a></td>\n";
      ++counter;
    }
    htmlFile << "</tr>\n";
    htmlFile << "</table>\n";
    htmlFile << "<hr>\n";
    htmlFile << "<a href=\"" << rootOutputFileName_ << "\">Download Root file</a>\n";
    htmlFile << "</body>\n";
    htmlFile << "</html>\n";
    htmlFile.close();
  }
}

void appendHtmlMainPage(std::string tubeName, std::vector<std::string>& imageNamesEvt, std::vector<std::string>& imageNamesReel)
{
  using namespace std;
  ofstream htmlFile(htmlFileName_, ios::out | ios::app);
  if(htmlFile.is_open())
  {
    htmlFile << "<h3>Tube name: " << tubeName << "</h3>\n";
    htmlFile << "<table>\n";
    htmlFile << "<tr>\n";
    int counter = 0;
    for(std::vector<std::string>::const_iterator imageName = imageNamesEvt.begin(); imageName != imageNamesEvt.end();
        ++imageName)
    {
      if(counter % newRowEvery_ == 0)
      {
        htmlFile << "</tr>\n";
        htmlFile << "<tr>\n";
      }
      htmlFile << "<td><a href=\"" << *imageName << "\"><img width=" << thumbnailSize_ << " src=\"" << *imageName << "\"></a></td>\n";
      ++counter;
    }
    for(std::vector<std::string>::const_iterator imageName = imageNamesReel.begin(); imageName != imageNamesReel.end();
        ++imageName)
    {
      if(counter % newRowEvery_ == 0)
      {
        htmlFile << "</tr>\n";
        htmlFile << "<tr>\n";
      }
      htmlFile << "<td><a href=\"" << *imageName << "\"><img width=" << thumbnailSize_ << " src=\"" << *imageName << "\"></a></td>\n";
      ++counter;
    }
    htmlFile << "</tr>\n";
    htmlFile << "</table>\n";
    htmlFile << "<hr>\n";
  }
}
//
void finishHtml(std::set<std::string> tubeNameSet)
{
  using namespace std;
  ofstream htmlFile(htmlFileName_, ios::out | ios::app);
  if(htmlFile.is_open())
  {
    htmlFile << "<hr>\n";
    htmlFile << "<h2>Histogram Avgs. vs. Event</h2>\n";
    for(set<string>::const_iterator tubeItr = tubeNameSet.begin(); tubeItr != tubeNameSet.end(); ++tubeItr)
    {
      htmlFile << "<a href=\"" << htmlDirName_ << "/" << *tubeItr << "_histAvgsVsEvent.html" << "\">Tube " << *tubeItr << "</a>\n";
      htmlFile << "<br>\n";
    }
    htmlFile << "<hr>\n";
    htmlFile << "<h2>Histogram Avgs. vs. Reel Position</h2>\n";
    for(set<string>::const_iterator tubeItr = tubeNameSet.begin(); tubeItr != tubeNameSet.end(); ++tubeItr)
    {
      htmlFile << "<a href=\"" << htmlDirName_ <<  "/" << *tubeItr << "_histAvgsVsReel.html" << "\">Tube " << *tubeItr << "</a>\n";
      htmlFile << "<br>\n";
    }
    htmlFile << "<hr>\n";
    htmlFile << "<h2>Mean/RMS Maps</h2>\n";
    int mapThumbSize = 500;
    if(firstEventHistMeanMaps[0]->GetEntries() > 0)
      htmlFile << "<a href=\"" << plotsDirName_ <<  "/" << firstEventHistMeanMaps[0]->GetName() << ".png"
        << "\"><img width=" << thumbnailSize_ << " src=\"" << plotsDirName_ <<  "/" << firstEventHistMeanMaps[0]->GetName()
        << ".png" << "\"></a>\n";
    if(firstEventHistMeanMaps[1]->GetEntries() > 0)
      htmlFile << "<a href=\"" << plotsDirName_ <<  "/" << firstEventHistMeanMaps[1]->GetName() << ".png"
        << "\"><img width=" << thumbnailSize_ << " src=\"" << plotsDirName_ <<  "/" << firstEventHistMeanMaps[1]->GetName()
        << ".png" << "\"></a>\n";
    htmlFile << "<br>\n";
    if(firstEventHistMeanMaps[2]->GetEntries() > 0)
      htmlFile << "<a href=\"" << plotsDirName_ <<  "/" << firstEventHistMeanMaps[2]->GetName() << ".png"
        << "\"><img width=" << thumbnailSize_ << " src=\"" << plotsDirName_ <<  "/" << firstEventHistMeanMaps[2]->GetName()
        << ".png" << "\"></a>\n";
    htmlFile << "<br>\n";
    if(firstEventHistRMSMaps[0]->GetEntries() > 0)
      htmlFile << "<a href=\"" << plotsDirName_ <<  "/" << firstEventHistRMSMaps[0]->GetName() << ".png"
        << "\"><img width=" << mapThumbSize << " src=\"" << plotsDirName_ <<  "/" << firstEventHistRMSMaps[0]->GetName()
        << ".png" << "\"></a>\n";
    if(firstEventHistRMSMaps[1]->GetEntries() > 0)
      htmlFile << "<a href=\"" << plotsDirName_ <<  "/" << firstEventHistRMSMaps[1]->GetName() << ".png"
        << "\"><img width=" << mapThumbSize << " src=\"" << plotsDirName_ <<  "/" << firstEventHistRMSMaps[1]->GetName()
        << ".png" << "\"></a>\n";
    if(firstEventHistRMSMaps[2]->GetEntries() > 0)
      htmlFile << "<a href=\"" << plotsDirName_ <<  "/" << firstEventHistRMSMaps[2]->GetName() << ".png"
        << "\"><img width=" << mapThumbSize << " src=\"" << plotsDirName_ <<  "/" << firstEventHistRMSMaps[2]->GetName()
        << ".png" << "\"></a>\n";
    htmlFile << "<br>\n";
    htmlFile << "<br>\n";
    htmlFile << "<hr>\n";
    htmlFile << "<a href=\"" << rootOutputFileName_ << "\">Download Root file</a>\n";
    htmlFile << "</body>\n";
    htmlFile << "</html>\n";
    htmlFile.close();
  }
}
//
std::string getBlockEventDirName(int event)
{
  int numEventsPerDir = 1000;
  int blockDirNum = (event-1) / numEventsPerDir;
  //int firstEventNum = blockDirNum*numEventsPerDir + 1;
  int firstEventNum = blockDirNum*numEventsPerDir + 1;
  int lastEventNum = (blockDirNum+1)*numEventsPerDir;
  std::string superDirName = "events";
  superDirName+=intToString(firstEventNum);
  superDirName+="to";
  superDirName+=intToString(lastEventNum);
  return superDirName;
}


// ------------ method called once each job just after ending the event loop  ------------
int main(int argc, char ** argv)
{
  // parse arguments
  if(argc < 2)
  {
    std::cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  // get the python configuration
  const edm::ParameterSet& process = edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");
  // now get each parameterset
  const edm::ParameterSet& ana = process.getParameter<edm::ParameterSet>("HCALSourceDataMonPlots");

  
  rootInputFileName_ = ana.getParameter<std::string>("RootInputFileName");
  rootOutputFileName_ = ana.getParameter<std::string>("RootOutputFileName");
  plotsDirName_ =ana.getParameter<std::string>("PlotsDirName");
  htmlDirName_ = ana.getParameter<std::string>("HtmlDirName");
  htmlFileName_ = ana.getParameter<std::string>("HtmlFileName");
  newRowEvery_ = ana.getParameter<int>("NewRowEvery");
  thumbnailSize_ = ana.getParameter<int>("ThumbnailSize");
  outputRawHistograms_ = ana.getParameter<bool>("OutputRawHistograms");
  selectDigiBasedOnTubeName_ = ana.getParameter<bool>("SelectDigiBasedOnTubeName");
  maxEvents_ = ana.getParameter<int>("MaxEvents");

  using namespace std;

  //const int32_t in_detector_mask=0x00100000;

  // do initialization
  rootInputFile_ = new TFile(rootInputFileName_.c_str());
  eventTree_ = (TTree*) rootInputFile_->Get("eventTree");
  cout << "RootInputFile: " << rootInputFileName_ << endl;
  cout << "eventTree: found entries = " << eventTree_->GetEntries() << endl;
  eventTree_->SetBranchAddress("eventNum",&treeEventNum_);
  eventTree_->SetBranchAddress("orbitNum",&treeOrbitNum_);
  eventTree_->SetBranchAddress("bx",&treeBx_);
  eventTree_->SetBranchAddress("index",&treeIndex_);
  eventTree_->SetBranchAddress("msgCounter",&treeMsgCounter_);
  eventTree_->SetBranchAddress("motorCurrent",&treeMotorCurrent_);
  eventTree_->SetBranchAddress("motorVoltage",&treeMotorVoltage_);
  eventTree_->SetBranchAddress("reelPos",&treeReelPos_);
  eventTree_->SetBranchAddress("timestamp1",&treeTimestamp1_);
  eventTree_->SetBranchAddress("triggerTimestamp",&treeTriggerTimestamp_);
  eventTree_->SetBranchAddress("tubeName",treeTubeName_);
  eventTree_->SetBranchAddress("driverStatus",&treeDriverStatus_);
  eventTree_->SetBranchAddress("nChInEvent",&treeNChInEvent_);
  //eventTree_->SetBranchAddress("chDenseIndex",treeChDenseIndex_);
  eventTree_->SetBranchAddress("detieta",detieta_);
  eventTree_->SetBranchAddress("detiphi",detiphi_);
  eventTree_->SetBranchAddress("detidepth",detidepth_);
  eventTree_->SetBranchAddress("chHistMean",treeChHistMean_);
  eventTree_->SetBranchAddress("chHistRMS",treeChHistRMS_);
  eventTree_->SetBranchAddress("chHistBinContentCap0",treeChHistBinContentCap0_);
  eventTree_->SetBranchAddress("chHistBinContentCap1",treeChHistBinContentCap1_);
  eventTree_->SetBranchAddress("chHistBinContentCap2",treeChHistBinContentCap2_);
  eventTree_->SetBranchAddress("chHistBinContentCap3",treeChHistBinContentCap3_);

  evtNumbers_.reserve(500000);
  orbitNumberSecs_.reserve(500000);
  indexVals_.reserve(500000);
  motorCurrentVals_.reserve(500000);
  reelVals_.reserve(500000);

  map<pair<string,HcalDetId>, RawHistoData*> rawHistoDataMap;
  set<string> garageHistNameSet;
  set<string> absorberHistNameSet;
  set<string> garageHistUnevenNameSet;
  set<string> absorberHistUnevenNameSet;
  set<string> tubeNameSet;
  set<HcalDetId> emptyChannels;
  map<HcalDetId, float> firstEventHistMeanValsMap;
  map<HcalDetId, float> firstEventHistRMSValsMap;
  map<string, set<int> > tubeNameToAssociatedChannelDenseIndexMap;
  TFile* outputRootFile = new TFile(rootOutputFileName_.c_str(),"recreate");
  outputRootFile->cd();
  firstEventHistMeanMaps[0] = new TH2F("firstEventHistMeanMapDepth1","histMean  d1;i#eta;i#phi",85,-42.5,42.5,72,0.5,72.5);
  firstEventHistRMSMaps[0] = new TH2F("firstEventHistRMSMapDepth1","histRMS  d1;i#eta;i#phi",85,-42.5,42.5,72,0.5,72.5);
  firstEventHistMeanMaps[1] = new TH2F("firstEventHistMeanMapDepth2","histMean  d2;i#eta;i#phi",85,-42.5,42.5,72,0.5,72.5);
  firstEventHistRMSMaps[1] = new TH2F("firstEventHistRMSMapDepth2","histRMS  d2;i#eta;i#phi",85,-42.5,42.5,72,0.5,72.5);
  firstEventHistMeanMaps[2] = new TH2F("firstEventHistMeanMapDepth3","histMean  d3;i#eta;i#phi",85,-42.5,42.5,72,0.5,72.5);
  firstEventHistRMSMaps[2] = new TH2F("firstEventHistRMSMapDepth3","histRMS  d3;i#eta;i#phi",85,-42.5,42.5,72,0.5,72.5);
  TH1F* firstEventHistMeanDistHEM = new TH1F("firstEventHistMeanHEM","histMean HEM",640,0,31);

  //int NHbins = 32;
  //float binsArray[33]  =   {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
  //                    17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 32.0, 35.0, 38.0, 41.0, 45.0, 49.0, 53.0,
  //                    58.0, 63.0, 64.0};

  // make plots dir
  int status = mkdir(plotsDirName_.c_str(), S_IRWXU);
  if(status != 0 && errno != EEXIST)
  {
    cout << "Creating directory: '" << plotsDirName_ << "' failed; can't continue." << endl;
    return -1;
  }
  // make html dir
  status = mkdir(htmlDirName_.c_str(), S_IRWXU);
  if(status != 0 && errno != EEXIST)
  {
    cout << "Creating directory: '" << htmlDirName_ << "'  failed; can't continue." << endl;
    return -1;
  }


  // disable branches
  eventTree_->SetBranchStatus("*",0);
  eventTree_->SetBranchStatus("tubeName",1);
  //eventTree_->SetBranchStatus("chDenseIndex",1);
  eventTree_->SetBranchStatus("nChInEvent",1);
  eventTree_->SetBranchStatus("chHistMean",1);
  eventTree_->SetBranchStatus("chHistRMS",1);
  cout << "Running over all events in tree" << endl;
  // loop over tree
  int nevents = eventTree_->GetEntries();
  cout << "Beginning loop over events --> found " << nevents << " events." << endl;
  for(int evt=0; evt < nevents; ++evt)
  {
    if((evt+1) > maxEvents_)
      break;

    eventTree_->GetEntry(evt);
    if(treeNChInEvent_ > MAXCHPEREVENT)
    {
      cout << "ERROR: nChInEvent " << treeNChInEvent_ << " > MAXCHPEREVENT=" << MAXCHPEREVENT << "; can't read tree." << endl;
      return -2;
    }

    string tubeName = string(treeTubeName_);
    tubeNameSet.insert(tubeName);


    if(evt < 100)
    {
      // loop over channels in event
      for(int nCh = 0; nCh < treeNChInEvent_; ++nCh)
      {
        //HcalDetId detId = HcalDetId::detIdFromDenseIndex(treeChDenseIndex_[nCh]);
        HcalDetId detId(HcalForward,detieta_[nCh],detiphi_[nCh],detidepth_[nCh]);
	if(firstEventHistMeanValsMap[detId] == 0)
        {
          firstEventHistMeanValsMap[detId] = treeChHistMean_[nCh];
          firstEventHistRMSValsMap[detId] = treeChHistRMS_[nCh];
        }
      }
    }

//
//    //// loop over channels in event
//    //for(int nCh = 0; nCh < treeNChInEvent_; ++nCh)
//    //{
//    //  HcalDetId detId = HcalDetId::detIdFromDenseIndex(treeChDenseIndex_[nCh]);
//
//    //  if(selectDigiBasedOnTubeName_)
//    //    if(!isDigiAssociatedToSourceTube(detId,tubeName))
//    //      continue;
//
//    //  outputRootFile->cd();
//  ////    RawHistoData* thisHistoData = rawHistoDataMap.insert(make_pair(make_pair(tubeName,detId), new RawHistoData(tubeName,detId,50000))).first->second;
//    //  TH1F* thisChannelHist = 0;
//    //  TH1F* thisChannelUnevenBinsHist = 0;
//    //  string histName = "hist_Ieta";
//    //  histName+=intToString(detId.ieta());
//    //  histName+="_Iphi";
//    //  histName+=intToString(detId.iphi());
//    //  histName+="_Depth";
//    //  histName+=intToString(detId.depth());
//    //  histName+="_";
//    //  histName+=tubeName;
//    //  string histNameUneven = histName;
//    //  histNameUneven+="_uneven";
//    //  if(fabs(treeReelPos_) < 5)
//    //  {
//    //    histName+="_sourceInGarage";
//    //    //thisChannelHist = (TH1F*)gDirectory->GetList()->FindObject(histName.c_str());
//    //    //if(!thisChannelHist)
//    //    //{
//    //    //  thisChannelHist = new TH1F(histName.c_str(),histName.c_str(),33,0,32);
//    //    //  thisChannelHist->Sumw2();
//    //    //  garageHistNameSet.insert(histName);
//    //    //}
//    //    histNameUneven+="_sourceInGarage";
//    //    //thisChannelUnevenBinsHist = (TH1F*)gDirectory->GetList()->FindObject(histNameUneven.c_str());
//    //    //if(!thisChannelUnevenBinsHist)
//    //    //{
//    //    //  string histTitleUneven = histNameUneven;
//    //    //  histTitleUneven+=";linearized QIE counts";
//    //    //  thisChannelUnevenBinsHist = new TH1F(histNameUneven.c_str(),histTitleUneven.c_str(),NHbins,binsArray);
//    //    //  thisChannelUnevenBinsHist->Sumw2();
//    //    //  garageHistUnevenNameSet.insert(histNameUneven);
//    //    //}
//    //  }
//    //  else if(treeDriverStatus_ & in_detector_mask)
//    //  {
//    //    histName+="_sourceInAbsorber";
//    //    //thisChannelHist = (TH1F*)gDirectory->GetList()->FindObject(histName.c_str());
//    //    //if(!thisChannelHist)
//    //    //{
//    //    //  thisChannelHist = new TH1F(histName.c_str(),histName.c_str(),33,0,32);
//    //    //  thisChannelHist->Sumw2();
//    //    //  absorberHistNameSet.insert(histName);
//    //    //}
//    //    histNameUneven+="_sourceInAbsorber";
//    //    //thisChannelUnevenBinsHist = (TH1F*)gDirectory->GetList()->FindObject(histNameUneven.c_str());
//    //    //if(!thisChannelUnevenBinsHist)
//    //    //{
//    //    //  string histTitleUneven = histNameUneven;
//    //    //  histTitleUneven+=";linearized QIE counts";
//    //    //  thisChannelUnevenBinsHist = new TH1F(histNameUneven.c_str(),histTitleUneven.c_str(),NHbins,binsArray);
//    //    //  thisChannelUnevenBinsHist->Sumw2();
//    //    //  absorberHistUnevenNameSet.insert(histNameUneven);
//    //    //}
//    //  }
//
//    //  // make hist
//    //  histName = getRawHistName(treeEventNum_,detId.ieta(),detId.iphi(),detId.depth());
//    //  tempHist->Reset();
//    //  tempHist->SetNameTitle(histName.c_str(),histName.c_str());
//    //  for(int ibin=0; ibin<32; ibin++)
//    //  {
//    //    float binValSum = treeChHistBinContentCap0_[nCh][ibin];
//    //    binValSum+=treeChHistBinContentCap1_[nCh][ibin];
//    //    binValSum+=treeChHistBinContentCap2_[nCh][ibin];
//    //    binValSum+=treeChHistBinContentCap3_[nCh][ibin];
//    //    for(int content = 0; content < binValSum; ++content)
//    //    {
//    //      tempHist->Fill(ibin);
//    //      //if(thisChannelHist != 0)
//    //      //{
//    //      //  thisChannelHist->Fill(ibin);
//    //      //}
//    //    }
//
//    //    if(thisChannelUnevenBinsHist)
//    //    {
//    //      if(ibin < 15) thisChannelUnevenBinsHist->SetBinContent(ibin+1,thisChannelUnevenBinsHist->GetBinContent(ibin+1)+binValSum);
//    //      if(14<ibin && ibin<22) thisChannelUnevenBinsHist->SetBinContent(ibin+1,thisChannelUnevenBinsHist->GetBinContent(ibin+1)+binValSum/2.0);
//    //      if(21<ibin && ibin<26) thisChannelUnevenBinsHist->SetBinContent(ibin+1,thisChannelUnevenBinsHist->GetBinContent(ibin+1)+binValSum/3.0);
//    //      if(25<ibin && ibin<29) thisChannelUnevenBinsHist->SetBinContent(ibin+1,thisChannelUnevenBinsHist->GetBinContent(ibin+1)+binValSum/4.0);
//    //      if(28<ibin) thisChannelUnevenBinsHist->SetBinContent(ibin+1,thisChannelUnevenBinsHist->GetBinContent(ibin+1)+binValSum/5.0);
//    //    }
//
//    //  }
//    //  // compute this from the uneven hists --> SIC Nov8 2013
//    //  //if(thisChannelUnevenBinsHist)
//    //  //{
//    //  //  thisHistoData->histoAverages.push_back(thisChannelUnevenBinsHist->GetMean());
//    //  //  thisHistoData->histoRMSs.push_back(thisChannelUnevenBinsHist->GetRMS());
//    //  //}
//
//
//    // 
//    //  if(outputRawHistograms_)
//    //  {
//    //    subDir->cd();
//    //    tempHist->Write();
//    //  }
//    //}
  }

  eventTree_->SetBranchStatus("*",0);
  eventTree_->SetBranchStatus("tubeName",1);
  eventTree_->SetBranchStatus("chHistMean",1);
  eventTree_->SetBranchStatus("reelPos",1);
  eventTree_->SetBranchStatus("eventNum",1);
  eventTree_->SetBranchStatus("nChInEvent",1);
  //eventTree_->SetBranchStatus("chDenseIndex",1);
  eventTree_->SetBranchStatus("detieta",1);
  eventTree_->SetBranchStatus("detiphi",1);
  eventTree_->SetBranchStatus("detidepth",1);
  eventTree_->SetBranchStatus("chHistBinContentCap0",1);
  eventTree_->SetBranchStatus("chHistBinContentCap1",1);
  eventTree_->SetBranchStatus("chHistBinContentCap2",1);
  eventTree_->SetBranchStatus("chHistBinContentCap3",1);
  // now make plots of avgVal vs. event number
  startHtml();
  cout << "Beginning loop over tubes." << endl;
  // make plots of chAvg vs. reel and event for each tube
  for(set<string>::const_iterator tubeItr = tubeNameSet.begin(); tubeItr != tubeNameSet.end(); ++tubeItr)
  {
    vector<string> imageNamesThisTube; 
    vector<string> reelImageNamesThisTube;

    string thisTube = *tubeItr;
    cout << "Found tube: " << thisTube << endl;

    map<HcalDetId,vector<float> > mapHistMeansForCh;
    map<HcalDetId,vector<float> > mapEventNumsForCh;
    map<HcalDetId,vector<float> > mapReelPosForCh;
    map<HcalDetId,TH1F*> mapRawHistForCh;

    int nevents = eventTree_->GetEntries();
    for(int evt=0; evt < nevents; ++evt)
    {
      if((evt+1) > maxEvents_)
        break;

      eventTree_->GetEntry(evt);
      // require that tube being source is the same
      // only get the events from this tube
      if(treeTubeName_ != thisTube)
        continue;

      // make dirs
      outputRootFile->cd();
      TDirectory* subDir = 0;
      if(outputRawHistograms_)
      {
        TDirectory* tubeDir = (TDirectory*) outputRootFile->GetDirectory(thisTube.c_str());
        if(!tubeDir)
          tubeDir = outputRootFile->mkdir(thisTube.c_str());
        string blockDirName = getBlockEventDirName(treeEventNum_);
        string blockDirPath = thisTube;
        blockDirPath+="/";
        blockDirPath+=blockDirName;
        TDirectory* blockEventDir = (TDirectory*) outputRootFile->GetDirectory(blockDirPath.c_str());
        if(!blockEventDir)
          blockEventDir = tubeDir->mkdir(blockDirName.c_str());
        string directoryName = "event";
        directoryName+=intToString(treeEventNum_);
        string dirPath = blockDirPath;
        dirPath+="/";
        dirPath+=directoryName;
        subDir = (TDirectory*) outputRootFile->GetDirectory(dirPath.c_str());
        if(!subDir)
          subDir = blockEventDir->mkdir(directoryName.c_str());
        subDir->cd();
      }
      // end of make dirs
      for(int nCh = 0; nCh < treeNChInEvent_; ++nCh)
      {
        //HcalDetId detId = HcalDetId::detIdFromDenseIndex(treeChDenseIndex_[nCh]);
        HcalDetId detId(HcalForward,detieta_[nCh],detiphi_[nCh],detidepth_[nCh]);
        if(selectDigiBasedOnTubeName_)
          if(!isDigiAssociatedToSourceTube(detId,thisTube))
            continue;

        mapHistMeansForCh[detId].push_back(treeChHistMean_[nCh]);
        mapEventNumsForCh[detId].push_back(treeEventNum_);
        mapReelPosForCh[detId].push_back(treeReelPos_);
        // raw hist
        if(outputRawHistograms_)
        {
          string histName = getRawHistName(treeEventNum_,detId.ieta(),detId.iphi(),detId.depth());
          TH1F* tempHist = new TH1F("tempHist","tempHist",32,0,31);
          tempHist->SetDirectory(0);
          tempHist->SetNameTitle(histName.c_str(),histName.c_str());
          for(int ibin=0; ibin<32; ibin++)
          {
            float binValSum = treeChHistBinContentCap0_[nCh][ibin];
            binValSum+=treeChHistBinContentCap1_[nCh][ibin];
            binValSum+=treeChHistBinContentCap2_[nCh][ibin];
            binValSum+=treeChHistBinContentCap3_[nCh][ibin];
            tempHist->SetBinContent(ibin+1,binValSum);
            //for(int content = 0; content < binValSum; ++content)
            //{
            //  tempHist->Fill(ibin);
            //  //if(thisChannelHist != 0)
            //  //{
            //  //  thisChannelHist->Fill(ibin);
            //  //}
            //}
          }
          tempHist->Write();
          delete tempHist;
        }
      }
    }

    outputRootFile->cd();
    TH2F* histMeanMaps[3];
    TH2F* histRMSMaps[3];
    TH2F* histOccupancyMaps[3];
    for(int i=0; i<3; ++i)
    {
      string histAvgMapName="histAvgMapAllEventsDepth";
      histAvgMapName+=intToString(i+1);
      histAvgMapName+="_";
      histAvgMapName+=thisTube;
      string histAvgMapTitle="histMean depth ";
      histAvgMapTitle+=intToString(i+1);
      histAvgMapTitle+=";i#eta;i#phi";
      histMeanMaps[i] = new TH2F(histAvgMapName.c_str(),histAvgMapTitle.c_str(),85,-42.5,42.5,72,0.5,72.5);
      string histRMSMapName="histRMSMapAllEventsDepth";
      histRMSMapName+=intToString(i+1);
      histRMSMapName+="_";
      histRMSMapName+=thisTube;
      string histRMSMapTitle="histRMS depth ";
      histRMSMapTitle+=intToString(i+1);
      histRMSMapTitle+=";i#eta;i#phi";
      histRMSMaps[i]= new TH2F(histRMSMapName.c_str(),histRMSMapTitle.c_str(),85,-42.5,42.5,72,0.5,72.5);
      string histOccMapName="histOccupancyMapAllEventsDepth";
      histOccMapName+=intToString(i+1);
      histOccMapName+="_";
      histOccMapName+=thisTube;
      string histOccMapTitle="histOccupancy depth ";
      histOccMapTitle+=intToString(i+1);
      histOccMapTitle+=";i#eta;i#phi";
      histOccupancyMaps[i] = new TH2F(histOccMapName.c_str(),histOccMapTitle.c_str(),85,-43,42,72,0.0,72.0);
    }
    // fill histAvg vs reel and event; dist of hist avgs
    for(map<HcalDetId,vector<float> >::const_iterator mapItr = mapHistMeansForCh.begin(); mapItr != mapHistMeansForCh.end(); ++mapItr)
    {
      HcalDetId thisDetId = mapItr->first;
      // hist avgs
      string histAvgDistName = getGraphName(thisDetId,thisTube);
      histAvgDistName+="histAvgDistAllEvents";
      TH1F* thisPlot = new TH1F(histAvgDistName.c_str(),histAvgDistName.c_str(),3200,0,31);
      // fill plot
      for(std::vector<float>::const_iterator i = mapHistMeansForCh[thisDetId].begin(); i != mapHistMeansForCh[thisDetId].end(); ++i)
        thisPlot->Fill(*i);
      float yavg = thisPlot->GetMean();
      float yrms = thisPlot->GetRMS();
      histMeanMaps[thisDetId.depth()-1]->Fill(thisDetId.ieta(),thisDetId.iphi(),yavg);
      histRMSMaps[thisDetId.depth()-1]->Fill(thisDetId.ieta(),thisDetId.iphi(),yrms);
      histOccupancyMaps[thisDetId.depth()-1]->Fill(thisDetId.ieta(),thisDetId.iphi(),1);
      TCanvas* canvas = new TCanvas("canvas","canvas",900,600);
      canvas->cd();
      thisPlot->Draw();
      thisPlot->GetXaxis()->SetTitle("hist. mean [ADC]");
      thisPlot->Write();
      TGraph* thisGraph = new TGraph(mapEventNumsForCh[thisDetId].size(),&(*mapEventNumsForCh[thisDetId].begin()),&(*mapHistMeansForCh[thisDetId].begin()));
      string graphName = getGraphName(thisDetId,thisTube);
      thisGraph->SetTitle(graphName.c_str());
      thisGraph->SetName(graphName.c_str());
      thisGraph->Draw();
      thisGraph->GetXaxis()->SetTitle("Event");
      thisGraph->GetYaxis()->SetTitle("hist. mean [ADC]");
      thisGraph->GetYaxis()->SetRangeUser(yavg-0.5,yavg+0.5);
      thisGraph->Draw("ap");
      thisGraph->Write();
      std::string fullPath = plotsDirName_;
      fullPath+="/";
      fullPath+=graphName;
      fullPath+=".png";
      canvas->Print(fullPath.c_str());
      imageNamesThisTube.push_back(fullPath);
      delete thisGraph;
      TGraph* reelGraph = new TGraph(mapReelPosForCh[thisDetId].size(),&(*mapReelPosForCh[thisDetId].begin()),&(*mapHistMeansForCh[thisDetId].begin()));
      string reelGraphName = getGraphName(thisDetId,thisTube);
      reelGraphName+="reelPosition";
      reelGraph->SetTitle(reelGraphName.c_str());
      reelGraph->SetName(reelGraphName.c_str());
      canvas->cd();
      reelGraph->Draw();
      reelGraph->GetXaxis()->SetTitle("Reel [mm]");
      reelGraph->GetYaxis()->SetTitle("hist. mean [ADC]");
      reelGraph->GetYaxis()->SetRangeUser(yavg-0.5,yavg+0.5);
      reelGraph->Draw("ap");
      reelGraph->Write();
      fullPath = plotsDirName_;
      fullPath+="/";
      fullPath+=reelGraphName;
      fullPath+=".png";
      canvas->Print(fullPath.c_str());
      reelImageNamesThisTube.push_back(fullPath);
      delete reelGraph;
      delete canvas;
    }
    // make images -- maps
    TCanvas* canvasMap = new TCanvas("canvasMap","canvasMap",900,600);
    gStyle->SetOptStat(11);
    for(int i=0; i<3; ++i)
    {
      gStyle->SetPaintTextFormat(".2f");
      histMeanMaps[i]->SetMinimum(histMeanMaps[i]->GetMinimum(0)-0.1);
      histMeanMaps[i]->Write();
      histMeanMaps[i]->Draw("colztext"); gPad->Update();
      TPaveStats *st = (TPaveStats*)histMeanMaps[i]->GetListOfFunctions()->FindObject("stats");
      st->SetX1NDC(0.8);st->SetX2NDC(0.995);st->SetY1NDC(0.93);st->SetY2NDC(0.995);histMeanMaps[i]->Draw("colztext");
      // hardcoded range
      histMeanMaps[i]->GetXaxis()->SetRangeUser(-30,-15);
      histMeanMaps[i]->GetYaxis()->SetRangeUser(29,43);
      string fullMapPath = plotsDirName_;
      fullMapPath+="/";
      fullMapPath+=histMeanMaps[i]->GetName();
      fullMapPath+=".png";
      canvasMap->Print(fullMapPath.c_str());
      gStyle->SetPaintTextFormat(".4f");
      histRMSMaps[i]->SetMinimum(histRMSMaps[i]->GetMinimum(0)-0.0005);
      histRMSMaps[i]->Write();
      histRMSMaps[i]->Draw("colztext"); gPad->Update();
      st = (TPaveStats*)histRMSMaps[i]->GetListOfFunctions()->FindObject("stats");
      st->SetX1NDC(0.8);st->SetX2NDC(0.995);st->SetY1NDC(0.93);st->SetY2NDC(0.995);histRMSMaps[i]->Draw("colztext");
      // hardcoded range
      histRMSMaps[i]->GetXaxis()->SetRangeUser(-30,-15);
      histRMSMaps[i]->GetYaxis()->SetRangeUser(29,43);
      fullMapPath = plotsDirName_;
      fullMapPath+="/";
      fullMapPath+=histRMSMaps[i]->GetName();
      fullMapPath+=".png";
      canvasMap->Print(fullMapPath.c_str());
      histOccupancyMaps[i]->Write();
    }
    appendHtml(thisTube,imageNamesThisTube,thisTube+"_histAvgsVsEvent.html");
    appendHtml(thisTube,reelImageNamesThisTube,thisTube+"_histAvgsVsReel.html");
    if(selectDigiBasedOnTubeName_)
      appendHtmlMainPage(thisTube,imageNamesThisTube,reelImageNamesThisTube);
    }
  cout << "Ending loop over tubes." << endl;


  cout << "Making driver info plots." << endl;
  eventTree_->SetBranchStatus("*",1);
  // make driver info graphs
  TDirectory* dInfoPlotsDir = outputRootFile->mkdir("driverInfoPlots");
  dInfoPlotsDir->cd();
  //
  eventTree_->Draw("eventNum:88.9e-6*(orbitNum+bx/3564)","","goff");
  TGraph* eventNumVsOrbitNumSecsGraph = new TGraph(eventTree_->GetSelectedRows(), eventTree_->GetV2(), eventTree_->GetV1());
  eventNumVsOrbitNumSecsGraph->Draw();
  eventNumVsOrbitNumSecsGraph->GetXaxis()->SetTitle("orbit [s]");
  eventNumVsOrbitNumSecsGraph->GetYaxis()->SetTitle("event");
  eventNumVsOrbitNumSecsGraph->SetName("naiveEventNumVsOrbitNumSecsGraph");
  eventNumVsOrbitNumSecsGraph->SetTitle("");
  eventNumVsOrbitNumSecsGraph->Write();

  //TGraph* messageCounterVsOrbitNumGraph = new TGraph(messageCounterVals_.size(),&(*orbitNumberSecs_.begin()),&(*messageCounterVals_.begin()));
  //messageCounterVsOrbitNumGraph->SetName("messageCounterVsOrbitNumGraph");
  //messageCounterVsOrbitNumGraph->Draw();
  //messageCounterVsOrbitNumGraph->GetXaxis()->SetTitle("orbit [s]");
  //messageCounterVsOrbitNumGraph->GetYaxis()->SetTitle("message");
  //messageCounterVsOrbitNumGraph->SetTitle("");
  //messageCounterVsOrbitNumGraph->Write();

  eventTree_->Draw("index:88.9e-6*(orbitNum+bx/3564)","","goff");
  TGraph* indexVsOrbitNumSecsGraph = new TGraph(eventTree_->GetSelectedRows(), eventTree_->GetV2(), eventTree_->GetV1());
  indexVsOrbitNumSecsGraph->SetName("indexVsOrbitNumSecsGraph");
  indexVsOrbitNumSecsGraph->Draw();
  indexVsOrbitNumSecsGraph->GetXaxis()->SetTitle("orbit [s]");
  indexVsOrbitNumSecsGraph->GetYaxis()->SetTitle("index");
  indexVsOrbitNumSecsGraph->SetTitle("");
  indexVsOrbitNumSecsGraph->Write();

  eventTree_->Draw("motorCurrent:88.9e-6*(orbitNum+bx/3564)","","goff");
  TGraph* motorCurrentVsOrbitNumSecsGraph = new TGraph(eventTree_->GetSelectedRows(), eventTree_->GetV2(), eventTree_->GetV1());
  motorCurrentVsOrbitNumSecsGraph->SetName("motorCurrentVsOrbitNumSecsGraph");
  motorCurrentVsOrbitNumSecsGraph->Draw();
  motorCurrentVsOrbitNumSecsGraph->GetXaxis()->SetTitle("orbit [s]");
  motorCurrentVsOrbitNumSecsGraph->GetYaxis()->SetTitle("motor current [mA]");
  motorCurrentVsOrbitNumSecsGraph->SetTitle("");
  motorCurrentVsOrbitNumSecsGraph->Write();

  //TGraph* motorVoltageVsOrbitNumGraph = new TGraph(motorVoltageVals_.size(),&(*orbitNumberSecs_.begin()),&(*motorVoltageVals_.begin()));
  //motorVoltageVsOrbitNumGraph->SetName("motorVoltageVsOrbitNumGraph");
  //motorVoltageVsOrbitNumGraph->Draw();
  //motorVoltageVsOrbitNumGraph->GetXaxis()->SetTitle("orbit");
  //motorVoltageVsOrbitNumGraph->GetYaxis()->SetTitle("motor voltage [V]");
  //motorVoltageVsOrbitNumGraph->SetTitle("");
  //motorVoltageVsOrbitNumGraph->Write();

  eventTree_->Draw("motorCurrent:reelPos","","goff");
  TGraph* motorCurrentVsReelPosGraph = new TGraph(eventTree_->GetSelectedRows(), eventTree_->GetV2(), eventTree_->GetV1());
  motorCurrentVsReelPosGraph->SetName("motorCurrentVsReelPosGraph");
  motorCurrentVsReelPosGraph->Draw();
  motorCurrentVsReelPosGraph->GetXaxis()->SetTitle("reel [mm]");
  motorCurrentVsReelPosGraph->GetYaxis()->SetTitle("motor current [mA]");
  motorCurrentVsReelPosGraph->SetTitle("");
  motorCurrentVsReelPosGraph->Write();

  //TGraph* motorVoltageVsReelPosGraph = new TGraph(motorVoltageVals_.size(),&(*reelVals_.begin()),&(*motorVoltageVals_.begin()));
  //motorVoltageVsReelPosGraph->SetName("motorVoltageVsReelPosGraph");
  //motorVoltageVsReelPosGraph->Draw();
  //motorVoltageVsReelPosGraph->GetXaxis()->SetTitle("reel [mm]");
  //motorVoltageVsReelPosGraph->GetYaxis()->SetTitle("motor voltage [V]");
  //motorVoltageVsReelPosGraph->SetTitle("");
  //motorVoltageVsReelPosGraph->Write();

  eventTree_->Draw("reelPos:88.9e-6*(orbitNum+bx/3564)","","goff");
  TGraph* reelVsOrbitNumSecsGraph = new TGraph(eventTree_->GetSelectedRows(), eventTree_->GetV2(), eventTree_->GetV1());
  reelVsOrbitNumSecsGraph->SetName("reelVsOrbitNumGraph");
  reelVsOrbitNumSecsGraph->Draw();
  reelVsOrbitNumSecsGraph->GetXaxis()->SetTitle("orbit [s]");
  reelVsOrbitNumSecsGraph->GetYaxis()->SetTitle("reel [mm]");
  reelVsOrbitNumSecsGraph->SetTitle("");
  reelVsOrbitNumSecsGraph->Write();

  //TGraph* triggerTimestampVsOrbitNumGraph = new TGraph(triggerTimeStampVals_.size(),&(*orbitNumberSecs_.begin()),&(*triggerTimeStampVals_.begin()));
  //triggerTimestampVsOrbitNumGraph->SetName("triggerTimestampVsOrbitNumGraph");
  //triggerTimestampVsOrbitNumGraph->GetXaxis()->SetTitle("orbit [s]");
  //triggerTimestampVsOrbitNumGraph->GetYaxis()->SetTitle("trigger timestamp [s]");
  //triggerTimestampVsOrbitNumGraph->SetTitle("");
  //triggerTimestampVsOrbitNumGraph->Write();

  //TGraph* timeStamp1VsOrbitNumGraph = new TGraph(timeStamp1Vals_.size(),&(*orbitNumberSecs_.begin()),&(*timeStamp1Vals_.begin()));
  //timeStamp1VsOrbitNumGraph->SetName("timeStamp1VsOrbitNumGraph");
  //timeStamp1VsOrbitNumGraph->GetXaxis()->SetTitle("orbit [s]");
  //timeStamp1VsOrbitNumGraph->GetYaxis()->SetTitle("timestamp1 [s]");
  //timeStamp1VsOrbitNumGraph->SetTitle("");
  //timeStamp1VsOrbitNumGraph->Write();

  //TGraph* triggerTimeStampVsTimeStamp1Graph = new TGraph(timeStamp1Vals_.size(),&(*timeStamp1Vals_.begin()),&(*triggerTimeStampVals_.begin()));
  //triggerTimeStampVsTimeStamp1Graph->SetName("triggerTimeStampVsTimeStamp1Graph");
  //triggerTimeStampVsTimeStamp1Graph->GetXaxis()->SetTitle("timestamp1 [s]");
  //triggerTimeStampVsTimeStamp1Graph->GetYaxis()->SetTitle("trigger timestamp [s]");
  //triggerTimeStampVsTimeStamp1Graph->SetTitle("");
  //triggerTimeStampVsTimeStamp1Graph->Write();

  // vs event number
  //TGraph* messageCounterVsEventNumGraph = new TGraph(messageCounterVals_.size(),&(*evtNumbers_.begin()),&(*messageCounterVals_.begin()));
  //messageCounterVsEventNumGraph->SetName("messageCounterVsEventNumGraph");
  //messageCounterVsEventNumGraph->Draw();
  //messageCounterVsEventNumGraph->GetXaxis()->SetTitle("event");
  //messageCounterVsEventNumGraph->GetYaxis()->SetTitle("message");
  //messageCounterVsEventNumGraph->SetTitle("");
  //messageCounterVsEventNumGraph->Write();

  eventTree_->Draw("index:eventNum","","goff");
  TGraph* indexVsEventNumGraph = new TGraph(eventTree_->GetSelectedRows(), eventTree_->GetV2(), eventTree_->GetV1());
  indexVsEventNumGraph->SetName("indexVsEventNumGraph");
  indexVsEventNumGraph->Draw();
  indexVsEventNumGraph->GetXaxis()->SetTitle("event");
  indexVsEventNumGraph->GetYaxis()->SetTitle("index");
  indexVsEventNumGraph->SetTitle("");
  indexVsEventNumGraph->Write();

  eventTree_->Draw("motorCurrent:eventNum","","goff");
  TGraph* motorCurrentVsEventNumGraph = new TGraph(eventTree_->GetSelectedRows(), eventTree_->GetV2(), eventTree_->GetV1());
  motorCurrentVsEventNumGraph->SetName("motorCurrentVsEventNumGraph");
  motorCurrentVsEventNumGraph->Draw();
  motorCurrentVsEventNumGraph->GetXaxis()->SetTitle("event");
  motorCurrentVsEventNumGraph->GetYaxis()->SetTitle("motor current [mA]");
  motorCurrentVsEventNumGraph->SetTitle("");
  motorCurrentVsEventNumGraph->Write();

  //TGraph* motorVoltageVsEventNumGraph = new TGraph(motorVoltageVals_.size(),&(*evtNumbers_.begin()),&(*motorVoltageVals_.begin()));
  //motorVoltageVsEventNumGraph->SetName("motorVoltageVsEventNumGraph");
  //motorVoltageVsEventNumGraph->Draw();
  //motorVoltageVsEventNumGraph->GetXaxis()->SetTitle("orbit");
  //motorVoltageVsEventNumGraph->GetYaxis()->SetTitle("motor voltage [V]");
  //motorVoltageVsEventNumGraph->SetTitle("");
  //motorVoltageVsEventNumGraph->Write();

  eventTree_->Draw("reelPos:eventNum","","goff");
  TGraph* reelVsEventNumGraph = new TGraph(eventTree_->GetSelectedRows(), eventTree_->GetV2(), eventTree_->GetV1());
  reelVsEventNumGraph->SetName("reelVsEventNumGraph");
  reelVsEventNumGraph->Draw();
  reelVsEventNumGraph->GetXaxis()->SetTitle("event");
  reelVsEventNumGraph->GetYaxis()->SetTitle("reel [mm]");
  reelVsEventNumGraph->SetTitle("");
  reelVsEventNumGraph->Write();

  //TGraph* triggerTimestampVsEventNumGraph = new TGraph(triggerTimeStampVals_.size(),&(*evtNumbers_.begin()),&(*triggerTimeStampVals_.begin()));
  //triggerTimestampVsEventNumGraph->SetName("triggerTimestampVsEventNumGraph");
  //triggerTimestampVsEventNumGraph->GetXaxis()->SetTitle("event");
  //triggerTimestampVsEventNumGraph->GetYaxis()->SetTitle("trigger timestamp [s]");
  //triggerTimestampVsEventNumGraph->SetTitle("");
  //triggerTimestampVsEventNumGraph->Write();

  //TGraph* timeStamp1VsEventNumGraph = new TGraph(timeStamp1Vals_.size(),&(*evtNumbers_.begin()),&(*timeStamp1Vals_.begin()));
  //timeStamp1VsEventNumGraph->SetName("timeStamp1VsEventNumGraph");
  //timeStamp1VsEventNumGraph->GetXaxis()->SetTitle("event");
  //timeStamp1VsEventNumGraph->GetYaxis()->SetTitle("timestamp1 [s]");
  //timeStamp1VsEventNumGraph->SetTitle("");
  //timeStamp1VsEventNumGraph->Write();
  cout << "Done making driver info plots." << endl;

  rootInputFile_->Close();

  cout << "Ended loop over events." << endl;
  cout << "Saw " << emptyChannels.size() << " empty channels total." << endl;

  for(map<HcalDetId, float>::const_iterator itr = firstEventHistMeanValsMap.begin(); itr != firstEventHistMeanValsMap.end(); ++itr)
  {
    HcalDetId detId = itr->first;
    float mean = itr->second;
    firstEventHistMeanMaps[detId.depth()-1]->Fill(detId.ieta(),detId.iphi(),mean);
    if(detId.subdet()==HcalEndcap && detId.ieta() < 0)
      firstEventHistMeanDistHEM->Fill(mean);
  }
  for(map<HcalDetId, float>::const_iterator itr = firstEventHistRMSValsMap.begin(); itr != firstEventHistRMSValsMap.end(); ++itr)
  {
    HcalDetId detId = itr->first;
    float rms = itr->second;
    firstEventHistRMSMaps[detId.depth()-1]->Fill(detId.ieta(),detId.iphi(),rms);
  }
  // write maps
  outputRootFile->cd();
  //
  firstEventHistMeanMaps[0]->Write();
  firstEventHistRMSMaps[0]->Write();
  firstEventHistMeanMaps[1]->Write();
  firstEventHistRMSMaps[1]->Write();
  firstEventHistMeanMaps[2]->Write();
  firstEventHistRMSMaps[2]->Write();
  firstEventHistMeanDistHEM->Write();
  // make images -- maps
  TCanvas* canvasMap = new TCanvas("canvasMap","canvasMap",900,600);
  gStyle->SetOptStat(11);
  firstEventHistMeanMaps[0]->Draw("colz"); gPad->Update();
  TPaveStats *st = (TPaveStats*)firstEventHistMeanMaps[0]->GetListOfFunctions()->FindObject("stats");
  st->SetX1NDC(0.8);st->SetX2NDC(0.995);st->SetY1NDC(0.93);st->SetY2NDC(0.995);firstEventHistMeanMaps[0]->Draw("colz");
  string fullMapPath = plotsDirName_;
  fullMapPath+="/";
  fullMapPath+=firstEventHistMeanMaps[0]->GetName();
  fullMapPath+=".png";
  canvasMap->Print(fullMapPath.c_str());
  firstEventHistMeanMaps[1]->Draw("colz"); gPad->Update();
  st = (TPaveStats*)firstEventHistMeanMaps[1]->GetListOfFunctions()->FindObject("stats");
  st->SetX1NDC(0.8);st->SetX2NDC(0.995);st->SetY1NDC(0.93);st->SetY2NDC(0.995);firstEventHistMeanMaps[1]->Draw("colz");
  fullMapPath = plotsDirName_;
  fullMapPath+="/";
  fullMapPath+=firstEventHistMeanMaps[1]->GetName();
  fullMapPath+=".png";
  canvasMap->Print(fullMapPath.c_str());
  firstEventHistMeanMaps[2]->Draw("colz"); gPad->Update();
  st = (TPaveStats*)firstEventHistMeanMaps[1]->GetListOfFunctions()->FindObject("stats");
  st->SetX1NDC(0.8);st->SetX2NDC(0.995);st->SetY1NDC(0.93);st->SetY2NDC(0.995);firstEventHistMeanMaps[1]->Draw("colz");
  fullMapPath = plotsDirName_;
  fullMapPath+="/";
  fullMapPath+=firstEventHistMeanMaps[2]->GetName();
  fullMapPath+=".png";
  canvasMap->Print(fullMapPath.c_str());
  //
  firstEventHistRMSMaps[0]->Draw("colz"); gPad->Update();
  st = (TPaveStats*)firstEventHistRMSMaps[0]->GetListOfFunctions()->FindObject("stats");
  st->SetX1NDC(0.8);st->SetX2NDC(0.995);st->SetY1NDC(0.93);st->SetY2NDC(0.995);firstEventHistRMSMaps[0]->Draw("colz");
  fullMapPath = plotsDirName_;
  fullMapPath+="/";
  fullMapPath+=firstEventHistRMSMaps[0]->GetName();
  fullMapPath+=".png";
  canvasMap->Print(fullMapPath.c_str());
  firstEventHistRMSMaps[1]->Draw("colz"); gPad->Update();
  st = (TPaveStats*)firstEventHistRMSMaps[1]->GetListOfFunctions()->FindObject("stats");
  st->SetX1NDC(0.8);st->SetX2NDC(0.995);st->SetY1NDC(0.93);st->SetY2NDC(0.995);firstEventHistRMSMaps[1]->Draw("colz");
  fullMapPath = plotsDirName_;
  fullMapPath+="/";
  fullMapPath+=firstEventHistRMSMaps[1]->GetName();
  fullMapPath+=".png";
  canvasMap->Print(fullMapPath.c_str());
  firstEventHistRMSMaps[2]->Draw("colz"); gPad->Update();
  st = (TPaveStats*)firstEventHistRMSMaps[1]->GetListOfFunctions()->FindObject("stats");
  st->SetX1NDC(0.8);st->SetX2NDC(0.995);st->SetY1NDC(0.93);st->SetY2NDC(0.995);firstEventHistRMSMaps[1]->Draw("colz");
  fullMapPath = plotsDirName_;
  fullMapPath+="/";
  fullMapPath+=firstEventHistRMSMaps[2]->GetName();
  fullMapPath+=".png";
  canvasMap->Print(fullMapPath.c_str());

  gStyle->SetOptStat(2222211);

//  // do quality checks
//  for(set<string>::const_iterator itr = garageHistUnevenNameSet.begin(); itr != garageHistUnevenNameSet.end(); ++itr)
//  {
//    TH1F* sourceGarageHist = (TH1F*)gDirectory->GetList()->FindObject(itr->c_str());
//    string abHistName = *itr;
//    abHistName = abHistName.substr(0,abHistName.find("Garage"));
//    cout << "abHistName=" << abHistName << endl;
//    abHistName+="Absorber";
//    cout << "abHistName2=" << abHistName << endl;
//    TH1F* sourceAbsorberHist = (TH1F*)gDirectory->GetList()->FindObject(abHistName.c_str());
//    if(!sourceAbsorberHist)
//    {
//      cout << "ERROR: could not find source absorber hist:" << abHistName << endl;
//      continue;
//    }
//    sourceGarageHist->Scale(sourceAbsorberHist->Integral(0,10)/sourceGarageHist->Integral(0,10));
//    // look for excess
//    if(sourceAbsorberHist->Integral(11,31)/sourceGarageHist->Integral(11,31) < 5)
//      cout << "ERROR: ratio of events in tail (bins 11-31) (sourceAbsorber/sourceGarage) < 5 for this hist: " << sourceAbsorberHist << endl;
//
//    TCanvas* t = new TCanvas("canvas","canvas",900,600);
//    t->cd();
//    t->SetLogy();
//    sourceGarageHist->SetStats(0);
//    sourceGarageHist->Draw();
//    sourceAbsorberHist->SetStats(0);
//    sourceAbsorberHist->SetLineColor(2);
//    sourceAbsorberHist->Draw("same");
//    string imgName = *itr;
//    imgName = imgName.substr(0,imgName.find("Garage"));
//    imgName+="AbsorberAndGarageOverlay";
//    string fullPath = plotsDirName_;
//    fullPath+="/";
//    fullPath+=imgName;
//    fullPath+=".png";
//    cout << "fullPath=" << fullPath << endl;
//    t->Print(fullPath.c_str());
//    delete t;
//  }
//  outputRootFile->cd();
//  // write per-channel source in/out hists; 
//  for(set<string>::const_iterator itr = garageHistNameSet.begin(); itr != garageHistNameSet.end(); ++itr)
//    gDirectory->GetList()->FindObject(itr->c_str())->Write();
//  for(set<string>::const_iterator itr = absorberHistNameSet.begin(); itr != absorberHistNameSet.end(); ++itr)
//    gDirectory->GetList()->FindObject(itr->c_str())->Write();
//  // uneven
//  for(set<string>::const_iterator itr = garageHistUnevenNameSet.begin(); itr != garageHistUnevenNameSet.end(); ++itr)
//    gDirectory->GetList()->FindObject(itr->c_str())->Write();
//  for(set<string>::const_iterator itr = absorberHistUnevenNameSet.begin(); itr != absorberHistUnevenNameSet.end(); ++itr)
//    gDirectory->GetList()->FindObject(itr->c_str())->Write();



  finishHtml(tubeNameSet);

  gROOT->GetListOfFiles()->Remove(outputRootFile);

  outputRootFile->Close();
  cout << "Closed root file." << endl;
}


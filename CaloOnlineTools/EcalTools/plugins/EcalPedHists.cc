/**
 * Module which outputs a root file of ADC counts (all three gains)
 *   of user-selected channels (defaults to channel 1) for 
 *   user-selected samples (defaults to samples 1,2,3) for
 *   user-selected supermodules.
 * 
 * \author S. Cooper
 *
 */

#include "CaloOnlineTools/EcalTools/plugins/EcalPedHists.h"

EcalPedHists::EcalPedHists(const edm::ParameterSet& ps) :
  runNum_(-1),
  fileName_ (ps.getUntrackedParameter<std::string>("fileName", std::string("ecalPedDigiDump"))),
  barrelDigiCollection_ (ps.getParameter<edm::InputTag> ("EBdigiCollection")),
  endcapDigiCollection_ (ps.getParameter<edm::InputTag> ("EEdigiCollection")),
  headerProducer_ (ps.getParameter<edm::InputTag> ("headerProducer"))
{
  using namespace std;

  fedMap_ = new EcalFedMap();
  histsFilled_ = false;
  //for(int i=601; i<655; ++i)
  //{
  //  listDefaults.push_back(i);
  //} 
  listFEDs_ = ps.getUntrackedParameter<vector<int> >("listFEDs");
  listEBs_ = ps.getUntrackedParameter<vector<string> >("listEBs");
 
  if(listFEDs_.size()==0)
  {
    allFEDsSelected_ = false;
    //if "actual" EB id given, then convert to FEDid and put in listFEDs_
    if(listEBs_.size() > 0)
    {
      listFEDs_.clear();
      for(vector<string>::const_iterator itr = listEBs_.begin(); itr != listEBs_.end(); ++itr)
      {
        listFEDs_.push_back(fedMap_->getFedFromSlice(*itr));
      }
    }
  }
  else if(listFEDs_[0]==-1)
  {
  // Apply no selection if -1 is passed in FED list
    allFEDsSelected_ = true;
    //debug
    //cout << "no selection on FEDs!" << endl;
    //inputIsOk_=false;
    //return;
    //listFEDs_ = listDefaults;
  }
  else
  {
    //in this case, listFEDs should be populated
    allFEDsSelected_ = false;
  }

  if(!allFEDsSelected_)
  {
    // Verify FED numbers are valid
    for (vector<int>::const_iterator intIter = listFEDs_.begin(); intIter != listFEDs_.end(); intIter++)
    {  
      if ( ((*intIter) < 601)||(654 < (*intIter)) )
      {  
        cout << "[EcalPedHists] FED value: " << (*intIter) << " found in listFEDs. "
          << " Valid range is 601-654. Returning." << endl;
        inputIsOk_ = false;
        return;
      }
      else
        theRealFedSet_.insert(*intIter);
    }
  }

  vector<int> listDefaults = vector<int>();
  listDefaults.clear();
  for(int i=1; i<1701; ++i)
  {
    listDefaults.push_back(i);
  }
  listChannels_ = ps.getUntrackedParameter<vector<int> >("listChannels", listDefaults);
  listDefaults.clear();
  // Get samples to plot (default to 1,2,3)
  listDefaults.push_back(0);
  listDefaults.push_back(1);
  listDefaults.push_back(2);
  listSamples_ = ps.getUntrackedParameter<vector<int> >("listSamples", listDefaults);
  
  inputIsOk_ = true;
  vector<int>::iterator intIter;
  
  // Verify crystal numbers are valid
  for (intIter = listChannels_.begin(); intIter != listChannels_.end(); ++intIter)
  { 
      //TODO: Fix crystal index checking?
      //if ( ((*intIter) < 1)||(1700 < (*intIter)) )       
      //{  
      //  cout << "[EcalPedHists] ic value: " << (*intIter) << " found in listChannels. "
      //  	  << " Valid range is 1-1700. Returning." << endl;
      //  inputIsOk_ = false;
      //  return;
      //}
  }
  // Verify sample numbers are valid
  for (intIter = listSamples_.begin(); intIter != listSamples_.end(); intIter++)
  {  
      if ( ((*intIter) < 1)||(10 < (*intIter)) )
      {  
	cout << "[EcalPedHists] sample number: " << (*intIter) << " found in listSamples. "
		  << " Valid range is 1-10. Returning." << endl;
	inputIsOk_ = false;
	return;
      }
  }

}

EcalPedHists::~EcalPedHists() 
{
}

void EcalPedHists::beginRun(edm::Run const &, edm::EventSetup const & c)
{
  edm::ESHandle<EcalElectronicsMapping> elecHandle;
  c.get<EcalMappingRcd>().get(elecHandle);
  ecalElectronicsMap_ = elecHandle.product();
}

void EcalPedHists::endJob(void)
{
  using namespace std;
  if(inputIsOk_)
  { 
    //debug
    //cout << "endJob:creating root file!" << endl;
    
    fileName_ += "-"+intToString(runNum_)+".graph.root";

    TFile root_file_(fileName_.c_str() , "RECREATE");
    //Loop over FEDs first
    for(set<int>::const_iterator FEDitr = theRealFedSet_.begin(); FEDitr != theRealFedSet_.end(); ++FEDitr)
    {
      if(!histsFilled_)
        break;
      string dir = fedMap_->getSliceFromFed(*FEDitr);
      TDirectory* FEDdir = gDirectory->mkdir(dir.c_str());
      FEDdir->cd();
      //root_file_.mkdir(dir.c_str());
      //root_file_.cd(dir.c_str());
      map<string,TH1F*> mapHistos = FEDsAndHistMaps_[*FEDitr];
      
      //Loop over channels; write histos and directory structure
      for (vector<int>::const_iterator itr = listChannels_.begin(); itr!=listChannels_.end(); itr++)
      {
        //debug
        //cout << "loop over channels" << endl;
        
        TH1F* hist = 0;
        string chnl = intToString(*itr);
        string name1 = "Cry";
        name1.append(chnl+"Gain1");
        string name2 = "Cry";
        name2.append(chnl+"Gain6");
        string name3 = "Cry";
        name3.append(chnl+"Gain12");
        hist = mapHistos[name1];
        // This is a sanity check only
        if(hist!=0)
        {
          string cryDirName = "Cry_"+chnl;
          TDirectory* cryDir = FEDdir->mkdir(cryDirName.c_str());
          cryDir->cd();
          hist->SetDirectory(cryDir);
          hist->Write();
          hist = mapHistos[name2];
          hist->SetDirectory(cryDir);
          hist->Write();
          hist = mapHistos[name3];
          hist->SetDirectory(cryDir);
          hist->Write();
          //root_file_.cd(dir.c_str());
          root_file_.cd();
        }
        else
        {
          cerr << "EcalPedHists: Error: This shouldn't happen!" << endl;
        }
      }
      root_file_.cd();
    }
    root_file_.Close();
  }
}

void EcalPedHists::analyze(const edm::Event& e, const edm::EventSetup& c)
{
  using namespace std;
  using namespace edm;

  if (!inputIsOk_)
    return;
  
  // loop over the headers, this is to detect missing FEDs if all are selected
  if(allFEDsSelected_)
  {
    edm::Handle<EcalRawDataCollection> DCCHeaders;
    try {
      e.getByLabel (headerProducer_, DCCHeaders);
    } catch ( std::exception& ex ) {
      edm::LogError ("EcalPedHists") << "Error! can't get the product " 
        << headerProducer_;
      return;
    }

    for (EcalRawDataCollection::const_iterator headerItr= DCCHeaders->begin();
        headerItr != DCCHeaders->end (); 
        ++headerItr) 
    {
      int FEDid = 600+headerItr->id();
      theRealFedSet_.insert(FEDid);
    }
  }

  // loop over fed list and make sure that there are histo maps
  for(set<int>::const_iterator fedItr = theRealFedSet_.begin(); fedItr != theRealFedSet_.end(); ++fedItr)
  {
    if(FEDsAndHistMaps_.find(*fedItr)==FEDsAndHistMaps_.end())
      initHists(*fedItr);
  }
  
  //debug
  //cout << "analyze...input is ok? " << inputIsOk_ << endl;
  
  bool barrelDigisFound = true;
  bool endcapDigisFound = true;
  // get the barrel digis
  // (one digi for each crystal)
  // TODO; SIC: fix this behavior
  Handle<EBDigiCollection> barrelDigis;
  try {
    e.getByLabel (barrelDigiCollection_, barrelDigis);
  } catch ( std::exception& ex ) 
  {
    edm::LogError ("EcalPedOffset") << "Error! can't get the product " 
      << barrelDigiCollection_;
    barrelDigisFound = false;
  }
  // get the endcap digis
  // (one digi for each crystal)
  // TODO; SIC: fix this behavior
  Handle<EEDigiCollection> endcapDigis;
  try {
    e.getByLabel (endcapDigiCollection_, endcapDigis);
  } catch ( std::exception& ex ) 
  {
    edm::LogError ("EcalPedOffset") << "Error! can't get the product " 
      << endcapDigiCollection_;
    endcapDigisFound = false;
  }
  
  if(barrelDigisFound)
    readEBdigis(barrelDigis);
  if(endcapDigisFound)
    readEEdigis(endcapDigis);
  if(!barrelDigisFound && !endcapDigisFound)
    edm::LogError ("EcalPedOffset") << "No digis found in the event!";
  
  if(runNum_==-1)
    runNum_ = e.id().run();
}

// insert the 3-entry hist map into the map keyed by FED number
void EcalPedHists::initHists(int FED)
{
  using namespace std;
  //using namespace edm;

  std::map<string,TH1F*> histMap;
  //debug
  //cout << "Initializing map for FED:" << *FEDitr << endl;
  for (vector<int>::const_iterator intIter = listChannels_.begin(); intIter != listChannels_.end(); ++intIter)
  { 
    //Put 3 histos (1 per gain) for the channel into the map
    string FEDid = intToString(FED);
    string chnl = intToString(*intIter);
    string title1 = "Gain1 ADC Counts for channel ";
    title1.append(chnl);
    string name1 = "Cry";
    name1.append(chnl+"Gain1");
    string title2 = "Gain6 ADC Counts for channel ";
    title2.append(chnl);
    string name2 = "Cry";
    name2.append(chnl+"Gain6");
    string title3 = "Gain12 ADC Counts for channel ";
    title3.append(chnl);
    string name3 = "Cry";
    name3.append(chnl+"Gain12");
    histMap.insert(make_pair(name1,new TH1F(name1.c_str(),title1.c_str(),75,175.0,250.0)));
    histMap[name1]->SetDirectory(0);
    histMap.insert(make_pair(name2,new TH1F(name2.c_str(),title2.c_str(),75,175.0,250.0)));
    histMap[name2]->SetDirectory(0);
    histMap.insert(make_pair(name3,new TH1F(name3.c_str(),title3.c_str(),75,175.0,250.0)));
    histMap[name3]->SetDirectory(0);
  }
  FEDsAndHistMaps_.insert(make_pair(FED,histMap));
}


void EcalPedHists::readEBdigis(edm::Handle<EBDigiCollection> digis)
{
  using namespace std;
  using namespace edm;
  //debug
  //cout << "readEBdigis" << endl;
  
  // Loop over digis
  for (EBDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); ++digiItr )
  {
    EBDetId detId = EBDetId(digiItr->id());
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(detId);
    int FEDid = 600+elecId.dccId();
    int crystalId = detId.ic();

    //debug
    //cout << "FEDid:" << FEDid << " cryId:" << crystalId << endl;
    //cout << "FEDid:" << FEDid << endl;
    //Select desired supermodules only
    set<int>::const_iterator fedIter = find(theRealFedSet_.begin(), theRealFedSet_.end(), FEDid);
    if (fedIter == theRealFedSet_.end())
      continue;

    // Select desired channels only
    vector<int>::iterator icIter;
    icIter = find(listChannels_.begin(), listChannels_.end(), crystalId);
    if (icIter == listChannels_.end())
      continue;

    // Get the adc counts from the selected samples and fill the corresponding histogram
    // Must subtract 1 from user-given sample list (e.g., user's sample 1 -> sample 0)
    for (vector<int>::iterator itr = listSamples_.begin(); itr!=listSamples_.end(); itr++)
    {
      histsFilled_ = true;
      map<string,TH1F*> mapHistos = FEDsAndHistMaps_[FEDid];
      string chnl = intToString(crystalId);
      string name1 = "Cry";
      name1.append(chnl+"Gain1");
      string name2 = "Cry";
      name2.append(chnl+"Gain6");
      string name3 = "Cry";
      name3.append(chnl+"Gain12");
      TH1F* hist = 0;
      if(((EBDataFrame)(*digiItr)).sample(*itr-1).gainId()==3)
        hist = mapHistos[name1];
      if(((EBDataFrame)(*digiItr)).sample(*itr-1).gainId()==2)
        hist = mapHistos[name2];
      if(((EBDataFrame)(*digiItr)).sample(*itr-1).gainId()==1)
        hist = mapHistos[name3];
      if(hist!=0)
        hist->Fill(((EBDataFrame)(*digiItr)).sample(*itr-1).adc());
      else
        cerr << "EcalPedHistDumper: Error: This shouldn't happen!" << endl;
    }
  }
}


void EcalPedHists::readEEdigis(edm::Handle<EEDigiCollection> digis)
{
  using namespace std;
  using namespace edm;
  //debug
  //cout << "readEEdigis" << endl;
  
  // Loop over digis
  for (EEDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); ++digiItr )
  {
    EEDetId detId = EEDetId(digiItr->id());
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(detId);
    int FEDid = 600+elecId.dccId();
    int crystalId = 10000*FEDid+100*elecId.towerId()+5*(elecId.stripId()-1)+elecId.xtalId();

    //Select desired FEDs only
    set<int>::const_iterator fedIter = find(theRealFedSet_.begin(), theRealFedSet_.end(), FEDid);
    if (fedIter == theRealFedSet_.end())
      continue;

    // Select desired channels only
    vector<int>::iterator icIter;
    icIter = find(listChannels_.begin(), listChannels_.end(), crystalId);
    if (icIter == listChannels_.end())
      continue;

    // Get the adc counts from the selected samples and fill the corresponding histogram
    // Must subtract 1 from user-given sample list (e.g., user's sample 1 -> sample 0)
    for (vector<int>::iterator itr = listSamples_.begin(); itr!=listSamples_.end(); itr++)
    {
      histsFilled_ = true;
      map<string,TH1F*> mapHistos = FEDsAndHistMaps_[FEDid];
      string chnl = intToString(crystalId);
      string name1 = "Cry";
      name1.append(chnl+"Gain1");
      string name2 = "Cry";
      name2.append(chnl+"Gain6");
      string name3 = "Cry";
      name3.append(chnl+"Gain12");
      TH1F* hist = 0;
      if(((EBDataFrame)(*digiItr)).sample(*itr-1).gainId()==3)
        hist = mapHistos[name1];
      if(((EBDataFrame)(*digiItr)).sample(*itr-1).gainId()==2)
        hist = mapHistos[name2];
      if(((EBDataFrame)(*digiItr)).sample(*itr-1).gainId()==1)
        hist = mapHistos[name3];
      if(hist!=0)
        hist->Fill(((EBDataFrame)(*digiItr)).sample(*itr-1).adc());
      else
        cerr << "EcalPedHistDumper: Error: This shouldn't happen!" << endl;
    }
  }
}


std::string EcalPedHists::intToString(int num)
{
  using namespace std;
  //
  // outputs the number into the string stream and then flushes
  // the buffer (makes sure the output is put into the stream)
  //
  ostringstream myStream; //creates an ostringstream object
  myStream << num << flush;
  return(myStream.str()); //returns the string form of the stringstream object
}



// -*- C++ -*-
//
// Package:   EcalMipGraphs 
// Class:     EcalMipGraphs 
// 
/**\class EcalMipGraphs EcalMipGraphs.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Th Nov 22 5:46:22 CEST 2007
// $Id: EcalMipGraphs.cc,v 1.5 2008/03/12 18:36:12 scooper Exp $
//
//

#include "CaloOnlineTools/EcalTools/plugins/EcalMipGraphs.h"

using namespace edm;
using namespace std;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalMipGraphs::EcalMipGraphs(const edm::ParameterSet& iConfig) :
  EcalUncalibratedRecHitCollection_ (iConfig.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection")),
  EBDigis_ (iConfig.getParameter<edm::InputTag>("EBDigiCollection")),
  headerProducer_ (iConfig.getParameter<edm::InputTag> ("headerProducer")),
  runNum_(-1),
  side_ (iConfig.getUntrackedParameter<int>("side", 3)),
  givenSeedCry_ (iConfig.getUntrackedParameter<int>("seedCry",0)),
  threshold_ (iConfig.getUntrackedParameter<double>("amplitudeThreshold", 12.0)),
  fileName_ (iConfig.getUntrackedParameter<std::string>("fileName", std::string("ecalMipGraphs")))
{
  vector<int> listDefaults;
  listDefaults.push_back(-1);
  
  maskedChannels_ = iConfig.getUntrackedParameter<vector<int> >("maskedChannels", listDefaults);
  maskedFEDs_ = iConfig.getUntrackedParameter<vector<int> >("maskedFEDs", listDefaults);

  vector<string> defaultMaskedEBs;
  defaultMaskedEBs.push_back("none");
  maskedEBs_ =  iConfig.getUntrackedParameter<vector<string> >("maskedEBs",defaultMaskedEBs);
  
  fedMap_ = new EcalFedMap();

  string title1 = "Jitter for all FEDs";
  string name1 = "JitterAllFEDs";
  allFedsTimingHist_ = new TH1F(name1.c_str(),title1.c_str(),14,-7,7);
  
  // load up the maskedFED list with the proper FEDids
  if(maskedFEDs_[0]==-1)
  {
    //if "actual" EB id given, then convert to FEDid and put in listFEDs_
    if(maskedEBs_[0] != "none")
    {
      maskedFEDs_.clear();
      for(vector<string>::const_iterator ebItr = maskedEBs_.begin(); ebItr != maskedEBs_.end(); ++ebItr)
      {
        maskedFEDs_.push_back(fedMap_->getFedFromSlice(*ebItr));
      }
    }
  }
  
  for (int i=0; i<10; i++)        abscissa[i] = i;
  naiveEvtNum_ = 0;

}


EcalMipGraphs::~EcalMipGraphs()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EcalMipGraphs::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get the headers
  // (one header for each supermodule)
  edm::Handle<EcalRawDataCollection> DCCHeaders;
  iEvent.getByLabel(headerProducer_, DCCHeaders);
  map<int,EcalDCCHeaderBlock> FEDsAndDCCHeaders_;

  for (EcalRawDataCollection::const_iterator headerItr= DCCHeaders->begin();
		  headerItr != DCCHeaders->end (); 
		  ++headerItr) 
  {
    FEDsAndDCCHeaders_[headerItr->id()+600] = *headerItr;
  }

  int ievt = iEvent.id().event();
  int graphCount = 0;
  
  naiveEvtNum_++;

  if(runNum_==-1)
  {
    runNum_ = iEvent.id().run();
    fileName_+=intToString(runNum_);
    fileName_+=".graph.root";
    file_ = TFile::Open(fileName_.c_str(),"RECREATE");
    eventsAndSeedCrys_ = new TNtuple("eventsSeedCrys","Events and Seed Crys Mapping","LV1A:ic:fed");
  }

  //We only want the 3x3's for this event...
  listAllChannels.clear();
  Handle<EcalUncalibratedRecHitCollection> hits;

  ESHandle<CaloTopology> caloTopo;
  iSetup.get<CaloTopologyRecord>().get(caloTopo);
  
  //TODO: improve try/catch behavior
  try
  {
    iEvent.getByLabel(EcalUncalibratedRecHitCollection_, hits);
    int neh = hits->size();
    LogDebug("EcalMipGraphs") << "event " << ievt << " hits collection size " << neh;
  }
  catch ( exception& ex)
  {
    LogWarning("EcalMipGraphs") << EcalUncalibratedRecHitCollection_ << " not available";
  }

  for (EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr)
  {
    EcalUncalibratedRecHit hit = (*hitItr);
    int ic = 0;
    int hashedIndex= 0;
    //EEDetId eeDet;
    //cout << "Subdetector field is: " << hit.id().subdetId() << endl;
    EBDetId ebDet = hit.id();
    //TODO: make it work for endcap FEDs also
    //if(ebDet.isValid())
    //{
    ic = ebDet.ic();
    hashedIndex = ebDet.hashedIndex();
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(ebDet);
    //}
    //else
    //{
    //  eeDet = hit.id();
    //  if(!eeDet.isValid())
    //  {
    //    LogWarning("EcalMipGraphs") << "Unable to find hashedIndex for hit in event " << ievt_;
    //    continue;
    //  }
    //  else
    //  {
    //    ic = eeDet.hashedIndex();
    //    elecId = ecalElectronicsMap->getElectronicsId(eeDet);
    //  }
    //}
    
    int FEDid = 600+elecId.dccId();

    float ampli = hit.amplitude();
    float jitter = hit.jitter();

    vector<int>::iterator result;
    result = find(maskedFEDs_.begin(), maskedFEDs_.end(), FEDid);
    if(result != maskedFEDs_.end())
    {
      LogWarning("EcalMipGraphs") << "skipping uncalRecHit for FED " << FEDid << " ; amplitude " << ampli;
      continue;
    }      

    result = find(maskedChannels_.begin(), maskedChannels_.end(), hashedIndex);
    if  (result != maskedChannels_.end())
    {
      LogWarning("EcalMipGraphs") << "skipping uncalRecHit for channel: " << ic << " in fed: " << FEDid << " with amplitude " << ampli ;
      continue;
    } 
    
    if(ampli > threshold_ && !givenSeedCry_)
    {
      // only produce output if no seed cry is given by user and amplitude makes cut
      LogWarning("EcalMipGraphs") << "channel: " << ic <<  " in fed: " << FEDid <<  "  ampli: " << ampli << " jitter " << jitter
        << " Event: " << ievt;
    }
    
    if(hashedIndex == givenSeedCry_ || (!givenSeedCry_ && ampli > threshold_))
    {
      eventsAndSeedCrys_->Fill(naiveEvtNum_, ic, FEDid);
      crysAndAmplitudesMap_[hashedIndex] = ampli;
      vector<DetId> neighbors = caloTopo->getWindow(ebDet,side_,side_);
      for(vector<DetId>::const_iterator itr = neighbors.begin(); itr != neighbors.end(); ++itr)
      {
        listAllChannels.insert(*itr);
      }
    }
    
    TH1F* timingHist = FEDsAndTimingHists_[FEDid];
    if(timingHist==0)
    {
      initHists(FEDid);
      timingHist = FEDsAndTimingHists_[FEDid];
    }
    
    timingHist->Fill(hit.jitter());
    allFedsTimingHist_->Fill(hit.jitter());
  }

  // retrieving crystal digi from Event
  edm::Handle<EBDigiCollection>  digis;
  iEvent.getByLabel(EBDigis_, digis);

  for(std::set<EBDetId>::const_iterator chnlItr = listAllChannels.begin(); chnlItr!= listAllChannels.end(); ++chnlItr)
  {
      //find digi we need  -- can't get find() to work; need DataFrame(DetId det) to work? 
      //TODO: use find(), launching it twice over EB and EE collections

    int ic = (*chnlItr).ic();
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(*chnlItr);
    int FEDid = 600+elecId.dccId();
    string sliceName = fedMap_->getSliceFromFed(FEDid);
    int hashedIndex = (*chnlItr).hashedIndex();
    EBDigiCollection::const_iterator digiItr = digis->begin();
    
    while(digiItr != digis->end() && ((*digiItr).id()!=*chnlItr))
    {
      ++digiItr;
    }
    if(digiItr==digis->end())
    {
      LogWarning("EcalMipGraphs") << "Cannot find digi for ic:" << ic
        << " FED:" << FEDid << " evt:" << naiveEvtNum_;
      continue;
    }
    
    //EBDataFrame df = (*digis)[hashedIndex];
    
    //cout << "the detId is: " << (*chnlItr).rawId() << endl;
    //cout << "the detId found: " << df.id().rawId() << endl;
    
    
    int gainId = FEDsAndDCCHeaders_[FEDid].getMgpaGain();
    int gainHuman;
    if      (gainId ==1) gainHuman =12;
    else if (gainId ==2) gainHuman =6;
    else if (gainId ==3) gainHuman =1;
    else                 gainHuman =-1; 

    int sample0GainId = EBDataFrame(*digiItr).sample(0).gainId();
    for (int i=0; (unsigned int)i< (*digiItr).size() ; ++i ) {
      EBDataFrame df(*digiItr); 
      ordinate[i] = df.sample(i).adc(); // accounf for possible gain !=12?
      if(df.sample(i).gainId()!=sample0GainId)
        LogWarning("EcalMipGraphs") << "Gain switch detected in evt:" <<
          naiveEvtNum_ << " sample:" << i << " ic:" << ic << " FED:" << FEDid;
    }

    TGraph oneGraph(10, abscissa,ordinate);
    string name = "Graph_ev" + intToString(naiveEvtNum_) + "_ic" + intToString(ic)
      + "_FED" + intToString(FEDid);
    string gainString = (gainId==1) ? "Free" : intToString(gainHuman);
    string title = "Event" + intToString(naiveEvtNum_) + "_lv1a" + intToString(ievt) +
      "_ic" + intToString(ic) + "_" + sliceName + "_gain" + gainString;
    map<int,float>::const_iterator itr;
    itr = crysAndAmplitudesMap_.find(hashedIndex);
    if(itr!=crysAndAmplitudesMap_.end())
      title+="_Amp"+intToString((int)itr->second);
    
    oneGraph.SetTitle(title.c_str());
    oneGraph.SetName(name.c_str());
    graphs.push_back(oneGraph);
    graphCount++;
  }
  
  if(graphs.size()==0)
    return;
  
  writeGraphs();
}

void EcalMipGraphs::writeGraphs()
{
  int graphCount = 0;
  file_->cd();
  std::vector<TGraph>::iterator gr_it;
  for (gr_it = graphs.begin(); gr_it !=  graphs.end(); gr_it++ )
  {
    graphCount++;
    if(graphCount % 100 == 0)
      LogInfo("EcalMipGraphs") << "Writing out graph " << graphCount << " of "
        << graphs.size(); 

    (*gr_it).Write(); 
  }
  
  graphs.clear();
}
  


// insert the hist map into the map keyed by FED number
void EcalMipGraphs::initHists(int FED)
{
  using namespace std;
  
  string title1 = "Jitter for ";
  title1.append(fedMap_->getSliceFromFed(FED));
  string name1 = "JitterFED";
  name1.append(intToString(FED));
  TH1F* timingHist = new TH1F(name1.c_str(),title1.c_str(),14,-7,7);
  FEDsAndTimingHists_[FED] = timingHist;
  FEDsAndTimingHists_[FED]->SetDirectory(0);
}

// ------------ method called once each job just before starting event loop  ------------
void 
EcalMipGraphs::beginJob(const edm::EventSetup& c)
{
  edm::ESHandle< EcalElectronicsMapping > handle;
  c.get< EcalMappingRcd >().get(handle);
  ecalElectronicsMap_ = handle.product();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalMipGraphs::endJob()
{
  writeGraphs();
  eventsAndSeedCrys_->Write();
  for(map<int,TH1F*>::const_iterator itr = FEDsAndTimingHists_.begin();
      itr != FEDsAndTimingHists_.end(); ++itr)
  {
    TH1F* hist = itr->second;
    if(hist!=0)
      hist->Write();
    else
    {
      cerr << "EcalMipGraphs: Error: This shouldn't happen!" << endl;
    }
  }
  allFedsTimingHist_->Write();
  file_->Close();
  std::string channels;
  for(std::vector<int>::const_iterator itr = maskedChannels_.begin();
      itr != maskedChannels_.end(); ++itr)
  {
    channels+=intToString(*itr);
    channels+=",";
  }
  
  LogWarning("EcalMipGraphs") << "Masked channels are: " << channels << " and that is all!";
}


std::string EcalMipGraphs::intToString(int num)
{
    using namespace std;
    ostringstream myStream;
    myStream << num << flush;
    return(myStream.str()); //returns the string form of the stringstream object
}


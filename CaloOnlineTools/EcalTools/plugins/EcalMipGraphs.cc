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
// $Id: EcalMipGraphs.cc,v 1.5 2008/05/06 18:38:08 scooper Exp $
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
  EBRecHitCollection_ (iConfig.getParameter<edm::InputTag>("EcalRecHitCollectionEB")),
  EERecHitCollection_ (iConfig.getParameter<edm::InputTag>("EcalRecHitCollectionEE")),
  EBDigis_ (iConfig.getParameter<edm::InputTag>("EBDigiCollection")),
  EEDigis_ (iConfig.getParameter<edm::InputTag>("EEDigiCollection")),
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
  graphCount = 0;

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
  listEBChannels.clear();
  listEEChannels.clear();
  Handle<EcalRecHitCollection> EBhits;
  Handle<EcalRecHitCollection> EEhits;
  ESHandle<CaloTopology> caloTopo;
  iSetup.get<CaloTopologyRecord>().get(caloTopo);
  iEvent.getByLabel(EBRecHitCollection_, EBhits);
  iEvent.getByLabel(EERecHitCollection_, EEhits);
  LogDebug("EcalMipGraphs") << "event " << ievt << " EBhits collection size " << EBhits->size();
  LogDebug("EcalMipGraphs") << "event " << ievt << " EEhits collection size " << EEhits->size();

  selectEBHits(EBhits, ievt, caloTopo);
  selectEEHits(EEhits, ievt, caloTopo);
  
  // Now, retrieve the crystal digi from the event
  edm::Handle<EBDigiCollection> EBdigisHandle;
  iEvent.getByLabel(EBDigis_, EBdigisHandle);
  edm::Handle<EEDigiCollection> EEdigisHandle;
  iEvent.getByLabel(EEDigis_, EEdigisHandle);
  
  selectEBDigis(EBdigisHandle, ievt, FEDsAndDCCHeaders_);
  selectEEDigis(EEdigisHandle, ievt, FEDsAndDCCHeaders_);
  
  if(graphs.size()==0)
    return;
  
  writeGraphs();
}

void EcalMipGraphs::selectEBDigis(edm::Handle<EBDigiCollection> EBdigisHandle, int ievt,
    map<int,EcalDCCHeaderBlock> FEDsAndDCCHeaders_)
{
  for(std::set<EBDetId>::const_iterator chnlItr = listEBChannels.begin(); chnlItr!= listEBChannels.end(); ++chnlItr)
  {
      //find digi we need  -- can't get find() to work; need DataFrame(DetId det) to work? 
      //TODO: use find(), launching it twice over EB and EE collections
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(*chnlItr);
    int FEDid = 600+elecId.dccId();
    int hashedIndex = chnlItr->hashedIndex();
    int ic = chnlItr->ic();
    string sliceName = fedMap_->getSliceFromFed(FEDid);
    EcalDigiCollection::const_iterator digiItr = EBdigisHandle->begin();
    while(digiItr != EBdigisHandle->end() && ((*digiItr).id()!=*chnlItr))
    {
      ++digiItr;
    }
    if(digiItr==EBdigisHandle->end())
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

    EBDataFrame df(*digiItr);
    double pedestal = 200;

    if(df.sample(0).gainId()!=1 || df.sample(1).gainId()!=1) continue; //goes to the next digi
    else {
      ordinate[0] = df.sample(0).adc();
      ordinate[1] = df.sample(1).adc();
      pedestal = (double)(ordinate[0]+ordinate[1])/(double)2;
    } 
    
    //debug
    //cout << "DCCGainId:" << dccGainId << " sample0 gain:" << sample0GainId << endl; 
    
    for (int i=0; (unsigned int)i< digiItr->size(); ++i ) {
      double gain = 12.; //gainHuman=1, so must x12
      if(df.sample(i).gainId()==1)
      {
        //edm::LogWarning("EcalMipGraphs") << "gain12 detected in ic:" << ic;
        gain = 1.; //gainHuman=12, so no factor needed
      }
      else if(df.sample(i).gainId()==2)
      {
        //edm::LogWarning("EcalMipGraphs") << "gain6 detected in ic:" << ic;
        gain = 2.; // gainHuman=6, so must x2.
      }
      //else if(df.sample(i).gainId()==3)
	//edm::LogWarning("EcalMipGraphs") << "gain1 detected in ic:" << ic;
        
      ordinate[i] = (int)(pedestal+(df.sample(i).adc()-pedestal)*gain);
    }

    TGraph oneGraph(10, abscissa,ordinate);
    string name = "Graph_ev" + intToString(naiveEvtNum_) + "_ic" + intToString(ic)
      + "_FED" + intToString(FEDid);
    string gainString = (gainId==1) ? "Free" : intToString(gainHuman);
    string title = "Event" + intToString(naiveEvtNum_) + "_lv1a" + intToString(ievt) +
      "_ic" + intToString(ic) + "_" + sliceName + "_gain" + gainString;
    
    float energy = 0;
    map<int,float>::const_iterator itr;
    itr = crysAndAmplitudesMap_.find(hashedIndex);
    if(itr!=crysAndAmplitudesMap_.end())
    {
      //edm::LogWarning("EcalMipGraphs")<< "itr->second(ampli)="<< itr->second;
      energy = (float) itr->second;
    }
    //else
    //edm::LogWarning("EcalMipGraphs") << "cry " << ic << "not found in ampMap";
    
    title+="_Energy"+floatToString(round(energy*1000));

    oneGraph.SetTitle(title.c_str());
    oneGraph.SetName(name.c_str());
    graphs.push_back(oneGraph);
    graphCount++;
  }
}

void EcalMipGraphs::selectEEDigis(edm::Handle<EEDigiCollection> EEdigisHandle, int ievt,
    map<int,EcalDCCHeaderBlock> FEDsAndDCCHeaders_)
{
  for(std::set<EEDetId>::const_iterator chnlItr = listEEChannels.begin(); chnlItr!= listEEChannels.end(); ++chnlItr)
  {
      //find digi we need  -- can't get find() to work; need DataFrame(DetId det) to work? 
      //TODO: use find(), launching it twice over EB and EE collections
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(*chnlItr);
    int FEDid = 600+elecId.dccId();
    int hashedIndex = chnlItr->hashedIndex();
    int ic = 10000*FEDid+100*elecId.towerId()+5*(elecId.stripId()-1)+elecId.xtalId();
    string sliceName = fedMap_->getSliceFromFed(FEDid);
    EcalDigiCollection::const_iterator digiItr = EEdigisHandle->begin();
    while(digiItr != EEdigisHandle->end() && ((*digiItr).id()!=*chnlItr))
    {
      ++digiItr;
    }
    if(digiItr==EEdigisHandle->end())
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
}

void EcalMipGraphs::selectEBHits(Handle<EcalRecHitCollection> EBhits,
    int ievt, ESHandle<CaloTopology> caloTopo)
{
  for (EcalRecHitCollection::const_iterator hitItr = EBhits->begin(); hitItr != EBhits->end(); ++hitItr)
  {
    EcalRecHit hit = (*hitItr);
    EBDetId det = hit.id();
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(det);
    int hashedIndex = det.hashedIndex();
    int ic = det.ic();
    int FEDid = 600+elecId.dccId();
    float ampli = hit.energy();
    float jitter = hit.time();

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
      vector<DetId> neighbors = caloTopo->getWindow(det,side_,side_);
      for(vector<DetId>::const_iterator itr = neighbors.begin(); itr != neighbors.end(); ++itr)
      {
        listEBChannels.insert(*itr);
      }
    }
    
    TH1F* timingHist = FEDsAndTimingHists_[FEDid];
    if(timingHist==0)
    {
      initHists(FEDid);
      timingHist = FEDsAndTimingHists_[FEDid];
    }
    timingHist->Fill(hit.time());
    allFedsTimingHist_->Fill(hit.time());
  }
}

void EcalMipGraphs::selectEEHits(Handle<EcalRecHitCollection> EEhits,
    int ievt, ESHandle<CaloTopology> caloTopo)
{
  for (EcalRecHitCollection::const_iterator hitItr = EEhits->begin(); hitItr != EEhits->end(); ++hitItr)
  {
    EcalRecHit hit = (*hitItr);
    EEDetId det = hit.id();
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(det);
    int hashedIndex = det.hashedIndex();
    int FEDid = 600+elecId.dccId();
    int ic = 10000*FEDid+100*elecId.towerId()+5*(elecId.stripId()-1)+elecId.xtalId();
    float ampli = hit.energy();
    float jitter = hit.time();

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
      vector<DetId> neighbors = caloTopo->getWindow(det,side_,side_);
      for(vector<DetId>::const_iterator itr = neighbors.begin(); itr != neighbors.end(); ++itr)
      {
        listEEChannels.insert(*itr);
      }
    }
    
    TH1F* timingHist = FEDsAndTimingHists_[FEDid];
    if(timingHist==0)
    {
      initHists(FEDid);
      timingHist = FEDsAndTimingHists_[FEDid];
    }
    
    timingHist->Fill(hit.time());
    allFedsTimingHist_->Fill(hit.time());
  }
  
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

std::string EcalMipGraphs::floatToString(float num)
{
    using namespace std;
    ostringstream myStream;
    myStream << num << flush;
    return(myStream.str()); //returns the string form of the stringstream object
}

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
// $Id: EcalMipGraphs.cc,v 1.11 2007/12/19 14:32:12 franzoni Exp $
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
  
  fedMap = new EcalFedMap();

  // load up the maskedFED list with the proper FEDids
  if(maskedFEDs_[0]==-1)
  {
    //if "actual" EB id given, then convert to FEDid and put in listFEDs_
    if(maskedEBs_[0] != "none")
    {
      maskedFEDs_.clear();
      for(vector<string>::const_iterator ebItr = maskedEBs_.begin(); ebItr != maskedEBs_.end(); ++ebItr)
      {
        maskedFEDs_.push_back(fedMap->getFedFromSlice(*ebItr));
      }
    }
  }
  
  for (int i=0; i<10; i++)        abscissa[i] = i;
  eventsAndSeedCrys_ = new TNtuple("eventsSeedCrys","Events and Seed Crys Mapping","LV1A:ic:fed");
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
  int ievt = iEvent.id().event();
  int graphCount = 0;
  //We only want the 3x3's for this event...
  listAllChannels.clear();
  auto_ptr<EcalElectronicsMapping> ecalElectronicsMap(new EcalElectronicsMapping);
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
    EcalElectronicsId elecId = ecalElectronicsMap->getElectronicsId(ebDet);
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
      eventsAndSeedCrys_->Fill(ievt, ic, FEDid);
      vector<DetId> neighbors = caloTopo->getWindow(ebDet,side_,side_);
      for(vector<DetId>::const_iterator itr = neighbors.begin(); itr != neighbors.end(); ++itr)
      {
        listAllChannels.insert(*itr);
      }
    }
  }

  // retrieving crystal digi from Event
  edm::Handle<EBDigiCollection>  digis;
  iEvent.getByLabel(EBDigis_, digis);

  for(std::set<EBDetId>::const_iterator chnlItr = listAllChannels.begin(); chnlItr!= listAllChannels.end(); ++chnlItr)
  {
      //find digi we need  -- can't get find() to work; need DataFrame(DetId det) to work? 
      //TODO: use find(), launching it twice over EB and EE collections

    EBDigiCollection::const_iterator digiItr = digis->begin();
    while(digiItr != digis->end() && ((*digiItr).id()!=*chnlItr))
    {
      ++digiItr;
    }
    if(digiItr==digis->end())
      continue;

    int ic = (*chnlItr).ic();
    EcalElectronicsId elecId = ecalElectronicsMap->getElectronicsId(*chnlItr);
    int FEDid = 600+elecId.dccId();
    string sliceName = fedMap->getSliceFromFed(FEDid);
    //int hashedIndex = (*chnlItr).hashedIndex();
    //EBDataFrame df = (*digis)[hashedIndex];
    
    //cout << "the detId is: " << (*chnlItr).rawId() << endl;
    //cout << "the detId found: " << df.id().rawId() << endl;
    
    for (int i=0; (unsigned int)i< (*digiItr).size() ; ++i ) {
      EBDataFrame df(*digiItr); 
      ordinate[i] = df.sample(i).adc();
    }

    TGraph oneGraph(10, abscissa,ordinate);
    string name = "Graph_ev" + intToString(ievt) + "_ic" + intToString(ic)
      + "_FED" + intToString(FEDid);
    string title = "Event" + intToString(ievt) + "_ic" + intToString(ic)
      + "_" + sliceName;
    
    oneGraph.SetTitle(title.c_str());
    oneGraph.SetName(name.c_str());
    graphs.push_back(oneGraph);
    graphCount++;
  }
  
  if(runNum_==-1)
  {
    runNum_ = iEvent.id().run();
    fileName_+=intToString(runNum_);
    fileName_+=".graph.root";
    file = TFile::Open(fileName_.c_str(),"RECREATE");
  }

  if(graphs.size()==0)
    return;
  
  writeGraphs();
}

void EcalMipGraphs::writeGraphs()
{
  int graphCount = 0;
  file->cd();
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
  



// ------------ method called once each job just before starting event loop  ------------
void 
EcalMipGraphs::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalMipGraphs::endJob()
{
  writeGraphs();
  eventsAndSeedCrys_->Write();
  file->Close();
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


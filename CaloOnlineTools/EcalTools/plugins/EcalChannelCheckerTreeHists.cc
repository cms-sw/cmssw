// -*- C++ -*-
//
// Package:   EcalChannelCheckerTreeHists 
// Class:     EcalChannelCheckerTreeHists 
// 
/**\class EcalChannelCheckerTreeHists EcalChannelCheckerTreeHists.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
//
// Original Author:  Seth COOPER
//          Author:  Caterina DOGLIONI
//         Created:  Th Nov 22 5:46:22 CEST 2007
// $Id: EcalChannelCheckerTreeHists.cc,v 1.9 2008/05/06 08:08:18 doglioni Exp $
//
//

// system include files

#include "CaloOnlineTools/EcalTools/plugins/EcalChannelCheckerTreeHists.h"

#include "TTree.h"
#include "TFile.h"

using namespace edm;
using namespace std;

//
// constants, enums and typedefs
//
#define MAX_XTALS 61200

//
// static data member definitions
//

//
// constructors and destructor
//
EcalChannelCheckerTreeHists::EcalChannelCheckerTreeHists(const edm::ParameterSet& iConfig) :
  EBDigis_ (iConfig.getParameter<edm::InputTag>("EBDigiCollection")),
  EBUncalibratedRecHitCollection_ (iConfig.getParameter<edm::InputTag>("EBUncalibratedRecHitCollection")),
  headerProducer_ (iConfig.getParameter<edm::InputTag>("headerProducer")),
  XtalJitterDir_ (fs_->mkdir("XtalJitter")),
  XtalAmpliDir_ (fs_->mkdir("XtalAmpli")),
  XtalPedDir_ (fs_->mkdir("XtalPed")),
  XtalPulseDir_ (fs_->mkdir("XtalPulse")),
  runNum_(-1),
  eventNum_(0)
{
  //cleaning up vectors
  for (int i=0; i<MAX_XTALS; i++) {
    v_h1_XtalJitter_[i]=0;
    v_h1_XtalAmpli_[i]=0;
    v_h1_XtalPed_[i]=0;
    v_prof_XtalPulse_[i]=0;
  }
  //making sure that hists keep track of overflows
  TH1::StatOverflows(1);

  fedMap_ = new EcalFedMap();
}

EcalChannelCheckerTreeHists::~EcalChannelCheckerTreeHists()
{
}

//
// member functions
//

// ------------ method called to for each event  ------------
  void
EcalChannelCheckerTreeHists::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //run number && event number
  runNum_=iEvent.id().run();
  //recalculating event number - some ST data had resetting counters due to DAQ
  eventNum_++;

  std::cout << "starting analysis for event " << eventNum_ << std::endl;

  // get the headers
  // (one header for each FED)
  edm::Handle<EcalRawDataCollection> DCCHeaders;
  iEvent.getByLabel(headerProducer_, DCCHeaders);
  map<int,EcalDCCHeaderBlock> FEDsAndDCCHeaders_;
  for (EcalRawDataCollection::const_iterator headerItr= DCCHeaders->begin();
      headerItr != DCCHeaders->end (); 
      ++headerItr) 
  {
    FEDsAndDCCHeaders_[headerItr->id()+600] = *headerItr;
  }

  // retrieving crystal digis
  edm::Handle<EBDigiCollection>  EBdigis;
  iEvent.getByLabel(EBDigis_, EBdigis);
  // retrieving uncalibrated recHits
  edm::Handle<EcalUncalibratedRecHitCollection> EBhits; 
  iEvent.getByLabel(EBUncalibratedRecHitCollection_, EBhits);
  
  for (EcalUncalibratedRecHitCollection::const_iterator hitItr = EBhits->begin();
      hitItr != EBhits->end(); ++hitItr)
  {
    EBDetId ebDetId = hitItr->id();
    int xtal_hashed = ebDetId.hashedIndex();
    double jitter = hitItr->jitter();
    double amplitude = hitItr->amplitude();
    //debug
    //cout << "crystal hash:" << xtal_hashed << " jitter:" << jitter << " ampli:" << amplitude << endl;
    
    //----------------XTALS ampli/jitter

    //booking jitter histograms
    if (!v_h1_XtalJitter_[xtal_hashed]) {
      //TFileDirectory::make histogram if it has never been booked before
      v_h1_XtalJitter_[xtal_hashed] = XtalJitterDir_.make<TH1D> (intToString(xtal_hashed).c_str(),intToString(xtal_hashed).c_str(),11,-6,5);
      //being paranoid about overflows
      v_h1_XtalJitter_[xtal_hashed]->StatOverflows(1);
      //being paranoid about pointers (but not really doing much)
      if (!v_h1_XtalJitter_[xtal_hashed]) std::cout << "TFileService (xtal jitter) had a problem and you will have a problem soon (segfault)" << std::endl;
    }

    //filling histogram and map
    v_h1_XtalJitter_[xtal_hashed]->Fill(jitter);
    prof2_XtalJitter_->Fill(ebDetId.iphi()-0.5, ebDetId.ieta(), jitter);

    //amplitude
    if (!v_h1_XtalAmpli_[xtal_hashed]) {
      v_h1_XtalAmpli_[xtal_hashed] = XtalAmpliDir_.make<TH1D> (intToString(xtal_hashed).c_str(),intToString(xtal_hashed).c_str(),500,0,500);
      v_h1_XtalAmpli_[xtal_hashed]->StatOverflows(1);
      if (!v_h1_XtalAmpli_[xtal_hashed]) std::cout << "TFileService (xtal ampli) had a problem and you will have a problem soon (segfault)" << std::endl;
    }

    v_h1_XtalAmpli_[xtal_hashed]->Fill(amplitude);
    prof2_XtalAmpli_->Fill(ebDetId.iphi()-0.5, ebDetId.ieta(), amplitude);

  }//end RecHit loop

  for(EBDigiCollection::const_iterator digiItr = EBdigis->begin();
      digiItr != EBdigis->end(); ++digiItr)
  {
    EBDetId ebDetId = digiItr->id();
    EBDataFrame digi(*digiItr);
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(ebDetId);
    //int FEDid = 600+elecId.dccId(); 
    //int ic = ebDetId.ic();
    int xtal_hashed = ebDetId.hashedIndex();

    //TODO: DCC gain checking needed or not?
    //int dccGainId = FEDsAndDCCHeaders_[FEDid].getMgpaGain();
    //int dccGainHuman;
    //double gain = 1.;
    //if      (dccGainId ==1) dccGainHuman =12;
    //else if (dccGainId ==2) dccGainHuman =6;
    //else if (dccGainId ==3) dccGainHuman =1;
    //else                 dccGainHuman =-1; 
    
    //double gain = 12.;
    double sampleADC[10];
    EBDataFrame df(*digiItr);

    double pedestal = 200;

    if(df.sample(0).gainId()!=1 || df.sample(1).gainId()!=1) continue; //goes to the next digi
    else {
      sampleADC[0] = df.sample(0).adc();
      sampleADC[1] = df.sample(1).adc();
      pedestal = (double)(sampleADC[0]+sampleADC[1])/(double)2;
    } 
    
    //debug
    //cout << "DCCGainId:" << dccGainId << " sample0 gain:" << sample0GainId << endl; 
    
    for (int i=0; (unsigned int)i< digiItr->size(); ++i ) {
      EBDataFrame df(*digiItr);
      //if(df.sample(i).gainId()!=sample0GainId)
      //  LogWarning("EcalChannelCheckerTreeHists") << "Gain switch detected in evt:" <<
      //    eventNum_ << " sample:" << i << " ic:" << ic << " FED:" << FEDid;
      //if(df.sample(i).gainId()!=dccGainId)
      //  LogWarning("EcalChannelCheckerTreeHists") << "Gain does not match DCC Header gain in evt:" <<
      //    eventNum_ << " sample:" << i << " ic:" << ic << " FED:" << FEDid;
      double gain = 12.;
      if(df.sample(i).gainId()==1)
        gain = 1.;
      else if(df.sample(i).gainId()==2)
        gain = 2.;
      sampleADC[i] = pedestal+(df.sample(i).adc()-pedestal)*gain;
    }

    //-------------------XTALS ped

    //std::cout << "pedestal=" << pedestal << std::endl;

    if (!v_h1_XtalPed_[xtal_hashed]) {
      v_h1_XtalPed_[xtal_hashed] = XtalPedDir_.make<TH1D> (intToString(xtal_hashed).c_str(),intToString(xtal_hashed).c_str(),1000,0,1000);
      v_h1_XtalPed_[xtal_hashed]->StatOverflows(1);
      if (!v_h1_XtalPed_[xtal_hashed]) std::cout << "TFileService (xtals ped) had a problem and you will have a problem soon (segfault)" << std::endl;
    }

    v_h1_XtalPed_[xtal_hashed]->Fill(pedestal);
    prof2_XtalPed_->Fill(ebDetId.iphi()-0.5, ebDetId.ieta(), pedestal);

    //-------------------XTALS pulse

    if (!v_prof_XtalPulse_[xtal_hashed]) {
      v_prof_XtalPulse_[xtal_hashed] = XtalPulseDir_.make<TProfile> (intToString(xtal_hashed).c_str(),intToString(xtal_hashed).c_str(),10,0.5,10.5);
      if (!v_prof_XtalPulse_[xtal_hashed]) std::cout << "TFileService (xtals pulse) had a problem and you will have a problem soon (segfault)" << std::endl;
    }
    for (int i=0; i<10; i++) {
      //double sample = digi.sample(i).adc();
      v_prof_XtalPulse_[xtal_hashed]->Fill(i+0.5,sampleADC[i],1);

    }
  }//end digi loop

}//end analyze


// ------------ method called once each job just before starting event loop  ------------
  void 
EcalChannelCheckerTreeHists::beginJob(const edm::EventSetup& c)
{
  edm::ESHandle< EcalElectronicsMapping > handle;
  c.get< EcalMappingRcd >().get(handle);
  ecalElectronicsMap_ = handle.product();

  //booking maps

  //Jitter
  std::string name = "Jitter_xtal_avg";
  std::string title = "Jitter (clock of max sample) per xtal (avg)";
  prof2_XtalJitter_=fs_->make<TProfile2D>(name.c_str(), title.c_str(), 360, 0 , 360, 170,-85 , 85);

  //Amplitude
  name = "Ampli_xtal_avg";
  title = "Max amplitude per xtal (avg)";
  prof2_XtalAmpli_=fs_->make<TProfile2D>(name.c_str(), title.c_str(), 360, 0, 360, 170,-85 , 85);

  //Pedestal
  name = "Ped_xtal_avg";
  title = "Pedestal (avg of first two samples) per xtal (avg)";
  prof2_XtalPed_=fs_->make<TProfile2D>(name.c_str(), title.c_str(), 360, 0, 360, 170,-85 , 85);

}



// ------------ method called once each job just after ending the event loop  ------------
  void 
EcalChannelCheckerTreeHists::endJob() 
{
  //TODO: PULSE FITTERS 

  makeTree();

  std::cout << "Writing files...this might take a while" << std::endl;

}


  void 
EcalChannelCheckerTreeHists::makeTree()
{

std::string treeName = "tree"+intToString(runNum_)+".root"; 		
  //TODO: PLACE TREE SOMEWHERE ELSE (.cfg)
  TFile f_xtalTree(treeName.c_str(), "RECREATE");

  //creating file-resident tree (scared of how it's going to be handled...)
  //TODO: find a better name
  TTree tree_xtal("xtal_tree","crystalTree");

  //variables
  int ic;
  char slice[5];
  int ieta;
  int iphi;
  int entries;
  int hashedIndex;
  int runNumber;	
  float ped_avg;
  float ped_rms;  
  float ampli_avg;
  float ampli_rms;
  float jitter_avg;
  float jitter_rms;
  float ampli_fracBelowThreshold;
  float entriesOverAvg;

  tree_xtal.Branch("ic" , &ic, "ic/I");
  tree_xtal.Branch("slice", slice, "slice/C");
  tree_xtal.Branch("ieta" , &ieta, "ieta/I");
  tree_xtal.Branch("iphi" , &iphi, "iphi/I");
  tree_xtal.Branch("runNumber",&runNumber, "runNumber/I");	
  tree_xtal.Branch("hashedIndex" , &hashedIndex, "hashedIndex/I");
  tree_xtal.Branch("ped_avg" , &ped_avg, "ped_avg/F");
  tree_xtal.Branch("ped_rms" , &ped_rms, "ped_rms/F");
  tree_xtal.Branch("ampli_avg" , &ampli_avg, "ampli_avg/F");
  tree_xtal.Branch("ampli_rms" , &ampli_rms, "ampli_rms/F");
  tree_xtal.Branch("jitter_avg" , &jitter_avg, "jitter_avg/F");
  tree_xtal.Branch("jitter_rms" , &jitter_rms, "jitter_rms/F");
  tree_xtal.Branch("ampli_fracBelowThreshold", &ampli_fracBelowThreshold, "ampli_fracBelowThreshold/F");
  tree_xtal.Branch("entries",&entries, "entries/I");
  tree_xtal.Branch("entriesOverAvg",&entriesOverAvg, "entriesOverAvg/F");

  //getting entries average

  float entryAvg=getEntriesAvg();

  //must check for anred crystals: do not fill the tree if any of the histogram is absent
  //TODO: REPORT SOMEWHERE IF ONE OF THE HISTOGRAMS IS MISSING (can it happen? it could, if different digi and recHit loops - e.g. skipping URH for ... errors)

  for (int i=0; i<MAX_XTALS; i++) {

    if (!v_h1_XtalAmpli_[i] || !v_h1_XtalPed_[i] || !v_h1_XtalJitter_[i]) continue;

    EBDetId detId = EBDetId::unhashIndex(i);
    ic = detId.ic();
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(detId);
    strcpy(slice,(fedMap_->getSliceFromFed(600+elecId.dccId())).c_str());
    ieta = detId.ieta();
    iphi = detId.iphi();
    runNumber = runNum_;	
    hashedIndex = detId.hashedIndex(); //TODO: check it is i
    ped_avg = v_h1_XtalPed_[i]->GetMean();
    ped_rms = v_h1_XtalPed_[i]->GetRMS();
    ampli_avg = v_h1_XtalAmpli_[i]->GetMean();
    ampli_rms = v_h1_XtalAmpli_[i]->GetRMS();
    jitter_avg = v_h1_XtalJitter_[i]->GetMean();
    jitter_rms = v_h1_XtalJitter_[i]->GetRMS();
    entries = (int)v_h1_XtalAmpli_[i]->GetEntries();
    entriesOverAvg = (float)entries/entryAvg;
    ampli_fracBelowThreshold = v_h1_XtalAmpli_[i]->Integral(0,10)/v_h1_XtalAmpli_[i]->Integral(0,10000);//FIXME: hardwired? overflows should be included though

    tree_xtal.Fill();

  }
  //saving the tree on file

  tree_xtal.Write();
  f_xtalTree.Close();

}



float EcalChannelCheckerTreeHists::getEntriesAvg() {

float avg=0, entrySum=0;
int counter=0;

for (int i=0; i<MAX_XTALS; i++) {
        
        if (!v_h1_XtalAmpli_[i]) continue;
        entrySum += v_h1_XtalAmpli_[i]->GetEntries();
        counter++ ;
        
} 

avg = entrySum/(float)counter;

return avg;

}

std::string EcalChannelCheckerTreeHists::intToString(int num)
{
  using namespace std;
  ostringstream myStream;
  myStream << num << flush;
  return(myStream.str()); //returns the string form of the stringstream object
}




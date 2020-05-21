// -*- C++ -*-
//
// Package:    EcalPulseShapeGrapher
// Class:      EcalPulseShapeGrapher
//
/**\class EcalPulseShapeGrapher EcalPulseShapeGrapher.cc Analyzers/EcalPulseShapeGrapher/src/EcalPulseShapeGrapher.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth Cooper
//         Created:  Tue Feb  5 11:35:45 CST 2008
//
//

#include "CaloOnlineTools/EcalTools/plugins/EcalPulseShapeGrapher.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalPulseShapeGrapher::EcalPulseShapeGrapher(const edm::ParameterSet& iConfig)
    : EBUncalibratedRecHitCollection_(iConfig.getParameter<edm::InputTag>("EBUncalibratedRecHitCollection")),
      EBDigis_(iConfig.getParameter<edm::InputTag>("EBDigiCollection")),
      EEUncalibratedRecHitCollection_(iConfig.getParameter<edm::InputTag>("EEUncalibratedRecHitCollection")),
      EEDigis_(iConfig.getParameter<edm::InputTag>("EEDigiCollection")),
      ampCut_(iConfig.getUntrackedParameter<int>("AmplitudeCutADC", 13)),
      rootFilename_(iConfig.getUntrackedParameter<std::string>("rootFilename", "pulseShapeGrapher")) {
  //now do what ever initialization is needed

  std::vector<int> listDefaults;
  listDefaults.push_back(-1);
  listChannels_ = iConfig.getUntrackedParameter<std::vector<int> >("listChannels", listDefaults);

  for (int listChannel : listChannels_) {
    std::string title = "Amplitude of cry " + intToString(listChannel);
    std::string name = "ampOfCry" + intToString(listChannel);
    ampHistMap_[listChannel] = new TH1F(name.c_str(), title.c_str(), 100, 0, 100);
    ampHistMap_[listChannel]->GetXaxis()->SetTitle("ADC");

    title = "Amplitude (over 13 ADC) of cry " + intToString(listChannel);
    name = "cutAmpOfCry" + intToString(listChannel);
    cutAmpHistMap_[listChannel] = new TH1F(name.c_str(), title.c_str(), 100, 0, 100);
    cutAmpHistMap_[listChannel]->GetXaxis()->SetTitle("ADC");

    title = "Pulse shape of cry " + intToString(listChannel);
    name = "PulseShapeCry" + intToString(listChannel);
    pulseShapeHistMap_[listChannel] = new TH2F(name.c_str(), title.c_str(), 10, 0, 10, 220, -20, 2);
    pulseShapeHistMap_[listChannel]->GetXaxis()->SetTitle("sample");

    title = "Raw Pulse shape of cry " + intToString(listChannel);
    name = "RawPulseShapeCry" + intToString(listChannel);
    rawPulseShapeHistMap_[listChannel] = new TH2F(name.c_str(), title.c_str(), 10, 0, 10, 500, 0, 500);
    rawPulseShapeHistMap_[listChannel]->GetXaxis()->SetTitle("sample");

    title = "Amplitude of first sample, cry " + intToString(listChannel);
    name = "AmpOfFirstSampleCry" + intToString(listChannel);
    firstSampleHistMap_[listChannel] = new TH1F(name.c_str(), title.c_str(), 300, 100, 400);
    firstSampleHistMap_[listChannel]->GetXaxis()->SetTitle("ADC");
  }

  fedMap_ = new EcalFedMap();

  for (int i = 0; i < 10; i++)
    abscissa[i] = i;
}

EcalPulseShapeGrapher::~EcalPulseShapeGrapher() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void EcalPulseShapeGrapher::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  int numHitsWithActivity = 0;

  //int eventNum = iEvent.id().event();
  //vector<EcalUncalibratedRecHit> sampleHitsPastCut;

  Handle<EcalUncalibratedRecHitCollection> EBHits;
  iEvent.getByLabel(EBUncalibratedRecHitCollection_, EBHits);
  Handle<EcalUncalibratedRecHitCollection> EEHits;
  iEvent.getByLabel(EEUncalibratedRecHitCollection_, EEHits);
  //cout << "event: " << eventNum << " sample hits collection size: " << sampleHits->size() << endl;
  //Handle<EcalUncalibratedRecHitCollection> fittedHits;
  //iEvent.getByLabel(FittedUncalibratedRecHitCollection_, fittedHits);
  Handle<EBDigiCollection> EBdigis;
  iEvent.getByLabel(EBDigis_, EBdigis);
  Handle<EEDigiCollection> EEdigis;
  iEvent.getByLabel(EEDigis_, EEdigis);
  //cout << "event: " << eventNum << " digi collection size: " << digis->size() << endl;

  unique_ptr<EcalElectronicsMapping> ecalElectronicsMap(new EcalElectronicsMapping);

  // Loop over the hits
  for (const auto& hitItr : *EBHits) {
    EcalUncalibratedRecHit hit = hitItr;
    float amplitude = hit.amplitude();
    EBDetId hitDetId = hit.id();

    // Get the Fedid
    EcalElectronicsId elecId = ecalElectronicsMap->getElectronicsId(hitDetId);
    int FEDid = 600 + elecId.dccId();
    string SMname = fedMap_->getSliceFromFed(FEDid);

    vector<int>::const_iterator itr = listChannels_.begin();
    while (itr != listChannels_.end() && (*itr) != hitDetId.hashedIndex()) {
      itr++;
    }
    if (itr == listChannels_.end())
      continue;

    ampHistMap_[hitDetId.hashedIndex()]->Fill(amplitude);
    //cout << "Cry hash:" << hitDetId.hashedIndex() << " amplitude: " << amplitude << endl;
    if (amplitude < ampCut_)
      continue;

    cutAmpHistMap_[hitDetId.hashedIndex()]->Fill(amplitude);
    numHitsWithActivity++;
    EBDigiCollection::const_iterator digiItr = EBdigis->begin();
    while (digiItr != EBdigis->end() && digiItr->id() != hitItr.id()) {
      digiItr++;
    }
    if (digiItr == EBdigis->end())
      continue;

    double sampleADC[10];
    EBDataFrame df(*digiItr);
    double pedestal = 200;

    if (df.sample(0).gainId() != 1 || df.sample(1).gainId() != 1)
      continue;  //goes to the next digi
    else {
      sampleADC[0] = df.sample(0).adc();
      sampleADC[1] = df.sample(1).adc();
      pedestal = (double)(sampleADC[0] + sampleADC[1]) / (double)2;
    }

    for (int i = 0; (unsigned int)i < digiItr->size(); ++i) {
      EBDataFrame df(*digiItr);
      double gain = 12.;
      if (df.sample(i).gainId() == 1)
        gain = 1.;
      else if (df.sample(i).gainId() == 2)
        gain = 2.;
      sampleADC[i] = pedestal + (df.sample(i).adc() - pedestal) * gain;
    }

    //cout << "1) maxsample amp:" << maxSampleAmp << " maxSampleIndex:" << maxSampleIndex << endl;
    for (int i = 0; i < 10; ++i) {
      //cout << "Filling hist for:" << hitDetId.hashedIndex() << " with sample:" << i
      //<< " amp:" <<(float)(df.sample(i).adc()-baseline)/(maxSampleAmp-baseline) << endl;
      //cout << "ADC of sample:" << df.sample(i).adc() << " baseline:" << baseline << " maxSampleAmp:"
      //<< maxSampleAmp << endl << endl;
      pulseShapeHistMap_[hitDetId.hashedIndex()]->Fill(i, (float)(sampleADC[i] - pedestal) / amplitude);
      rawPulseShapeHistMap_[hitDetId.hashedIndex()]->Fill(i, (float)(sampleADC[i]));
    }
    firstSampleHistMap_[hitDetId.hashedIndex()]->Fill(sampleADC[0]);
  }

  // Now do the same for the EE hits
  for (const auto& hitItr : *EEHits) {
    EcalUncalibratedRecHit hit = hitItr;
    float amplitude = hit.amplitude();
    EEDetId hitDetId = hit.id();

    // Get the Fedid
    EcalElectronicsId elecId = ecalElectronicsMap->getElectronicsId(hitDetId);
    int FEDid = 600 + elecId.dccId();
    string SMname = fedMap_->getSliceFromFed(FEDid);

    vector<int>::const_iterator itr = listChannels_.begin();
    while (itr != listChannels_.end() && (*itr) != hitDetId.hashedIndex()) {
      itr++;
    }
    if (itr == listChannels_.end())
      continue;

    ampHistMap_[hitDetId.hashedIndex()]->Fill(amplitude);
    //cout << "Cry hash:" << hitDetId.hashedIndex() << " amplitude: " << amplitude << endl;
    if (amplitude < ampCut_)
      continue;

    cutAmpHistMap_[hitDetId.hashedIndex()]->Fill(amplitude);
    numHitsWithActivity++;
    EEDigiCollection::const_iterator digiItr = EEdigis->begin();
    while (digiItr != EEdigis->end() && digiItr->id() != hitItr.id()) {
      digiItr++;
    }
    if (digiItr == EEdigis->end())
      continue;

    double sampleADC[10];
    EEDataFrame df(*digiItr);
    double pedestal = 200;

    if (df.sample(0).gainId() != 1 || df.sample(1).gainId() != 1)
      continue;  //goes to the next digi
    else {
      sampleADC[0] = df.sample(0).adc();
      sampleADC[1] = df.sample(1).adc();
      pedestal = (double)(sampleADC[0] + sampleADC[1]) / (double)2;
    }

    for (int i = 0; (unsigned int)i < digiItr->size(); ++i) {
      EEDataFrame df(*digiItr);
      double gain = 12.;
      if (df.sample(i).gainId() == 1)
        gain = 1.;
      else if (df.sample(i).gainId() == 2)
        gain = 2.;
      sampleADC[i] = pedestal + (df.sample(i).adc() - pedestal) * gain;
    }

    //cout << "1) maxsample amp:" << maxSampleAmp << " maxSampleIndex:" << maxSampleIndex << endl;
    for (int i = 0; i < 10; ++i) {
      //cout << "Filling hist for:" << hitDetId.hashedIndex() << " with sample:" << i
      //<< " amp:" <<(float)(df.sample(i).adc()-baseline)/(maxSampleAmp-baseline) << endl;
      //cout << "ADC of sample:" << df.sample(i).adc() << " baseline:" << baseline << " maxSampleAmp:"
      //<< maxSampleAmp << endl << endl;
      pulseShapeHistMap_[hitDetId.hashedIndex()]->Fill(i, (float)(sampleADC[i] - pedestal) / amplitude);
      rawPulseShapeHistMap_[hitDetId.hashedIndex()]->Fill(i, (float)(sampleADC[i]));
    }
    firstSampleHistMap_[hitDetId.hashedIndex()]->Fill(sampleADC[0]);
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void EcalPulseShapeGrapher::endJob() {
  rootFilename_ += ".root";
  file_ = new TFile(rootFilename_.c_str(), "RECREATE");
  TH1::AddDirectory(false);

  for (int listChannel : listChannels_) {
    ampHistMap_[listChannel]->Write();
    cutAmpHistMap_[listChannel]->Write();
    firstSampleHistMap_[listChannel]->Write();

    rawPulseShapeHistMap_[listChannel]->Write();
    TProfile* t2 = (TProfile*)(rawPulseShapeHistMap_[listChannel]->ProfileX());
    t2->Write();
    //TODO: fix the normalization so these are correct
    //pulseShapeHistMap_[*itr]->Write();
    //TProfile* t1 = (TProfile*) (pulseShapeHistMap_[*itr]->ProfileX());
    //t1->Write();
  }

  file_->Write();
  file_->Close();
}

std::string EcalPulseShapeGrapher::intToString(int num) {
  using namespace std;
  ostringstream myStream;
  myStream << num << flush;
  return (myStream.str());  //returns the string form of the stringstream object
}

#include "CalibTracker/SiStripDCS/test/Synchronization/SyncDCSO2O.h"

#include "CoralBase/TimeStamp.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

#include <iostream>
#include <algorithm>

#include "TFile.h"
#include "TCanvas.h"

SyncDCSO2O::SyncDCSO2O(const edm::ParameterSet& iConfig) {
  // get all digi collections
  digiProducersList_ = iConfig.getParameter<Parameters>("DigiProducersList");
}

SyncDCSO2O::~SyncDCSO2O() {
  std::cout << "Analyzed events with digis = " << timeInfo_.size() << std::endl;

  // First sort the timeInfo vector by time
  std::sort(timeInfo_.begin(), timeInfo_.end(), SortByTime());

  TFile* outputFile = new TFile("digisAndHVvsTime.root", "RECREATE");
  outputFile->cd();

  TH1F* digis = new TH1F("digis", "digis", timeInfo_.size(), 0, timeInfo_.size());
  TH1F* digisWithMasking = new TH1F("digisWithMasking", "digisWithMasking", timeInfo_.size(), 0, timeInfo_.size());
  TH1F* HVoff = new TH1F("HVoff", "HVoff", timeInfo_.size(), 0, timeInfo_.size());
  TH1F* time = new TH1F("time", "time", timeInfo_.size(), 0, timeInfo_.size());
  std::vector<TimeInfo>::const_iterator it = timeInfo_.begin();
  // Float_t * timeArray = new Float_t[timeInfo_.size()];
  unsigned int i = 1;
  for (; it != timeInfo_.end(); ++it, ++i) {
    digis->SetBinContent(i, it->digiOccupancy);
    digisWithMasking->SetBinContent(i, it->digiOccupancyWithMasking);
    HVoff->SetBinContent(i, it->HVoff);

    // Store only with seconds precision
    coral::TimeStamp coralTime(cond::time::to_boost(it->time));
    // N.B. we add 1 hour to the coralTime because it is the conversion from posix_time which is non-adjusted.
    // The shift of +1 gives the CERN time zone.
    TDatime date1(coralTime.year(),
                  coralTime.month(),
                  coralTime.day(),
                  coralTime.hour() + 1,
                  coralTime.minute(),
                  coralTime.second());
    // timeArray[i-1] = date1.Convert();
    time->SetBinContent(i, date1.Convert());
  }

  // TCanvas * canvas = new TCanvas("digiAndHVvsTimeCanvas", "digi and HVoff vs time", 1000, 800);
  // canvas->Draw();
  // canvas->cd();

  digis->Draw();
  digis->Write();

  digisWithMasking->Draw("same");
  digisWithMasking->Write();

  HVoff->Draw("same");
  HVoff->SetLineColor(2);
  HVoff->Write();

  time->Draw("same");
  time->Write();

  //   TGraph * digisGraph = buildGraph(digis, timeArray);
  //   digisGraph->Write("digisGraph");
  //   TGraph * digisWithMaskingGraph = buildGraph(digisWithMasking, timeArray);
  //   digisWithMaskingGraph->Write("digisWithMaskingGraph");
  //   TGraph * HVoffGraph = buildGraph(HVoff, timeArray);
  //   HVoffGraph->Write("HVoffGraph");

  outputFile->Write();
  outputFile->Close();

  // delete[] timeArray;
}

// ------------ method called to for each event  ------------
void SyncDCSO2O::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  ESHandle<SiStripDetVOff> detVOff;
  iSetup.get<SiStripDetVOffRcd>().get(detVOff);

  std::vector<uint32_t> detIds;
  detVOff->getDetIds(detIds);
  std::cout << "Number of DetIds with HV off = " << detIds.size() << std::endl;

  // stringstream ss;
  // detVOff->printSummary(ss);
  // std::cout << ss.str() << std::endl;

  // double seconds = (iEvent.time().value() >> 32);
  std::cout << "event time = " << iEvent.time().value() << std::endl;
  // std::cout << "event time in seconds = "  << seconds << std::endl;

  coral::TimeStamp coralTime(cond::time::to_boost(iEvent.time().value()));

  std::cout << "year = " << coralTime.year() << ", month = " << coralTime.month() << ", day = " << coralTime.day();
  // N.B. we add 1 hour to the coralTime because it is the conversion from posix_time which is non-adjusted.
  // The shift of +1 gives the CERN time zone.
  std::cout << ", hour = " << coralTime.hour() + 1 << ", minute = " << coralTime.minute()
            << ", second = " << coralTime.second();
  std::cout << ", nanosecond = " << coralTime.nanosecond() << std::endl;

  getDigis(iEvent);
  if (!(digiDetsetVector_[0].isValid()))
    std::cout << "NOT VALID DIGI COLLECTION 0" << std::endl;
  else {
    edm::DetSetVector<SiStripDigi>::const_iterator it = digiDetsetVector_[0]->begin();
    unsigned int totDigis = 0;
    unsigned int totDigisWithMasking = 0;
    for (; it != digiDetsetVector_[0]->end(); ++it) {
      totDigis += it->size();
      // Compute also number of digis masking detIds off according to DetVOff
      if (!(detVOff->IsModuleHVOff(it->detId()))) {
        totDigisWithMasking += it->size();
      }
    }
    std::cout << "digis = " << totDigis << std::endl;
    timeInfo_.push_back(TimeInfo(iEvent.time().value(), totDigis, totDigisWithMasking, detVOff->getHVoffCounts()));
  }
}

void SyncDCSO2O::getDigis(const edm::Event& iEvent) {
  using namespace edm;

  int icoll = 0;
  Parameters::iterator itDigiProducersList = digiProducersList_.begin();
  for (; itDigiProducersList != digiProducersList_.end(); ++itDigiProducersList) {
    std::string digiProducer = itDigiProducersList->getParameter<std::string>("DigiProducer");
    std::string digiLabel = itDigiProducersList->getParameter<std::string>("DigiLabel");
    // std::cout << "Reading digi for " << digiProducer << " with label: " << digiLabel << std::endl;
    iEvent.getByLabel(digiProducer, digiLabel, digiDetsetVector_[icoll]);
    icoll++;
  }
}

/// Build TGraphs with quantity vs time
TGraph* SyncDCSO2O::buildGraph(TH1F* histo, Float_t* timeArray) {
  unsigned int arraySize = histo->GetNbinsX();
  // Note that the array reported has [0] = underflow and [Nbins+1] = overflow
  Float_t* valueArray = (histo->GetArray()) + 1;

  TGraph* graph = new TGraph(arraySize, valueArray, timeArray);
  graph->Draw("A*");
  graph->GetXaxis()->SetTimeDisplay(1);
  graph->GetXaxis()->SetLabelOffset(0.02);
  graph->GetXaxis()->SetTimeFormat("#splitline{  %d}{%H:%M}");
  graph->GetXaxis()->SetTimeOffset(0, "gmt");
  graph->GetYaxis()->SetRangeUser(0, 16000);
  graph->GetXaxis()->SetTitle("day/hour");
  graph->GetXaxis()->SetTitleSize(0.03);
  graph->GetXaxis()->SetTitleColor(kBlack);
  graph->GetXaxis()->SetTitleOffset(1.80);
  graph->GetYaxis()->SetTitle("number of digis");
  graph->GetYaxis()->SetTitleSize(0.03);
  graph->GetYaxis()->SetTitleColor(kBlack);
  graph->GetYaxis()->SetTitleOffset(1.80);
  graph->SetTitle();

  return graph;
}

// ------------ method called once each job just before starting event loop  ------------
void SyncDCSO2O::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SyncDCSO2O::endJob() {}

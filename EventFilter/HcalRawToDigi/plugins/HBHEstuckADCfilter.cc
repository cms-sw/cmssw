// simple filter slecting events with all-equal 8 ADC counts > threshold
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>

class HBHEstuckADCfilter : public edm::one::EDFilter<> {
public:
  explicit HBHEstuckADCfilter(const edm::ParameterSet&);
  ~HBHEstuckADCfilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  edm::EDGetTokenT<QIE11DigiCollection> tok_qie11_;
  int thresholdADC_;
  bool writeList_;
  std::ofstream outfile_;
};

HBHEstuckADCfilter::HBHEstuckADCfilter(const edm::ParameterSet& conf)
    : tok_qie11_(consumes<QIE11DigiCollection>(conf.getParameter<edm::InputTag>("digiLabel"))),
      thresholdADC_(conf.getParameter<int>("thresholdADC")),
      writeList_(conf.getParameter<bool>("writeList")) {
  if (writeList_)
    outfile_.open("events_list_stuckADC.txt");
}

HBHEstuckADCfilter::~HBHEstuckADCfilter() {}

bool HBHEstuckADCfilter::filter(edm::Event& ev, const edm::EventSetup& set) {
  edm::Handle<QIE11DigiCollection> theDigis;
  ev.getByToken(tok_qie11_, theDigis);

  bool result = true;
  for (QIE11DigiCollection::const_iterator itr = theDigis->begin(); itr != theDigis->end(); itr++) {
    int tsize = (*itr).size();
    const QIE11DataFrame frame = *itr;

    bool flag = true;
    int adc0 = (frame[0]).adc();
    if (adc0 < thresholdADC_)
      flag = false;
    else {
      for (int i = 1; i < tsize; i++) {
        if ((frame[i]).adc() != adc0) {
          flag = false;
          break;
        }
      }
    }

    //  report explicitly
    if (flag) {
      const HcalDetId cell(itr->id());
      edm::LogWarning("HBHEstuckADCfilter") << "stuck ADC = " << adc0 << " in  " << cell << std::endl;
      result = false;
    }
  }
  if (!result && writeList_)
    outfile_ << ev.id().run() << ":" << ev.luminosityBlock() << ":" << ev.id().event() << std::endl;

  return result;
}

void HBHEstuckADCfilter::endJob() {
  if (writeList_)
    outfile_.close();
}

void HBHEstuckADCfilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiLabel", edm::InputTag("hcalDigis"));
  desc.add<int>("thresholdADC", 100);
  desc.add<bool>("writeList", true);
  descriptions.add("hbhestuckADCfilter", desc);
}

//define as a plug-in
DEFINE_FWK_MODULE(HBHEstuckADCfilter);

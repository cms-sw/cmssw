#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalUMNioDigi.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include <array>
#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <TH1F.h>

class HcalLaserTest : public edm::global::EDFilter<> {
public:
  explicit HcalLaserTest(const edm::ParameterSet&);
  virtual ~HcalLaserTest();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
private:
  virtual bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  virtual void endJob(void) override;
  
  // ----------member data ---------------------------
  edm::EDGetTokenT<HBHEDigiCollection>     inputTokenHBHE_;
  edm::EDGetTokenT<QIE10DigiCollection>    inputTokenHF_;
  edm::EDGetTokenT<HcalUMNioDigi>          ioTokenUMN_;
  double                                   minFracDiffHBHELaser_, minFracHFLaser_;
  int                                      minADCHBHE_, minADCHF_;

  bool                                     testMode_;
  mutable std::array<std::atomic<int>, 16> eventsByType_;
  mutable std::array<std::atomic<int>, 16> passedEventsByType_;
  TH1F                                    *h_hb1_, *h_hb2_, *h_hb3_, *h_hb4_;
  TH1F                                    *h_hb5_, *h_hf1_, *h_hf2_;
};


HcalLaserTest::HcalLaserTest(const edm::ParameterSet& config) :
  inputTokenHBHE_( consumes<HBHEDigiCollection>( config.getParameter<edm::InputTag>("InputHBHE") ) ),
  inputTokenHF_( consumes<QIE10DigiCollection>( config.getParameter<edm::InputTag>("InputHF") ) ),
  ioTokenUMN_( consumes<HcalUMNioDigi>( config.getParameter<edm::InputTag>("UMNioDigis") )),
  minFracDiffHBHELaser_(config.getParameter<double>("minFracDiffHBHELaser")),
  minFracHFLaser_(config.getParameter<double>("minFracHFLaser")),
  minADCHBHE_(config.getParameter<int>("minADCHBHE")),
  minADCHF_(config.getParameter<int>("minADCHF")),
  testMode_(config.getUntrackedParameter<bool>("testMode", false)),
  eventsByType_(),
  passedEventsByType_() {

  for (auto & i : eventsByType_)       i = 0; 
  for (auto & i : passedEventsByType_) i = 0; 

  edm::Service<TFileService> tfile;
  if ( !tfile.isAvailable() )
    throw cms::Exception("HcalLaserTest") << "TFileService unavailable: "
					  << "please add it to config file";
  h_hb1_ = tfile->make<TH1F>("hb1","Maximum ADC in HB (Good)",5000,0,100);
  h_hb2_ = tfile->make<TH1F>("hb2","Maximum ADC in HB (Bad)", 5000,0,100);
  h_hb3_ = tfile->make<TH1F>("hb3","Signal Channel fraction (Good)", 1000,0,1);
  h_hb4_ = tfile->make<TH1F>("hb4","Signal Channel fraction (Bad)",  1000,0,1);
  h_hb5_ = tfile->make<TH1F>("hb5","Signal Channel fraction (Diff)",1000,-1,1);
  h_hf1_ = tfile->make<TH1F>("hf1","Maximum ADC in HF",       5000,0,100);
  h_hf2_ = tfile->make<TH1F>("hf2","Signal Channel fraction (HF)", 1000,0,1);
}

HcalLaserTest::~HcalLaserTest() { }
 
void HcalLaserTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputHBHE",edm::InputTag("source"));
  desc.add<edm::InputTag>("InputHF",edm::InputTag("source"));
  desc.add<edm::InputTag>("UMNioDigis",edm::InputTag("UMNioDigis"));
  desc.add<int>("minADCHBHE",10);
  desc.add<int>("minADCHF",10);
  desc.add<double>("minFracDiffHBHELaser",0.3);
  desc.add<double>("minFracHFLaser",0.3);
  desc.addUntracked<bool>("testMode",false);
  descriptions.add("hcalLaserTest",desc);
}

bool HcalLaserTest::filter(edm::StreamID, edm::Event& iEvent, 
			   const edm::EventSetup& iSetup) const {
 
  edm::Handle<HBHEDigiCollection>     hbhe_digi; 
  iEvent.getByToken(inputTokenHBHE_,  hbhe_digi);

  edm::Handle<QIE10DigiCollection>    hf_digi; 
  iEvent.getByToken(inputTokenHF_,    hf_digi);

  // Count digis in good, bad RBXes.  ('bad' RBXes see no laser signal)
  double badrbxfracHBHE(0), goodrbxfracHBHE(0), goodrbxfracHF(0);
  int NbadHBHE = 72*3;             // 3 bad RBXes, 72 channels each
  int NgoodHBHE= 2592*2-NbadHBHE;  // remaining HBHE channels are 'good'
  int NgoodHF  = 864*4;
  int eventType = 0;
  int laserType = 0;

  // Verify proper functioning on calibration stream events, 
  // where the laser firing location "laserType"
  // is tagged by the UMNio board for eventType==14

  edm::Handle<HcalUMNioDigi> cumn;
  iEvent.getByToken(ioTokenUMN_, cumn);

  eventType = static_cast<int>(cumn->eventType());
  laserType = static_cast<int>(cumn->valueUserWord(0));

  if (eventType == 14) eventsByType_.at(laserType)++;

  if (testMode_)
    edm::LogVerbatim("HcalLaserTest") 
      << "hbhe digi collection size: " << hbhe_digi->size() << "\n"
      << "hf digi collection size: " << hf_digi->size() << "\n"
      << "Event type: " << eventType 
      << " Laser type: " << laserType << std::endl;

  for (HBHEDigiCollection::const_iterator hbhe = hbhe_digi->begin();  
       hbhe != hbhe_digi->end(); ++ hbhe){
    const HBHEDataFrame digi = (const HBHEDataFrame)(*hbhe);
    HcalDetId myid=(HcalDetId)digi.id();
    bool isbad(false); 

    bool passCut(false);
    int  maxdigiHB(0);
    for (int i=0; i<digi.size(); i++) 
      if(digi.sample(i).adc() > maxdigiHB) maxdigiHB = digi.sample(i).adc();
    if (maxdigiHB > minADCHBHE_) passCut = true;

    if ( myid.subdet()==HcalBarrel && myid.ieta()<0 ) {
      if      (myid.iphi()>=15 && myid.iphi()<=18) isbad=true;
      else if (myid.iphi()>=27 && myid.iphi()<=34) isbad=true;
    }

    if (isbad) h_hb2_->Fill(maxdigiHB);
    else       h_hb1_->Fill(maxdigiHB);

    if (passCut) {
      if (isbad) { 
	badrbxfracHBHE += 1.;
      } else goodrbxfracHBHE += 1.;
    }
  }
  goodrbxfracHBHE /= NgoodHBHE;
  badrbxfracHBHE  /= NbadHBHE;
  h_hb3_->Fill(goodrbxfracHBHE);
  h_hb4_->Fill(badrbxfracHBHE);
  h_hb5_->Fill(goodrbxfracHBHE-badrbxfracHBHE);
  
  for (QIE10DigiCollection::const_iterator hf = hf_digi->begin();  
       hf != hf_digi->end(); ++hf) {
    const QIE10DataFrame digi = (const QIE10DataFrame)(*hf);
    bool passCut(false);
    int  maxdigiHF(0);
    for (int i=0; i<digi.samples(); i++) 
      if(digi[i].adc() > maxdigiHF) maxdigiHF = digi[i].adc();
    if (maxdigiHF > minADCHF_) passCut = true;
    h_hf1_->Fill(maxdigiHF);

    if (passCut) {
      goodrbxfracHF += 1.;
    }
  }
  goodrbxfracHF /= NgoodHF;
  h_hf2_->Fill(goodrbxfracHF);

  if (testMode_) 
    edm::LogWarning("HcalLaserTest") 
      << "******************************************************************\n"
      << "goodrbxfracHBHE: " << goodrbxfracHBHE << " badrbxfracHBHE: " 
      << badrbxfracHBHE << " Size " << hbhe_digi->size() << "\n"
      << "goodrbxfracHF:   " << goodrbxfracHF   << " Size " << hf_digi->size()
      << "\n******************************************************************";
  
  if (((goodrbxfracHBHE-badrbxfracHBHE) < minFracDiffHBHELaser_) ||
      (goodrbxfracHF < minFracHFLaser_))  return false;
      
  passedEventsByType_.at(laserType)++;
  return true;
}

// ------------ method called once each job just after ending the event loop  ------------
void HcalLaserTest::endJob() {
  edm::LogVerbatim("HcalLaserTest") 
    << "Summary of filter decisions (passed/total): \n" 
    << passedEventsByType_.at(hc_Null)      << "/" << eventsByType_.at(hc_Null)      << "(No Calib), " 
    << passedEventsByType_.at(hc_Pedestal)  << "/" << eventsByType_.at(hc_Pedestal)  << "(Pedestal), " 
    << passedEventsByType_.at(hc_RADDAM)    << "/" << eventsByType_.at(hc_RADDAM)    << "(RADDAM), " 
    << passedEventsByType_.at(hc_HBHEHPD)   << "/" << eventsByType_.at(hc_HBHEHPD)   << "(HBHE/HPD), " 
    << passedEventsByType_.at(hc_HOHPD)     << "/" << eventsByType_.at(hc_HOHPD)     << "(HO/HPD), " 
    << passedEventsByType_.at(hc_HFPMT)     << "/" << eventsByType_.at(hc_HFPMT)     << "(HF/PMT), "  
    << passedEventsByType_.at(6)            << "/" << eventsByType_.at(6)            << "(ZDC), "  
    << passedEventsByType_.at(7)            << "/" << eventsByType_.at(7)            << "(HEPMega)\n"  
    << passedEventsByType_.at(8)            << "/" << eventsByType_.at(8)            << "(HEMMega), "  
    << passedEventsByType_.at(9)            << "/" << eventsByType_.at(9)            << "(HBPMega), "  
    << passedEventsByType_.at(10)           << "/" << eventsByType_.at(10)           << "(HBMMega), "  
    << passedEventsByType_.at(11)           << "/" << eventsByType_.at(11)           << "(Undefined), "  
    << passedEventsByType_.at(12)           << "/" << eventsByType_.at(12)           << "(CRF), "  
    << passedEventsByType_.at(13)           << "/" << eventsByType_.at(13)           << "(Calib), "  
    << passedEventsByType_.at(14)           << "/" << eventsByType_.at(14)           << "(Safe), "  
    << passedEventsByType_.at(15)           << "/" << eventsByType_.at(15)           << "(Undefined)";
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalLaserTest);

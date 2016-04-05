/*
 * \file L1TLayer1.cc
 *
 * N. Smith <nick.smith@cern.ch>
 */
//Modified by Bhawna Gomber <bhawna.gomber@cern.ch>

#include "DQM/L1TMonitor/interface/L1TLayer1.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

// Binning specification
namespace {
  constexpr unsigned int TPGETABINS = 57;
  constexpr float TPGETAMIN = -28.5;
  constexpr float TPGETAMAX = 28.5;

  constexpr unsigned int TPETABINSHCAL = 83;
  constexpr float TPETAMINHCAL = -41.5;
  constexpr float TPETAMAXHCAL = 41.5;

  constexpr unsigned int TPGPHIBINS = 72;
  constexpr float TPGPHIMIN = 0.5;
  constexpr float TPGPHIMAX = 72.5;

  constexpr unsigned int TPGEtbins = 255;
  constexpr float TPGEtMIN = 0.0;
  constexpr float TPGEtMAX = 255.0;

  // Will be used for et difference at some point
  // constexpr unsigned int TPGEtbins1 = 510;
  // constexpr float TPGEtMIN1 = -255.0;
  // constexpr float TPGEtMAX1 = 255.0;
};

L1TLayer1::L1TLayer1(const edm::ParameterSet & ps) :
  ecalTPSourceRecd_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceRecd"))),
  ecalTPSourceRecdLabel_(ps.getParameter<edm::InputTag>("ecalTPSourceRecd").label()),
  hcalTPSourceRecd_(consumes<HcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("hcalTPSourceRecd"))),
  hcalTPSourceRecdLabel_(ps.getParameter<edm::InputTag>("hcalTPSourceRecd").label()),
  ecalTPSourceSent_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceSent"))),
  ecalTPSourceSentLabel_(ps.getParameter<edm::InputTag>("ecalTPSourceSent").label()),
  hcalTPSourceSent_(consumes<HcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("hcalTPSourceSent"))),
  hcalTPSourceSentLabel_(ps.getParameter<edm::InputTag>("hcalTPSourceSent").label()),
  histFolder_(ps.getParameter<std::string>("histFolder")),
  tpFillThreshold_(ps.getUntrackedParameter<int>("etDistributionsFillThreshold", 0))
{
}

L1TLayer1::~L1TLayer1()
{
}

void L1TLayer1::dqmBeginRun(const edm::Run&, const edm::EventSetup&)
{
}

void L1TLayer1::analyze(const edm::Event & event, const edm::EventSetup & es)
{
  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsSent;
  event.getByToken(ecalTPSourceSent_, ecalTPsSent);
  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsRecd;
  event.getByToken(ecalTPSourceRecd_, ecalTPsRecd);

  SimpleTowerSet ecalSentSet;
  for ( const auto& tp : *ecalTPsSent ) {
    // Zero-suppress (if not already done before collection was saved)
    if ( tp.compressedEt() > 0 ) {
      ecalSentSet.emplace(tp.id().ieta(), tp.id().iphi(), tp.compressedEt());
    }
  }
  SimpleTowerSet ecalRecdSet;
  SimpleTowerSet ecalMasked;
  // Store link flags from received TPs (bits 13-15)
  // Shift so we access using bits 1-3
  // See EventFilter/L1TXRawToDigi/plugins/L1TCaloLayer1RawToDigi.cc L218
  for ( const auto& tp : *ecalTPsRecd ) {
    // Zero-suppress (if not already done before collection was saved)
    if ( tp.compressedEt() > 0 ) {
      ecalRecdSet.emplace(tp.id().ieta(), tp.id().iphi(), tp.compressedEt(), tp.sample(0).raw()>>13);
    }
    if ( ((tp.sample(0).raw()>>13) & 0b11) > 0 ) {
      // We will use this later to check if a mismatch is from masked link/tower
      // Sets are nice for this since find is O(log(n))
      ecalMasked.emplace(tp.id().ieta(), tp.id().iphi(), 0);
    }
  }

  SimpleTowerSet ecalSentNotRecdSet;
  std::set_difference(ecalSentSet.begin(), ecalSentSet.end(), ecalRecdSet.begin(), ecalRecdSet.end(), std::inserter(ecalSentNotRecdSet, ecalSentNotRecdSet.begin()));
  for ( const auto& tp : ecalSentNotRecdSet ) {
    ecalOccSentNotRecd_->Fill(tp.ieta_, tp.iphi_);
    ecalTPRawEtSentNotRecd_->Fill(tp.data_);

    SimpleTower tpNoEt(tp.ieta_, tp.iphi_, 0);
    if ( ecalMasked.find(tpNoEt) == ecalMasked.end() ) {
      ecalOccMaskedSentNotRecd_->Fill(tp.ieta_, tp.iphi_);
      updateMismatch(event, 0);
    }
  }

  SimpleTowerSet ecalRecdNotSentSet;
  std::set_difference(ecalRecdSet.begin(), ecalRecdSet.end(), ecalSentSet.begin(), ecalSentSet.end(), std::inserter(ecalRecdNotSentSet, ecalRecdNotSentSet.begin()));
  for ( const auto& tp : ecalRecdNotSentSet ) {
    ecalOccRecdNotSent_->Fill(tp.ieta_, tp.iphi_);
    ecalTPRawEtRecdNotSent_->Fill(tp.data_);

    SimpleTower tpNoEt(tp.ieta_, tp.iphi_, 0);
    if ( ecalMasked.find(tpNoEt) == ecalMasked.end() ) {
      ecalOccMaskedRecdNotSent_->Fill(tp.ieta_, tp.iphi_);
      updateMismatch(event, 1);
    }
  }

  SimpleTowerSet ecalSentAndRecdSet;
  std::set_intersection(ecalRecdSet.begin(), ecalRecdSet.end(), ecalSentSet.begin(), ecalSentSet.end(), std::inserter(ecalSentAndRecdSet, ecalSentAndRecdSet.begin()));
  for ( const auto& tp : ecalSentAndRecdSet ) {
    ecalOccSentAndRecd_->Fill(tp.ieta_, tp.iphi_);
    ecalTPRawEtSentAndRecd_->Fill(tp.data_);
  }

  for ( const auto& ecalTp : *ecalTPsRecd ) {
    if ( ecalTp.compressedEt() > tpFillThreshold_ ) {
      ecalTPRawEtRecd_->Fill(ecalTp.compressedEt());
      ecalOccRecd_->Fill(ecalTp.id().ieta(), ecalTp.id().iphi());
    }
    
    if(ecalTp.fineGrain()==1){
      ecalOccRecd_isFineGrainVB_->Fill(ecalTp.id().ieta(), ecalTp.id().iphi());
    }

    if(((ecalTp.sample(0).raw()>>13)&1)==1){
      ecalOccRecd_isECALTowerMasked_->Fill(ecalTp.id().ieta(), ecalTp.id().iphi());
    }
    
    if(((ecalTp.sample(0).raw()>>14)&1)==1){
      ecalOccRecd_isECALLinkMasked_->Fill(ecalTp.id().ieta(), ecalTp.id().iphi());
    }

    if(((ecalTp.sample(0).raw()>>15)&1)==1){
      ecalOccRecd_isECALLinkInError_->Fill(ecalTp.id().ieta(), ecalTp.id().iphi());
    }
  }

  for ( const auto& ecalTp : *ecalTPsSent ) {
    if ( ecalTp.compressedEt() > tpFillThreshold_ ) {
      ecalTPRawEtSent_->Fill(ecalTp.compressedEt());
      ecalOccSent_->Fill(ecalTp.id().ieta(), ecalTp.id().iphi());
    }
    if(ecalTp.fineGrain()==1){
      ecalOccSent_isFineGrainVB_->Fill(ecalTp.id().ieta(), ecalTp.id().iphi());
    }
  }



  edm::Handle<HcalTrigPrimDigiCollection> hcalTPsSent;
  event.getByToken(hcalTPSourceSent_, hcalTPsSent);
  edm::Handle<HcalTrigPrimDigiCollection> hcalTPsRecd;
  event.getByToken(hcalTPSourceRecd_, hcalTPsRecd);

  SimpleTowerSet hcalSentSet;
  for ( const auto& tp : *hcalTPsSent ) {
    // Zero-suppress (if not already done before collection was saved)
    if ( tp.SOI_compressedEt() > 0 ) {
      hcalSentSet.emplace(tp.id().ieta(), tp.id().iphi(), tp.SOI_compressedEt());
    }
  }
  SimpleTowerSet hcalRecdSet;
  SimpleTowerSet hcalMasked;
  // Store link flags from received TPs (bits 13-15, also bits 9-11 are misaligned, inerror, down, resp.)
  // See EventFilter/L1TXRawToDigi/plugins/L1TCaloLayer1RawToDigi.cc L261
  for ( const auto& tp : *hcalTPsRecd ) {
    // Zero-suppress (if not already done before collection was saved)
    if ( tp.SOI_compressedEt() > 0 ) {
      hcalRecdSet.emplace(tp.id().ieta(), tp.id().iphi(), tp.SOI_compressedEt(), tp.sample(0).raw()>>13);
    }
  }

  SimpleTowerSet hcalSentNotRecdSet;
  std::set_difference(hcalSentSet.begin(), hcalSentSet.end(), hcalRecdSet.begin(), hcalRecdSet.end(), std::inserter(hcalSentNotRecdSet, hcalSentNotRecdSet.begin()));
  for ( const auto& tp : hcalSentNotRecdSet ) {
    hcalOccSentNotRecd_->Fill(tp.ieta_, tp.iphi_);
    hcalTPRawEtSentNotRecd_->Fill(tp.data_);

    SimpleTower tpNoEt(tp.ieta_, tp.iphi_, 0);
    if ( hcalMasked.find(tpNoEt) == hcalMasked.end() ) {
      hcalOccMaskedSentNotRecd_->Fill(tp.ieta_, tp.iphi_);
      updateMismatch(event, 2);
    }
  }

  SimpleTowerSet hcalRecdNotSentSet;
  std::set_difference(hcalRecdSet.begin(), hcalRecdSet.end(), hcalSentSet.begin(), hcalSentSet.end(), std::inserter(hcalRecdNotSentSet, hcalRecdNotSentSet.begin()));
  for ( const auto& tp : hcalRecdNotSentSet ) {
    hcalOccRecdNotSent_->Fill(tp.ieta_, tp.iphi_);
    hcalTPRawEtRecdNotSent_->Fill(tp.data_);

    SimpleTower tpNoEt(tp.ieta_, tp.iphi_, 0);
    if ( hcalMasked.find(tpNoEt) == hcalMasked.end() ) {
      hcalOccMaskedRecdNotSent_->Fill(tp.ieta_, tp.iphi_);
      updateMismatch(event, 3);
    }
  }

  SimpleTowerSet hcalSentAndRecdSet;
  std::set_intersection(hcalRecdSet.begin(), hcalRecdSet.end(), hcalSentSet.begin(), hcalSentSet.end(), std::inserter(hcalSentAndRecdSet, hcalSentAndRecdSet.begin()));
  for ( const auto& tp : hcalSentAndRecdSet ) {
    hcalOccSentAndRecd_->Fill(tp.ieta_, tp.iphi_);
    hcalTPRawEtSentAndRecd_->Fill(tp.data_);
  }


  for ( const auto& hcalTp : *hcalTPsRecd ) {
    if ( hcalTp.SOI_compressedEt() > tpFillThreshold_ ) {
      hcalTPRawEtRecd_->Fill(hcalTp.SOI_compressedEt());
      hcalOccRecd_->Fill(hcalTp.id().ieta(), hcalTp.id().iphi());
    }
    if(hcalTp.SOI_fineGrain()){
      hcalOccRecd_hasFeatureBits_->Fill(hcalTp.id().ieta(), hcalTp.id().iphi());
    }
  
    if(((hcalTp.sample(0).raw()>>13)&1)==1){
      hcalOccRecd_isHCALTowerMasked_->Fill(hcalTp.id().ieta(), hcalTp.id().iphi());
    }
  
    if(((hcalTp.sample(0).raw()>>14)&1)==1){
      hcalOccRecd_isHCALLinkMasked_->Fill(hcalTp.id().ieta(), hcalTp.id().iphi());
    }
  
    if(((hcalTp.sample(0).raw()>>15)&1)==1){
      hcalOccRecd_isHCALLinkInError_->Fill(hcalTp.id().ieta(), hcalTp.id().iphi());
    }
  }

  for ( const auto& hcalTp : *hcalTPsSent ) {
    if ( hcalTp.SOI_compressedEt() > tpFillThreshold_ ) {
      hcalTPRawEtSent_->Fill(hcalTp.SOI_compressedEt());
      hcalOccSent_->Fill(hcalTp.id().ieta(), hcalTp.id().iphi());
    }
    if(hcalTp.SOI_fineGrain()==1){
      hcalOccSent_hasFeatureBits_->Fill(hcalTp.id().ieta(), hcalTp.id().iphi());
    }
  }
}


void L1TLayer1::updateMismatch(const edm::Event& e, int mismatchType) {
  auto id = e.id();
  std::string eventString{std::to_string(id.run()) + ":" + std::to_string(id.luminosityBlock()) + ":" + std::to_string(id.event())};
  last20MismatchArray_.at(lastMismatchIndex_) = {eventString, mismatchType};

  // Ugly way to loop backwards through the last 20 mismatches
  for (size_t ibin=1, imatch=lastMismatchIndex_; ibin<=20; ibin++, imatch=(imatch+19)%20) {
    last20Mismatches_->getTH2F()->GetYaxis()->SetBinLabel(ibin, last20MismatchArray_.at(imatch).first.c_str());
    for(int itype=0; itype<4; ++itype) {
      int binContent = itype==last20MismatchArray_.at(imatch).second;
      last20Mismatches_->setBinContent(itype+1, ibin, binContent);
    }
  }

  lastMismatchIndex_ = (lastMismatchIndex_+1) % 20;
}


void L1TLayer1::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& run , const edm::EventSetup& es) 
{
  auto sourceString = [](std::string label){return " (source: "+label+")";};

  ibooker.setCurrentFolder(histFolder_+"/ECalEtDistributions");

  ecalTPRawEtSent_ = ibooker.book1D("ecalTPRawEtSent",
                                           "ECal Raw Et sent"+sourceString(ecalTPSourceSentLabel_),
                                           TPGEtbins, TPGEtMIN, TPGEtMAX);

  ecalTPRawEtRecd_ = ibooker.book1D("ecalTPRawEtRecd",
                                           "ECal Raw Et received"+sourceString(ecalTPSourceRecdLabel_),
                                           TPGEtbins, TPGEtMIN, TPGEtMAX);

  ecalTPRawEtRecdNotSent_ = ibooker.book1D("ecalTPRawEtRecdNotSent",
                                                  "ECal Raw Et Received NOT Sent",
                                                  TPGEtbins, TPGEtMIN, TPGEtMAX);

  ecalTPRawEtSentNotRecd_ = ibooker.book1D("ecalTPRawEtSentNotRecd",
                                                  "ECal Raw Et Sent NOT Received",
                                                  TPGEtbins, TPGEtMIN, TPGEtMAX);

  ecalTPRawEtSentAndRecd_ = ibooker.book1D("ecalTPRawEtSentAndRecd",
                                                "ECal Raw Et Sent AND Recd",
                                                TPGEtbins, TPGEtMIN, TPGEtMAX);


  ibooker.setCurrentFolder(histFolder_+"/ECalOccupancies");

  ecalOccRecd_isECALTowerMasked_ = ibooker.book2D("ecalOccRecd_isECALTowerMasked", 
                                                            "ECal TP Occupancy received for the ECAL Masked towers"+sourceString(ecalTPSourceRecdLabel_),
                                                            TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccRecd_isFineGrainVB_ = ibooker.book2D("ecalOccRecd_isFineGrainVB", 
                                                        "ECal TP Occupancy received for the fine grain veto"+sourceString(ecalTPSourceRecdLabel_),
                                                        TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccSent_isFineGrainVB_ = ibooker.book2D("ecalOccSent_isFineGrainVB", 
                                                        "ECal TP Occupancy sent for the fine grain veto"+sourceString(ecalTPSourceSentLabel_),
                                                        TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccRecd_isECALLinkMasked_ = ibooker.book2D("ecalOccRecd_isECALLinkMasked", 
                                                           "ECal TP Occupancy received for the ECAL Masked Links"+sourceString(ecalTPSourceRecdLabel_),
                                                           TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccRecd_isECALLinkInError_ = ibooker.book2D("ecalOccRecd_isECALLinkInError", 
                                                            "ECal TP Occupancy received for the ECAL Misaligned, Inerror and Down Links"+sourceString(ecalTPSourceRecdLabel_),
                                                            TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccRecd_ = ibooker.book2D("ecalOccRecd", 
                                          "ECal TP Occupancy received"+sourceString(ecalTPSourceRecdLabel_),
                                          TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccSent_ = ibooker.book2D("ecalOccSent", 
                                          "ECal TP Occupancy sent"+sourceString(ecalTPSourceSentLabel_),
                                          TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccRecdNotSent_ = ibooker.book2D("ecalOccRecdNotSent", 
                                             "ECal TP Occupancy Received NOT Sent",
                                             TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccSentNotRecd_ = ibooker.book2D("ecalOccSentNotRecd", 
                                             "ECal TP Occupancy Sent NOT Received",
                                             TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccSentAndRecd_ = ibooker.book2D("ecalOccSentAndRecd", 
                                             "ECal TP Occupancy Sent AND Received",
                                           TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);


  ibooker.setCurrentFolder(histFolder_+"/ECalOccupancies/Masked");

  ecalOccMaskedRecdNotSent_ = ibooker.book2D("ecalOccMaskedRecdNotSent", 
                                             "ECal Masked TP Occupancy Received NOT Sent",
                                             TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccMaskedSentNotRecd_ = ibooker.book2D("ecalOccMaskedSentNotRecd", 
                                             "ECal Masked TP Occupancy Sent NOT Received",
                                             TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  ecalOccMaskedSentAndRecd_ = ibooker.book2D("ecalOccMaskedSentAndRecd", 
                                             "ECal Masked TP Occupancy Sent AND Received",
                                           TPGETABINS, TPGETAMIN, TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);


  ibooker.setCurrentFolder(histFolder_+"/HCalEtDistributions");

  hcalTPRawEtSent_ = ibooker.book1D("hcalTPRawEtSent",
                                           "Hcal Raw Et sent"+sourceString(hcalTPSourceSentLabel_),
                                           TPGEtbins, TPGEtMIN, TPGEtMAX);

  hcalTPRawEtRecd_ = ibooker.book1D("hcalTPRawEtRecd",
                                           "Hcal Raw Et received"+sourceString(hcalTPSourceRecdLabel_),
                                           TPGEtbins, TPGEtMIN, TPGEtMAX);

  hcalTPRawEtRecdNotSent_ = ibooker.book1D("hcalTPRawEtRecdNotSent",
                                                  "Hcal Raw Et Received NOT Sent",
                                                  TPGEtbins, TPGEtMIN, TPGEtMAX);

  hcalTPRawEtSentNotRecd_ = ibooker.book1D("hcalTPRawEtSentNotRecd",
                                                  "Hcal Raw Et Sent NOT Received",
                                                  TPGEtbins, TPGEtMIN, TPGEtMAX);

  hcalTPRawEtSentAndRecd_ = ibooker.book1D("hcalTPRawEtSentAndRecd",
                                                "Hcal Raw Et Sent AND Received",
                                                TPGEtbins, TPGEtMIN, TPGEtMAX);


  ibooker.setCurrentFolder(histFolder_+"/HCalOccupancies");

  hcalOccRecd_isHCALTowerMasked_ = ibooker.book2D("hcalOccRecd_isHCALTowerMasked", 
                                                            "Hcal TP Occupancy received for the HCAL Masked towers"+sourceString(hcalTPSourceRecdLabel_),
                                                            TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  hcalOccRecd_hasFeatureBits_ = ibooker.book2D("hcalOccRecd_hasFeatureBits", 
                                                        "Hcal TP Occupancy received for the feature bits"+sourceString(hcalTPSourceRecdLabel_),
                                                        TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  hcalOccSent_hasFeatureBits_ = ibooker.book2D("hcalOccSent_hasFeatureBits", 
                                                        "Hcal TP Occupancy sent for the feature bits"+sourceString(hcalTPSourceSentLabel_),
                                                        TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);


  hcalOccRecd_isHCALLinkMasked_ = ibooker.book2D("hcalOccRecd_isHCALLinkMasked", 
                                                           "Hcal TP Occupancy received for the HCAL Masked Links"+sourceString(hcalTPSourceRecdLabel_),
                                                           TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  hcalOccRecd_isHCALLinkInError_ = ibooker.book2D("hcalOccRecd_isHCALLinkInError", 
                                                            "Hcal TP Occupancy received for the HCAL Misaligned, Inerror and Down Links"+sourceString(hcalTPSourceRecdLabel_),
                                                            TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  hcalOccRecd_ = ibooker.book2D("hcalOccRecd", 
                                          "Hcal TP Occupancy received"+sourceString(hcalTPSourceRecdLabel_),
                                          TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  hcalOccSent_ = ibooker.book2D("hcalOccSent", 
                                          "Hcal TP Occupancy sent"+sourceString(hcalTPSourceSentLabel_),
                                          TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  hcalOccRecdNotSent_ = ibooker.book2D("hcalOccRecdNotSent", 
                                             "HCal TP Occupancy Received NOT Sent",
                                             TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  hcalOccSentNotRecd_ = ibooker.book2D("hcalOccSentNotRecd", 
                                             "HCal TP Occupancy Sent NOT Received",
                                             TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  hcalOccSentAndRecd_ = ibooker.book2D("hcalOccSentAndRecd", 
                                             "HCal TP Occupancy Sent AND Received",
                                           TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);


  ibooker.setCurrentFolder(histFolder_+"/HCalOccupancies/Masked");

  hcalOccMaskedRecdNotSent_ = ibooker.book2D("hcalOccMaskedRecdNotSent", 
                                             "HCal Masked TP Occupancy Received NOT Sent",
                                             TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  hcalOccMaskedSentNotRecd_ = ibooker.book2D("hcalOccMaskedSentNotRecd", 
                                             "HCal Masked TP Occupancy Sent NOT Received",
                                             TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

  hcalOccMaskedSentAndRecd_ = ibooker.book2D("hcalOccMaskedSentAndRecd", 
                                             "HCal Masked TP Occupancy Sent AND Received",
                                           TPETABINSHCAL, TPETAMINHCAL, TPETAMAXHCAL, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);


  ibooker.setCurrentFolder(histFolder_+"/Mismatch");

  last20Mismatches_ = ibooker.book2D("last20Mismatches", 
                                             "Log of last 20 mismatches (use json tool to copy/paste)",
                                             4, 0, 4, 20, 0, 20);
  last20Mismatches_->getTH2F()->GetXaxis()->SetBinLabel(1, "Ecal TP Sent Not Received");
  last20Mismatches_->getTH2F()->GetXaxis()->SetBinLabel(2, "Ecal TP Received Not Sent");
  last20Mismatches_->getTH2F()->GetXaxis()->SetBinLabel(3, "Hcal TP Sent Not Received");
  last20Mismatches_->getTH2F()->GetXaxis()->SetBinLabel(4, "Hcal TP Received Not Sent");
  for (size_t i=0; i<20; ++i) last20MismatchArray_.at(i) = {"-", -1};
  for (size_t i=1; i<=20; ++i) last20Mismatches_->getTH2F()->GetYaxis()->SetBinLabel(i, "-");
}

void L1TLayer1::beginLuminosityBlock(const edm::LuminosityBlock& ls,const edm::EventSetup& es)
{
}


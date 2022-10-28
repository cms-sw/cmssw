// -*- C++ -*-
//
// Package:    HcalLaserHBHEHFFilter2012
// Class:      HcalLaserHBHEHFFilter2012
//
/**\class HcalLaserHBHEHFFilter2012 HcalLaserHBHEHFFilter2012.cc UserCode/HcalLaserHBHEFilter2012/src/HcalLaserHBHEHFFilter2012.cc

 Description: [filters out HBHE laser events based on number of HBHE calib digis above threshold and the distribution of HBHE calib digis]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:
//         Created:  Fri Oct 19 13:15:44 EDT 2012
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"

//
// class declaration
//

class HcalLaserHBHEHFFilter2012 : public edm::one::EDFilter<> {
public:
  explicit HcalLaserHBHEHFFilter2012(const edm::ParameterSet&);
  ~HcalLaserHBHEHFFilter2012() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  // Flag to activate laser filter for HBHE
  bool filterHBHE_;
  // set minimum number of HBHE Calib events that causes an event to be considered a bad (i.e., HBHE laser) event
  int minCalibChannelsHBHELaser_;
  // minimum difference in fractional occupancies between 'good' and 'bad' HBHE regions (i.e., regions whose RBXes
  // receive laser signals and those whose RBXes see no laser) necessary to declare an event as a laser event.
  // In laser events, good fractional occupancy is generally near 1, while bad fractional occupancy is
  // considerably less
  double minFracDiffHBHELaser_;

  // minimum integrated charge needed for a hit to count as an occupied calib channel
  double HBHEcalibThreshold_;
  // time slices used when integrating calib charges
  std::vector<int> CalibTS_;

  // Flag to activate laser filter for HF
  bool filterHF_;
  // set minimum number of HF Calib events that causes an event to be considered a bad (i.e., HF laser) event
  int minCalibChannelsHFLaser_;

  edm::InputTag digiLabel_;
  edm::EDGetTokenT<HcalCalibDigiCollection> tok_calib_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;

  // if set to true, then the run:LS:event for any event failing the cut will be printed out
  bool verbose_;
  // prefix will be printed before any event if verbose mode is true, in order to make searching for events easier
  std::string prefix_;
  bool WriteBadToFile_;
  bool forceFilterTrue_;
  std::ofstream outfile_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalLaserHBHEHFFilter2012::HcalLaserHBHEHFFilter2012(const edm::ParameterSet& ps) {
  //now do what ever initialization is needed
  filterHBHE_ = ps.getParameter<bool>("filterHBHE");
  minCalibChannelsHBHELaser_ = ps.getParameter<int>("minCalibChannelsHBHELaser");
  minFracDiffHBHELaser_ = ps.getParameter<double>("minFracDiffHBHELaser");
  HBHEcalibThreshold_ = ps.getParameter<double>("HBHEcalibThreshold");
  CalibTS_ = ps.getParameter<std::vector<int> >("CalibTS");

  filterHF_ = ps.getParameter<bool>("filterHF");
  minCalibChannelsHFLaser_ = ps.getParameter<int>("minCalibChannelsHFLaser");

  digiLabel_ = ps.getParameter<edm::InputTag>("digiLabel");
  tok_calib_ = consumes<HcalCalibDigiCollection>(digiLabel_);
  tok_hbhe_ = consumes<HBHEDigiCollection>(digiLabel_);

  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
  prefix_ = ps.getUntrackedParameter<std::string>("prefix", "");
  WriteBadToFile_ = ps.getUntrackedParameter<bool>("WriteBadToFile", false);
  forceFilterTrue_ = ps.getUntrackedParameter<bool>("forceFilterTrue", false);
  if (WriteBadToFile_)
    outfile_.open("badHcalLaserList_hcalfilter.txt");
}  // HcalLaserHBHEHFFilter2012::HcalLaserHBHEHFFilter2012  constructor

HcalLaserHBHEHFFilter2012::~HcalLaserHBHEHFFilter2012() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HcalLaserHBHEHFFilter2012::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Step 1:: try to get calib digi and HBHE collections.
  // Return true if collection not found?  Or false?  What should default behavior be?
  edm::Handle<HcalCalibDigiCollection> calib_digi;
  if (!(iEvent.getByToken(tok_calib_, calib_digi))) {
    edm::LogWarning("HcalLaserFilter2012") << digiLabel_ << " calib_digi not available";
    return true;
  }

  if (!(calib_digi.isValid())) {
    edm::LogWarning("HcalLaserFilter2012") << digiLabel_ << " calib_digi is not valid";
    return true;
  }

  edm::Handle<HBHEDigiCollection> hbhe_digi;
  if (!(iEvent.getByToken(tok_hbhe_, hbhe_digi))) {
    edm::LogWarning("HcalLaserFilter2012") << digiLabel_ << " hbhe_digi not available";
    return true;
  }

  if (!(hbhe_digi.isValid())) {
    edm::LogWarning("HcalLaserHBHEHFFilter2012") << digiLabel_ << " hbhe_digi is not valid";
    return true;
  }

  // Step 2:  Count HBHE digi calib channels
  int ncalibHBHE = 0;  // this will track number of HBHE digi channels
  int ncalibHF = 0;    // this will track number of HF digi channels

  for (HcalCalibDigiCollection::const_iterator Calibiter = calib_digi->begin(); Calibiter != calib_digi->end();
       ++Calibiter) {
    const HcalCalibDataFrame digi = (const HcalCalibDataFrame)(*Calibiter);
    if (digi.zsMarkAndPass())
      continue;  // skip digis labeled as "mark and pass" in NZS events
    HcalCalibDetId myid = (HcalCalibDetId)digi.id();
    if (filterHBHE_ && (myid.hcalSubdet() == HcalBarrel || myid.hcalSubdet() == HcalEndcap)) {
      if (myid.calibFlavor() == HcalCalibDetId::HOCrosstalk)
        continue;
      // Compute charge in current channel (for relevant TS only)
      // If total charge in channel exceeds threshold, increment count of calib channels
      double thischarge = 0;
      for (unsigned int i = 0; i < CalibTS_.size(); ++i) {
        thischarge += digi[CalibTS_[i]].nominal_fC();
        if (thischarge > HBHEcalibThreshold_) {
          ++ncalibHBHE;
          break;
        }
      }

      if (ncalibHBHE >= minCalibChannelsHBHELaser_) {
        if (verbose_)
          std::cout << prefix_ << iEvent.id().run() << ":" << iEvent.luminosityBlock() << ":" << iEvent.id().event()
                    << std::endl;
        if (WriteBadToFile_)
          outfile_ << iEvent.id().run() << ":" << iEvent.luminosityBlock() << ":" << iEvent.id().event() << std::endl;
        if (forceFilterTrue_)
          return true;  // if special input boolean set, always return true, regardless of filter decision
        else
          return false;
      }  // if  (ncalibHBHE>=minCalibChannelsHBHELaser_)

    } else if (filterHF_ && (myid.hcalSubdet() == HcalForward)) {
      ++ncalibHF;
      if (ncalibHF >= minCalibChannelsHFLaser_) {
        if (verbose_)
          std::cout << prefix_ << iEvent.id().run() << ":" << iEvent.luminosityBlock() << ":" << iEvent.id().event()
                    << std::endl;
        if (WriteBadToFile_)
          outfile_ << iEvent.id().run() << ":" << iEvent.luminosityBlock() << ":" << iEvent.id().event() << std::endl;
        if (forceFilterTrue_)
          return true;  // if special input boolean set, always return true, regardless of filter decision
        else
          return false;
      }
    }
  }

  if (filterHBHE_) {
    // Step 3:  Look at distribution of HBHE hits
    // Count digis in good, bad RBXes.  ('bad' RBXes see no laser signal)
    double badrbxfrac = 0.;
    double goodrbxfrac = 0.;
    int Nbad = 72 * 3;            // 3 bad RBXes, 72 channels each
    int Ngood = 2592 * 2 - Nbad;  // remaining HBHE channels are 'good'
    for (HBHEDigiCollection::const_iterator hbhe = hbhe_digi->begin(); hbhe != hbhe_digi->end(); ++hbhe) {
      const HBHEDataFrame digi = (const HBHEDataFrame)(*hbhe);
      HcalDetId myid = (HcalDetId)digi.id();
      bool isbad = false;  // assume channel is not bad
      if (myid.subdet() == HcalBarrel && myid.ieta() < 0) {
        if (myid.iphi() >= 15 && myid.iphi() <= 18)
          isbad = true;
        else if (myid.iphi() >= 27 && myid.iphi() <= 34)
          isbad = true;
      }
      if (isbad == true)
        badrbxfrac += 1.;
      else
        goodrbxfrac += 1.;
    }
    goodrbxfrac /= Ngood;
    badrbxfrac /= Nbad;
    if (goodrbxfrac - badrbxfrac > minFracDiffHBHELaser_) {
      if (verbose_)
        std::cout << prefix_ << iEvent.id().run() << ":" << iEvent.luminosityBlock() << ":" << iEvent.id().event()
                  << std::endl;
      if (WriteBadToFile_)
        outfile_ << iEvent.id().run() << ":" << iEvent.luminosityBlock() << ":" << iEvent.id().event() << std::endl;

      if (forceFilterTrue_)
        return true;  // if special input boolean set, always return true, regardless of filter decision
      else
        return false;
    }
  }
  // Step 4:  HBHEHF laser tests passed.  return true
  return true;

}  // HcalLaserHBHEHFFilter2012::filter

// ------------ method called once each job just after ending the event loop  ------------
void HcalLaserHBHEHFFilter2012::endJob() {
  if (WriteBadToFile_)
    outfile_.close();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalLaserHBHEHFFilter2012::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(HcalLaserHBHEHFFilter2012);

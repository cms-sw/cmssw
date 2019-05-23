// -*- C++ -*-
//
// Package:    HLTHcalLaserMisfireFilter
// Class:      HLTHcalLaserMisfireFilter
//
/**\class HLTHcalLaserMisfireFilter HLTHcalLaserMisfireFilter.cc

Description: Filter to select misfires of the HCAL laser.

Implementation:
Three HBHE RBX's ("bad") do not see the laser signal. Laser events are 
selected by finding events with a large fraction of digis in HBHE above
a given threshold that do not have signals in the "bad" RBXes.

For HF events are selected if a large fraction of digis in HF are above 
a given threshold
*/

// system include files
#include <memory>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "HLTHcalLaserMisfireFilter.h"

//
// constructors and destructor
//
HLTHcalLaserMisfireFilter::HLTHcalLaserMisfireFilter(const edm::ParameterSet& config) {
  inputHBHE_ = config.getParameter<edm::InputTag>("InputHBHE");
  inputHF_ = config.getParameter<edm::InputTag>("InputHF");
  minFracDiffHBHELaser_ = config.getParameter<double>("minFracDiffHBHELaser");
  minFracHFLaser_ = config.getParameter<double>("minFracHFLaser");
  minADCHBHE_ = config.getParameter<int>("minADCHBHE");
  minADCHF_ = config.getParameter<int>("minADCHF"), testMode_ = config.getUntrackedParameter<bool>("testMode", false);

  inputTokenHBHE_ = consumes<HBHEDigiCollection>(inputHBHE_);
  inputTokenHF_ = consumes<QIE10DigiCollection>(inputHF_);
}

HLTHcalLaserMisfireFilter::~HLTHcalLaserMisfireFilter() {}

void HLTHcalLaserMisfireFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputHBHE", edm::InputTag("source"));
  desc.add<edm::InputTag>("InputHF", edm::InputTag("source"));
  desc.add<int>("minADCHBHE", 10);
  desc.add<int>("minADCHF", 10);
  desc.add<double>("minFracDiffHBHELaser", 0.3);
  desc.add<double>("minFracHFLaser", 0.3);
  desc.addUntracked<bool>("testMode", false);
  descriptions.add("hltHcalLaserMisfireFilter", desc);
}

bool HLTHcalLaserMisfireFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<HBHEDigiCollection> hbhe_digi;
  iEvent.getByToken(inputTokenHBHE_, hbhe_digi);

  edm::Handle<QIE10DigiCollection> hf_digi;
  iEvent.getByToken(inputTokenHF_, hf_digi);

  // Count digis in good, bad RBXes.  ('bad' RBXes see no laser signal)
  double badrbxfracHBHE(0), goodrbxfracHBHE(0), allrbxfracHF(0);
  int NbadHBHE = 72 * 3;                // 3 bad RBXes, 72 channels each
  int NgoodHBHE = 2592 * 2 - NbadHBHE;  // remaining HBHE channels are 'good'
  int NallHF = 864 * 4;

  for (auto hbhe = hbhe_digi->begin(); hbhe != hbhe_digi->end(); ++hbhe) {
    const HBHEDataFrame digi = (const HBHEDataFrame)(*hbhe);
    HcalDetId myid = (HcalDetId)digi.id();
    bool isbad(false);  // assume channel is not bad

    bool passCut(false);
    int maxdigiHB(0);
    for (int i = 0; i < digi.size(); i++)
      if (digi.sample(i).adc() > maxdigiHB)
        maxdigiHB = digi.sample(i).adc();
    if (maxdigiHB > minADCHBHE_)
      passCut = true;

    // Three RBX's in HB do not receive any laser light (HBM5, HBM8, HBM9)
    // They correspond to iphi = 15:18, 27:30, 31:34 respectively and
    // ieta < 0
    if (myid.subdet() == HcalBarrel && myid.ieta() < 0) {
      if (myid.iphi() >= 15 && myid.iphi() <= 18)
        isbad = true;
      else if (myid.iphi() >= 27 && myid.iphi() <= 34)
        isbad = true;
    }

    if (passCut) {
      if (isbad) {
        badrbxfracHBHE += 1.;
      } else
        goodrbxfracHBHE += 1.;
    }
  }
  goodrbxfracHBHE /= NgoodHBHE;
  badrbxfracHBHE /= NbadHBHE;

  for (auto hf = hf_digi->begin(); hf != hf_digi->end(); ++hf) {
    const QIE10DataFrame digi = (const QIE10DataFrame)(*hf);
    bool passCut(false);
    int maxdigiHF(0);
    for (int i = 0; i < digi.samples(); i++)
      if (digi[i].adc() > maxdigiHF)
        maxdigiHF = digi[i].adc();
    if (maxdigiHF > minADCHF_)
      passCut = true;

    if (passCut) {
      allrbxfracHF += 1.;
    }
  }
  allrbxfracHF /= NallHF;

  if (testMode_)
    edm::LogVerbatim("Report") << "******************************************************************\n"
                               << "goodrbxfracHBHE: " << goodrbxfracHBHE << " badrbxfracHBHE: " << badrbxfracHBHE
                               << " Size " << hbhe_digi->size() << "\n"
                               << "allrbxfracHF:    " << allrbxfracHF << " Size " << hf_digi->size()
                               << "\n******************************************************************";

  if (((goodrbxfracHBHE - badrbxfracHBHE) < minFracDiffHBHELaser_) || (allrbxfracHF < minFracHFLaser_))
    return false;

  return true;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTHcalLaserMisfireFilter);

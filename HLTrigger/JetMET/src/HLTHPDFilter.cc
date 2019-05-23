/* \class HLTHPDFilter
 *
 *
 * Fedor Ratnikov (UMd) May 19, 2008
 */

#include "HLTrigger/JetMET/interface/HLTHPDFilter.h"

#include <cmath>

#include <set>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"
#include "TH2F.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace {
  enum Partition { HBM = 0, HBP = 1, HEM = 2, HEP = 3 };
  std::pair<Partition, int> hpdId(HcalDetId fId) {
    int hpd = fId.iphi();
    Partition partition = HBM;
    if (fId.subdet() == HcalBarrel) {
      partition = fId.ieta() > 0 ? HBP : HBM;
    } else if (fId.subdet() == HcalEndcap) {
      partition = fId.ieta() > 0 ? HEP : HEM;
      if ((fId.iphi() - 1) % 4 < 2) {  // 1,2
        switch (fId.ieta()) {          // 1->2
          case 22:
          case 24:
          case 26:
          case 28:
            hpd = +1;
            break;
          case 29:
            if (fId.depth() == 1 || fId.depth() == 3)
              hpd += 1;
            break;
          default:
            break;
        }
      } else {                 // 3,4
        switch (fId.ieta()) {  // 3->4
          case 21:
          case 23:
          case 25:
          case 27:
            hpd += 1;
            break;
          case 29:
            if (fId.depth() == 2)
              hpd += 1;
            break;
          default:
            break;
        }
      }
    }
    return std::pair<Partition, int>(partition, hpd);
  }
}  // namespace

HLTHPDFilter::HLTHPDFilter(const edm::ParameterSet& iConfig)
    : mInputTag(iConfig.getParameter<edm::InputTag>("inputTag")),
      mEnergyThreshold(iConfig.getParameter<double>("energy")),
      mHPDSpikeEnergyThreshold(iConfig.getParameter<double>("hpdSpikeEnergy")),
      mHPDSpikeIsolationEnergyThreshold(iConfig.getParameter<double>("hpdSpikeIsolationEnergy")),
      mRBXSpikeEnergyThreshold(iConfig.getParameter<double>("rbxSpikeEnergy")),
      mRBXSpikeUnbalanceThreshold(iConfig.getParameter<double>("rbxSpikeUnbalance")) {
  m_theRecHitCollectionToken = consumes<HBHERecHitCollection>(mInputTag);
}

HLTHPDFilter::~HLTHPDFilter() = default;

void HLTHPDFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag", edm::InputTag("hltHbhereco"));
  desc.add<double>("energy", -99.0);
  desc.add<double>("hpdSpikeEnergy", 10.0);
  desc.add<double>("hpdSpikeIsolationEnergy", 1.0);
  desc.add<double>("rbxSpikeEnergy", 50.0);
  desc.add<double>("rbxSpikeUnbalance", 0.2);
  descriptions.add("hltHPDFilter", desc);
}

bool HLTHPDFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (mHPDSpikeEnergyThreshold <= 0 && mRBXSpikeEnergyThreshold <= 0)
    return true;  // nothing to filter
  // get hits
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(m_theRecHitCollectionToken, hbhe);

  // collect energies
  float hpdEnergy[4][73];
  for (auto& i : hpdEnergy)
    for (size_t j = 0; j < 73; ++j)
      i[j] = 0;

  // select hist above threshold
  for (unsigned i = 0; i < hbhe->size(); ++i) {
    if ((*hbhe)[i].energy() > mEnergyThreshold) {
      std::pair<Partition, int> hpd = hpdId((*hbhe)[i].id());
      hpdEnergy[int(hpd.first)][hpd.second] += (*hbhe)[i].energy();
    }
  }

  // not single HPD spike
  if (mHPDSpikeEnergyThreshold > 0) {
    for (auto& partition : hpdEnergy) {
      for (size_t i = 1; i < 73; ++i) {
        if (partition[i] > mHPDSpikeEnergyThreshold) {
          int hpdPlus = i + 1;
          if (hpdPlus == 73)
            hpdPlus = 1;
          int hpdMinus = i - 1;
          if (hpdMinus == 0)
            hpdMinus = 72;
          double maxNeighborEnergy = fmax(partition[hpdPlus], partition[hpdMinus]);
          if (maxNeighborEnergy < mHPDSpikeIsolationEnergyThreshold)
            return false;  // HPD spike found
        }
      }
    }
  }

  // not RBX flash
  if (mRBXSpikeEnergyThreshold > 0) {
    for (auto& partition : hpdEnergy) {
      for (size_t rbx = 1; rbx < 19; ++rbx) {
        int ifirst = (rbx - 1) * 4 - 1;
        int iend = (rbx - 1) * 4 + 3;
        double minEnergy = 0;
        double maxEnergy = -1;
        double totalEnergy = 0;
        for (int irm = ifirst; irm < iend; ++irm) {
          int hpd = irm;
          if (hpd <= 0)
            hpd = 72 + hpd;
          totalEnergy += partition[hpd];
          if (minEnergy > maxEnergy) {
            minEnergy = maxEnergy = partition[hpd];
          } else {
            if (partition[hpd] < minEnergy)
              minEnergy = partition[hpd];
            if (partition[hpd] > maxEnergy)
              maxEnergy = partition[hpd];
          }
        }
        if (totalEnergy > mRBXSpikeEnergyThreshold) {
          if (minEnergy / maxEnergy > mRBXSpikeUnbalanceThreshold)
            return false;  // likely HPF flash
        }
      }
    }
  }
  return true;
}

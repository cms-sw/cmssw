// -*- C++ -*-
//
// Package:    L1TMuonCaloSumProducer
// Class:      L1TMuonCaloSumProducer
//
/**\class L1TMuonCaloSumProducer L1TMuonCaloSumProducer.cc L1Trigger/L1TGlobalMuon/plugins/L1TMuonCaloSumProducer.cc

 Description: takes generated muons and fills them in the expected collections for the MicroGMT

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Joschka Philip Lingemann,40 3-B01,+41227671598,
//         Created:  Thu Oct  3 10:12:30 CEST 2013
// $Id$
//
//


// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TMuon/interface/MuonCaloSumFwd.h"
#include "DataFormats/L1TMuon/interface/MuonCaloSum.h"
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "TMath.h"
#include "TRandom3.h"

//
// class declaration
//
using namespace l1t;

class L1TMuonCaloSumProducer : public edm::EDProducer {
   public:
      explicit L1TMuonCaloSumProducer(const edm::ParameterSet&);
      ~L1TMuonCaloSumProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override ;
      virtual void endJob() override ;

      virtual void beginRun(const edm::Run&, edm::EventSetup const&) override ;
      virtual void endRun(const edm::Run&, edm::EventSetup const&) override ;
      virtual void beginLuminosityBlock(const edm::LuminosityBlock&, edm::EventSetup const&) override ;
      virtual void endLuminosityBlock(const edm::LuminosityBlock&, edm::EventSetup const&) override ;

      edm::EDGetTokenT <CaloTowerBxCollection> m_caloTowerToken;
      edm::InputTag m_caloLabel;

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
L1TMuonCaloSumProducer::L1TMuonCaloSumProducer(const edm::ParameterSet& iConfig) {
  //register your inputs:
  m_caloLabel = iConfig.getParameter<edm::InputTag> ("caloStage2Layer2Label");
  m_caloTowerToken = consumes <CaloTowerBxCollection> (m_caloLabel);
  //register your products
  produces<MuonCaloSumBxCollection>("TriggerTowerSums");
  produces<MuonCaloSumBxCollection>("TriggerTower2x2s");
}


L1TMuonCaloSumProducer::~L1TMuonCaloSumProducer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//


// ------------ method called to produce the data  ------------
void
L1TMuonCaloSumProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  std::auto_ptr<MuonCaloSumBxCollection> towerSums (new MuonCaloSumBxCollection());
  std::auto_ptr<MuonCaloSumBxCollection> tower2x2s (new MuonCaloSumBxCollection());

  edm::Handle<CaloTowerBxCollection> caloTowers;

  if (iEvent.getByToken(m_caloTowerToken, caloTowers)) {
    int detamax = 4;
    int dphimax = 4;

    for (int bx = caloTowers->getFirstBX(); bx <= caloTowers->getLastBX(); ++bx) {
      std::map<int, MuonCaloSum> sums;
      std::map<int, MuonCaloSum> regs;

      for (auto it = caloTowers->begin(bx); it != caloTowers->end(bx); ++it) {
        const CaloTower& twr = *it;
        int hwEta = twr.hwEta();
        if (std::abs(hwEta) > 27) {
          continue;
        }
        int hwPt = twr.hwPt();
        if (hwPt < 1) {
          continue;
        }
        int hwPhi = twr.hwPhi();

        // calculating tower2x2s
        int ieta2x2 = (hwEta + 27) / 2;
        int iphi2x2 = hwPhi / 2;
        int muon_idx = iphi2x2 * 28 + ieta2x2;
        if (regs.count(muon_idx) == 0) {
          regs[muon_idx] = MuonCaloSum(hwPt, iphi2x2, ieta2x2, muon_idx);
        } else {
          regs.at(muon_idx).setEtBits(regs.at(muon_idx).etBits() + hwPt);
        }

        // std::cout << "iphi; phi " << hwPhi << "; " << phi << " .. ieta; eta" << hwEta << "; " << twr.eta() << std::endl;

        // calculating towerSums
        int ietamax = hwEta + detamax + 1;
        for (int ieta = hwEta-detamax; ieta < ietamax; ++ieta) {
          if (std::abs(ieta) > 27) {
            continue;
          }
          int ietamu = (ieta + 27) / 2;
          int iphimax = hwPhi + dphimax + 1;
          for (int iphi = hwPhi-dphimax; iphi < iphimax; ++iphi) {
            int iphiwrapped = iphi;
            if (iphiwrapped < 0) {
              iphiwrapped += 72;
            } else if (iphiwrapped > 71) {
              iphiwrapped -= 72;
            }
            int iphimu = iphiwrapped / 2;
            int idxmu = iphimu * 28 + ietamu;
            if (sums.count(idxmu) == 0) {
              sums[idxmu] = MuonCaloSum(hwPt, iphimu, ietamu, idxmu);
            } else {
              sums.at(idxmu).setEtBits(sums.at(idxmu).etBits() + hwPt);
            }
          }
        }
      }

      // fill towerSums output collection for this BX
      for (auto it = sums.begin(); it != sums.end(); ++it) {
        if (it->second.etBits() > 0) {
          MuonCaloSum sum = MuonCaloSum(it->second);
          // convert Et to correct scale:
          if (sum.etBits() > 31) {
            sum.setEtBits(31);
          }
          towerSums->push_back(bx, sum);
        }
      }
      // fill tower2x2s output collection for this BX
      for (auto it = regs.begin(); it != regs.end(); ++it) {
        if (it->second.etBits() > 0) {
          tower2x2s->push_back(bx, it->second);
        }
      }
    }
  } else {
    LogWarning("GlobalMuon") << "CaloTowers not found. Producing empty collections." << std::endl;
  }

  iEvent.put(towerSums, "TriggerTowerSums");
  iEvent.put(tower2x2s, "TriggerTower2x2s");

}

// ------------ method called once each job just before starting event loop  ------------
void
L1TMuonCaloSumProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TMuonCaloSumProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TMuonCaloSumProducer::beginRun(const edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void
L1TMuonCaloSumProducer::endRun(const edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
L1TMuonCaloSumProducer::beginLuminosityBlock(const edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
L1TMuonCaloSumProducer::endLuminosityBlock(const edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TMuonCaloSumProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonCaloSumProducer);

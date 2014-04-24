//  Original Author:   Stilianos Kesisoglou - Institute of Nuclear and Particle
// Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
//          Created:   Wed Nov 21 11:24:39 EET 2012
//
//      Nov 21 2012:   First version of the code. Based on the old
// "EcalDeadChannelRecoveryProducers.cc" code
//

// system include files
#include <memory>
#include <string>
#include <cstdio>

// Geometry
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/test/plugins/EcalDeadChannelRecoveryProducers.h"

using namespace cms;
using namespace std;

template <typename DetIdT>
EcalDeadChannelRecoveryProducers<DetIdT>::EcalDeadChannelRecoveryProducers(
    const edm::ParameterSet& ps) {
  //now do what ever other initialization is needed
  CorrectDeadCells_ = ps.getParameter<bool>("CorrectDeadCells");
  CorrectionMethod_ = ps.getParameter<std::string>("CorrectionMethod");
  reducedHitCollection_ = ps.getParameter<std::string>("reducedHitCollection");
  DeadChannelFileName_ = ps.getParameter<std::string>("DeadChannelsFile");
  Sum8GeVThreshold_ = ps.getParameter<double>("Sum8GeVThreshold");

  hitToken_ =
      consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("hitTag"));
  produces<EcalRecHitCollection>(reducedHitCollection_);
}

template <typename DetIdT>
EcalDeadChannelRecoveryProducers<DetIdT>::~EcalDeadChannelRecoveryProducers() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to produce the data  ------------
template <typename DetIdT>
void EcalDeadChannelRecoveryProducers<DetIdT>::produce(
    edm::Event& evt, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);

  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByToken(hitToken_, rhcHandle);
  if (!(rhcHandle.isValid())) {
    //  std::cout << "could not get a handle on the EcalRecHitCollection!" <<
    // std::endl;
    return;
  }
  const EcalRecHitCollection* hit_collection = rhcHandle.product();

  // create an auto_ptr to a EcalRecHitCollection, copy the RecHits into it and
  // put it in the Event:
  std::auto_ptr<EcalRecHitCollection> redCollection(new EcalRecHitCollection);
  deadChannelCorrector.setCaloTopology(theCaloTopology.product());

  //
  //  Double loop over EcalRecHit collection and "dead" cell RecHits.
  //  If we step into a "dead" cell call "deadChannelCorrector::correct()"
  //
  for (EcalRecHitCollection::const_iterator it = hit_collection->begin();
       it != hit_collection->end(); ++it) {
    std::vector<EBDetId>::const_iterator CheckDead = ChannelsDeadID.begin();
    bool OverADeadRecHit = false;
    while (CheckDead != ChannelsDeadID.end()) {
      if (it->detid() == *CheckDead) {
        OverADeadRecHit = true;
        bool AcceptRecHit = true;
        EcalRecHit hit = deadChannelCorrector.correct(
            it->detid(), *hit_collection, CorrectionMethod_, Sum8GeVThreshold_,
            &AcceptRecHit);

        if (hit.energy() != 0 and AcceptRecHit == true) {
          hit.setFlag(EcalRecHit::kNeighboursRecovered);
        } else {
          // recovery failed
          hit.setFlag(EcalRecHit::kDead);
        }

        redCollection->push_back(hit);
        break;
      }
      CheckDead++;
    }
    if (!OverADeadRecHit) {
      redCollection->push_back(*it);
    }
  }

  evt.put(redCollection, reducedHitCollection_);
}

// method called once each job just before starting event loop  ------------
template <> void EcalDeadChannelRecoveryProducers<EBDetId>::beginJob() {
  //Open the DeadChannel file, read it.
  FILE* DeadCha;
  printf("Dead Channels FILE: %s\n", DeadChannelFileName_.c_str());
  DeadCha = fopen(DeadChannelFileName_.c_str(), "r");

  int fileStatus = 0;
  int ieta = -10000;
  int iphi = -10000;

  while (fileStatus != EOF) {

    fileStatus = fscanf(DeadCha, "%d %d\n", &ieta, &iphi);

    //  Problem reading Dead Channels file
    if (ieta == -10000 || iphi == -10000) {
      break;
    }

    if (EBDetId::validDetId(ieta, iphi)) {
      EBDetId cell(ieta, iphi);
      ChannelsDeadID.push_back(cell);
    }

  }  //end while

  fclose(DeadCha);
}

template <> void EcalDeadChannelRecoveryProducers<EEDetId>::beginJob() {
  //Open the DeadChannel file, read it.
  FILE* DeadCha;
  printf("Dead Channels FILE: %s\n", DeadChannelFileName_.c_str());
  DeadCha = fopen(DeadChannelFileName_.c_str(), "r");

  int fileStatus = 0;
  int ix = -10000;
  int iy = -10000;
  int iz = -10000;
  while (fileStatus != EOF) {

    fileStatus = fscanf(DeadCha, "%d %d %d\n", &ix, &iy, &iz);

    //  Problem reading Dead Channels file
    if (ix == -10000 || iy == -10000 || iz == -10000) {
      break;
    }

    if (EEDetId::validDetId(ix, iy, iz)) {
      EEDetId cell(ix, iy, iz);
      ChannelsDeadID.push_back(cell);
    }

  }  //end while

  fclose(DeadCha);
}

// ------------ method called once each job just after ending the event loop
// ------------
template <typename DetIdT>
void EcalDeadChannelRecoveryProducers<DetIdT>::endJob() {}

typedef class EcalDeadChannelRecoveryProducers<EBDetId>
    EBDeadChannelRecoveryProducers;
typedef class EcalDeadChannelRecoveryProducers<EEDetId>
    EEDeadChannelRecoveryProducers;

DEFINE_FWK_MODULE(EBDeadChannelRecoveryProducers);
DEFINE_FWK_MODULE(EEDeadChannelRecoveryProducers);

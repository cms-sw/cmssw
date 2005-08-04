#ifndef UncorrTowersMETProducer_h
#define UncorrTowersMETProducer_h

/** \class UncorrTowersMETProducer
 *
 * UncorrTowersMETProducer is the EDProducer subclass which runs 
 * the UncorrTowersMETAlgo MET finding algorithm.
 *
 * \author Michael Schmitt, The University of Florida
 *
 * \version 1st Version May 14, 2005
 *
 */

#include <vector>
#include <cstdlib>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMET/METAlgorithms/interface/TowerMETAlgo.h"


namespace cms {

  class UncorrTowersMETProducer: public edm::EDProducer {

  public:

    explicit UncorrTowersMETProducer(const edm::ParameterSet&);
    explicit UncorrTowersMETProducer();
    virtual ~UncorrTowersMETProducer();

    virtual void produce(edm::Event&, const edm::EventSetup&);

  private:

    TowerMETAlgo alg_;

  };

}

#endif // UncorrTowersMETProducer_h

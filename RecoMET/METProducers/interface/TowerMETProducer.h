#ifndef TowerMETProducer_h
#define TowerMETProducer_h

/** \class TowerMETProducer
 *
 * TowerMETProducer is the EDProducer subclass which runs 
 * the TowerMETAlgo MET finding algorithm.
 *
 * \author M. Schmitt, R. Cavanaugh, The University of Florida
 *
 * \version 1st Version May 14, 2005
 *
 */

#include <vector>
#include <cstdlib>
#include <string.h>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMET/METAlgorithms/interface/TowerMETAlgo.h"


namespace cms 
{
  class TowerMETProducer: public edm::EDProducer 
    {
    public:
      explicit TowerMETProducer(const edm::ParameterSet&);
      explicit TowerMETProducer();
      virtual ~TowerMETProducer();
      virtual void produce(edm::Event&, const edm::EventSetup&);
    private:
      TowerMETAlgo alg_;
      std::string inputLabel;
    };
}

#endif // TowerMETProducer_h

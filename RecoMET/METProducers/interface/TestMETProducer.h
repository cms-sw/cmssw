#ifndef TestMETProducer_h
#define TestMETProducer_h

/** \class TestMETProducer
 *
 * TestMETProducer is the EDProducer subclass which runs 
 * the TestMETAlgo MET finding algorithm.
 *
 * \author R. Cavanaugh, The University of Florida
 *
 * \version 1st Version May 14, 2005
 *
 */

#include <vector>
#include <cstdlib>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMET/METAlgorithms/interface/TestMETAlgo.h"


namespace cms 
{
  class TestMETProducer: public edm::EDProducer 
    {
    public:
      explicit TestMETProducer(const edm::ParameterSet&);
      explicit TestMETProducer();
      virtual ~TestMETProducer();
      virtual void produce(edm::Event&, const edm::EventSetup&);
    private:
      TestMETAlgo alg_;
    };
}

#endif // TestMETProducer_h

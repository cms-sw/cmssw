#ifndef CorrMETProducer_h
#define CorrMETProducer_h

/** \class CorrMETProducer
 *
 * CorrMETProducer is the EDProducer subclass which runs 
 * the CorrMETAlgo MET finding algorithm.
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

#include "RecoMET/METAlgorithms/interface/CorrMETAlgo.h"


namespace cms 
{
  class CorrMETProducer: public edm::EDProducer 
    {
    public:
      explicit CorrMETProducer(const edm::ParameterSet&);
      explicit CorrMETProducer();
      virtual ~CorrMETProducer();
      virtual void produce(edm::Event&, const edm::EventSetup&);
    private:
      CorrMETAlgo alg_;
    };
}

#endif // CorrMETProducer_h

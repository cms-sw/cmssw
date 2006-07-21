#ifndef CandAlgos_CandCombiner_h
#define CandAlgos_CandCombiner_h
/** \class cand::modules::CandCombiner
 *
 * performs all possible and selected combinations
 * of particle pairs using the TwoBodyCombiner 
 * or ThreeBodyCombiner utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.7 $
 *
 * $Id: CandCombiner.h,v 1.7 2006/04/10 08:28:01 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
#include "PhysicsTools/CandUtils/interface/ThreeBodyCombiner.h"
#include "PhysicsTools/CandAlgos/src/decayParser.h"
#include <string>

namespace edm {
  class ParameterSet;
}

namespace cand {
  namespace modules {
    
    class CandCombiner : public edm::EDProducer {
    public:
      /// constructor from parameter set
      explicit CandCombiner( const edm::ParameterSet & );    
      /// destructor
      ~CandCombiner();
      
    private:
      /// process an event
      void produce( edm::Event& e, const edm::EventSetup& );
      /// label vector
      std::vector<cand::parser::ConjInfo> labels_;
      /// combiner utilities
      std::auto_ptr<TwoBodyCombiner> combiner2_;
      std::auto_ptr<ThreeBodyCombiner> combiner3_;
    };
  }
}

#endif

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
 * \version $Revision: 1.8 $
 *
 * $Id: CandCombiner.h,v 1.8 2006/07/21 10:35:14 fabozzi Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "PhysicsTools/CandUtils/interface/NBodyCombiner.h"
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
      /// combiner utility
      std::auto_ptr<NBodyCombiner> combiner_;
    };
  }
}

#endif

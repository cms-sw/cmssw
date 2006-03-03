#ifndef CandAlgos_CandCombiner_h
#define CandAlgos_CandCombiner_h
/** \class candmodules::CandCombiner
 *
 * performs all possible and selected combinations
 * of particle pairs using the TwoBoddyCombiner utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision$
 *
 * $Id: Track.h,v 1.12 2006/03/01 12:23:40 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
#include "PhysicsTools/CandAlgos/src/decayParser.h"
#include <string>

namespace edm {
  class ParameterSet;
}

namespace candmodules {

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
    std::vector<candcombiner::ConjInfo> labels_;
    /// combiner utility
    std::auto_ptr<TwoBodyCombiner> combiner_;
    /// labels of the two source candidate collections
    std::string source1_, source2_;
  };

}

#endif

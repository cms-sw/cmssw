#ifndef __CommonTools_ParticleFlow_interface_EventHypothesis__
#define __CommonTools_ParticleFlow_interface_EventHypothesis__


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <map>
#include <string>


namespace pf2pat {
  
  class EventHypothesis {

  public:
    /// the name of the parameter set is given to the event hypothesis
    /// the parameter set must contain a vstring named "sequence"
    /// and at least a parameter set named after each string in the 
    /// sequence. 
    EventHypothesis( const edm::ParameterSet& ps);
    

  private:
    // need a base class for all algorithms/producers
    typedef std::string Producer;
    typedef std::map< std::string, Producer >  Producers;
    typedef std::vector< std::string > Sequence; 
    
    /// unique name
    std::string   name_;
    
    /// map of producers, indexed by producer name
    Producers     producers_;

    /// sequence of producers 
    Sequence      sequence_;
    
  };

}

#endif

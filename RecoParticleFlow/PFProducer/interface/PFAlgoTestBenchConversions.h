#ifndef RecoParticleFlow_PFAlgoTestBenchConversions_PFAlgoTestBenchConversions_h
#define RecoParticleFlow_PFAlgoTestBenchConversions_PFAlgoTestBenchConversions_h 

#include <iostream>




#include "RecoParticleFlow/PFProducer/interface/PFAlgo.h"

/// \brief Particle Flow Algorithm test bench for the electron team
/*!
  \author Florian Beaudette, Daniele Benedetti, Michele Pioppi
  \date January 2006
*/


class PFAlgoTestBenchConversions : public PFAlgo {

 public:

  /// constructor
  PFAlgoTestBenchConversions() {}

  /// destructor
  virtual ~PFAlgoTestBenchConversions() {}
  
 
 protected:

  /// process one block. can be reimplemented in more sophisticated 
  /// algorithms
  virtual void processBlock( const reco::PFBlockRef& blockref,
                             std::list<reco::PFBlockRef>& hcalBlockRefs, 
			     std::list<reco::PFBlockRef>& ecalBlockRefs );
  

 private:

};


#endif



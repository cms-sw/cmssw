#ifndef RecoParticleFlow_PFAlgoTestBenchElectrons_PFAlgoTestBenchElectrons_h
#define RecoParticleFlow_PFAlgoTestBenchElectrons_PFAlgoTestBenchElectrons_h 

#include <iostream>
#include <vector>
//#include <pair>


#include "RecoParticleFlow/PFProducer/interface/PFAlgo.h"

/// \brief Particle Flow Algorithm test bench for the electron team
/*!
  \author Florian Beaudette, Daniele Benedetti, Michele Pioppi
  \date January 2006
*/


class PFAlgoTestBenchElectrons : public PFAlgo {

 public:

  /// constructor
  PFAlgoTestBenchElectrons() {}

  /// destructor
  ~PFAlgoTestBenchElectrons() override {}
  

 protected:

  /// process one block. can be reimplemented in more sophisticated 
  /// algorithms
  void processBlock( const reco::PFBlockRef& blockref,
                             std::list<reco::PFBlockRef>& hcalBlockRefs, 
			     std::list<reco::PFBlockRef>& ecalBlockRefs ) override;
  

 private:

};


#endif



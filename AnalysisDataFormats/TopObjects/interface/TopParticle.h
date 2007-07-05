//
// $Id: TopParticle.h,v 1.2 2007/06/23 07:09:29 lowette Exp $
//

#ifndef TopObjects_TopParticle_h
#define TopObjects_TopParticle_h

/**
  \class    TopParticle TopParticle.h "AnalysisDataFormats/TopObjects/interface/TopParticle.h"
  \brief    High-level top particle container

   TopParticle contains a particle as a TopObject

  \author   Steven Lowette
  \version  $Id: TopParticle.h,v 1.2 2007/06/23 07:09:29 lowette Exp $
*/


#include "DataFormats/Candidate/interface/Particle.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"


typedef reco::Particle TopParticleType;


class TopParticle : public TopObject<TopParticleType> {
  
  public:

    TopParticle();
    TopParticle(const TopParticleType & aParticle);
    virtual ~TopParticle();
          
};


#endif

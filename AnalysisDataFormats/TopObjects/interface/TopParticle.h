//
// Author:  Steven Lowette
// Created: Thu May  3 14:41:12 PDT 2007
//
// $Id: TopParticle.h,v 1.1 2007/05/04 01:08:38 lowette Exp $
//

#ifndef TopObjects_TopParticle_h
#define TopObjects_TopParticle_h

/**
  \class    TopParticle TopParticle.h "AnalysisDataFormats/TopObjects/interface/TopParticle.h"
  \brief    High-level top particle container

   TopParticle contains a particle as a TopObject

  \author   Steven Lowette
  \version  $Id: TopParticle.h,v 1.1 2007/05/04 01:08:38 lowette Exp $
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

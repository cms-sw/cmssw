//
// Author:  Steven Lowette
// Created: Thu May  3 14:41:12 PDT 2007
//
// $Id$
//

#ifndef TopParticle_h
#define TopParticle_h

/**
  \class    TopParticle TopParticle.h "AnalysisDataFormats/TopObjects/interface/TopParticle.h"
  \brief    High-level top particle container

   TopParticle contains a particle as a TopObject

  \author   Steven Lowette
  \version  $Id$
*/


#include "DataFormats/Candidate/interface/Particle.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"


typedef TopObject<reco::Particle> TopParticle;


#endif

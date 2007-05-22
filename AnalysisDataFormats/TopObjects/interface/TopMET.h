//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopMET.h,v 1.1 2007/05/04 01:08:38 lowette Exp $
//

#ifndef TopMET_h
#define TopMET_h

/**
  \class    TopMET TopMET.h "AnalysisDataFormats/TopObjects/interface/TopMET.h"
  \brief    High-level top MET container

   TopMET contains a missing ET 4-vector as a TopObject

  \author   Steven Lowette
  \version  $Id: TopMET.h,v 1.1 2007/05/04 01:08:38 lowette Exp $
*/


#include "DataFormats/METReco/interface/CaloMET.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"



typedef reco::CaloMET METType;

class TopMET : public TopObject<METType> 
{
   
   public:
      TopMET();
      TopMET(METType);
      virtual ~TopMET();
            
      void 		setGenMET(Particle);
      void    		setFitMET(TopParticle);
      
      reco::Particle	getGenMET() const;
      TopParticle  	getFitMET() const;
      
   protected:
      Particle	   genMET;
      TopParticle  fitMET;
      
};


#endif

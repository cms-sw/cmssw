//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopJet.h,v 1.1 2007/05/04 01:08:38 lowette Exp $
//

#ifndef TopObjects_TopJet_h
#define TopObjects_TopJet_h

/**
  \class    TopJet TopJet.h "AnalysisDataFormats/TopObjects/interface/TopJet.h"
  \brief    High-level top jet container

   TopJet contains a jet as a TopObject

  \author   Steven Lowette
  \version  $Id: TopJet.h,v 1.1 2007/05/04 01:08:38 lowette Exp $
*/


#include "DataFormats/JetReco/interface/CaloJet.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"


typedef reco::CaloJet JetType;

class TopJet : public TopObject<JetType>
{
   
   public:
      TopJet();
      TopJet(JetType);
      virtual ~TopJet();
            
      void 		setGenJet(Particle);
      void    		setRecJet(JetType);
      void    		setFitJet(TopParticle);
      void    		setBdiscriminant(double);
      
      reco::Particle	getGenJet() const;
      JetType 		getRecJet() const;
      TopParticle  	getFitJet() const;
      double  		getBdiscriminant() const;
      
   protected:
      Particle   	genJet;
      JetType     	recJet;
      TopParticle 	fitJet;
      double      	bdiscr;
      
};

#endif

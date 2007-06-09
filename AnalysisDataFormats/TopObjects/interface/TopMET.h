//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopMET.h,v 1.2 2007/05/22 16:36:50 heyninck Exp $
//

#ifndef TopMET_h
#define TopMET_h

/**
  \class    TopMET TopMET.h "AnalysisDataFormats/TopObjects/interface/TopMET.h"
  \brief    High-level top MET container

   TopMET contains a missing ET 4-vector as a TopObject

  \author   Steven Lowette
  \version  $Id: TopMET.h,v 1.2 2007/05/22 16:36:50 heyninck Exp $
*/


#include "DataFormats/METReco/interface/CaloMET.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"


typedef reco::CaloMET METType;


class TopMET : public TopObject<METType> {
  
  public:

    TopMET();
    TopMET(METType);
    virtual ~TopMET();
          
    reco::Particle getGenMET() const;
    TopParticle    getFitMET() const;
    void           setGenMET(Particle);
    void           setFitMET(TopParticle);

  protected:

    reco::Particle genMET;
    TopParticle    fitMET;

};


#endif

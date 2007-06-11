//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopMET.h,v 1.3 2007/06/09 01:18:48 lowette Exp $
//

#ifndef TopMET_h
#define TopMET_h

/**
  \class    TopMET TopMET.h "AnalysisDataFormats/TopObjects/interface/TopMET.h"
  \brief    High-level top MET container

   TopMET contains a missing ET 4-vector as a TopObject

  \author   Steven Lowette
  \version  $Id: TopMET.h,v 1.3 2007/06/09 01:18:48 lowette Exp $
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
	// solve for neutrino Pz constraining to the W mass in W -> mu + nu
	// type defines how to choose the roots:
	// type = 1: the closest nu_pz to mu_pz if real roots,
	//           or just the real part if solution is complex.
	// type = 2: pending
	// type = 3: pending
	TopParticle    getPz(TopParticle lepton, int type=0);
	// return true if the solution is complex
	bool           isSolutionComplex() { return iscomplex_; }
	
  protected:

    reco::Particle genMET;
    TopParticle    fitMET;
	bool           iscomplex_;
	

};


#endif

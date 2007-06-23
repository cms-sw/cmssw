//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopMET.h,v 1.4 2007/06/11 21:00:37 yumiceva Exp $
//

#ifndef TopObjects_TopMET_h
#define TopObjects_TopMET_h

/**
  \class    TopMET TopMET.h "AnalysisDataFormats/TopObjects/interface/TopMET.h"
  \brief    High-level top MET container

   TopMET contains a missing ET 4-vector as a TopObject

  \author   Steven Lowette
  \version  $Id: TopMET.h,v 1.4 2007/06/11 21:00:37 yumiceva Exp $
*/


#include "DataFormats/METReco/interface/CaloMET.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"


typedef reco::CaloMET TopMETType;


class TopMET : public TopObject<TopMETType> {

  friend class TopMETProducer;
  friend class TtSemiKinFitterEMom;
  friend class TtSemiKinFitterEtEtaPhi;
  friend class TtSemiKinFitterEtThetaPhi;
  friend class StKinFitterEMom;
  friend class StKinFitterEtEtaPhi;
  friend class StKinFitterEtThetaPhi;

  public:

    TopMET();
    TopMET(const TopMETType & aMET);
    virtual ~TopMET();
          
    reco::Particle getGenMET() const;
    TopParticle    getFitMET() const;
    // solve for neutrino Pz constraining to the W mass in W -> mu + nu
    // type defines how to choose the roots:
    // type = 1: the closest nu_pz to mu_pz if real roots,
    //           or just the real part if solution is complex.
    // type = 2: pending
    // type = 3: pending
    TopParticle    getPz(const TopParticle & lepton, int type = 0);
    // return true if the solution is complex
    bool           isSolutionComplex() const;

  protected:

    void setGenMET(const Particle & gm);
    void setFitMET(const TopParticle & fm);

  protected:

    std::vector<reco::Particle> genMET_;
    std::vector<TopParticle> fitMET_;
    bool isComplex_;

};


#endif

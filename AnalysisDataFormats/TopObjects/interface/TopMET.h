//
// $Id: TopMET.h,v 1.6 2007/07/05 23:09:31 lowette Exp $
//

#ifndef TopObjects_TopMET_h
#define TopObjects_TopMET_h

/**
  \class    TopMET TopMET.h "AnalysisDataFormats/TopObjects/interface/TopMET.h"
  \brief    High-level top MET container

   TopMET contains a missing ET 4-vector as a TopObject

  \author   Steven Lowette
  \version  $Id: TopMET.h,v 1.6 2007/07/05 23:09:31 lowette Exp $
*/


#include "DataFormats/METReco/interface/CaloMET.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"


typedef reco::CaloMET TopMETType;


class TopMET : public TopObject<TopMETType> {

  friend class TopMETProducer;

  public:

    TopMET();
    TopMET(const TopMETType & aMET);
    virtual ~TopMET();
          
    reco::Particle getGenMET() const;
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

  protected:

    std::vector<reco::Particle> genMET_;
    bool isComplex_;

};


#endif

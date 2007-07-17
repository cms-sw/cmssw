//
// $Id: TopMET.h,v 1.7 2007/07/05 23:49:59 lowette Exp $
//

#ifndef TopObjects_TopMET_h
#define TopObjects_TopMET_h

/**
  \class    TopMET TopMET.h "AnalysisDataFormats/TopObjects/interface/TopMET.h"
  \brief    High-level top MET container

   TopMET contains a missing ET 4-vector as a TopObject

  \author   Steven Lowette
  \version  $Id: TopMET.h,v 1.7 2007/07/05 23:49:59 lowette Exp $
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

  protected:

    void setGenMET(const Particle & gm);

  protected:

    std::vector<reco::Particle> genMET_;

};


#endif

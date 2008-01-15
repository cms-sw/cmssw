//
// $Id: MET.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
//

#ifndef DataFormats_PatCandidates_MET_h
#define DataFormats_PatCandidates_MET_h

/**
  \class    MET MET.h "DataFormats/PatCandidates/interface/MET.h"
  \brief    Analysis-level MET class

   MET implements an analysis-level missing energy class as a 4-vector
   within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: MET.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
*/


#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"


namespace pat {


  typedef reco::CaloMET METType;


  class MET : public PATObject<METType> {

    friend class PATMETProducer;

    public:

      MET();
      MET(const METType & aMET);
      virtual ~MET();

      reco::Particle getGenMET() const;

    protected:

      void setGenMET(const reco::Particle & gm);

    protected:

      std::vector<reco::Particle> genMET_;

  };


}

#endif

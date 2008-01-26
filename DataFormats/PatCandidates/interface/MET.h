//
// $Id: MET.h,v 1.4 2008/01/22 21:58:14 lowette Exp $
//

#ifndef DataFormats_PatCandidates_MET_h
#define DataFormats_PatCandidates_MET_h

/**
  \class    MET MET.h "DataFormats/PatCandidates/interface/MET.h"
  \brief    Analysis-level MET class

   MET implements an analysis-level missing energy class as a 4-vector
   within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: MET.h,v 1.4 2008/01/22 21:58:14 lowette Exp $
*/


#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"


namespace pat {


  typedef reco::CaloMET METType;


  class MET : public PATObject<METType> {

    public:

      MET();
      MET(const METType & aMET);
      MET(const edm::RefToBase<METType> & aMETRef);
      virtual ~MET();

      const reco::Particle * genMET() const;

      void setGenMET(const reco::Particle & gm);

    protected:

      std::vector<reco::Particle> genMET_;

  };


}

#endif

//
// $Id: MET.h,v 1.8 2008/03/05 14:47:33 fronga Exp $
//

#ifndef DataFormats_PatCandidates_MET_h
#define DataFormats_PatCandidates_MET_h

/**
  \class    pat::MET MET.h "DataFormats/PatCandidates/interface/MET.h"
  \brief    Analysis-level MET class

   MET implements an analysis-level missing energy class as a 4-vector
   within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: MET.h,v 1.8 2008/03/05 14:47:33 fronga Exp $
*/


#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace pat {


  typedef reco::CaloMET METType;


  class MET : public PATObject<METType> {

    public:

      MET();
      MET(const METType & aMET);
      MET(const edm::RefToBase<METType> & aMETRef);
      virtual ~MET();

      const reco::GenMET * genMET() const;

      void setGenMET(const reco::GenMET & gm);

    protected:

      std::vector<reco::GenMET> genMET_;

  };


}

#endif

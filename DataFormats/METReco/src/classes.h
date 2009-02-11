// File:           classes.h
// Description:  
// Author:         Unknown
// Creation Date:  Unknown
//
//--------------------------------------------
//Modified:     29.10.2008 by R.Cavanaugh, UIC/Fermilab
//Description:  include Particle Flow MET
//

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/Common/interface/EDProduct.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/PxPyPzE4D.h" 
#include <boost/cstdint.hpp> 
#include <Rtypes.h> 
#include "DataFormats/METReco/interface/METFwd.h" 
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h" 
#include "DataFormats/METReco/interface/CaloMETFwd.h" 
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h" 
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETFwd.h" 
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include <vector>

namespace {
  namespace {
    edm::Wrapper<reco::MET> dummy1;
    edm::Wrapper<reco::METCollection> dummy2;
    edm::Wrapper< std::vector<reco::MET> > dummy3;
    std::vector<reco::MET> dummy4;

    CaloMETRef r1;
    CaloMETRefProd rp1;
    CaloMETRefVector rv1;
    edm::Wrapper<reco::CaloMET> dummy5;
    edm::Wrapper<reco::CaloMETCollection> dummy7;
    edm::Wrapper< std::vector<reco::CaloMET> > dummy8;
    std::vector<reco::CaloMET> dummy9;
    edm::reftobase::Holder<reco::Candidate,reco::CaloMETRef> rtb1;

    GenMETRef r2;
    GenMETRefProd rp2;
    GenMETRefVector rv2;
    edm::Wrapper<reco::GenMET> dummy10;
    edm::Wrapper<reco::GenMETCollection> dummy11;
    edm::Wrapper< std::vector<reco::GenMET> > dummy12;
    std::vector<reco::GenMET> dummy13;
    edm::reftobase::Holder<reco::Candidate,reco::GenMETRef> rtb2;

    METRef r3;
    METRefProd rp3;
    METRefVector rv3;
    edm::Wrapper<reco::MET> dummy14;
    edm::Wrapper<reco::METCollection> dummy15;
    edm::Wrapper< std::vector<reco::MET> > dummy16;
    std::vector<reco::MET> dummy17;
    edm::reftobase::Holder<reco::Candidate,reco::METRef> rtb3;

    PFMETRef r4;
    PFMETRefProd rp4;
    PFMETRefVector rv4;
    edm::Wrapper<reco::PFMET> dummy18;
    edm::Wrapper<reco::PFMETCollection> dummy19;
    edm::Wrapper< std::vector<reco::PFMET> > dummy20;
    std::vector<reco::PFMET> dummy21;
    edm::reftobase::Holder<reco::Candidate,reco::PFMETRef> rtb4;

    edm::reftobase::RefHolder<reco::METRef> dummy22;
    edm::reftobase::RefHolder<reco::CaloMETRef> dummy23;
    edm::reftobase::RefHolder<reco::GenMETRef> dummy24;

    // used for backward compatibility
    std::vector<CommonMETData> bcv1;
  }
}

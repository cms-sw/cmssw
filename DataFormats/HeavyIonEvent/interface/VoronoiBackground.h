//
// $Id: VoronoiBackground.h,v 1.12 2010/08/23 16:42:01 nart Exp $
//

#ifndef DataFormats_VoronoiBackground_h
#define DataFormats_VoronoiBackground_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <string>
#include <vector>

namespace reco { class VoronoiBackground {
public:
      VoronoiBackground();
      VoronoiBackground(double pt0, double pt1, double pt2, double mt0, double mt1, double mt2);
      virtual ~VoronoiBackground();

  double pt() const{
     return pt_corrected;
  }

protected:

  double pt_preeq;
  double pt_posteq;
  double pt_corrected;

  double mt_preeq;
  double mt_posteq;
  double mt_corrected;

  double voronoi_area;
  //RefVector of neighbors...

};

 typedef edm::ValueMap<reco::VoronoiBackground> VoronoiMap;
 typedef edm::Ref<reco::CandidateView> CandidateViewRef;


}

#endif 



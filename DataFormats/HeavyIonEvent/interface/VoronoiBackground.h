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
      VoronoiBackground(double pt0, double pt1, double pt2, double mt0, double mt1, double mt2, double v);
      virtual ~VoronoiBackground();

  double pt() const{ return pt_corrected; }
  double pt_equalized() const{ return pt_posteq; }
  double pt_initial() const{ return pt_preeq; }

  double mt() const{ return mt_corrected; }
  double mt_equalized() const{ return mt_posteq; }
  double mt_initial() const{ return mt_preeq; }

protected:

  double pt_preeq;
  double pt_posteq;
  double pt_corrected;

  double mt_preeq;
  double mt_posteq;
  double mt_corrected;

  double voronoi_area;

};

 typedef edm::ValueMap<reco::VoronoiBackground> VoronoiMap;
 typedef edm::Ref<reco::CandidateView> CandidateViewRef;


}

#endif 



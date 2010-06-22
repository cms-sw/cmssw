#ifndef MuonReco_MuonQuality_h
#define MuonReco_MuonQuality_h

#include "DataFormats/Math/interface/Point3D.h"
namespace reco {
    struct MuonQuality {
      ///
      /// value of the kink algorithm applied to the inner track stub
      float trkKink;
      /// value of the kink algorithm applied to the global track
      float glbKink;
      /// chi2 value for the inner track stub with respect to the global track
      float trkRelChi2;
      /// chi2 value for the outer track stub with respect to the global track
      float staRelChi2;
      
      /// Kink position for the tracker stub and global track
      math::XYZPoint tkKink_position;
      math::XYZPoint glbKink_position;
      
      MuonQuality():
	trkKink(0), glbKink(0),
	trkRelChi2(0), staRelChi2(0)
      { }
       
    };
}
#endif

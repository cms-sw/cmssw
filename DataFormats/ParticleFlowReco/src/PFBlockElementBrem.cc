#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/Common/interface/Ref.h" 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include <iomanip>

using namespace reco;
using namespace std;


PFBlockElementBrem::PFBlockElementBrem(const GsfPFRecTrackRef& gsfref, const double DeltaP, const double SigmaDeltaP, const uint indTrajPoint):
  PFBlockElement( BREM ),
  GsftrackRefPF_( gsfref ), 
  GsftrackRef_( gsfref->gsfTrackRef() ),
  deltaP_(DeltaP),
  sigmadeltaP_(SigmaDeltaP),
  indPoint_(indTrajPoint){}


void PFBlockElementBrem::Dump(ostream& out, 
                               const char* tab ) const {
  
  if(! out ) return;
 
  if( !GsftrackRefPF_.isNull() ) {

    double charge = ((*GsftrackRefPF()).PFRecBrem()[(indPoint_-2)]).charge();
    double dp =  deltaP_;
    double sigmadp = sigmadeltaP_;
    int indextrj = (indPoint_-2);
    out<<setprecision(0);
    out<<tab<<setw(7)<<"charge="<<setw(3)<<charge;
    out<<setprecision(3);
    out<<setiosflags(ios::right);
    out<<setiosflags(ios::fixed);
    out<<", DeltaP=  "<< dp;
    out<<", SigmaDeltaP=  " << sigmadp;
    out<<", Traj Point=  " << indextrj;
    out<<resetiosflags(ios::right|ios::fixed); }

}


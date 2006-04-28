#include "PixelTrackBuilder.h"
#include "DataFormats/TrackReco/interface/HelixParameters.h"

reco::Track * PixelTrackBuilder::build(
      const Measurement1D & pt,
      const Measurement1D & phi, 
      const Measurement1D & cotTheta,
      const Measurement1D & tip,  
      const Measurement1D & zip,
      float chi2,
      int   charge,
      const std::vector<const TrackingRecHit* >& hits) const 
{
  float valPt = pt.value();
  //
  //momentum
  //
  math::XYZVector mom( valPt*cos( phi.value()),
                       valPt*sin( phi.value()),
                       valPt*cotTheta.value());

  //
  // point of the closest approax to Beam line
  //
  cout << "TIP value: " <<  tip.value() << endl;
  math::XYZPoint  vtx(  tip.value()*cos( phi.value()),
                        tip.value()*sin( phi.value()),
                        zip.value());
  cout <<"vertex: " << vtx << endl;

  // temporary fix!
  vtx = math::XYZPoint(0.,0.,vtx.z());
  //
  //errors (dummy)
  //
  math::Error<6>::type cov; //FIXME - feel

  cout <<" momentum: " << mom << endl;

  int nhits = hits.size();

  return new reco::Track( chi2,         // chi2
                          2*nhits-5,  // dof
                          nhits,      // foundHits
                          0,
                          0,          //lost hits
                          charge,
                          vtx,
                          mom,
                          cov);

}


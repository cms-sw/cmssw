#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/Common/interface/Ref.h" 
#include "Math/Vector3D.h"

#include <iomanip>

using namespace reco;
using namespace std;

void PFBlockElementCluster::Dump(ostream& out, 
                                 const char* tab ) const {
  
  if(! out ) return;
  // need to convert the math::XYZPoint data member of the PFCluster class=
  // to a displacement vector: 
  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D<double> , ROOT::Math::DefaultCoordinateSystemTag > 
    clusterPos( clusterRef_->position().X(), clusterRef_->position().Y(),clusterRef_->position().Z() );

  clusterPos = clusterPos.Unit();
  double E = clusterRef_->energy();
  clusterPos *= E;
  double ET = sqrt (clusterPos.X()*clusterPos.X() + clusterPos.Y()*clusterPos.Y());

  out << setprecision(3);
  out << tab<<setw(7)<<"layer="<<setw(3)<<clusterRef_->layer();
  out << setiosflags(ios::right);
  out << setiosflags(ios::fixed);
  out << setw(4) <<", ET =" << setw(7) << ET;
  out << setw(4) <<", E ="  << setw(7) << E;
  out << " (eta,phi,z)= (";
  out << clusterRef_->position().Eta()<<",";
  out << clusterRef_->position().Phi()<<",";
  out << clusterRef_->position().Z()  <<")";
  out << resetiosflags(ios::right|ios::fixed);
}

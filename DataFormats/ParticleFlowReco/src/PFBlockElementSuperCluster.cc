#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Common/interface/Ref.h" 
#include "Math/Vector3D.h"

#include <iomanip>

using namespace reco;
using namespace std;

void PFBlockElementSuperCluster::Dump(ostream& out, 
				      const char* tab ) const {
  
  if(! out ) return;
  // need to convert the math::XYZPoint data member of the PFCluster class=
  // to a displacement vector: 
  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D<double> , ROOT::Math::DefaultCoordinateSystemTag > 
    clusterPos( superClusterRef_->position().X(), superClusterRef_->position().Y(),superClusterRef_->position().Z() );

  clusterPos = clusterPos.Unit();
  double E = superClusterRef_->energy();
  clusterPos *= E;
  double ET = sqrt (clusterPos.X()*clusterPos.X() + clusterPos.Y()*clusterPos.Y());

  out << setprecision(3);
  out << setiosflags(ios::right);
  out << setiosflags(ios::fixed);
  out << setw(4) <<", ET =" << setw(7) << ET;
  out << setw(4) <<", E ="  << setw(7) << E;
  out << " (eta,phi,z)= (";
  out << superClusterRef_->position().Eta()<<",";
  out << superClusterRef_->position().Phi()<<",";
  out << superClusterRef_->position().Z()  <<")";
  out << resetiosflags(ios::right|ios::fixed);
}

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/Common/interface/Ref.h" 

#include <iomanip>

using namespace reco;
using namespace std;

void PFBlockElementCluster::Dump(ostream& out, 
				 const char* tab ) const {

  if(! out ) return;
  out<<setprecision(3);
  out<<tab<<setw(10)<<"layer="<<setw(2)<<clusterRef_->layer();
  out<<setiosflags(ios::right);
  out<<setiosflags(ios::fixed);
  out<<setw(4)<<", E ="<<setw(7)<<clusterRef_->energy();
  out<<" (eta,phi)= (";
  out<<clusterRef_->positionXYZ().Eta()<<",";
  out<<clusterRef_->positionXYZ().Phi()<<")";
  out<<resetiosflags(ios::right|ios::fixed);
}

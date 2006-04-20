#include "DataFormats/VertexReco/interface/TrackPrint.h"

using namespace std;

ostream& operator<<(ostream& os, const reco::Track & tk) {
  
  os << "track parameters" << endl;
  {
    os << "(d_xy, phi_0, q/pT, d_z, p_z/p_T) = ";
    {
      for (int i = 0; i < 5; i++) {
        os.precision(6); os.width(13); os<<tk.parameter(i);
      }
    }
  }
  os << endl;
  os << "covariance" << endl;
  {
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++) {
	os.precision(6); os.width(13); os<<tk.covariance(i, j);
      }
      os << endl;
    }
  }
  os << endl;
  
  os << "covariance (x,y,z,px,py,pz)" << endl;
  {
    reco::TrackBase::PosMomError err = tk.posMomError();
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
	os.precision(6); os.width(13); os<< err(i,j);
      }
      os << endl;
    }
  }
  os << endl;
  
  return os;
}

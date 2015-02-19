#include "PhysicsTools/Heppy/interface/Apc.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"


#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
#include <stdlib.h>

namespace heppy {

  double Apc::getApcJetMetMin( const std::vector<double>& et,
                               const std::vector<double>& px,
                               const std::vector<double>& py,
                               const double metx, const double mety) {
    
    if(et.size()==0) return -1.;
    
    // Momentum sums in transverse plane
    const double ht = accumulate( et.begin(), et.end(), 0. );
    
    // jets are pt-sorted
    double firstjet_phi = atan2(py[0], px[0]);

    double apcjet(0.), apcmet(0.), apcjetmet(0.);
    double apcjetmetmin(0.);

    for (size_t i = 0; i < et.size(); i++) {
      double jet_phi = atan2(py[i], px[i]);
      double met_phi = atan2(mety, metx);

      double dphisignaljet = fabs(deltaPhi(jet_phi,firstjet_phi));
      double dphimet       = fabs(deltaPhi(jet_phi, met_phi));
      
      apcjet    += et[i] * cos(dphisignaljet/2.0);
      apcmet    += et[i] * sin(dphimet/2.0);
      apcjetmet += et[i] * cos(dphisignaljet/2.0) * sin(dphimet/2.0);
      }
    
    
    std::vector<double> apcjetvector;
    std::vector<double> apcjetmetvector;
    for (size_t j = 0; j < et.size(); j++) {
      apcjetvector.push_back(0.);
      apcjetmetvector.push_back(0.);
      double jet_phi_j = atan2(py[j], px[j]);
      for (size_t i = 0; i < et.size(); i++) {
        double jet_phi_i = atan2(py[i], px[i]);
        double dphi_jet = fabs(deltaPhi(jet_phi_i, jet_phi_j));
        double met_phi = atan2(mety, metx);
        double dphimet = fabs(deltaPhi(jet_phi_i, met_phi));
        
        apcjetvector.back()    += et[i] * cos(dphi_jet/2.0);
        apcjetmetvector.back() += et[i] * cos(dphi_jet/2.0) * sin(dphimet/2.0);
      }
    }
    if (apcjetvector.size() > 0 && apcjetmetvector.size() > 0) {
      apcjetmetmin = *min_element(apcjetmetvector.begin(), apcjetmetvector.end());
    }
    
  
    if (ht != 0) return apcjetmetmin / ht;
    else return -1.;

  }
}

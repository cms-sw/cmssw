#include "../interface/TowerProjector.h"

namespace reco{


  void newCaloPoint(const Particle::Vector& direction,Particle::Point& caloPoint_){
       // "direction" is direction of calorimeter object seen from origin
      // It should be normalised to unit length.

      static const double depth = 0.1; // one for all relative depth of the reference point between ECAL begin and HCAL end
      static const double R_BARREL = (1.-depth)*143.+depth*407.;
      static const double Z_ENDCAP = (1.-depth)*320.+depth*568.; // 1/2(EEz+HEz)
      static const double R_FORWARD = Z_ENDCAP / sqrt (cosh(3.)*cosh(3.) -1.); // eta=3
      static const double Z_FORWARD = 1120.+depth*165.;
      static const double R_INNER = Z_FORWARD / sqrt (cosh(5.2)*sinh(5.2) - 1.); // eta = 5.2
      static const double Z_BIG = 1.e5;
            
      // Check which subdetector the energy is deposited.

      double a_z = fabs(direction.z());
      double rho =      direction.rho();

      if (a_z < rho * (Z_ENDCAP/R_BARREL)) {
	// Barrel
        caloPoint_ = direction * (R_BARREL/rho);        
      
      } else if (a_z < rho * (Z_ENDCAP/R_FORWARD)) {
        // Endcap
        caloPoint_ = direction * (Z_ENDCAP/a_z);        

      } else if (a_z < rho * (Z_FORWARD/R_INNER)) {
        // Forward
        caloPoint_ = direction * (Z_FORWARD/a_z);        
      } else {
        // Outside acceptance
        caloPoint_ = direction * (Z_BIG/a_z);
      }
    }

  void physicsP4 (const Particle::Point &vertex, const Particle &inParticle, Particle::LorentzVector &returnVector) {
    Particle::Point caloPoint;
    newCaloPoint(inParticle.momentum().unit(),caloPoint); // Jet position in Calo.
    Particle::Vector physicsDir = caloPoint - vertex;
    double p = inParticle.momentum().r();
    Particle::Vector p3 = p * physicsDir.unit();
    Particle::LorentzVector p4(p3.x(), p3.y(), p3.z(), inParticle.energy());
    // Set parameters for "Candidate" also.
    //  this->SetP4(p4);
    //  this->SetVertex(vertex);
    returnVector=p4;
  }

  
}

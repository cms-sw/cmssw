#ifndef DataFormats_PatCandidates_CandKinResolution_h
#define DataFormats_PatCandidates_CandKinResolution_h
#include <vector>
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace pat {
  class CandKinResolution  {
		/** 
		  <h2>Parametrizations</h2>
                   (lowercase means values, uppercase means fixed parameters)<br />
		   <b>Cart</b> = (px, py, pz, m)                KinFitter uses (px, py, pz, m/M0) with M0 = mass of the starting p4  <br />
		   <b>ECart</b> = (px, py, pz, e)               as in KinFitter <br />
		   <b>MCCart</b> = (px, py, pz, M)              as in KinFitter <br />
		   <b>Spher</b> = (p, theta, phi, m)            KinFitter uses (p, theta, phi, m/M0) with M0 = mass of the starting p4  <br />
		   <b>ESpher</b> = (p, theta, phi, e)           KinFitter uses (p, theta, phi, e/E0) with E0 = energy of the starting  <br />
                   <b>MCSpher</b> = (p, eta, phi, M)            as in KinFitter <br />
                   <b>MCPInvSpher</b> = (1/p, theta, phi, M)    as in KinFitter <br />
		   <b>EtEtaPhi</b> = (et, eta, phi, M == 0)     as in KinFitter <br />
		   <b>EtThetaPhi</b> = (et, theta, phi, M == 0) as in KinFitter <br />
                   <b>MomDev</b> = (p/P0, dp_theta, dp_phi, m/M0), so that P = [0]*|P0|*u_r + [1]*u_theta + [2]*u_phi <br />
                        the "u_<xyz>" are polar unit vectors around the initial momentum P0, their directions are: <br />
                        u_r ~ P0, u_phi ~ u_z x u_r, u_theta ~ u_r x u_phi  M0 is the mass of the initial 4-momentum.<br />
                   <b>EMomDev</b> = (p/P0, dp_theta, dp_phi, E/E0) with the P defined as for MomDev <br />
                   <b>MCMomDev</b> = (p/P0, dp_theta, dp_phi, M)   with the P defined as for MomDev <br />
                   <b>EScaledMomDev</b> = (p/P0, dp_theta, dp_phi,E/P=E0/P0) with the P defined as for MomDev, fixed E/p to E0/P0 <br />
                   <br />
		*/
     public:
        typedef math::XYZTLorentzVector LorentzVector;
        typedef float Scalar;

        enum Parametrization { Invalid=0, 
                // 4D = 0xN4
                Cart          = 0x04, 
                ECart         = 0x14, 
                Spher         = 0x24, 
                ESpher        = 0x34, 
                MomDev        = 0x44, 
                EMomDev       = 0x54, 
                // 3D =0xN3
                MCCart        = 0x03, 
                MCSpher       = 0x13, 
                MCPInvSpher   = 0x23, 
                EtEtaPhi      = 0x33, 
                EtThetaPhi    = 0x43,
                MCMomDev      = 0x53, 
                EScaledMomDev = 0x63
                };
        CandKinResolution() ; 

        /// Create a resolution object given a parametrization code, 
        /// a covariance matrix (streamed as a vector) and a vector of constraints.
        ///
        /// In the vector you can put either the full triangular block or just the diagonal terms
        ///
        /// The triangular block should be written in a way that the constructor
        ///  <code>AlgebraicSymMatrixNN(covariance.begin(), covariance.end())</code>
        /// works (N = 3 or 4)
        CandKinResolution(Parametrization parametrization, const std::vector<Scalar> &covariances, 
                            const std::vector<Scalar> &constraints = std::vector<Scalar>()) ;

        /// Fill in a cresolution object given a parametrization code, a covariance matrix and a vector of constraints.
        CandKinResolution(Parametrization parametrization, const AlgebraicSymMatrix44 &covariance,
                            const std::vector<Scalar> &constraints = std::vector<Scalar>()) ;
        ~CandKinResolution() ;

        /// Return the code of the parametrization used in this object
        Parametrization parametrization() const { return parametrization_; }

        /// Returns the number of free parameters in this parametrization
        uint32_t dimension() const { 
            return dimensionFrom(parametrization_);
        }

        /// Returns the full covariance matrix
        const AlgebraicSymMatrix44 & covariance()  const { 
            return covmatrix_; 
        }

        /// The constraints associated with this parametrization
        const std::vector<Scalar> & constraints() const { return constraints_; }

        /// Resolution on eta, given the 4-momentum of the associated Candidate
	double resolEta(const LorentzVector &p4) const ;

        /// Resolution on theta, given the 4-momentum of the associated Candidate
	double resolTheta(const LorentzVector &p4) const ;

        /// Resolution on phi, given the 4-momentum of the associated Candidate
	double resolPhi(const LorentzVector &p4) const ;

        /// Resolution on energy, given the 4-momentum of the associated Candidate
	double resolE(const LorentzVector &p4) const ;

        /// Resolution on et, given the 4-momentum of the associated Candidate
	double resolEt(const LorentzVector &p4) const ;

        /// Resolution on the invariant mass, given the 4-momentum of the associated Candidate
        /// Warning: returns 0 for mass-constrained parametrizations.
	double resolM(const LorentzVector &p4) const ;

        /// Resolution on p, given the 4-momentum of the associated Candidate
	double resolP(const LorentzVector &p4) const ;

        /// Resolution on pt, given the 4-momentum of the associated Candidate
	double resolPt(const LorentzVector &p4) const ;

        /// Resolution on 1/p, given the 4-momentum of the associated Candidate
	double resolPInv(const LorentzVector &p4) const ;

        /// Resolution on px, given the 4-momentum of the associated Candidate
	double resolPx(const LorentzVector &p4) const ;

        /// Resolution on py, given the 4-momentum of the associated Candidate
	double resolPy(const LorentzVector &p4) const ;

        /// Resolution on pz, given the 4-momentum of the associated Candidate
	double resolPz(const LorentzVector &p4) const ;

        static int dimensionFrom(Parametrization parametrization) {
          return (static_cast<uint32_t>(parametrization) & 0x0F);
        }

        static void fillMatrixFrom( Parametrization parametrization, const std::vector<Scalar>& covariances,
                                    AlgebraicSymMatrix44& covmatrix);

     private:
        // persistent 
        /// Parametrization code
        Parametrization parametrization_;
        /// Matrix, streamed as a vector
        std::vector<Scalar> covariances_;
        /// Constraints
        std::vector<Scalar> constraints_;

        // transient

        /// Transient copy of the full 4x4 covariance matrix
        AlgebraicSymMatrix44 covmatrix_;

        //methods

        /// Fill matrix from vector
        void fillMatrix() ;

        /// Fill vectoor from matrix
        void fillVector() ;
  };

  typedef std::vector<CandKinResolution>   CandKinResolutionCollection;
  typedef edm::ValueMap<CandKinResolution> CandKinResolutionValueMap;
}

#endif

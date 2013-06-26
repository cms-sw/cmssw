#include "CommonTools/ParticleFlow/interface/PFMETAlgo.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;
using namespace reco;
using namespace math;
using namespace pf2pat;

PFMETAlgo::PFMETAlgo(const edm::ParameterSet& iConfig) {
  

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  hfCalibFactor_ = 
    iConfig.getParameter<double>("hfCalibFactor");

}



PFMETAlgo::~PFMETAlgo() { }

reco::MET  PFMETAlgo::produce( const reco::PFCandidateCollection& pfCandidates) {
  
  double sumEx = 0;
  double sumEy = 0;
  double sumEt = 0;

  for( unsigned i=0; i<pfCandidates.size(); i++ ) {

    const reco::PFCandidate& cand = pfCandidates[i];
    
    double E = cand.energy();

    /// HF calibration factor (in 31X applied by PFProducer)
    if( cand.particleId()==PFCandidate::h_HF || 
	cand.particleId()==PFCandidate::egamma_HF ) 
      E *= hfCalibFactor_;

    double phi = cand.phi();
    double cosphi = cos(phi);
    double sinphi = sin(phi);

    double theta = cand.theta();
    double sintheta = sin(theta);
    
    double et = E*sintheta;
    double ex = et*cosphi;
    double ey = et*sinphi;
    
    sumEx += ex;
    sumEy += ey;
    sumEt += et;
  }
  
  double Et = sqrt( sumEx*sumEx + sumEy*sumEy);
  XYZTLorentzVector missingEt( -sumEx, -sumEy, 0, Et);
  
  if(verbose_) {
    cout<<"PFMETAlgo: mEx, mEy, mEt = "
	<< missingEt.X() <<", "
	<< missingEt.Y() <<", "
	<< missingEt.T() <<endl;
  }

  XYZPoint vertex; // dummy vertex
  return MET(sumEt, missingEt, vertex);

 
}



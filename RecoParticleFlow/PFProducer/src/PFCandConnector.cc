#include "RecoParticleFlow/PFProducer/interface/PFCandConnector.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h" 
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h" 

using namespace reco;
using namespace std;

const double PFCandConnector::pion_mass2 = 0.0194;

const reco::PFCandidate::Flags PFCandConnector::fT_TO_DISP_ = PFCandidate::T_TO_DISP;
const reco::PFCandidate::Flags PFCandConnector::fT_FROM_DISP_ = PFCandidate::T_FROM_DISP;

std::auto_ptr<reco::PFCandidateCollection>
PFCandConnector::connect(std::auto_ptr<PFCandidateCollection>& pfCand) {
   

  if(pfC_.get() ) pfC_->clear();
  else 
    pfC_.reset( new PFCandidateCollection );

  bMask_.clear();
  bMask_.resize(pfCand->size(), false);

   
  // loop on primary
  if (bCorrect_){
    if(debug_){
      cout << "" << endl;
      cout << "==================== ------------------------------ ===============" << endl;
      cout << "====================         Cand Connector         ===============" << endl;
      cout << "==================== ------------------------------ ===============" << endl;
      cout << "====================   \tfor " <<  pfCand->size() << " Candidates\t =============" << endl; 
    }

    for( unsigned int ce1=0; ce1 < pfCand->size(); ++ce1){
      if ( isPrimaryNucl(pfCand->at(ce1)) ){

	if (debug_) 
	  cout << "" << endl << "Nuclear Interaction w Primary Candidate " << ce1 
	       << " " << pfCand->at(ce1) << endl
	       << " based on the Track " << pfCand->at(ce1).trackRef().key()
	       << " w pT = " << pfCand->at(ce1).trackRef()->pt()
	       << " #pm " << pfCand->at(ce1).trackRef()->ptError() << " %" << endl;


	analyseNuclearWPrim(pfCand, ce1);
	    
	if (debug_) 
	  cout << "After Connection the candidate " << ce1 
	       << " is " << pfCand->at(ce1) << endl << endl;


      }

    }

    for( unsigned int ce1=0; ce1 < pfCand->size(); ++ce1){
      if ( !bMask_[ce1] && isSecondaryNucl(pfCand->at(ce1)) ){
	if (debug_) 
	  cout << "" << endl << "Nuclear Interaction w no Primary Candidate " << ce1 
	       << " " << pfCand->at(ce1) << endl
	       << " based on the Track " << pfCand->at(ce1).trackRef().key()
	       << " w pT = " << pfCand->at(ce1).trackRef()->pt()
	       << " #pm " << pfCand->at(ce1).trackRef()->ptError() << " %" << endl;
		
	analyseNuclearWSec(pfCand, ce1);

	if (debug_) 
	  cout << "After Connection the candidate " << ce1 
	       << " is " << pfCand->at(ce1) << endl << endl;

      }
    }   

  }
    

  for( unsigned int ce1=0; ce1 < pfCand->size(); ++ce1)
    if (!bMask_[ce1]) pfC_->push_back(pfCand->at(ce1));


  if(debug_ && bCorrect_) cout << "==================== ------------------------------ ===============" << endl<< endl << endl;
   
  return pfC_;


}

void 
PFCandConnector::analyseNuclearWPrim(std::auto_ptr<PFCandidateCollection>& pfCand, unsigned int ce1) {


  PFDisplacedVertexRef ref1, ref2, ref1_bis;  

  PFCandidate primaryCand = pfCand->at(ce1);

  // ------- look for the little friends -------- //

  math::XYZTLorentzVectorD momentumPrim = primaryCand.p4();
  math::XYZTLorentzVectorD momentumSec;

  ref1 = primaryCand.displacedVertexRef(fT_TO_DISP_);

  for( unsigned int ce2=0; ce2 < pfCand->size(); ++ce2) {
    if (ce2 != ce1 && isSecondaryNucl(pfCand->at(ce2))){
       
      ref2 = (pfCand->at(ce2)).displacedVertexRef(fT_FROM_DISP_);

      if (ref1 == ref2) {
	 
	if (debug_) cout << "\t here is a Secondary Candidate " << ce2  
			 << " " << pfCand->at(ce2) << endl
			 << "\t based on the Track " << pfCand->at(ce2).trackRef().key()
			 << " w pT = " << pfCand->at(ce2).trackRef()->pt()
			 << " #pm " << pfCand->at(ce2).trackRef()->ptError() << " %" << endl;
	if(isPrimaryNucl(pfCand->at(ce2))){
	  if (debug_) cout << "\t\t but it is also a Primary Candidate " << ce2 << endl;

	  ref1_bis = (pfCand->at(ce2)).displacedVertexRef(fT_TO_DISP_);
	  if(ref1_bis.isNonnull()) analyseNuclearWPrim(pfCand, ce2);  
	}
	momentumSec += (pfCand->at(ce2)).p4();
	bMask_[ce2] = true;
      
      }
    }
  }

  double dpt = pfCand->at(ce1).trackRef()->ptError()/pfCand->at(ce1).trackRef()->pt()*100; 
  
  
  if( ( ref1->isTherePrimaryTracks() ||
	( ref1->isThereMergedTracks() && momentumPrim.E() > momentumSec.E() ) ) 
      && dpt<20 ) {
      

    if (bCalibPrimary_){
	double factor = rescaleFactor( momentumPrim.pt() ); 
	momentumSec += (1-factor)*momentumPrim;
    }


	pfCand->at(ce1).rescaleMomentum(momentumSec.E()/momentumPrim.E()); 
	return;

  }						
    
  if (bCalibSecondary_){
    double factor = rescaleFactor( momentumSec.pt() );
    momentumSec *= factor;  
  }

  math::XYZVector prim =  ref1->primaryDirection();
     
  if (prim.Mag2() < 0.1){
    pfCand->at(ce1).setP4(momentumSec);
  } else {
    double px = momentumSec.P()*prim.x();
    double py = momentumSec.P()*prim.y();
    double pz = momentumSec.P()*prim.z();
    double E  = sqrt(px*px + py*py + pz*pz + pion_mass2);
     
    math::XYZTLorentzVectorD momentum(px, py, pz, E);
     
    pfCand->at(ce1).setP4(momentum);
  }
   

}




void 
PFCandConnector::analyseNuclearWSec(std::auto_ptr<PFCandidateCollection>& pfCand, unsigned int ce1){

  PFDisplacedVertexRef ref1, ref2;  

  PFCandidate secondaryCand = pfCand->at(ce1);

  // ------- look for the little friends -------- //

  math::XYZTLorentzVectorD momentumSec = secondaryCand.p4();

  ref1 = secondaryCand.displacedVertexRef(fT_FROM_DISP_);

  for( unsigned int ce2=ce1+1; ce2 < pfCand->size(); ++ce2) {
    if (isSecondaryNucl(pfCand->at(ce2))){
      ref2 = (pfCand->at(ce2)).displacedVertexRef(fT_FROM_DISP_);

      if (ref1 == ref2) {
	 
	if (debug_) cout << "\t here is a Secondary Candidate " << ce2  
			 << " " << pfCand->at(ce2) << endl
			 << "\t based on the Track " << pfCand->at(ce2).trackRef().key()
			 << " w pT = " << pfCand->at(ce2).trackRef()->pt()
			 << " #pm " << pfCand->at(ce2).trackRef()->ptError() << " %" << endl;

	momentumSec += (pfCand->at(ce2)).p4();
	bMask_[ce2] = true;
      }
    }
  }
   
  
  math::XYZVector primDir =  ref1->primaryDirection();
    
  if (bCalibSecondary_){
    double factor = rescaleFactor( momentumSec.pt() );
    momentumSec *= factor;  
  }


  if (primDir.Mag2() < 0.1){
    pfCand->at(ce1).setP4(momentumSec);
  } else {
    double px = momentumSec.P()*primDir.x();
    double py = momentumSec.P()*primDir.y();
    double pz = momentumSec.P()*primDir.z();
    double E  = sqrt(px*px + py*py + pz*pz + pion_mass2);
     
    math::XYZTLorentzVectorD momentum(px, py, pz, E);
     
    pfCand->at(ce1).setP4(momentum);
  }

}

bool 
PFCandConnector::isSecondaryNucl( const PFCandidate& pf ) const {
  
  PFDisplacedVertexRef ref1;
  // nuclear
  if( pf.flag( fT_FROM_DISP_ ) ) {
    ref1 = pf.displacedVertexRef(fT_FROM_DISP_);
    ref1->Dump();
    if (!ref1.isNonnull()) return false;
    else if (ref1->isNucl() || ref1->isNucl_Loose() || ref1->isNucl_Kink())
    return true;
  }
  
  return false;
}

bool 
PFCandConnector::isPrimaryNucl( const PFCandidate& pf ) const {

  PFDisplacedVertexRef ref1;
  
  // nuclear
  if( pf.flag( fT_TO_DISP_ ) ) {
    ref1 = pf.displacedVertexRef(fT_TO_DISP_);
    ref1->Dump();

    if (!ref1.isNonnull()) return false;
    else if (ref1->isNucl()|| ref1->isNucl_Loose() || ref1->isNucl_Kink())
    return true;
  }
  
  return false;
}


double
PFCandConnector::rescaleFactor( const double pt ) const {

  double factor = 1/( fConst_ - fNorm_*exp( -fExp_*pt ) );

  return factor;

}

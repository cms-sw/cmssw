#include "RecoParticleFlow/PFProducer/interface/PFCandConnector.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h" 
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h" 
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h" 


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

  //  debug_ = true;

  // loop on primary
  if (bCorrect_){
    if(debug_){
      cout << "" << endl;
      cout << "==================== ------------------------------ ===============" << endl;
      cout << "====================         Cand Connector         ===============" << endl;
      cout << "==================== ------------------------------ ===============" << endl;
      cout << "====================   \tfor " <<  pfCand->size() << " Candidates\t =============" << endl; 
      cout << "==================== primary calibrated " << bCalibPrimary_ << "=============" << endl;
    }

    for( unsigned int ce1=0; ce1 < pfCand->size(); ++ce1){
      if ( isPrimaryNucl(pfCand->at(ce1)) ){

	if (debug_) 
	  cout << "" << endl << "Nuclear Interaction w Primary Candidate " << ce1 
	       << " " << pfCand->at(ce1) << endl
	       << " based on the Track " << pfCand->at(ce1).trackRef().key()
	       << " w pT = " << pfCand->at(ce1).trackRef()->pt()
	       << " #pm " << pfCand->at(ce1).trackRef()->ptError()/pfCand->at(ce1).trackRef()->pt()*100 << " %" 
	       << " ECAL = " << pfCand->at(ce1).ecalEnergy() 
	       << " HCAL = " << pfCand->at(ce1).hcalEnergy() << endl;

	if (debug_) (pfCand->at(ce1)).displacedVertexRef(fT_TO_DISP_)->Dump();

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
	       << " #pm " << pfCand->at(ce1).trackRef()->ptError() << " %" 
	       << " ECAL = " << pfCand->at(ce1).ecalEnergy() 
	       << " HCAL = " << pfCand->at(ce1).hcalEnergy() 
	       << " dE(Trk-CALO) = " << pfCand->at(ce1).trackRef()->p()-pfCand->at(ce1).ecalEnergy()-pfCand->at(ce1).hcalEnergy() 
	       << " Nmissing hits = " << pfCand->at(ce1).trackRef()->trackerExpectedHitsOuter().numberOfHits() << endl;

	if (debug_) (pfCand->at(ce1)).displacedVertexRef(fT_FROM_DISP_)->Dump();
		
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

  momentumSec = momentumPrim/momentumPrim.E()*(primaryCand.ecalEnergy() + primaryCand.hcalEnergy());

  map<double, math::XYZTLorentzVectorD> candidatesWithTrackExcess;
  map<double, math::XYZTLorentzVectorD> candidatesWithoutCalo;
  
    
  ref1 = primaryCand.displacedVertexRef(fT_TO_DISP_);
  
  for( unsigned int ce2=0; ce2 < pfCand->size(); ++ce2) {
    if (ce2 != ce1 && isSecondaryNucl(pfCand->at(ce2))){
      
      ref2 = (pfCand->at(ce2)).displacedVertexRef(fT_FROM_DISP_);
      
      if (ref1 == ref2) {
	
	if (debug_) cout << "\t here is a Secondary Candidate " << ce2  
			 << " " << pfCand->at(ce2) << endl
			 << "\t based on the Track " << pfCand->at(ce2).trackRef().key()
			 << " w p = " << pfCand->at(ce2).trackRef()->p()
			 << " w pT = " << pfCand->at(ce2).trackRef()->pt()
			 << " #pm " << pfCand->at(ce2).trackRef()->ptError() << " %"
			 << " ECAL = " << pfCand->at(ce2).ecalEnergy() 
			 << " HCAL = " << pfCand->at(ce2).hcalEnergy() 
			 << " dE(Trk-CALO) = " << pfCand->at(ce2).trackRef()->p()-pfCand->at(ce2).ecalEnergy()-pfCand->at(ce2).hcalEnergy() 
			 << " Nmissing hits = " << pfCand->at(ce2).trackRef()->trackerExpectedHitsOuter().numberOfHits() << endl;

	if(isPrimaryNucl(pfCand->at(ce2))){
	  if (debug_) cout << "\t\t but it is also a Primary Candidate " << ce2 << endl;
	  
	  ref1_bis = (pfCand->at(ce2)).displacedVertexRef(fT_TO_DISP_);
	  if(ref1_bis.isNonnull()) analyseNuclearWPrim(pfCand, ce2);  
	}

	double caloEn = pfCand->at(ce2).ecalEnergy() + pfCand->at(ce2).hcalEnergy();
	double deltaEn =  pfCand->at(ce2).p4().E() - caloEn;
	int nMissOuterHits = pfCand->at(ce2).trackRef()->trackerExpectedHitsOuter().numberOfHits();
	

	if (deltaEn > 1  && nMissOuterHits > 1) {
	  math::XYZTLorentzVectorD momentumToAdd = pfCand->at(ce2).p4()*caloEn/pfCand->at(ce2).p4().E();
	  momentumSec += momentumToAdd;
	  if (debug_) cout << "The difference track-calo s really large and the track miss at least 2 hits. A secondary NI may have happened. Let's trust the calo energy" << endl << "add " << momentumToAdd << endl;
	  
	} else {
	  if (caloEn > 0.01 && deltaEn > 1  && nMissOuterHits > 0) {
	    math::XYZTLorentzVectorD momentumExcess = pfCand->at(ce2).p4()*deltaEn/pfCand->at(ce2).p4().E();
	    candidatesWithTrackExcess[pfCand->at(ce2).trackRef()->pt()/pfCand->at(ce2).trackRef()->ptError()] =  momentumExcess;
	  }
	  else if(caloEn < 0.01) candidatesWithoutCalo[pfCand->at(ce2).trackRef()->pt()/pfCand->at(ce2).trackRef()->ptError()] = pfCand->at(ce2).p4();
	  momentumSec += (pfCand->at(ce2)).p4();
	}

	bMask_[ce2] = true;


	
      }
    }
  }
  
  
  // We have more primary energy than secondary: reject all secondary tracks which have no calo energy attached.
  
  
  if (momentumPrim.E() < momentumSec.E()){
    
    if(debug_) cout << "Size of 0 calo Energy secondary candidates" << candidatesWithoutCalo.size() << endl;
    for( map<double, math::XYZTLorentzVectorD>::iterator iter = candidatesWithoutCalo.begin(); iter != candidatesWithoutCalo.end() && momentumPrim.E() < momentumSec.E(); iter++)
      if (momentumSec.E() > iter->second.E()+0.1) {
	momentumSec -= iter->second;   
    
	if(debug_) cout << "\t Remove a SecondaryCandidate with 0 calo energy " << iter->second << endl;
	
	if(debug_) cout << "momentumPrim.E() = " << momentumPrim.E() << " and momentumSec.E() = " <<  momentumSec.E() << endl; 
	    
      }

  }

      
  if (momentumPrim.E() < momentumSec.E()){
    if(debug_) cout << "0 Calo Energy rejected but still not sufficient. Size of not enough calo Energy secondary candidates" << candidatesWithTrackExcess.size() << endl;
     for( map<double, math::XYZTLorentzVectorD>::iterator iter = candidatesWithTrackExcess.begin(); iter != candidatesWithTrackExcess.end() && momentumPrim.E() < momentumSec.E(); iter++)
      if (momentumSec.E() > iter->second.E()+0.1) momentumSec -= iter->second;   
     
  }


    

  double dpt = pfCand->at(ce1).trackRef()->ptError()/pfCand->at(ce1).trackRef()->pt()*100; 

  if (momentumSec.E() < 0.1) {
    bMask_[ce1] = true;
    return;
  }
  
  // Rescale the secondary candidates to account for the loss of energy, but only if we can trust the primary track:
  // if it has more energy than secondaries and is precise enough and secondary exist and was not eaten or rejected during the PFAlgo step.
  
  if( ( (ref1->isTherePrimaryTracks() && dpt<dptRel_PrimaryTrack_)  || (ref1->isThereMergedTracks() && dpt<dptRel_MergedTrack_) ) && momentumPrim.E() > momentumSec.E() && momentumSec.E() > 0.1) {
      
    if (bCalibPrimary_){
      double factor = rescaleFactor( momentumPrim.Pt(), momentumSec.E()/momentumPrim.E()); 
      if (debug_) cout << "factor = " << factor << endl;
      if (factor*momentumPrim.Pt() < momentumSec.Pt()) momentumSec = momentumPrim;
      else momentumSec += (1-factor)*momentumPrim;
    }
    
    double px = momentumPrim.Px()*momentumSec.P()/momentumPrim.P();
    double py = momentumPrim.Py()*momentumSec.P()/momentumPrim.P();
    double pz = momentumPrim.Pz()*momentumSec.P()/momentumPrim.P();
    double E  = sqrt(px*px + py*py + pz*pz + pion_mass2);
    math::XYZTLorentzVectorD momentum(px, py, pz, E);
    pfCand->at(ce1).setP4(momentum);
    
    return;
    
  } else {

    math::XYZVector primDir =  ref1->primaryDirection();
    
    if (primDir.Mag2() < 0.1){
      pfCand->at(ce1).setP4(momentumSec);
      return;
    } else {
      double px = momentumSec.P()*primDir.x();
      double py = momentumSec.P()*primDir.y();
      double pz = momentumSec.P()*primDir.z();
      double E  = sqrt(px*px + py*py + pz*pz + pion_mass2);
      
      math::XYZTLorentzVectorD momentum(px, py, pz, E);
      pfCand->at(ce1).setP4(momentum);
      return;
    }
  }
  
  

}




void 
PFCandConnector::analyseNuclearWSec(std::auto_ptr<PFCandidateCollection>& pfCand, unsigned int ce1){

  PFDisplacedVertexRef ref1, ref2;  


  // Check if the track excess was not too large and track may miss some outer hits. This may point to a secondary NI.

  double caloEn = pfCand->at(ce1).ecalEnergy() + pfCand->at(ce1).hcalEnergy();
  double deltaEn =  pfCand->at(ce1).p4().E() - caloEn;
  int nMissOuterHits = pfCand->at(ce1).trackRef()->trackerExpectedHitsOuter().numberOfHits();  


  ref1 = pfCand->at(ce1).displacedVertexRef(fT_FROM_DISP_);

  // ------- check if an electron or a muon vas spotted as incoming track -------- //
  // ------- this mean probably that the NI was fake thus we do not correct it -------- /

  if (ref1->isTherePrimaryTracks() || ref1->isThereMergedTracks()){
    
    std::vector<reco::Track> refittedTracks = ref1->refittedTracks();
    for(unsigned it = 0; it < refittedTracks.size(); it++){
      reco::TrackBaseRef primaryBaseRef = ref1->originalTrack(refittedTracks[it]);  
      if (ref1->isIncomingTrack(primaryBaseRef))
	if (debug_) cout << "There is a Primary track ref with pt = " << primaryBaseRef->pt()<< endl;

      for( unsigned int ce=0; ce < pfCand->size(); ++ce){
	//	  cout << "PFCand Id = " << (pfCand->at(ce)).particleId() << endl;
	if ((pfCand->at(ce)).particleId() == reco::PFCandidate::e || (pfCand->at(ce)).particleId() == reco::PFCandidate::mu) {

	  if (debug_) cout << " It is an electron and it has a ref to a track " << (pfCand->at(ce)).trackRef().isNonnull() << endl;
	  

	  if ( (pfCand->at(ce)).trackRef().isNonnull() ){
	    reco::TrackRef tRef = (pfCand->at(ce)).trackRef();
	    reco::TrackBaseRef bRef(tRef);
	    if (debug_) cout << "With Track Ref pt = " << (pfCand->at(ce)).trackRef()->pt() << endl;

	    if (bRef == primaryBaseRef) {
	      if (debug_ && (pfCand->at(ce)).particleId() == reco::PFCandidate::e) cout << "It is a NI from electron. NI Discarded. Just release the candidate." << endl; 
	      if (debug_ && (pfCand->at(ce)).particleId() == reco::PFCandidate::mu) cout << "It is a NI from muon. NI Discarded. Just release the candidate" << endl; 
	
	      // release the track but take care of not overcounting bad tracks. In fact those tracks was protected against destruction in 
	      // PFAlgo. Now we treat them as if they was treated in PFAlgo

	      if (caloEn < 0.1 && pfCand->at(ce1).trackRef()->ptError() > ptErrorSecondary_) { 
		cout << "discarded track since no calo energy and ill measured" << endl; 
		bMask_[ce1] = true;
	      }
	      if (caloEn > 0.1 && deltaEn >ptErrorSecondary_  && pfCand->at(ce1).trackRef()->ptError() > ptErrorSecondary_) { 
		cout << "rescaled momentum of the track since no calo energy and ill measured" << endl;
		
		double factor = caloEn/pfCand->at(ce1).p4().E();
		pfCand->at(ce1).rescaleMomentum(factor);
	      }

	      return;
	    }
	  }
	}
      }
    }
  }


  PFCandidate secondaryCand = pfCand->at(ce1);

  math::XYZTLorentzVectorD momentumSec = secondaryCand.p4();

  if (deltaEn > ptErrorSecondary_  && nMissOuterHits > 1) {
    math::XYZTLorentzVectorD momentumToAdd = pfCand->at(ce1).p4()*caloEn/pfCand->at(ce1).p4().E();
    momentumSec = momentumToAdd;
    if (debug_) cout << "The difference track-calo s really large and the track miss at least 2 hits. A secondary NI may have happened. Let's trust the calo energy" << endl << "add " << momentumToAdd << endl;  
  }


  // ------- look for the little friends -------- //
  for( unsigned int ce2=ce1+1; ce2 < pfCand->size(); ++ce2) {
    if (isSecondaryNucl(pfCand->at(ce2))){
      ref2 = (pfCand->at(ce2)).displacedVertexRef(fT_FROM_DISP_);

      if (ref1 == ref2) {
	 
	if (debug_) cout << "\t here is a Secondary Candidate " << ce2  
			 << " " << pfCand->at(ce2) << endl
			 << "\t based on the Track " << pfCand->at(ce2).trackRef().key()
			 << " w pT = " << pfCand->at(ce2).trackRef()->pt()
			 << " #pm " << pfCand->at(ce2).trackRef()->ptError() << " %" 
			 << " ECAL = " << pfCand->at(ce2).ecalEnergy() 
			 << " HCAL = " << pfCand->at(ce2).hcalEnergy() 
			 << " dE(Trk-CALO) = " << pfCand->at(ce2).trackRef()->p()-pfCand->at(ce2).ecalEnergy()-pfCand->at(ce2).hcalEnergy() 
			 << " Nmissing hits = " << pfCand->at(ce2).trackRef()->trackerExpectedHitsOuter().numberOfHits() << endl;

	double caloEn = pfCand->at(ce2).ecalEnergy() + pfCand->at(ce2).hcalEnergy();
	double deltaEn =  pfCand->at(ce2).p4().E() - caloEn;
	int nMissOuterHits = pfCand->at(ce2).trackRef()->trackerExpectedHitsOuter().numberOfHits();  
	if (deltaEn > ptErrorSecondary_ && nMissOuterHits > 1) {
	  math::XYZTLorentzVectorD momentumToAdd = pfCand->at(ce2).p4()*caloEn/pfCand->at(ce2).p4().E();
	  momentumSec += momentumToAdd;
	  if (debug_) cout << "The difference track-calo s really large and the track miss at least 2 hits. A secondary NI may have happened. Let's trust the calo energy" << endl << "add " << momentumToAdd << endl;  
	} else {
	  momentumSec += (pfCand->at(ce2)).p4();	
	}

	bMask_[ce2] = true;
      }
    }
  }
  
  


  math::XYZVector primDir =  ref1->primaryDirection();

  if (primDir.Mag2() < 0.1){
    pfCand->at(ce1).setP4(momentumSec);
    return;
  } else {
    double px = momentumSec.P()*primDir.x();
    double py = momentumSec.P()*primDir.y();
    double pz = momentumSec.P()*primDir.z();
    double E  = sqrt(px*px + py*py + pz*pz + pion_mass2);
     
    math::XYZTLorentzVectorD momentum(px, py, pz, E);
     
    pfCand->at(ce1).setP4(momentum);
    return;
  }

}

bool 
PFCandConnector::isSecondaryNucl( const PFCandidate& pf ) const {
  
  PFDisplacedVertexRef ref1;
  // nuclear
  if( pf.flag( fT_FROM_DISP_ ) ) {
    ref1 = pf.displacedVertexRef(fT_FROM_DISP_);
    //    ref1->Dump();
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
    //ref1->Dump();

    if (!ref1.isNonnull()) return false;
    else if (ref1->isNucl()|| ref1->isNucl_Loose() || ref1->isNucl_Kink())
    return true;
  }
  
  return false;
}


double
PFCandConnector::rescaleFactor( const double pt, const double cFrac ) const {


  /*
    LOG NORMAL FIT
 FCN=35.8181 FROM MIGRAD    STATUS=CONVERGED     257 CALLS         258 TOTAL
 EDM=8.85763e-09    STRATEGY= 1      ERROR MATRIX ACCURATE
  EXT PARAMETER                                   STEP         FIRST
  NO.   NAME      VALUE            ERROR          SIZE      DERIVATIVE
   1  p0           7.99434e-01   2.77264e-02   6.59108e-06   9.80247e-03
   2  p1           1.51303e-01   2.89981e-02   1.16775e-05   6.99035e-03
   3  p2          -5.03829e-01   2.87929e-02   1.90070e-05   1.37015e-03
   4  p3           4.54043e-01   5.00908e-02   3.17625e-05   3.86622e-03
   5  p4          -4.61736e-02   8.07940e-03   3.25775e-06  -1.37247e-02
  */


  /*
    FCN=34.4051 FROM MIGRAD    STATUS=CONVERGED     221 CALLS         222 TOTAL
    EDM=1.02201e-09    STRATEGY= 1  ERROR MATRIX UNCERTAINTY   2.3 per cent

   fConst
   1  p0           7.99518e-01   2.23519e-02   1.41523e-06   4.05975e-04
   2  p1           1.44619e-01   2.39398e-02  -7.68117e-07  -2.55775e-03

   fNorm
   3  p2          -5.16571e-01   3.12362e-02   5.74932e-07   3.42292e-03
   4  p3           4.69055e-01   5.09665e-02   1.94353e-07   1.69031e-03

   fExp
   5  p4          -5.18044e-02   8.13458e-03   4.29815e-07  -1.07624e-02
  */

  double fConst, fNorm, fExp;

  fConst = fConst_[0] + fConst_[1]*cFrac;
  fNorm = fNorm_[0] - fNorm_[1]*cFrac;
  fExp = fExp_[0];  

  double factor = fConst - fNorm*exp( -fExp*pt );

  return factor;

}

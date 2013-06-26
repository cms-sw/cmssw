#ifndef DataFormats_BTauReco_PFIsolatedTauTagInfo_h
#define DataFormats_BTauReco_PFIsolatedTauTagInfo_h

/* class PFIsolatedTauTagInfo
 * Extended object for the Particle Flow Tau Isolation algorithm,
 * contains the result and the methods used in the PFConeIsolation Algorithm
 * created: Apr 21 2007,
 * revised: Jun 23 2007,
 * authors: Simone Gennai, Ludovic Houchu
 */

#include <math.h>

#include "DataFormats/BTauReco/interface/RefMacros.h"

#include "Math/GenVector/PxPyPzE4D.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"






const int PFChargedHadrCand_codenumber=1;
const int PFNeutrHadrCand_codenumber=5;
const int PFGammaCand_codenumber=4;
 
const int PFRecTrack_codenumber=1;
const int PFRecECALClus_codenumber=4;
const int PFRecHCALClus_codenumber=5;

namespace reco{ 
  class PFIsolatedTauTagInfo : public BaseTagInfo{
  public:
    PFIsolatedTauTagInfo() {}
    PFIsolatedTauTagInfo(PFCandidateRefVector PFCands){
      initialPFCands_=PFCands;
      PFCands_=PFCands;
      for(PFCandidateRefVector::const_iterator iPFCand=PFCands_.begin();iPFCand!=PFCands_.end();iPFCand++){
	if ((**iPFCand).particleId()==PFChargedHadrCand_codenumber) initialPFChargedHadrCands_.push_back(*iPFCand);
	if ((**iPFCand).particleId()==PFNeutrHadrCand_codenumber) initialPFNeutrHadrCands_.push_back(*iPFCand);
	if ((**iPFCand).particleId()==PFGammaCand_codenumber) initialPFGammaCands_.push_back(*iPFCand);
      }
      PFChargedHadrCands_=initialPFChargedHadrCands_;
      PFNeutrHadrCands_=initialPFNeutrHadrCands_;
      PFGammaCands_=initialPFGammaCands_;
      alternatLorentzVect_.SetPx(NAN);
      alternatLorentzVect_.SetPy(NAN);
      alternatLorentzVect_.SetPz(NAN);
      alternatLorentzVect_.SetE(NAN);
      passedtrackerisolation_=false;
      passedECALisolation_=false;
    }
    virtual ~PFIsolatedTauTagInfo(){};
    virtual PFIsolatedTauTagInfo* clone()const{return new PFIsolatedTauTagInfo(*this);}
    
    //get the PFCandidates's which compose the PF jet and may be have been filtered by filterPFChargedHadrCands(.,.,.,.,.,.), filterPFNeutrHadrCands(.), filterPFGammaCands(.) member functions
    const PFCandidateRefVector& PFCands() const {return PFCands_;}
    const PFCandidateRefVector& PFChargedHadrCands() const {return PFChargedHadrCands_;}
    const PFCandidateRefVector& PFNeutrHadrCands() const {return PFNeutrHadrCands_;}
    const PFCandidateRefVector& PFGammaCands() const {return PFGammaCands_;}
    
    // rec. jet Lorentz-vector combining charged hadr. PFCandidate's and gamma PFCandidate's  
    math::XYZTLorentzVector alternatLorentzVect()const{return(alternatLorentzVect_);} 
    void setalternatLorentzVect(math::XYZTLorentzVector x){alternatLorentzVect_=x;}

    //the reference to the PFJet
    const PFJetRef& pfjetRef()const{return PFJetRef_;}
    void setpfjetRef(const PFJetRef x){PFJetRef_=x;}

    //JetTag::discriminator() computed with the parameters taken from the RecoTauTag/PFConeIsolation/data/pfConeIsolation.cfi file
    
    // true if a lead. PFCandidate exists and no charged hadron PFCandidate was found in an DR isolation ring around it (DR isolation ring limits defined in the RecoTauTag/PFConeIsolation/data/pfConeIsolation.cfi file
    bool passedtrackerisolation()const{return passedtrackerisolation_;}
    void setpassedtrackerisolation(bool x){passedtrackerisolation_=x;}

    // true if a lead. PFCandidate exists and no gamma PFCandidate was found in an DR isolation ring around it (DR isolation ring limits defined in the RecoTauTag/PFConeIsolation/data/pfConeIsolation.cfi file
    bool passedECALisolation()const{return passedECALisolation_;}
    void setpassedECALisolation(bool x){passedECALisolation_=x;}

    //methods to be used to recompute the isolation with a new set of parameters
    double discriminatorByIsolPFCandsN(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN=0)const;
    double discriminatorByIsolPFCandsN(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN=0)const;
    double discriminatorByIsolPFChargedHadrCandsN(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN=0)const;
    double discriminatorByIsolPFChargedHadrCandsN(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN=0)const;
    double discriminatorByIsolPFNeutrHadrCandsN(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN=0)const;
    double discriminatorByIsolPFNeutrHadrCandsN(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN=0)const;
    double discriminatorByIsolPFGammaCandsN(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN=0)const;
    double discriminatorByIsolPFGammaCandsN(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,int IsolPFCands_maxN=0)const;
    double discriminatorByIsolPFCandsEtSum(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum=0)const;
    double discriminatorByIsolPFCandsEtSum(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum=0)const;
    double discriminatorByIsolPFChargedHadrCandsEtSum(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum=0)const;
    double discriminatorByIsolPFChargedHadrCandsEtSum(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum=0)const;
    double discriminatorByIsolPFNeutrHadrCandsEtSum(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum=0)const;
    double discriminatorByIsolPFNeutrHadrCandsEtSum(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum=0)const;
    double discriminatorByIsolPFGammaCandsEtSum(float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum=0)const;
    double discriminatorByIsolPFGammaCandsEtSum(math::XYZVector myVector,float matchingcone_size,float signalcone_size,float isolcone_size,bool useOnlyChargedHadrforleadPFCand,float minPt_leadPFCand,float minPt_PFCand,float IsolPFCands_maxEtSum=0)const;
    
    // return all PFCandidate's in a cone of size "conesize" around a direction "myVector" 
    const PFCandidateRefVector PFCandsInCone(const math::XYZVector myVector,const float conesize,const float minPt)const;
    const PFCandidateRefVector PFChargedHadrCandsInCone(const math::XYZVector myVector,const float conesize,const float minPt)const;
    const PFCandidateRefVector PFNeutrHadrCandsInCone(const math::XYZVector myVector,const float conesize,const float minPt)const;
    const PFCandidateRefVector PFGammaCandsInCone(const math::XYZVector myVector,const float conesize,const float minPt)const;
    
    // return all PFCandidate's in a band defined by inner(size "innercone_size") and outer(size "outercone_size") cones around a direction "myVector" 
    const PFCandidateRefVector PFCandsInBand(const math::XYZVector myVector,const float innercone_size,const float outercone_size,const float minPt)const;
    const PFCandidateRefVector PFChargedHadrCandsInBand(const math::XYZVector myVector,const float innercone_size,const float outercone_size,const float minPt)const;
    const PFCandidateRefVector PFNeutrHadrCandsInBand(const math::XYZVector myVector,const float innercone_size,const float outercone_size,const float minPt)const;
    const PFCandidateRefVector PFGammaCandsInBand(const math::XYZVector myVector,const float innercone_size,const float outercone_size,const float minPt)const;
    
    //return the leading PFCandidate in a given cone around the jet axis or a given direction
    const PFCandidateRef leadPFCand(const float matchingcone_size, const float minPt)const;
    const PFCandidateRef leadPFCand(const math::XYZVector myVector,const float matchingcone_size, const float minPt)const;  
    const PFCandidateRef leadPFChargedHadrCand(const float matchingcone_size, const float minPt)const;
    const PFCandidateRef leadPFChargedHadrCand(const math::XYZVector myVector,const float matchingcone_size, const float minPt)const;  
    const PFCandidateRef leadPFNeutrHadrCand(const float matchingcone_size, const float minPt)const;
    const PFCandidateRef leadPFNeutrHadrCand(const math::XYZVector myVector,const float matchingcone_size, const float minPt)const;  
    const PFCandidateRef leadPFGammaCand(const float matchingcone_size, const float minPt)const;
    const PFCandidateRef leadPFGammaCand(const math::XYZVector myVector,const float matchingcone_size, const float minPt)const;  
 
    void filterPFChargedHadrCands(double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2,double ChargedHadrCand_tktorefpointDZ,bool UsePVconstraint,double PVtx_Z,bool UseOnlyChargedHadr_for_LeadCand,double LeadChargedHadrCandtoJet_MatchingConeSize,double LeadChargedHadrCand_minPt);
    void filterPFNeutrHadrCands(double NeutrHadrCand_HcalclusminEt);
    void filterPFGammaCands(double GammaCand_EcalclusminEt);
    void removefilters();
  private:
    PFJetRef PFJetRef_;
    PFCandidateRefVector initialPFCands_;
    PFCandidateRefVector PFCands_;
    PFCandidateRefVector initialPFChargedHadrCands_;
    PFCandidateRefVector PFChargedHadrCands_;
    PFCandidateRefVector initialPFNeutrHadrCands_;
    PFCandidateRefVector PFNeutrHadrCands_;
    PFCandidateRefVector initialPFGammaCands_;
    PFCandidateRefVector PFGammaCands_;
    math::XYZTLorentzVector alternatLorentzVect_;
    bool passedtrackerisolation_;
    bool passedECALisolation_;
  };

  DECLARE_EDM_REFS( PFIsolatedTauTagInfo )

}

#endif // DataFormats_BTauReco_PFIsolatedTauTagInfo_h

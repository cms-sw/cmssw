#ifndef BTauReco_PFTauTagIsolation_h
#define BTauReco_PFTauTagIsolation_h

/* class PFIsolatedTauTagInfo
 * Extended object for the Particle Flow Tau Isolation algorithm,
 * contains the result and the methods used in the PFConeIsolation Algorithm
 * created: Apr 21 2007,
 * revised: May 10 2007,
 * authors: Simone Gennai, Ludovic Houchu
 */
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/PFIsolatedTauTagInfoFwd.h"
#include "DataFormats/JetReco/interface/GenericJet.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "math.h"

using namespace std;
using namespace edm;
using namespace reco;

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
      discriminator_=NAN;
    }
    virtual ~PFIsolatedTauTagInfo(){};
    virtual PFIsolatedTauTagInfo* clone()const{return new PFIsolatedTauTagInfo(*this);}
    
    //get the PFCandidates's which compose the PF jet and may be have been filtered by filterPFChargedHadrCands(.,.,.,.,.,.), filterPFNeutrHadrCands(.), filterPFGammaCands(.) member functions
    const PFCandidateRefVector& PFCands() const {return PFCands_;}
    const PFCandidateRefVector& PFChargedHadrCands() const {return PFChargedHadrCands_;}
    const PFCandidateRefVector& PFNeutrHadrCands() const {return PFNeutrHadrCands_;}
    const PFCandidateRefVector& PFGammaCands() const {return PFGammaCands_;}
    
    //the reference to the GenericJet
    const GenericJetRef& genericjetRef()const{return GenericJetRef_;}
    void setgenericjetRef(const GenericJetRef x){GenericJetRef_=x;}

    //default discriminator, computed with the parameters taken from the RecoTauTag/PFConeIsolation/data/pfConeIsolation.cfi file
    double discriminator()const{return discriminator_;}
    void setdiscriminator(double x){discriminator_=x;}

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
 
    void filterPFChargedHadrCands(double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2,double ChargedHadrCand_tkmaxPVtxDZ,double PVtx_Z);
    void filterPFNeutrHadrCands(double NeutrHadrCand_HcalclusminEt);
    void filterPFGammaCands(double GammaCand_EcalclusminEt);
    void removefilters();
  private:
    GenericJetRef GenericJetRef_;
    PFCandidateRefVector initialPFCands_;
    PFCandidateRefVector PFCands_;
    PFCandidateRefVector initialPFChargedHadrCands_;
    PFCandidateRefVector PFChargedHadrCands_;
    PFCandidateRefVector initialPFNeutrHadrCands_;
    PFCandidateRefVector PFNeutrHadrCands_;
    PFCandidateRefVector initialPFGammaCands_;
    PFCandidateRefVector PFGammaCands_;
    double discriminator_;
  };
}

#endif

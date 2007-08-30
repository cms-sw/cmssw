#ifndef DataFormats_TauReco_PFTauTagInfo_h
#define DataFormats_TauReco_PFTauTagInfo_h

/* class PFTauTagInfo
 * the object of this class is created by RecoTauTag/RecoTau PFRecoTauTagInfoProducer EDProducer starting from JetTrackAssociations <a PFJet,a list of Track's> object
 *                          is the initial object for building a PFTau object
 * created: Aug 29 2007,
 * revised: 
 * authors: Ludovic Houchu
 */

#include <math.h>

#include "Math/GenVector/PxPyPzE4D.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTauTagInfoFwd.h"
#include "DataFormats/TauReco/interface/TauTagInfo.h"

using namespace std;
using namespace edm;
using namespace reco;

namespace reco{ 
  class PFTauTagInfo : public TauTagInfo {
  public:
    PFTauTagInfo(){
      alternatLorentzVect_.SetPx(NAN);
      alternatLorentzVect_.SetPy(NAN);
      alternatLorentzVect_.SetPz(NAN);
      alternatLorentzVect_.SetE(NAN);
    }
    virtual ~PFTauTagInfo(){};
    virtual PFTauTagInfo* clone()const{return new PFTauTagInfo(*this);}
    
    //get the PFCandidates's which compose the PF jet and were filtered by RecoTauTag/RecoTau/ PFRecoTauTagInfoAlgorithm::filteredPFChargedHadrCands(.,.,.,.,.,.), filteredPFNeutrHadrCands(.), filteredPFGammaCands(.) functions
    const PFCandidateRefVector& PFCands() const {return PFCands_;}
    void  setPFCands(const PFCandidateRefVector x){PFCands_=x;}
    const PFCandidateRefVector& PFChargedHadrCands() const {return PFChargedHadrCands_;}
    void  setPFChargedHadrCands(const PFCandidateRefVector x){PFChargedHadrCands_=x;}
    const PFCandidateRefVector& PFNeutrHadrCands() const {return PFNeutrHadrCands_;}
    void  setPFNeutrHadrCands(const PFCandidateRefVector x){PFNeutrHadrCands_=x;}
    const PFCandidateRefVector& PFGammaCands() const {return PFGammaCands_;}
    void  setPFGammaCands(const PFCandidateRefVector x){PFGammaCands_=x;}
    
    //the reference to the PFJet
    const PFJetRef& pfjetRef()const{return PFJetRef_;}
    void setpfjetRef(const PFJetRef x){PFJetRef_=x;}

  private:
    PFJetRef PFJetRef_;
    PFCandidateRefVector PFCands_;
    PFCandidateRefVector PFChargedHadrCands_;
    PFCandidateRefVector PFNeutrHadrCands_;
    PFCandidateRefVector PFGammaCands_;
    math::XYZTLorentzVector alternatLorentzVect_;
  };
}
#endif


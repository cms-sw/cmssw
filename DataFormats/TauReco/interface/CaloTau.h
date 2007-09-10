#ifndef DataFormats_TauReco_CaloTau_h
#define DataFormats_TauReco_CaloTau_h

/* class CaloTau
 * the object of this class is created by RecoTauTag/RecoTau CaloRecoTauProducer EDProducer starting from the CaloTauTagInfo object,
 *                          is a hadronic tau-jet candidate -built from a calo. jet- that analysts manipulate;
 * authors: Simone Gennai (simone.gennai@cern.ch), Ludovic Houchu (Ludovic.Houchu@cern.ch)
 * created: Jun 21 2007,
 * revised: Sep 4 2007
 */
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"

#include <limits>

using namespace reco;

namespace reco {
  class CaloTau : public BaseTau {
  public:
    CaloTau();
    CaloTau(Charge q, const LorentzVector &, const Point & = Point( 0, 0, 0 ) );
    virtual ~CaloTau(){}
    CaloTau* clone()const;
    
    const CaloTauTagInfoRef& caloTauTagInfoRef()const;
    void setcaloTauTagInfoRef(const CaloTauTagInfoRef);
    
    // signed transverse impact parameter significance of the lead Track
    float leadTracksignedSipt()const;
    void setleadTracksignedSipt(const float&);
    
    // invariant mass of the signal Tracks system
    float signalTracksInvariantMass()const;
    void setsignalTracksInvariantMass(const float&);
    
    // invariant mass of the Tracks system
    float TracksInvariantMass()const;
    void setTracksInvariantMass(const float&);

    // sum of Pt of the isolation annulus Tracks
    float isolationTracksPtSum()const;
    void setisolationTracksPtSum(const float&);
   
    // sum of Et of the isolation annulus ECAL hits
    float isolationECALhitsEtSum()const;
    void setisolationECALhitsEtSum(const float&);
  
    // Et of the highest Et HCAL hit
    float highestEtHCALhitEt()const;
    void sethighestEtHCALhitEt(const float&);    
  private:
    // check overlap with another candidate
    virtual bool overlap(const Candidate&d)const;
    CaloTauTagInfoRef CaloTauTagInfoRef_;  
    float leadTracksignedSipt_;
    float signalTracksInvariantMass_;
    float TracksInvariantMass_; 
    float isolationTracksPtSum_;
    float isolationECALhitsEtSum_;
    float highestEtHCALhitEt_;
  };
}
#endif

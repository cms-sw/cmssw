#ifndef DataFormats_TauReco_CaloTau_h
#define DataFormats_TauReco_CaloTau_h

/* class CaloTau
 * the object of this class is created by RecoTauTag/RecoTau CaloRecoTauProducer EDProducer starting from the CaloTauTagInfo object,
 *                          is a hadronic tau-jet candidate -built from a calo. jet- that analysts manipulate;
 * authors: Simone Gennai (simone.gennai@cern.ch), Ludovic Houchu (Ludovic.Houchu@cern.ch)
 * created: Jun 21 2007,
 * revised: Feb 20 2007
 */
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"

#include <limits>

namespace reco {
  class CaloTau : public BaseTau {
  public:
    CaloTau();
    CaloTau(Charge q, const LorentzVector &, const Point & = Point( 0, 0, 0 ) );
    ~CaloTau() override{}
    CaloTau* clone()const override;
    
    const CaloTauTagInfoRef& caloTauTagInfoRef()const;
    void setcaloTauTagInfoRef(const CaloTauTagInfoRef);

    const CaloJetRef rawJetRef() const;

    // signed transverse impact parameter significance of leading Track ; NaN if no leading Track
    float leadTracksignedSipt()const;
    void setleadTracksignedSipt(const float&);

    // sum of Et of HCAL hits inside a 3x3 calo. tower matrix centered on direction of propag. leading Track - ECAL inner surf. contact point ; NaN if no leading Track or if invalid propag. leading Track - ECAL inner surf. contact point
    float leadTrackHCAL3x3hitsEtSum()const;
    void setleadTrackHCAL3x3hitsEtSum(const float&);

    // |DEta| between direction of propag. leading Track - ECAL inner surf. contact point and direction of highest Et hit among HCAL hits inside a 3x3 calo. tower matrix centered on direction of propag. leading Track - ECAL inner surf. contact point ; NaN if no leading Track or if invalid propag. leading Track - ECAL inner surf. contact point
    float leadTrackHCAL3x3hottesthitDEta()const;
    void setleadTrackHCAL3x3hottesthitDEta(const float&);
    
    // invariant mass of the system of Tracks inside a signal cone around leading Track ; NaN if no leading Track
    float signalTracksInvariantMass()const;
    void setsignalTracksInvariantMass(const float&);
    
    // invariant mass of the system of Tracks ; NaN if no Track
    float TracksInvariantMass()const;
    void setTracksInvariantMass(const float&);

    // sum of Pt of the Tracks inside a tracker isolation annulus around leading Track ; NaN if no leading Track
    float isolationTracksPtSum()const;
    void setisolationTracksPtSum(const float&);
   
    // sum of Et of ECAL RecHits inside an ECAL isolation annulus around leading Track ; NaN if no leading Track 
    float isolationECALhitsEtSum()const;
    void setisolationECALhitsEtSum(const float&);
  
    // Et of the highest Et HCAL hit
    float maximumHCALhitEt()const;
    void setmaximumHCALhitEt(const float&);    
  private:
    // check overlap with another candidate
    bool overlap(const Candidate&d)const override;
    CaloTauTagInfoRef CaloTauTagInfoRef_;  
    float leadTracksignedSipt_;
    float leadTrackHCAL3x3hitsEtSum_;
    float leadTrackHCAL3x3hottesthitDEta_;
    float signalTracksInvariantMass_;
    float TracksInvariantMass_; 
    float isolationTracksPtSum_;
    float isolationECALhitsEtSum_;
    float maximumHCALhitEt_;
  };
}
#endif

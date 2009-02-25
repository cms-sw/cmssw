#ifndef JetProducers_BasePilupSubtractionJetProducer_h
#define JetProducers_BasePilupSubtractionJetProducer_h

/** \class BasePilupSubtractionJetProducer
 *
 * BasePilupSubtractionJetProducer is a base class for JetProducers.
 * It handles generic manipulations of input and output collections
 *
 * \author Fedor Ratnikov (UMd) Aug. 22, 2006
 * $Id: BasePilupSubtractionJetProducer.h,v 1.5 2008/07/16 15:01:07 kodolova Exp $
 *
 * Modifications:
 *   Sal Rappoccio (JHU): Added anomalous cell cuts
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <map>
#include <vector>
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace cms
{
  class BasePilupSubtractionJetProducer : public edm::EDProducer
  {
  public:

    BasePilupSubtractionJetProducer(const edm::ParameterSet& ps);

    /**Default destructor*/
    virtual ~BasePilupSubtractionJetProducer();
    /**Produces the EDM products*/
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
    /** jet type */
    std::string jetType () const {return mJetType;}
    
    int ieta(const reco::Candidate*);
    int iphi(const reco::Candidate*);
    
    void beginJob( const edm::EventSetup& iSetup);

    // abstract method to be set up in actual implementations
    /** run algorithm itself */
    virtual bool runAlgorithm (const  JetReco::InputCollection& fInput,  JetReco::OutputCollection* fOutput) = 0;
    
    void calculate_pedestal(const JetReco::InputCollection&);
//    reco::CandidateCollection subtract_pedestal(const JetReco::InputCollection&);
    JetReco::InputCollection subtract_pedestal(const JetReco::InputCollection&);

  private:
    edm::InputTag mSrc;
    std::string mJetType;
    bool mVerbose;
    double mEtInputCut;
    double mEInputCut;
    double mEtJetInputCut;
    double nSigmaPU;
    double radiusPU;
    std::map<int,double> esigma;
    std::map<int,double> emean;  
    std::map<int,int> geomtowers;
    std::map<int,int> ntowers_with_jets;
    std::vector<HcalDetId> allgeomid;
    const CaloGeometry* geo;
    int ietamax;
    int ietamin;
    // Including anomalous cell cuts
    uint maxBadEcalCells;
    uint maxRecoveredEcalCells;
    uint maxProblematicEcalCells;
    uint maxBadHcalCells;
    uint maxRecoveredHcalCells;
    uint maxProblematicHcalCells;

  };
}

#endif

#ifndef JetProducers_BaseJetProducer_h
#define JetProducers_BaseJetProducer_h

/** \class BaseJetProducer
 *
 * BaseJetProducer is a base class for JetProducers.
 * It handles generic manipulations of input and output collections
 *
 * \author Fedor Ratnikov (UMd) Aug. 22, 2006
 * $Id: BaseJetProducer.h,v 1.5 2008/06/23 14:35:15 oehler Exp $
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace cms
{
  class BaseJetProducer : public edm::EDProducer
  {
  public:

    BaseJetProducer(const edm::ParameterSet& ps);

    /**Default destructor*/
    virtual ~BaseJetProducer();
    /**Produces the EDM products*/
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
    /** jet type */
    std::string jetType () const {return mJetType;}

    // abstract method to be set up in actual implementations
    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput) = 0;
    reco::Particle::Point vertex;

  private:
    edm::InputTag mSrc;
    std::string mJetType;
    bool mVerbose;
    double mEtInputCut;
    double mEInputCut;
    double mJetPtMin;
    int mVertexCaloJet;
    //should be: 0 (none), 1 (correct constituents), 2 (correct final jets), 3 (use built-in method of CaloTowers)
    edm::InputTag mPVCollection;
  };
}

#endif

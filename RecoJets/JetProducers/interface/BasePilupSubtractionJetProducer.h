#ifndef JetProducers_BasePilupSubtractionJetProducer_h
#define JetProducers_BasePilupSubtractionJetProducer_h

/** \class BasePilupSubtractionJetProducer
 *
 * BasePilupSubtractionJetProducer is a base class for JetProducers.
 * It handles generic manipulations of input and output collections
 *
 * \author Fedor Ratnikov (UMd) Aug. 22, 2006
 * $Id: BasePilupSubtractionJetProducer.h,v 1.1 2006/08/22 22:11:40 fedor Exp $
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include <map>
#include <vector>

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

    // abstract method to be set up in actual implementations
    /** run algorithm itself */
    virtual bool runAlgorithm (const  JetReco::InputCollection& fInput,  JetReco::OutputCollection* fOutput) = 0;
    
    void calculate_pedestal(JetReco::InputCollection&);
    reco::CandidateCollection subtract_pedestal(JetReco::InputCollection&);

  private:
    edm::InputTag mSrc;
    std::string mJetType;
    bool mVerbose;
    double mEtInputCut;
    double mEInputCut;

    std::map<int,double> esigma;
    std::map<int,double> emean;
    std::map<double,int> ietamap;
  };
}

#endif

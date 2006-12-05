#ifndef JetProducers_BaseJetProducer_h
#define JetProducers_BaseJetProducer_h

/** \class BaseJetProducer
 *
 * BaseJetProducer is a base class for JetProducers.
 * It handles generic manipulations of input and output collections
 *
 * \author Fedor Ratnikov (UMd) Aug. 22, 2006
 * $Id: BaseJetProducer.h,v 1.1 2006/08/22 22:11:40 fedor Exp $
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"

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
    /** init branches and set alias name */
    void initBranch (const std::string& fName);
    /** jet type */
    std::string jetType () const {return jetType_;}

    // abstract method to be set up in actual implementations
    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput) = 0;

  private:
    edm::InputTag src_;
    std::string jetType_;
  };
}

#endif

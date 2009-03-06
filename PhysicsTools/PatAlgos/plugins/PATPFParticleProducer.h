//
// $Id: PATPFParticleProducer.h,v 1.10 2008/07/08 21:24:50 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATPFParticleProducer_h
#define PhysicsTools_PatAlgos_PATPFParticleProducer_h

/**
  \class    pat::PATPFParticleProducer PATPFParticleProducer.h "PhysicsTools/PatAlgos/interface/PATPFParticleProducer.h"
  \brief    Produces pat::PFParticle's

   The PATPFParticleProducer produces analysis-level pat::PFParticle's starting from
   a collection of objects of PFParticleType.

  \author   Steven Lowette, Roger Wolf
  \version  $Id: PATPFParticleProducer.h,v 1.10 2008/07/08 21:24:50 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include "DataFormats/PatCandidates/interface/PFParticle.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"

#include <string>


namespace pat {


  class ObjectResolutionCalc;
  class LeptonLRCalc;


  class PATPFParticleProducer : public edm::EDProducer {

    public:

      explicit PATPFParticleProducer(const edm::ParameterSet & iConfig);
      ~PATPFParticleProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      void 
	fetchCandidateCollection(edm::Handle< edm::View<PFParticleType> >& c, 
				 const edm::InputTag& tag, 
				 const edm::Event& iSetup) const;

      // configurables
      edm::InputTag pfCandidateSrc_;
      bool          embedPFCandidate_;
      bool          addGenMatch_;
      bool          embedGenMatch_;
      edm::InputTag genMatchSrc_;
      // tools
      GreaterByPt<PFParticle>      pTComparator_;


 
  };


}

#endif

/** \class HLTAnalCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/17 03:37:47 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTAnalCand.h"

#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<typeinfo>

//
// constructors and destructor
//
 
HLTAnalCand::HLTAnalCand(const edm::ParameterSet& iConfig)
{
   src_ = iConfig.getParameter< std::string > ("src");
}

HLTAnalCand::~HLTAnalCand()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTAnalCand::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace reco;

   edm::LogInfo("Analyze start") << "Input source: " << src_;


   // get hold of products from Event

   edm::Handle<HLTFilterObjectWithRefs> ref;
   iEvent.getByLabel(src_,ref);

   HLTParticle particle;
   const Candidate* candidate;

   const unsigned int n(ref->numberParticles());
   edm::LogVerbatim("Analyze") << "Size = " << n;
   for (unsigned int i=0; i!=n; i++) {
     particle=ref->getParticle(i);
     candidate=(ref->getParticleRef(i)).get();
     edm::LogVerbatim("Analyze") << i << " E: " 
               << particle.energy() << " " << candidate->energy() << " "  
               << typeid(*candidate).name();
   }

   edm::LogInfo("Analyze stop ") << "Input source: " << src_;
}

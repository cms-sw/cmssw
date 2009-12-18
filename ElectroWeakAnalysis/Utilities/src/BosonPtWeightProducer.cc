#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include <Math/VectorUtil.h>

//
// class declaration
//
class BosonPtWeightProducer : public edm::EDProducer {
   public:
      explicit BosonPtWeightProducer(const edm::ParameterSet&);
      ~BosonPtWeightProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      edm::InputTag genTag_;
      std::vector<double> bosonPtBinEdges_;
      std::vector<double> ptWeights_;
};


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
/////////////////////////////////////////////////////////////////////////////////////
BosonPtWeightProducer::BosonPtWeightProducer(const edm::ParameterSet& pset) {
      genTag_ = pset.getUntrackedParameter<edm::InputTag> ("GenTag", edm::InputTag("generator"));

  // Pt bin edges
      std::vector<double> defPtEdges;
      defPtEdges.push_back(0.);
      defPtEdges.push_back(999999.);
      bosonPtBinEdges_ = pset.getUntrackedParameter<std::vector<double> > ("BosonPtBinEdges",defPtEdges);
      unsigned int ninputs_expected = bosonPtBinEdges_.size()-1;

  // Distortions in muon momentum
      std::vector<double> defWeights;
      defWeights.push_back(1.);
      ptWeights_ = pset.getUntrackedParameter<std::vector<double> > ("PtWeights",defWeights);
      if (ptWeights_.size()==1 && ninputs_expected>1) {
            for (unsigned int i=1; i<ninputs_expected; i++){ ptWeights_.push_back(ptWeights_[0]);}
      }

      produces<double>();
} 

/////////////////////////////////////////////////////////////////////////////////////
BosonPtWeightProducer::~BosonPtWeightProducer(){}

/////////////////////////////////////////////////////////////////////////////////////
void BosonPtWeightProducer::beginJob(const edm::EventSetup&) {}

/////////////////////////////////////////////////////////////////////////////////////
void BosonPtWeightProducer::endJob(){}

/////////////////////////////////////////////////////////////////////////////////////
void BosonPtWeightProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {

      if (iEvent.isRealData()) return;

      edm::Handle<reco::GenParticleCollection> genParticles;
      iEvent.getByLabel("genParticles", genParticles);
      unsigned int gensize = genParticles->size();

      std::auto_ptr<double> weight (new double);

      // Set as default weight the asymptotic value at high pt (i.e. value of last bin)
      (*weight) = ptWeights_[ptWeights_.size()-1];

      unsigned int nbins = bosonPtBinEdges_.size()-1;
      for(unsigned int i = 0; i<gensize; ++i) {
            const reco::GenParticle& part = (*genParticles)[i];
            int id = part.pdgId();
            if (id!=23 && abs(id)!=24) continue;
            int status = part.status();
            if (status!=3) continue;
            double pt = part.pt();
            if (pt>bosonPtBinEdges_[0] && pt<bosonPtBinEdges_[nbins]) {
                  for (unsigned int j=1; j<=nbins; ++j) {
                        if (pt>bosonPtBinEdges_[j]) continue;
                        (*weight) = ptWeights_[j-1];
                        break;
                  }
            }
            break;
      }

      iEvent.put(weight);
}

DEFINE_FWK_MODULE(BosonPtWeightProducer);


//
// Original Author:  Simon Knutzen
//         Created:  Mon Apr 15 18:03:26 CEST 2013
// $Id: MatchTool.cc,v 1.5 2013/05/14 13:25:46 knutzen Exp $
//
//


#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include <math.h>
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h" 
#include "RecoTauTag/TauAnalysisTools/interface/ExpressionNtuple.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Candidate/interface/Candidate.h"

//#include <GeneratorTau.h>

#include "RecoTauTag/TauAnalysisTools/interface/TauTrigMatch.h"


class MatchTool : public edm::EDAnalyzer {
   public:
      explicit MatchTool(const edm::ParameterSet&);
      ~MatchTool();


   private:

      uct::ExpressionNtuple<TauTrigMatch> ntuple_;

      edm::InputTag tauSrc_;
      edm::InputTag triggerSrc_;
      double maxDR_;
      std::vector< std::string > filtNames ;

      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

};



MatchTool::MatchTool(const edm::ParameterSet& iConfig):
    ntuple_(iConfig.getParameterSet("ntuple"))
{
      edm::Service<TFileService> fs;

      ntuple_.initialize(*fs);

      tauSrc_       =   iConfig.getParameter<edm::InputTag>("tauTag");
      triggerSrc_   =   iConfig.getParameter<edm::InputTag>("trigTag");
      maxDR_        =   iConfig.getParameter<double>("maxDR");
      filtNames      =   iConfig.getParameter< std::vector<std::string> >("filterNames");


}



MatchTool::~MatchTool()
{ 

}


// Get collection of generator particles with status 2

std::vector<const reco::GenParticle*> getGenParticleCollection(const edm::Event& evt) {
    std::vector<const reco::GenParticle*> output;
    edm::Handle< std::vector<reco::GenParticle> > handle;
    evt.getByLabel("genParticles", handle);
    // Loop over objects in current collection
    for (size_t j = 0; j < handle->size(); ++j) {
      const reco::GenParticle& object = handle->at(j);
      //if(fabs(object.pdgId())==15 && object.status() == 2) output.push_back(&object);
      if(object.status() == 2) output.push_back(&object);
    }
  return output;
}


// Get collection of pat::taus
//
std::vector<const pat::Tau*> getRecoCandCollections(const edm::Event& evt, const edm::InputTag& collection) {
    std::vector<const pat::Tau*> output;
    edm::Handle< std::vector<pat::Tau> > handle;
    evt.getByLabel(collection, handle);
    // Loop over objects in current collection
    for (size_t j = 0; j < handle->size(); ++j) {
      const pat::Tau& object = handle->at(j);
      if(object.pt()>15. && fabs(object.eta())< 3.){
        output.push_back(&object);
      }
    }
  return output;
}


// Get vector of collections of TriggerFilterObjects
//
std::vector<const reco::Candidate*> getTrigObjCandCollections(const edm::Event& evt, const edm::InputTag& collection, const std::string& filtername) {
  std::vector<const reco::Candidate*> output;
    edm::Handle<pat::TriggerEvent> triggerEv;
    evt.getByLabel(collection, triggerEv);
    pat::TriggerObjectRefVector FilterObjects = triggerEv->filterObjects(filtername);

    for (unsigned int i = 0; i < FilterObjects.size(); i++) {
      const reco::Candidate &object = dynamic_cast< const reco::Candidate& >( *(FilterObjects.at(i)) );
      output.push_back(&object);
    }
  return output;
}

// Method to find the best match between tag tau and trigger filter object. The best matched filter object will be returned. If there is no match within a DR < 0.5, a null pointer is returned
const reco::Candidate* findBestMatch(const pat::Tau* TagTauObj,
    std::vector<const reco::Candidate*>& FilterSelection, double maxDR) {
  const reco::Candidate* output = NULL;
  double bestDeltaR = -1;
  for (size_t i = 0; i < FilterSelection.size(); ++i) {
    double deltaR = reco::deltaR(*TagTauObj, *FilterSelection[i]);
    if (deltaR < maxDR) {
      if (!output || deltaR < bestDeltaR) {
        output = FilterSelection[i];
        bestDeltaR = deltaR;
      }
    }
  }
  return output;
}

// Method to find the best match between tag tau and gen object. The best matched gen tau object will be returned. If there is no match within a DR < 0.5, a null pointer is returned
const reco::GenParticle* findBestGenMatch(const pat::Tau* TagTauObj,
    std::vector<const reco::GenParticle*>& GenPart, double maxDR) {
  const reco::GenParticle* output = NULL;
  double bestDeltaR = -1;
  for (size_t i = 0; i < GenPart.size(); ++i) {
    double deltaR = reco::deltaR(*TagTauObj, *GenPart[i]);
    if (deltaR < maxDR) {
      if (!output || deltaR < bestDeltaR) {
        output = GenPart[i];
        bestDeltaR = deltaR;
      }
    }
  }
  return output;
}

void
MatchTool::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    
    std::vector<const pat::Tau*> tauObjects = getRecoCandCollections(iEvent, tauSrc_);
    std::vector<const reco::GenParticle*> GenObjects = getGenParticleCollection(iEvent);
    std::vector<std::vector<const reco::Candidate*>> allTrigObjects;
   

    for(unsigned int i = 0; i < filtNames.size(); i++){

         std::vector<const reco::Candidate*> trigObjects = getTrigObjCandCollections(iEvent, triggerSrc_, filtNames[i]);
         allTrigObjects.push_back(trigObjects); // enter collection of trigger objects for each filter in a vector
    }

    std::vector<TauTrigMatch*> matches;
    TauTrigMatch *theMatch = NULL;

    for(unsigned int i = 0; i<tauObjects.size(); i++){

            const pat::Tau* TagTau = tauObjects[i];
            std::vector<const reco::Candidate* > allBestFilterMatches;
            const reco::Candidate* bestGenMatch = findBestGenMatch(TagTau,GenObjects, maxDR_) ;

            for(unsigned int j = 0; j < allTrigObjects.size(); j++){

                const reco::Candidate* bestFilterMatch = findBestMatch(TagTau, allTrigObjects[j], maxDR_);
                allBestFilterMatches.push_back(bestFilterMatch); // enter the best matched trigger object for each filter into a vector 
                   
            }    
                
            theMatch = new TauTrigMatch(TagTau,&allBestFilterMatches,bestGenMatch,matches.size(),tauObjects.size()); // create a TauTrigMatch object for each tag tau
            matches.push_back(theMatch); 

    }


    for (size_t i = 0; i < matches.size(); ++i) {
       ntuple_.fill(*matches.at(i));  // create TTree
    }
}


void 
MatchTool::beginJob()
{
}

void 
MatchTool::endJob() 
{
}

DEFINE_FWK_MODULE(MatchTool);

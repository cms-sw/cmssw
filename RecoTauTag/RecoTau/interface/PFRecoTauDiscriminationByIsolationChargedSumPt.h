#ifndef RecoTauTag_RecoTau_PFRecoTauDiscriminationByIsolationChargedSumPt_H_
#define RecoTauTag_RecoTau_PFRecoTauDiscriminationByIsolationChargedSumPt_H_

/* class PFRecoTauDiscriminationByIsolationChargedSumPt
 * created : Jul 23 2007,
 * revised : Jan 27 2009,
 * contributors : Ludovic Houchu (Ludovic.Houchu@cern.ch ; IPHC, Strasbourg), Christian Veelken (veelken@fnal.gov ; UC Davis), 
 *                Evan K. Friis (friis@physics.ucdavis.edu ; UC Davis)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"

using namespace std; 
using namespace edm;
using namespace edm::eventsetup; 
using namespace reco;

class PFRecoTauDiscriminationByIsolationChargedSumPt : public EDProducer {
   public:
      explicit PFRecoTauDiscriminationByIsolationChargedSumPt(const ParameterSet& iConfig){   
         PFTauProducer_                              = iConfig.getParameter<InputTag>("PFTauProducer");
         maxChargedSumPt_                            = iConfig.getParameter<double>("MaxChargedSumPt");
         minPtForInclusion_                          = iConfig.getParameter<double>("MinPtForObjectInclusion");
         ManipulateTracks_insteadofChargedHadrCands_ = iConfig.getParameter<bool>("ManipulateTracks_insteadofChargedHadrCands");

         produces<PFTauDiscriminator>();
      }
      ~PFRecoTauDiscriminationByIsolationChargedSumPt(){
         //delete ;
      } 
      virtual void produce(Event&, const EventSetup&);
   private:  
      InputTag PFTauProducer_;
      double maxChargedSumPt_;
      double minPtForInclusion_;
      bool ManipulateTracks_insteadofChargedHadrCands_;
};
#endif


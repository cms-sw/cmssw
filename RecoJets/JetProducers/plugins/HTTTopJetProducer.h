#ifndef RecoJets_JetProducers_HTTTopJetProducer_h
#define RecoJets_JetProducers_HTTTopJetProducer_h


/* *********************************************************


 * \class CATopJetProducer
 * Jet producer to produce top jets using the C-A algorithm to break
 * jets into subjets as described here:
 * "Top-tagging: A Method for Identifying Boosted Hadronic Tops"
 * David E. Kaplan, Keith Rehermann, Matthew D. Schwartz, Brock Tweedie
 * arXiv:0806.0848v1 [hep-ph] 

  \brief Jet producer to run the CATopJetAlgorithm

  \author   Salvatore Rappoccio
  \version  

         Notes on implementation:

	 Because the BaseJetProducer only allows the user to produce
	 one jet collection at a time, this algorithm cannot
	 fit into that paradigm. 

	 All of the "hard" jets are of type BasicJet, since
	 they are "jets of jets". The subjets will be either
	 CaloJets, GenJets, etc.

	 In order to avoid a templatization of the entire
	 EDProducer itself, we only use a templated method
	 to write out the subjets to the event record,
	 and to use that information to write out the
	 hard jets to the event record.

	 This templated method is called "write_outputs". It
	 relies on a second templated method called "write_specific",
	 which relies on some template specialization to create
	 different specific objects (i.e. CaloJets, BasicJets, GenJets, etc). 

 ************************************************************/




#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoJets/JetProducers/plugins/FastjetJetProducer.h"

#include "RecoJets/JetAlgorithms/interface/HEPTopTaggerWrapperV2.h"

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CLHEP/Random/RandomEngine.h"

#include "fastjet/SISConePlugin.hh"


namespace cms
{
  class HTTTopJetProducer : public FastjetJetProducer
  {
  public:

    HTTTopJetProducer(const edm::ParameterSet& ps);

    ~HTTTopJetProducer() override {}

    void produce( edm::Event& iEvent, const edm::EventSetup& iSetup ) override;

    void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup ) override;

    void addHTTTopJetTagInfoCollection( edm::Event& iEvent, 
						const edm::EventSetup& iSetup,
						edm::OrphanHandle<reco::BasicJetCollection> & oh) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::unique_ptr<fastjet::HEPTopTaggerV2>        fjHEPTopTagger_;

    // Below are all configurable options. 
    // Parenthesis indicates if this is enforced by the tagger itself or by the producer

    bool optimalR_; // Should the MultiR version of the tagger be used? (tagger)
    bool qJets_; // Should Q-jets be used? (tagger/producer)

    double minFatjetPt_; // Only process fatjets larger pT with the tagger [GeV] (producer)
    double minSubjetPt_; // Minimal pT for subjets [GeV] (tagger)
    double minCandPt_;   // Minimal pT to return a candidate [GeV] (tagger)
 
    double maxFatjetAbsEta_; // Only process fatjets with smaller |eta| with the tagger. (producer)

    double subjetMass_; // Mass above which subjets are further unclustered (tagger)
    double muCut_; // Mass drop threshold (tagger)

    double filtR_; // maximal filtering radius
    int filtN_; // number of filtered subjets to use
    
    // HEPTopTagger Mode (tagger):
    // 0: do 2d-plane, return candidate with delta m_top minimal
    // 1: return candidate with delta m_top minimal IF passes 2d plane
    // 2: do 2d-plane, return candidate with max dj_sum
    // 3: return candidate with max dj_sum IF passes 2d plane
    // 4: return candidate built from leading three subjets after unclustering IF passes 2d plane
    // Note: Original HTT was mode==1    
    int mode_; 

    // Top Quark mass window in GeV (tagger)
    double minCandMass_;
    double maxCandMass_;
    
    double massRatioWidth_; // One sided width of the A-shaped window around m_W/m_top in % (tagger)
    double minM23Cut_; // minimal value of m23/m123 (tagger)
    double minM13Cut_; // minimal value of atan(m13/m12) (tagger)
    double maxM13Cut_; // maximal value of atan(m13/m12) (tagger)

    double maxR_; // maximal fatjet size for MultiR tagger (tagger)
    double minR_; // minimal fatjet size for MultiR tagger (tagger)
        
    bool rejectMinR_; // set Ropt to zero when the candidate never
		      // leaves the window around the initial mass

    bool verbose_;

  };

}


#endif

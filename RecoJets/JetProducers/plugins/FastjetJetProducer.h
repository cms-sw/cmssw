#ifndef RecoJets_JetProducers_plugins_FastjetJetProducer_h
#define RecoJets_JetProducers_plugins_FastjetJetProducer_h

#include "RecoJets/JetProducers/interface/JetSpecific.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"

class FastjetJetProducer : public VirtualJetProducer
{

public:
  //
  // construction/destruction
  //
  explicit FastjetJetProducer(const edm::ParameterSet& iConfig);
  virtual ~FastjetJetProducer();

  virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );

  
protected:

  //
  // member functions
  //

  virtual void produceTrackJets( edm::Event & iEvent, const edm::EventSetup & iSetup );
  virtual void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup );

 private:

  // trackjet clustering parameters
  bool useOnlyVertexTracks_;
  bool useOnlyOnePV_;
  float dzTrVtxMax_;
  float dxyTrVtxMax_;
  int minVtxNdof_;
  float maxVtxZ_;

  // jet trimming parameters
  bool useMassDropTagger_;    /// Mass-drop tagging for boosted Higgs
  bool useFiltering_;         /// Jet filtering technique
  bool useTrimming_;          /// Jet trimming technique
  bool usePruning_;           /// Jet pruning technique
  bool useCMSBoostedTauSeedingAlgorithm_; /// algorithm for seeding reconstruction of boosted Taus (similar to mass-drop tagging)
  double muCut_;              /// for mass-drop tagging, m0/mjet (m0 = mass of highest mass subjet)
  double yCut_;               /// for mass-drop tagging, symmetry cut: min(pt1^2,pt2^2) * dR(1,2) / mjet > ycut
  double rFilt_;              /// for filtering, trimming: dR scale of sub-clustering
  int    nFilt_;              /// for filtering, pruning: number of subjets expected
  double trimPtFracMin_;      /// for trimming: constituent minimum pt fraction of full jet
  double zCut_;               /// for pruning: constituent minimum pt fraction of parent cluster
  double RcutFactor_;         /// for pruning: constituent dR * pt/2m < rcut_factor

  double subjetPtMin_;        /// for CMSBoostedTauSeedingAlgorithm : subjet pt min
  double muMin_;              /// for CMSBoostedTauSeedingAlgorithm : min mass-drop
  double muMax_;              /// for CMSBoostedTauSeedingAlgorithm : max mass-drop
  double yMin_;               /// for CMSBoostedTauSeedingAlgorithm : min asymmetry
  double yMax_;               /// for CMSBoostedTauSeedingAlgorithm : max asymmetry
  double dRMin_;              /// for CMSBoostedTauSeedingAlgorithm : min dR
  double dRMax_;              /// for CMSBoostedTauSeedingAlgorithm : max dR
  int    maxDepth_;           /// for CMSBoostedTauSeedingAlgorithm : max depth for descending into clustering sequence


  // tokens for the data access
  edm::EDGetTokenT<edm::View<reco::RecoChargedRefCandidate> > input_chrefcand_token_;
    
};


#endif

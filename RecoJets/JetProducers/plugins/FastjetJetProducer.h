#ifndef RecoJets_JetProducers_plugins_FastjetJetProducer_h
#define RecoJets_JetProducers_plugins_FastjetJetProducer_h

#include "RecoJets/JetProducers/interface/JetSpecific.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"

#include <fastjet/tools/Transformer.hh>

// a function returning
//   min(Rmax, deltaR_factor * deltaR(j1,j2))
// where j1 and j2 are the 2 subjets of j
// if the jet does not have exactly 2 pieces, Rmax is used.
class DynamicRfilt : public fastjet::FunctionOfPseudoJet<double>{
public:
  // default ctor
  DynamicRfilt(double Rmax, double deltaR_factor) : _Rmax(Rmax), _deltaR_factor(deltaR_factor){}

  // action of the function
  double result(const fastjet::PseudoJet &j) const{
    if (! j.has_pieces()) return _Rmax;

    std::vector<fastjet::PseudoJet> pieces = j.pieces();
    if (! pieces.size()==2) return _Rmax;

    double deltaR = pieces[0].delta_R(pieces[1]);
    return std::min(_Rmax, _deltaR_factor * deltaR);
  }

private:
  double _Rmax, _deltaR_factor;
};



class FastjetJetProducer : public VirtualJetProducer
{

public:
  // typedefs
  typedef fastjet::Transformer         transformer;
  typedef std::unique_ptr<transformer> transformer_ptr;
  typedef std::vector<transformer_ptr> transformer_coll;
  //
  // construction/destruction
  //
  explicit FastjetJetProducer(const edm::ParameterSet& iConfig);
  virtual ~FastjetJetProducer();

  virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );

  // typedefs
  typedef boost::shared_ptr<DynamicRfilt>  DynamicRfiltPtr;

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
  bool useDynamicFiltering_;  /// Use dynamic filtering radius (as in arXiv:0802.2470)
  bool useTrimming_;          /// Jet trimming technique
  bool usePruning_;           /// Jet pruning technique
  bool useCMSBoostedTauSeedingAlgorithm_; /// algorithm for seeding reconstruction of boosted Taus (similar to mass-drop tagging)
  bool useKtPruning_;         /// Use Kt clustering algorithm for pruning (default is Cambridge/Aachen)
  bool useConstituentSubtraction_; /// constituent subtraction technique
  bool useSoftDrop_;          /// Soft drop
  bool correctShape_;         /// Correct the shape of the jets
  double muCut_;              /// for mass-drop tagging, m0/mjet (m0 = mass of highest mass subjet)
  double yCut_;               /// for mass-drop tagging, symmetry cut: min(pt1^2,pt2^2) * dR(1,2) / mjet > ycut
  double rFilt_;              /// for filtering, trimming: dR scale of sub-clustering
  DynamicRfiltPtr rFiltDynamic_; /// for dynamic filtering radius (as in arXiv:0802.2470)
  double rFiltFactor_;        /// for dynamic filtering radius (as in arXiv:0802.2470)
  int    nFilt_;              /// for filtering, pruning: number of subjets expected
  double trimPtFracMin_;      /// for trimming: constituent minimum pt fraction of full jet
  double zCut_;               /// for pruning OR soft drop: constituent minimum pt fraction of parent cluster
  double RcutFactor_;         /// for pruning: constituent dR * pt/2m < rcut_factor
  double csRho_EtaMax_;       /// for constituent subtraction : maximum rapidity for ghosts
  double csRParam_;           /// for constituent subtraction : R parameter for KT alg in jet median background estimator
  double beta_;               /// for soft drop : beta (angular exponent)
  double R0_;                 /// for soft drop : R0 (angular distance normalization - should be set to jet radius in most cases)
  double gridMaxRapidity_;    /// for shape subtraction, get the fixed-grid rho
  double gridSpacing_;        /// for shape subtraction, get the grid spacing


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

#ifndef PatAlgos_PATPrimaryVertexSelector_H_
#define PatAlgos_PATPrimaryVertexSelector_H_

/**
  \class    pat::PATPrimaryVertexSelector PATPrimaryVertexSelector.h "PhysicsTools/PatAlgos/interface/PATPrimaryVertexSelector.h"

   The PATPrimaryVertexSelector is used together with an ObjectSelector to clean and
   sort a collection of primary vertices. The code is very close to what is done in
   SusyAnalyzer: it allows a selection based on the (normalized) chi2 of the vertex fit,
   the position, the multiplicity and the pt-sum of the associated tracks. The tracks
   entering in the calculation of the last two quantities can be restricted in eta.
   The output collection is sorted by the sum of the track pts.

*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class PATPrimaryVertexSelector {
public:
  typedef reco::VertexCollection collection;
  typedef std::vector<const reco::Vertex*> container;
  typedef container::const_iterator const_iterator;
  PATPrimaryVertexSelector (const edm::ParameterSet& cfg, edm::ConsumesCollector && iC);
  /// needed for use with an ObjectSelector
  const_iterator begin() const { return selected_.begin(); }
  /// needed for use with an ObjectSelector
  const_iterator end() const { return selected_.end(); }
  /// needed for use with an ObjectSelector
  void select(const edm::Handle<collection>&, const edm::Event&, const edm::EventSetup&);
  /// needed for use with an ObjectSelector
  size_t size() const { return selected_.size(); }
  /// operator used in sorting the selected vertices
  bool operator() (const reco::Vertex*, const reco::Vertex*) const;
private:
  /// access to track-related vertex quantities (multiplicity and pt-sum)
  void getVertexVariables (const reco::Vertex&, unsigned int&, double&) const;
  /// track selection
  bool acceptTrack (const reco::Track&) const;

private:
  container selected_;               /// container of selected vertices
  unsigned int multiplicityCut_;     /// minimum multiplicity of (selected) associated tracks
  float ptSumCut_;                   /// minimum pt sum o (selected) associated tracks
  float trackEtaCut_;                /// eta cut used for the track selection
  float chi2Cut_;                    /// cut on the normalized chi2
  float dr2Cut_;                     /// cut on the (squared) transverse position
  float dzCut_;                      /// cut on the longitudinal position
};

#endif


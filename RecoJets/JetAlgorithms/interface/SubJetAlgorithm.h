#ifndef RecoJets_JetAlgorithms_CATopJetAlgorithm2_h
#define RecoJets_JetAlgorithms_CATopJetAlgorithm2_h

#include <vector>

#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"
#include "FWCore/Framework/interface/Event.h"
#include "RecoJets/JetAlgorithms/interface/FastPrunePlugin.hh"
#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>
#include <fastjet/GhostedAreaSpec.hh>

class SubJetAlgorithm {
public:
  SubJetAlgorithm(double ptMin,
                  unsigned int subjets,
                  double zcut,
                  double rcut_factor,
                  std::shared_ptr<fastjet::JetDefinition> fjJetDefinition,
                  bool doAreaFastjet,
                  std::shared_ptr<fastjet::GhostedAreaSpec> fjActiveArea,
                  double voronoiRfact)
      : ptMin_(ptMin),
        nSubjets_(subjets),
        zcut_(zcut),
        rcut_factor_(rcut_factor),
        fjJetDefinition_(fjJetDefinition),
        doAreaFastjet_(doAreaFastjet),
        fjActiveArea_(fjActiveArea),
        voronoiRfact_(voronoiRfact) {}

  void set_zcut(double z);
  void set_rcut_factor(double r);
  double zcut() const { return zcut_; }
  double rcut_factor() const { return rcut_factor_; }

  /// Find the ProtoJets from the collection of input Candidates.
  void run(const std::vector<fastjet::PseudoJet>& cell_particles, std::vector<CompoundPseudoJet>& hardjetsOutput);

private:
  double ptMin_;        //<! lower pt cut on which jets to reco
  int nSubjets_;        //<! number of subjets to produce.
  double zcut_;         //<! zcut parameter (see arXiv:0903.5081). Only relevant if pruning is enabled.
  double rcut_factor_;  //<! r-cut factor (see arXiv:0903.5081).
  std::shared_ptr<fastjet::JetDefinition> fjJetDefinition_;  //<! jet definition to use
  bool doAreaFastjet_;                                       //<! whether or not to use the fastjet area
  std::shared_ptr<fastjet::GhostedAreaSpec> fjActiveArea_;   //<! fastjet area spec
  double voronoiRfact_;                                      //<! fastjet voronoi area R factor
};

#endif

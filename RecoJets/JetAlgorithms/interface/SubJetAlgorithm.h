#ifndef RecoJets_JetAlgorithms_CATopJetAlgorithm2_h
#define RecoJets_JetAlgorithms_CATopJetAlgorithm2_h

#include <vector>

#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"
#include "FWCore/Framework/interface/Event.h"

#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>


class SubJetAlgorithm{
 public:
  SubJetAlgorithm(edm::InputTag mSrc,
		    int algorithm,
		    double centralEtaCut,            
		    double ptMin,                    
		    double jetsize,
		    unsigned int subjets,
		    bool pruning,
		    double zcut=0.1) :
    mSrc_          (mSrc          ),
    algorithm_     (algorithm     ),
    centralEtaCut_ (centralEtaCut ), 
    ptMin_         (ptMin         ),         
    jetsize_       (jetsize       ),
    nSubjets_      (subjets       ),
    enable_pruning_(pruning       ),
    zcut_          (zcut          )
      { }

  bool get_pruning()const;
  void set_zcut(double z);
  void set_rcut_factor(double r);

    /// Find the ProtoJets from the collection of input Candidates.
    void run( const std::vector<fastjet::PseudoJet> & cell_particles, 
	      std::vector<CompoundPseudoJet> & hardjetsOutput,
	      const edm::EventSetup & c);


 private:

  edm::InputTag       mSrc_;          //<! calo tower input source
  int                 algorithm_;     //<! 0 = KT, 1 = CA, 2 = anti-KT
  double              centralEtaCut_; //<! eta for defining "central" jets
  double              ptMin_;	      //<! lower pt cut on which jets to reco
  double              jetsize_;	      //<!
  int                 nSubjets_;      //<! number of subjets to produce.
  bool                enable_pruning_;//<! flag whether pruning is enabled (see arXiv:0903.5081)
  double              zcut_;          //<! zcut parameter (see arXiv:0903.5081). Only relevant if pruning is enabled.
  double              rcut_factor_;   //<! r-cut factor (see arXiv:0903.5081).
  
};

#endif

#ifndef JetAlgorithms_CMSmidpointAlgorithm_h
#define JetAlgorithms_CMSmidpointAlgorithm_h

/** \class CMSmidpointAlgorithm
 *
 * prototype porting from code in RecoJets
 *
 ************************************************************/
#include <vector>
#include "CLHEP/Vector/LorentzVector.h"
#include "PhysicsTools/JetExamples/interface/JetableObjectHelper.h"
#include "PhysicsTools/JetExamples/interface/ProtoJet.h"
#include "PhysicsTools/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/Handle.h"

class CMSmidpointAlgorithm 
{
 public:
  /// Default constructor which defines the default values of the algorithm parameters
  ///
  /// theSeedThreshold:    minimum ET in GeV of an CaloTower that can seed a jet.
  /// theTowerThreshold:   minimum ET in GeV of an CaloTower that is included in a jet
  /// theConeRadius:       nominal radius of the jet in eta-phi space
  /// theConeAreaFraction: multiplier to reduce the search cone area during the iteration phase.
  ///                      introduced by CDF in 2002 to avoid energy loss due to proto-jet migration. 
  ///                      Actively being discussed in 2005.
  ///                      A value of 1.0 gives the original run II algorithm in hep-ex/0005012.
  ///                      CDF value of 0.25 gives new search cone of 0.5 theConeRadius during iteration, 
  ///                      but keeps the final cone at theConeRadius for the last iteration.
  /// theMaxPairSize:      Maximum size of proto-jet pair, triplet, etc for defining midpoint.
  ///                      Both CDF and D0 use 2.
  /// theMaxIterations:    Maximum number of iterations before finding a stable cone.
  /// theOverlapThreshold: When two proto-jets overlap, this is the merging threshold on the fraction of PT in the 
  ///                      overlap region compared to the lower Pt Jet. 
  ///                      If the overlapPt/lowerJetPt > theOverlapThreshold the 2 proto-jets will be merged into one
  ///                      final jet.
  ///                      if the overlapPt/lowerJetPt < theOverlapThreshold the towers in the two proto-jets will be
  ///                      seperated into two distinct sets of towers: two final jets.
  ///                      D0 has used 0.5, CDF has used both 0.5 and 0.75 in run 2, and 0.75 in run 1.
  /// theDebugLevel:       integer level of diagnostic printout: 0 = no printout, 1 = minimal printout, 2 = pages per event.

   // This default constructor is temporary; all configuration
   // parameters must be passed in to the constructor so that they can
   // be traced in the new EDM.
  
  typedef edm::Handle<reco::CandidateCollection> Handle;

  CMSmidpointAlgorithm(const Handle & ctcp) :
    theSeedThreshold(3.0),
    theTowerThreshold(1.0),
    theConeRadius(0.5),
    theConeAreaFraction(1.0),
    theMaxPairSize(2),
    theMaxIterations(100),
    theOverlapThreshold(0.75),
    theDebugLevel(0)
  { }

  /// Constructor takes as input all the values of the algorithm that the user can change, and the CaloTower Collection pointer.
  CMSmidpointAlgorithm(double st, double tt, double cr, double caf, 
		       int mps, int mi, double ot, int dl) : // , const CaloTowerCollection *ctcp):
    theSeedThreshold(st),
    theTowerThreshold(tt),
    theConeRadius(cr),
    theConeAreaFraction(caf),
    theMaxPairSize(mps),
    theMaxIterations(mi),
    theOverlapThreshold(ot),
    theDebugLevel(dl)
  { }

  /// Runs the algorithm and returns a list of caloJets. 
  /// The user declares the vector and calls this method.
  void run(const Handle & theCtcp, reco::CandidateCollection& caloJets);


 private:
  /// Find the list of proto-jets from the seeds.  
  /// Called by run, but public method to allow testing and studies.
  void findStableConesFromSeeds(JetableObjectHelper& theHelper,
				const Handle & theCtcp,
				std::vector<ProtoJet> & protoJets);

  /// Iterate the proto-jet center until it is stable.  
  /// Called by findStableConesFromSeeds and findStableConesFromMidPoints but public method to allow testing and studies.
  void iterateCone(JetableObjectHelper& theHelper,
	   const Handle & theCtcp,
		   double startRapidity, double startPhi, double startPt, bool reduceConeSize, std::vector<ProtoJet> & protoJets);

  /// Add to the list of proto-jets the list of midpoints.
  /// Called by run but public method to allow testing and studies.
  void findStableConesFromMidPoints(JetableObjectHelper& theHelper,
				    const Handle & theCtcp,
				    std::vector<ProtoJet>& protoJets);

  /// Add proto-jets to pairs, triplets, etc, prior to finding their midpoints.
  /// Called by findStableConesFromMidPoints but public method to allow testing and studies.
  void addClustersToPairs(JetableObjectHelper& theHelper,
			  const Handle & theCtcp,
			  std::vector<int>& testPair, std::vector<std::vector<int> >& pairs,
			  std::vector<std::vector<bool> >& distanceOK, int maxClustersInPair);

  /// Split and merge the proto-jets, assigning each tower in the protojets to one and only one final jet.
  ///  Called by run but public method to allow testing and studies.
  void splitAndMerge(JetableObjectHelper& theHelper,
		     const Handle & theCtcp,
		     std::vector<ProtoJet>& protoJets, std::vector<ProtoJet>& finalJets);


  double theSeedThreshold;
  double theTowerThreshold;
  double theConeRadius;
  double theConeAreaFraction;
  int    theMaxPairSize;
  int    theMaxIterations;
  double theOverlapThreshold;
  int    theDebugLevel;
};

#endif

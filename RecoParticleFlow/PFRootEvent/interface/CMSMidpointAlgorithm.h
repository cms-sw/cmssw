#ifndef JetAlgorithms_CMSMidpointAlgorithm_h
#define JetAlgorithms_CMSMidpointAlgorithm_h

/** \class CMSMidpointAlgorithm
 *
 * CMSMidpointAlgorithm is an algorithm for CMS jet reconstruction
 * baded on the CDF midpoint algorithm. 
 *
 * The algorithm is documented in the proceedings of the Physics at 
 * RUN II: QCD and Weak Boson Physics Workshop: hep-ex/0005012.
 *
 * The algorithm runs off of generic Candidates 
 *
 * \author Robert M Harris, Fermilab
 *
 * \version   1st Version Feb. 4, 2005  Based on the CDF Midpoint Algorithm code by Matthias Toennesmann.
 * \version   2nd Version Apr. 6, 2005  Modifications toward integration in new EDM.
 * \version   3rd Version Oct. 19, 2005 Modified to work with real CaloTowers from Jeremy Mans
 * \version   F.Ratnikov, Mar. 8, 2006. Work from Candidate
 * $Id: CMSMidpointAlgorithm.h,v 1.1 2009/08/24 14:35:59 srappocc Exp $
 *
 ************************************************************/


#include "RecoParticleFlow/PFRootEvent/interface/JetRecoTypes.h"

class CMSMidpointAlgorithm 
{
 public:
  typedef std::vector<ProtoJet*> InternalCollection;

  /// Default constructor
  CMSMidpointAlgorithm () :
    theSeedThreshold(3.0),
    theConeRadius(0.5),
    theConeAreaFraction(1.0),
    theMaxPairSize(2),
    theMaxIterations(100),
    theOverlapThreshold(0.75),
    theDebugLevel(0)
  { }

  /** Constructor takes as input all the values of the algorithm that the user can change.
  \param fSeedThreshold:    minimum ET in GeV of an CaloTower that can seed a jet.
  \param fTowerThreshold:   minimum ET in GeV of an CaloTower that is included in a jet
  \param fConeRadius:       nominal radius of the jet in eta-phi space
  \param fConeAreaFraction: multiplier to reduce the search cone area during the iteration phase.
                        introduced by CDF in 2002 to avoid energy loss due to proto-jet migration. 
                           Actively being discussed in 2005.
                           A value of 1.0 gives the original run II algorithm in hep-ex/0005012.
                           CDF value of 0.25 gives new search cone of 0.5 theConeRadius during iteration, 
                           but keeps the final cone at theConeRadius for the last iteration.
  \param fMaxPairSize:      Maximum size of proto-jet pair, triplet, etc for defining midpoint.
                           Both CDF and D0 use 2.
  \param fMaxIterations:    Maximum number of iterations before finding a stable cone.
  \param fOverlapThreshold: When two proto-jets overlap, this is the merging threshold on the fraction of PT in the 
                           overlap region compared to the lower Pt Jet. 
                           If the overlapPt/lowerJetPt > theOverlapThreshold the 2 proto-jets will be merged into one
                           final jet.
                           if the overlapPt/lowerJetPt < theOverlapThreshold the towers in the two proto-jets will be
                           seperated into two distinct sets of towers: two final jets.
                           D0 has used 0.5, CDF has used both 0.5 and 0.75 in run 2, and 0.75 in run 1.
  \param fDebugLevel:       integer level of diagnostic printout: 0 = no printout, 1 = minimal printout, 2 = pages per event.
   */
  CMSMidpointAlgorithm(double fSeedThreshold, double fConeRadius, double fConeAreaFraction, 
		       int fMaxPairSize, int fMaxIterations, double fOverlapThreshold, int fDebugLevel) : 
    theSeedThreshold(fSeedThreshold),
    theConeRadius(fConeRadius),
    theConeAreaFraction(fConeAreaFraction),
    theMaxPairSize(fMaxPairSize),
    theMaxIterations(fMaxIterations),
    theOverlapThreshold(fOverlapThreshold),
    theDebugLevel(fDebugLevel)
  { }

  /// Runs the algorithm and returns a list of caloJets. 
  /// The user declares the vector and calls this method.
  void run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput);

 private:
  /// Find the list of proto-jets from the seeds.  
  /// Called by run, but public method to allow testing and studies.
  void findStableConesFromSeeds(const JetReco::InputCollection& fInput, InternalCollection* fOutput);

  /// Iterate the proto-jet center until it is stable.  
  /// Called by findStableConesFromSeeds and findStableConesFromMidPoints but public method to allow testing and studies.
  void iterateCone(const JetReco::InputCollection& fInput,
		   double startRapidity, double startPhi, double startPt, bool reduceConeSize, 
		   InternalCollection* fOutput);

  /// Add to the list of proto-jets the list of midpoints.
  /// Called by run but public method to allow testing and studies.
  void findStableConesFromMidPoints(const JetReco::InputCollection& fInput, InternalCollection* fOutput);

  /// Add proto-jets to pairs, triplets, etc, prior to finding their midpoints.
  /// Called by findStableConesFromMidPoints but public method to allow testing and studies.
  void addClustersToPairs(const JetReco::InputCollection& fInput,
			  std::vector<int>& testPair, std::vector<std::vector<int> >& pairs,
			  std::vector<std::vector<bool> >& distanceOK, int maxClustersInPair);

  /// Split and merge the proto-jets, assigning each tower in the protojets to one and only one final jet.
  ///  Called by run but public method to allow testing and studies.
  void splitAndMerge(const JetReco::InputCollection& fInput,
		     InternalCollection* fProtoJets, JetReco::OutputCollection* fFinalJets);


  double theSeedThreshold;
  double theConeRadius;
  double theConeAreaFraction;
  int    theMaxPairSize;
  int    theMaxIterations;
  double theOverlapThreshold;
  int    theDebugLevel;
};

#endif

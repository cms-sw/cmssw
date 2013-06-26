#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentAlgorithm_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentAlgorithm_h

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentSetup.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTrackRefitter.h"

#include <set>

/// The main class for the Kalman alignment algorithm. It is the stage on which all the protagonists
/// are playing: the refitter, the trajectory factory and the updator.
/// See E.Widl, R.Fr¨uhwirth, W.Adam, A Kalman Filter for Track-based Alignment, CMS NOTE-2006/022
/// for details.

class AlignableNavigator;
class AlignmentParameterSelector;
class TrajectoryFitter;

namespace reco { class BeamSpot; }

class KalmanAlignmentAlgorithm : public AlignmentAlgorithmBase
{

public:

  typedef TrajectoryFactoryBase::ReferenceTrajectoryPtr ReferenceTrajectoryPtr;
  typedef TrajectoryFactoryBase::ReferenceTrajectoryCollection ReferenceTrajectoryCollection;
  typedef TrajectoryFactoryBase::ExternalPredictionCollection ExternalPredictionCollection;

  typedef KalmanAlignmentTracklet::TrackletPtr TrackletPtr;
  typedef std::vector< TrackletPtr > TrackletCollection;

  typedef KalmanAlignmentSetup AlignmentSetup;
  typedef std::vector< AlignmentSetup* > AlignmentSetupCollection;


  KalmanAlignmentAlgorithm( const edm::ParameterSet& config );
  virtual ~KalmanAlignmentAlgorithm( void );

  /// Dummy implementation.
  /// Needed for inheritance of TrackProducerBase, but we don't produce anything.
  virtual void produce( edm::Event&, const edm::EventSetup& ) {}

  virtual void initialize( const edm::EventSetup& setup, 
			   AlignableTracker* tracker,
                           AlignableMuon* muon,
			   AlignableExtras* extras,
			   AlignmentParameterStore* store );

  virtual void terminate( const edm::EventSetup& setup );

  virtual void run( const edm::EventSetup& setup, const EventInfo &eventInfo );

  inline bool operator()( const Alignable* a1, const Alignable* a2 ) const { return ( a1->id() < a2->id() ); }

private:

  void initializeAlignmentParameters( const edm::EventSetup& setup );

  void initializeAlignmentSetups( const edm::EventSetup& setup );

  void applyAlignmentParameters( Alignable* ali, AlignmentParameters* par, bool applyPar, bool applyCov ) const;

  void getComponents( Alignable* ali, std::vector<Alignable*>& comps ) const;

  void mergeResults( void ) const;

  void setAPEToZero( void );

  inline const PropagationDirection getDirection( const std::string& dir ) const
    { return ( dir == "alongMomentum" ) ? alongMomentum : oppositeToMomentum; }

  inline const PropagationDirection oppositeDirection( const PropagationDirection dir ) const
    { return ( dir == alongMomentum ) ? oppositeToMomentum : alongMomentum; }

  const KalmanAlignmentSetup::SortingDirection getSortingDirection( const std::string& sortDir ) const;

  edm::ParameterSet theConfiguration;

  AlignmentSetupCollection theAlignmentSetups;

  KalmanAlignmentTrackRefitter* theRefitter;

  AlignmentParameterStore* theParameterStore;
  AlignableNavigator* theNavigator;
  AlignmentParameterSelector* theSelector;

  AlignableTracker* theTracker;

  bool theMergerFlag;
};

#endif

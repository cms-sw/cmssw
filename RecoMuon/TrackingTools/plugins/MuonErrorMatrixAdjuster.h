#ifndef RecoMuon_TrackingTools_MuonErrorMatrixAdjuster_H
#define RecoMuon_TrackingTools_MuonErrorMatrixAdjuster_H

/** \class MuonErrorMatrixAdjuster
 *
 * EDProducer which duplicatesa collection of track, adjusting their error matrix
 *
 * track collection is retrieve from the event, duplicated, while the error matrix is corrected
 * rechit are copied into a new collection
 * track extra is also copied and error matrix are corrected by the same scale factors
 *
 *
 * \author Jean-Roch Vlimant  UCSB
 * \author Finn Rebassoo      UCSB
 */

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "FWCore/Utilities/interface/InputTag.h"

class FreeTrajectoryState;
class MuonErroMatrix;
class MagneticField;
class IdealMagneticFieldRecord;
class MuonErrorMatrix;
class TrackerTopologyRcd;
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "FWCore/Framework/interface/ESHandle.h"

//
// class decleration
//

class MuonErrorMatrixAdjuster : public edm::stream::EDProducer<> {
public:
  /// constructor
  explicit MuonErrorMatrixAdjuster(const edm::ParameterSet&);
  /// destructor
  ~MuonErrorMatrixAdjuster() override;

private:
  /// framework method
  void produce(edm::Event&, const edm::EventSetup&) override;

  /// return a corrected error matrix
  reco::TrackBase::CovarianceMatrix fix_cov_matrix(const reco::TrackBase::CovarianceMatrix& error_matrix,
                                                   const GlobalVector& momentum);
  /// mutliply revised_matrix (first argument)  by second matrix TERM by TERM
  void multiply(reco::TrackBase::CovarianceMatrix& revised_matrix,
                const reco::TrackBase::CovarianceMatrix& scale_matrix);
  /// divide the num_matrix  (first argument) by second matrix, TERM by TERM
  bool divide(reco::TrackBase::CovarianceMatrix& num_matrix, const reco::TrackBase::CovarianceMatrix& denom_matrix);

  /// create a corrected reco::Track from itself and trajectory state (redundant information)
  reco::Track makeTrack(const reco::Track& recotrack_orig, const FreeTrajectoryState& PCAstate);

  /// make a selection on the reco:Track. (dummy for the moment)
  bool selectTrack(const reco::Track& recotrack_orig);

  /// create a track extra for the newly created recotrack, scaling the outer/inner measurment error matrix by the scale matrix recotrack/recotrack_orig
  reco::TrackExtra* makeTrackExtra(const reco::Track& recotrack_orig,
                                   reco::Track& recotrack,
                                   reco::TrackExtraCollection& TEcol);

  /// attached rechits to the newly created reco::Track and reco::TrackExtra
  bool attachRecHits(const reco::Track& recotrack_orig,
                     reco::Track& recotrack,
                     reco::TrackExtra& trackextra,
                     TrackingRecHitCollection& RHcol,
                     const TrackerTopology& ttopo);

  // ----------member data ---------------------------
  /// log category: MuonErrorMatrixAdjuster
  std::string theCategory;

  /// input tag of the reco::Track collection to be corrected
  edm::InputTag theTrackLabel;

  /// instrance name of the created track collecion. rechit and trackextra have no instance name
  std::string theInstanceName;

  /// select the rescaling or replacing method to correct the error matrix
  bool theRescale;

  /// holds the error matrix parametrization
  std::unique_ptr<MuonErrorMatrix> theMatrixProvider;

  /// hold on to the magnetic field
  edm::ESHandle<MagneticField> theField;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theFieldToken;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> theHttopoToken;

  /// get reference before put track extra to the event, in order to create edm::Ref
  edm::RefProd<reco::TrackExtraCollection> theRefprodTE;
  edm::Ref<reco::TrackExtraCollection>::key_type theTEi;

  /// get reference before put rechit to the event, in order to create edm::Ref
  edm::RefProd<TrackingRecHitCollection> theRefprodRH;
  edm::Ref<TrackingRecHitCollection>::key_type theRHi;
};

#endif

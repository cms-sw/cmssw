#ifndef RecoMuon_GlobalTrackingTools_MuonErrorMatrixAdjuster_H
#define RecoMuon_GlobalTrackingTools_MuonErrorMatrixAdjuster_H
// -*- C++ -*-
//
// Package:    MuonErrorMatrixAdjuster
// Class:      MuonErrorMatrixAdjuster
// 
/**\class MuonErrorMatrixAdjuster

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Finn Rebassoo
//         Created:  Mon Aug 13 16:13:44 CDT 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h>

#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/TrackReco/interface/TrackExtra.h>
#include <DataFormats/TrackingRecHit/interface/TrackingRecHit.h>

#include <string>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "RecoMuon/GlobalTrackingTools/interface/ErrorMatrix.h"

#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>

#include <MagneticField/Records/interface/IdealMagneticFieldRecord.h>
#include <MagneticField/Engine/interface/MagneticField.h>
#include "TString.h"
#include "TMath.h"
#include  <FWCore/Framework/interface/ESHandle.h>

//
// class decleration
//

class MuonErrorMatrixAdjuster : public edm::EDProducer {
   public:
      explicit MuonErrorMatrixAdjuster(const edm::ParameterSet&);
      ~MuonErrorMatrixAdjuster();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  reco::TrackBase::CovarianceMatrix fix_cov_matrix(const reco::TrackBase::CovarianceMatrix& error_matrix,const GlobalVector& momentum);
  void multiply(reco::TrackBase::CovarianceMatrix & revised_matrix, const reco::TrackBase::CovarianceMatrix & scale_matrix);
  bool divide(reco::TrackBase::CovarianceMatrix & num_matrix, const reco::TrackBase::CovarianceMatrix & denom_matrix);

  reco::Track makeTrack(const reco::Track & recotrack_orig,
			const FreeTrajectoryState & PCAstate);
  bool selectTrack(const reco::Track & recotrack_orig);
  reco::TrackExtra * makeTrackExtra(const reco::Track & recotrack_orig,
				    reco::Track & recotrack,
				    reco::TrackExtraCollection& TEcol);

  bool attachRecHits(const reco::Track & recotrack_orig,
		     reco::Track & recotrack,
		     reco::TrackExtra & trackextra,
		     TrackingRecHitCollection& RHcol);

  // ----------member data ---------------------------
  std::string theCategory;
  std::string theInstanceName;
  bool theRescale;
  edm::InputTag theTrackLabel;
  
  edm::ParameterSet theMatrixProvider_pset;
  //  std::string theErrorMatrixRootFile;
  ErrorMatrix * theMatrixProvider;

  edm::ESHandle<MagneticField> theField;

  edm::RefProd<reco::TrackExtraCollection> theRefprodTE;
  edm::Ref<reco::TrackExtraCollection>::key_type theTEi;

  edm::RefProd<TrackingRecHitCollection> theRefprodRH;
  edm::Ref<TrackingRecHitCollection>::key_type theRHi;

};

#endif

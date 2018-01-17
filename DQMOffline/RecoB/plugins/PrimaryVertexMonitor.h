#ifndef PrimaryVertexMonitor_H
#define PrimaryVertexMonitor_H

#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"


/** \class PrimaryVertexMonitor
 *
 *
 */

namespace {
  struct Histograms {
    ConcurrentMonitorElement nbvtx, nbgvtx, nbtksinvtx[2], trksWeight[2], score[2];
    ConcurrentMonitorElement tt[2];
    ConcurrentMonitorElement xrec[2], yrec[2], zrec[2], xDiff[2], yDiff[2], xerr[2], yerr[2], zerr[2];
    ConcurrentMonitorElement xerrVsTrks[2], yerrVsTrks[2], zerrVsTrks[2];
    ConcurrentMonitorElement ntracksVsZ[2];
    ConcurrentMonitorElement vtxchi2[2], vtxndf[2], vtxprob[2], nans[2];
    ConcurrentMonitorElement type[2];
    ConcurrentMonitorElement bsX, bsY, bsZ, bsSigmaZ, bsDxdz, bsDydz, bsBeamWidthX, bsBeamWidthY, bsType;

    ConcurrentMonitorElement sumpt, ntracks, weight, chi2ndf, chi2prob;
    ConcurrentMonitorElement dxy, dxy2, dz, dxyErr, dzErr;
    ConcurrentMonitorElement dxyVsPhi_pt1, dzVsPhi_pt1;
    ConcurrentMonitorElement dxyVsEta_pt1, dzVsEta_pt1;
    ConcurrentMonitorElement dxyVsPhi_pt10, dzVsPhi_pt10;
    ConcurrentMonitorElement dxyVsEta_pt10, dzVsEta_pt10;
  };
}

class PrimaryVertexMonitor : public DQMGlobalEDAnalyzer<Histograms> {
   public:
      explicit PrimaryVertexMonitor(const edm::ParameterSet& pSet);

      ~PrimaryVertexMonitor() override;

      void bookHistograms(DQMStore::ConcurrentBooker &, edm::Run const&, edm::EventSetup const&, Histograms &) const override;
      void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms const&) const override;

   private:

  void pvTracksPlots(const Histograms &, const reco::Vertex & v) const;
  void vertexPlots(const Histograms &, const reco::Vertex & v, const reco::BeamSpot& beamSpot, int i) const;

  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  edm::EDGetTokenT<reco::BeamSpot>         beamspotToken_;
  using VertexScore = edm::ValueMap<float>;
  edm::EDGetTokenT<VertexScore>        scoreToken_;
  
  edm::InputTag vertexInputTag_, beamSpotInputTag_;

  edm::ParameterSet conf_;

  std::string dqmLabel;

  std::string TopFolderName_;
  std::string AlignmentLabel_;
  int ndof_;
  mutable std::atomic_flag errorPrinted_;
  
  // the histos


};


#endif

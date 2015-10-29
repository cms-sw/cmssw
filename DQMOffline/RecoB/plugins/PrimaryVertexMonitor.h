#ifndef PrimaryVertexMonitor_H
#define PrimaryVertexMonitor_H

#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"


/** \class PrimaryVertexMonitor
 *
 *
 */

class PrimaryVertexMonitor : public DQMEDAnalyzer {
   public:
      explicit PrimaryVertexMonitor(const edm::ParameterSet& pSet);

      ~PrimaryVertexMonitor();

      virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
      virtual void analyze(const edm::Event &, const edm::EventSetup &) override;

   private:

  void pvTracksPlots(const reco::Vertex & v);
  void vertexPlots(const reco::Vertex & v, const reco::BeamSpot& beamSpot, int i);

  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  edm::EDGetTokenT<reco::BeamSpot>         beamspotToken_;
  
  edm::InputTag vertexInputTag_, beamSpotInputTag_;

  edm::ParameterSet conf_;

  std::string dqmLabel;

  std::string TopFolderName_;
  std::string AlignmentLabel_;

  // the histos
  MonitorElement *nbvtx, *nbgvtx, *nbtksinvtx[2], *trksWeight[2];
  MonitorElement *tt[2];
  MonitorElement *xrec[2] , *yrec[2], *zrec[2], *xDiff[2] , *yDiff[2], *xerr[2] , *yerr[2], *zerr[2] ;
  MonitorElement *xerrVsTrks[2] , *yerrVsTrks[2], *zerrVsTrks[2] ;
  MonitorElement * ntracksVsZ[2];
  MonitorElement *vtxchi2[2] , *vtxndf[2], *vtxprob[2] , *nans[2];
  MonitorElement *type[2];
  MonitorElement *bsX, *bsY, *bsZ, *bsSigmaZ, *bsDxdz, *bsDydz, *bsBeamWidthX, *bsBeamWidthY, *bsType;

  MonitorElement *sumpt, *ntracks, *weight, *chi2ndf, *chi2prob;
  MonitorElement *dxy, *dxy2, *dz, *dxyErr, *dzErr;
  MonitorElement *dxyVsPhi_pt1, *dzVsPhi_pt1;
  MonitorElement *dxyVsEta_pt1, *dzVsEta_pt1;
  MonitorElement *dxyVsPhi_pt10, *dzVsPhi_pt10;
  MonitorElement *dxyVsEta_pt10, *dzVsEta_pt10;


};


#endif

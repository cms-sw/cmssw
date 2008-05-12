/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/04/30 14:06:32 $
 *  $Revision: 1.6 $
 *  \author Suchandra Dutta , Giorgia Mila
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackingMonitor.h"
#include <string>

TrackingMonitor::TrackingMonitor(const edm::ParameterSet& iConfig) {
  dqmStore_ = edm::Service<DQMStore>().operator->();
  conf_ = iConfig;
}

TrackingMonitor::~TrackingMonitor() { }

void TrackingMonitor::beginJob(edm::EventSetup const& iSetup) {
  using namespace edm;

  std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 

  dqmStore_->setCurrentFolder(MEFolderName);

  //    
  int    TKNoBin = conf_.getParameter<int>("TkSizeBin");
  double TKNoMin = conf_.getParameter<double>("TkSizeMin");
  double TKNoMax = conf_.getParameter<double>("TkSizeMax");

  int    TKHitBin = conf_.getParameter<int>("RecHitBin");
  double TKHitMin = conf_.getParameter<double>("RecHitMin");
  double TKHitMax = conf_.getParameter<double>("RecHitMax");

  int    Chi2Bin  = conf_.getParameter<int>("Chi2Bin");
  double Chi2Min  = conf_.getParameter<double>("Chi2Min");
  double Chi2Max  = conf_.getParameter<double>("Chi2Max");

  int    PhiBin   = conf_.getParameter<int>("PhiBin");
  double PhiMin   = conf_.getParameter<double>("PhiMin");
  double PhiMax   = conf_.getParameter<double>("PhiMax");

  int    EtaBin   = conf_.getParameter<int>("EtaBin");
  double EtaMin   = conf_.getParameter<double>("EtaMin");
  double EtaMax   = conf_.getParameter<double>("EtaMax");

  int    ThetaBin = conf_.getParameter<int>("ThetaBin");
  double ThetaMin = conf_.getParameter<double>("ThetaMin");
  double ThetaMax = conf_.getParameter<double>("ThetaMax");


  histname = "NumberOfTracks_";
  NumberOfTracks = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKNoBin, TKNoMin, TKNoMax);

  histname = "NumberOfRecHitsPerTrack_";
  NumberOfRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrack->setAxisTitle("Number of RecHits of each track");

  histname = "NumberOfMeanRecHitsPerTrack_";
  NumberOfMeanRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfMeanRecHitsPerTrack->setAxisTitle("Mean number of RecHits per track");


  histname = "Chi2_";
  Chi2 = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, Chi2Bin, Chi2Min, Chi2Max);
  Chi2->setAxisTitle("Chi2 of each track");

  histname = "Chi2overDoF_";
  Chi2overDoF = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoF->setAxisTitle("Chi2 over nr. of degrees of freedom of each track");

  histname = "DistanceOfClosestApproach_";
  DistanceOfClosestApproach = dqmStore_->book1D(histname+AlgoName,histname+AlgoName,100, -0.5, 0.5);
  DistanceOfClosestApproach->setAxisTitle("Track distance of closest approach");

  histname = "DistanceOfClosestApproachVsTheta_";
  DistanceOfClosestApproachVsTheta = dqmStore_->book2D(histname+AlgoName,histname+AlgoName, ThetaBin, ThetaMin, ThetaMax, 100, -0.4, 0.4);
  DistanceOfClosestApproachVsTheta->setAxisTitle("Track polar angle",1);
  DistanceOfClosestApproachVsTheta->setAxisTitle("Track distance of closest approach",2);
  histname = "DistanceOfClosestApproachVsPhi_";
  DistanceOfClosestApproachVsPhi = dqmStore_->book2D(histname+AlgoName,histname+AlgoName, PhiBin, PhiMin, PhiMax, 100, -0.5, 0.5);
  
  DistanceOfClosestApproachVsPhi->setAxisTitle("Track azimuthal angle",1);
  DistanceOfClosestApproachVsPhi->setAxisTitle("Track distance of closest approach",2);
  histname = "DistanceOfClosestApproachVsEta_";
  DistanceOfClosestApproachVsEta = dqmStore_->book2D(histname+AlgoName,histname+AlgoName, EtaBin, EtaMin, EtaMax, 100, -0.5, 0.5);
  DistanceOfClosestApproachVsEta->setAxisTitle("Track pseudorapidity",1);
  DistanceOfClosestApproachVsEta->setAxisTitle("Track distance of closest approach",2);

  histname = "xPointOfClosestApproach_";
  xPointOfClosestApproach = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, 20, -20, 20);
  xPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the x-axis");

  histname = "yPointOfClosestApproach_";
  yPointOfClosestApproach = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, 20, -20, 20);
  yPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the y-axis");

  histname = "zPointOfClosestApproach_";
  zPointOfClosestApproach = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, 50, -100, 100);
  zPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the z-axis");

  std::string StateName = conf_.getParameter<std::string>("MeasurementState");
  if (StateName == "All") {
    bookHistosForState("OuterSurface");
    bookHistosForState("InnerSurface");
    bookHistosForState("ImpactPoint");
  } else {
    bookHistosForState(StateName);
  }
}
//
// -- Analyse
//
void TrackingMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  
  // std::string TrackProducer = conf_.getParameter<std::string>("TrackProducer");
  //  std::string TrackLabel = conf_.getParameter<std::string>("TrackLabel");
  InputTag TrackProducer = conf_.getParameter<edm::InputTag>("TrackProducer");
  
  Handle<reco::TrackCollection> trackCollection;
  //  iEvent.getByLabel(TrackProducer, TrackLabel, trackCollection);
  iEvent.getByLabel(TrackProducer, trackCollection);
  if (!trackCollection.isValid()) return;


  NumberOfTracks->Fill(trackCollection->size());
  std::string StateName = conf_.getParameter<std::string>("MeasurementState");
  
  int totalRecHits = 0;
  for (reco::TrackCollection::const_iterator track = trackCollection->begin(); track!=trackCollection->end(); ++track) {

    NumberOfRecHitsPerTrack->Fill(track->recHitsSize());
    totalRecHits += track->recHitsSize();

    Chi2->Fill(track->chi2());
    Chi2overDoF->Fill(track->normalizedChi2());

    DistanceOfClosestApproach->Fill(track->d0());
    DistanceOfClosestApproachVsTheta->Fill(track->theta(), track->d0());
    DistanceOfClosestApproachVsPhi->Fill(track->phi(), track->d0());
    DistanceOfClosestApproachVsEta->Fill(track->eta(), track->d0());
    
    xPointOfClosestApproach->Fill(track->vertex().x());
    yPointOfClosestApproach->Fill(track->vertex().y());
    zPointOfClosestApproach->Fill(track->vertex().z());
    

    if (StateName == "All") {
      fillHistosForState(iSetup, (*track), std::string("OuterSurface"));
      fillHistosForState(iSetup, (*track), std::string("InnerSurface"));
      fillHistosForState(iSetup, (*track), std::string("ImactPoint"));
    } else {
      fillHistosForState(iSetup, (*track), StateName);
    }
  }

  double meanrechits = 0;
  // check that track size to avoid division by zero.
  if (trackCollection->size()) meanrechits = static_cast<double>(totalRecHits)/static_cast<double>(trackCollection->size());
  NumberOfMeanRecHitsPerTrack->Fill(meanrechits);
}


void TrackingMonitor::endJob(void) {
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dqmStore_->showDirStructure();
    dqmStore_->save(outputFileName);
  }
}

// 
// book histograms at differnt measurement points
// 
void TrackingMonitor::bookHistosForState(std::string sname) {

  std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 

  int    TKHitBin   = conf_.getParameter<int>("RecHitBin");
  double TKHitMin   = conf_.getParameter<double>("RecHitMin");
  double TKHitMax   = conf_.getParameter<double>("RecHitMax");

  int    Chi2Bin    = conf_.getParameter<int>("Chi2Bin");
  double Chi2Min    = conf_.getParameter<double>("Chi2Min");
  double Chi2Max    = conf_.getParameter<double>("Chi2Max");

  int    PhiBin     = conf_.getParameter<int>("PhiBin");
  double PhiMin     = conf_.getParameter<double>("PhiMin");
  double PhiMax     = conf_.getParameter<double>("PhiMax");

  int    EtaBin     = conf_.getParameter<int>("EtaBin");
  double EtaMin     = conf_.getParameter<double>("EtaMin");
  double EtaMax     = conf_.getParameter<double>("EtaMax");

  int    ThetaBin   = conf_.getParameter<int>("ThetaBin");
  double ThetaMin   = conf_.getParameter<double>("ThetaMin");
  double ThetaMax   = conf_.getParameter<double>("ThetaMax");

  int    TrackPtBin = conf_.getParameter<int>("TrackPtBin");
  double TrackPtMin = conf_.getParameter<double>("TrackPtMin");
  double TrackPtMax = conf_.getParameter<double>("TrackPtMax");

  int    TrackPxBin = conf_.getParameter<int>("TrackPxBin");
  double TrackPxMin = conf_.getParameter<double>("TrackPxMin");
  double TrackPxMax = conf_.getParameter<double>("TrackPxMax");

  int    TrackPyBin = conf_.getParameter<int>("TrackPyBin");
  double TrackPyMin = conf_.getParameter<double>("TrackPyMin");
  double TrackPyMax = conf_.getParameter<double>("TrackPyMax");

  int    TrackPzBin = conf_.getParameter<int>("TrackPzBin");
  double TrackPzMin = conf_.getParameter<double>("TrackPzMin");
  double TrackPzMax = conf_.getParameter<double>("TrackPzMax");

  int    ptErrBin   = conf_.getParameter<int>("ptErrBin");
  double ptErrMin   = conf_.getParameter<double>("ptErrMin");
  double ptErrMax   = conf_.getParameter<double>("ptErrMax");

  int    pxErrBin   = conf_.getParameter<int>("pxErrBin");
  double pxErrMin   = conf_.getParameter<double>("pxErrMin");
  double pxErrMax   = conf_.getParameter<double>("pxErrMax");

  int    pyErrBin   = conf_.getParameter<int>("pyErrBin");
  double pyErrMin   = conf_.getParameter<double>("pyErrMin");
  double pyErrMax   = conf_.getParameter<double>("pyErrMax");

  int    pzErrBin   = conf_.getParameter<int>("pzErrBin");
  double pzErrMin   = conf_.getParameter<double>("pzErrMin");
  double pzErrMax   = conf_.getParameter<double>("pzErrMax");

  int    pErrBin    = conf_.getParameter<int>("pErrBin");
  double pErrMin    = conf_.getParameter<double>("pErrMin");
  double pErrMax    = conf_.getParameter<double>("pErrMax");

  int    phiErrBin  = conf_.getParameter<int>("phiErrBin");
  double phiErrMin  = conf_.getParameter<double>("phiErrMin");
  double phiErrMax  = conf_.getParameter<double>("phiErrMax");

  int    etaErrBin  = conf_.getParameter<int>("etaErrBin");
  double etaErrMin  = conf_.getParameter<double>("etaErrMin");
  double etaErrMax  = conf_.getParameter<double>("etaErrMax");

  std::string htag;

  if (sname == "default") htag = AlgoName;
  else                    htag = sname + "_" + AlgoName; 
  dqmStore_->setCurrentFolder(MEFolderName);

  TkParameterMEs tkmes;

  histname = "NumberOfRecHitsPerTrackVsPhi_" + htag;
  tkmes.NumberOfRecHitsPerTrackVsPhi = dqmStore_->book2D(histname, histname, PhiBin, PhiMin, PhiMax, TKHitBin, TKHitMin, TKHitMax);
  tkmes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Track azimuthal angle",1);
  tkmes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Number of RecHits of each track",2);

  histname = "NumberOfRecHitsPerTrackVsTheta_" + htag;
  tkmes.NumberOfRecHitsPerTrackVsTheta = dqmStore_->book2D(histname, histname, ThetaBin, ThetaMin, ThetaMax, TKHitBin, TKHitMin, TKHitMax);
  tkmes.NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Track polar angle",1);
  tkmes.NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Number of RecHits of each track",2);

  histname = "NumberOfRecHitsPerTrackVsEta_" + htag;
  tkmes.NumberOfRecHitsPerTrackVsEta = dqmStore_->book2D(histname, histname, EtaBin, EtaMin, EtaMax, TKHitBin, TKHitMin, TKHitMax);
  tkmes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Track pseudorapidity",1);
  tkmes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Number of RecHits of each track",2);

  histname = "Chi2overDoFVsTheta_" + htag;
  tkmes.Chi2overDoFVsTheta = dqmStore_->book2D(histname, histname, ThetaBin, ThetaMin, ThetaMax, Chi2Bin, Chi2Min, Chi2Max);
  tkmes.Chi2overDoFVsTheta->setAxisTitle("Track polar angle",1);
  tkmes.Chi2overDoFVsTheta->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);

  histname = "Chi2overDoFVsPhi_" + htag;
  tkmes.Chi2overDoFVsPhi   = dqmStore_->book2D(histname, histname, PhiBin, PhiMin, PhiMax, Chi2Bin, Chi2Min, Chi2Max);
  tkmes.Chi2overDoFVsPhi->setAxisTitle("Track azimuthal angle",1);
  tkmes.Chi2overDoFVsPhi->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);

  histname = "Chi2overDoFVsEta_" + htag;
  tkmes.Chi2overDoFVsEta   = dqmStore_->book2D(histname, histname, EtaBin, EtaMin, EtaMax, Chi2Bin, Chi2Min, Chi2Max);
  tkmes.Chi2overDoFVsEta->setAxisTitle("Track pseudorapidity",1);
  tkmes.Chi2overDoFVsEta->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);

  histname = "TrackPt_" + htag;
  tkmes.TrackPt = dqmStore_->book1D(histname, histname, TrackPtBin, TrackPtMin, TrackPtMax);
  tkmes.TrackPt->setAxisTitle("Transverse track momentum");

  histname = "TrackPx_" + htag;
  tkmes.TrackPx = dqmStore_->book1D(histname, histname, TrackPxBin, TrackPxMin, TrackPxMax);
  tkmes.TrackPx->setAxisTitle("x component of track momentum");

  histname = "TrackPy_" + htag;
  tkmes.TrackPy = dqmStore_->book1D(histname, histname, TrackPyBin, TrackPyMin, TrackPyMax);
  tkmes.TrackPy->setAxisTitle("y component of track momentum");

  histname = "TrackPz_" + htag;
  tkmes.TrackPz = dqmStore_->book1D(histname, histname, TrackPzBin, TrackPzMin, TrackPzMax);
  tkmes.TrackPz->setAxisTitle("z component of track momentum");

  histname = "TrackPhi_" + htag;
  tkmes.TrackPhi = dqmStore_->book1D(histname, histname, PhiBin, PhiMin, PhiMax);
  tkmes.TrackPhi->setAxisTitle("Track azimuthal angle");

  histname = "TrackEta_" + htag;
  tkmes.TrackEta = dqmStore_->book1D(histname, histname, EtaBin, EtaMin, EtaMax);
  tkmes.TrackEta->setAxisTitle("Track pseudorapidity");

  histname = "TrackTheta_" + htag;
  tkmes.TrackTheta = dqmStore_->book1D(histname, histname, ThetaBin, ThetaMin, ThetaMax);
  tkmes.TrackTheta->setAxisTitle("Track polar angle");

  histname = "TrackPtErrOverPt_" + htag;
  tkmes.TrackPtErr = dqmStore_->book1D(histname, histname, ptErrBin, ptErrMin, ptErrMax);
  tkmes.TrackPtErr->setAxisTitle("ptErr/pt");
  
  histname = "TrackPxErrOverPx_" + htag;
  tkmes.TrackPxErr = dqmStore_->book1D(histname, histname, pxErrBin, pxErrMin, pxErrMax);
  tkmes.TrackPxErr->setAxisTitle("pxErr/px");
  
  histname = "TrackPyErrOverPy_" + htag;
  tkmes.TrackPyErr = dqmStore_->book1D(histname, histname, pyErrBin, pyErrMin, pyErrMax);
  tkmes.TrackPyErr->setAxisTitle("pyErr/py");
  
  histname = "TrackPzErrOverPz_" + htag;
  tkmes.TrackPzErr = dqmStore_->book1D(histname, histname, pzErrBin, pzErrMin, pzErrMax);
  tkmes.TrackPzErr->setAxisTitle("pzErr/pz");
  
  histname = "TrackPErrOverP_" + htag;
  tkmes.TrackPErr = dqmStore_->book1D(histname, histname, pErrBin, pErrMin, pErrMax);
  tkmes.TrackPErr->setAxisTitle("pErr/p");
  
  histname = "TrackPhiErr_" + htag;
  tkmes.TrackPhiErr = dqmStore_->book1D(histname, histname, phiErrBin, phiErrMin, phiErrMax);
  tkmes.TrackPhiErr->setAxisTitle("phiErr");
  
  histname = "TrackEtaErr_" + htag;
  tkmes.TrackEtaErr = dqmStore_->book1D(histname, histname, etaErrBin, etaErrMin, etaErrMax);
  tkmes.TrackEtaErr->setAxisTitle("etaErr");

  // now put the MEs in the map
  TkParameterMEMap.insert(std::make_pair(sname, tkmes));

}
// 
// fill histograms at differnt measurement points
// 
void TrackingMonitor::fillHistosForState(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname) {


  //get the kinematic parameters
  double px, py, pz, pt, theta, phi, eta;
  double pxerror, pyerror, pzerror, pterror, perror, phierror, etaerror;

  if (sname == "default") {

    px    = track.px();
    py    = track.py();
    pz    = track.pz();
    pt    = track.pt();
    phi   = track.phi();
    theta = track.theta();
    eta   = track.eta();
    
    pterror  = (pt) ? track.ptError()/pt : 0.0;
    pxerror  = -1.0;
    pyerror  = -1.0;
    pzerror  = -1.0;
    perror   = -1.0;
    phierror = track.phiError();
    etaerror = track.etaError();
  } else {

    edm::ESHandle<TransientTrackBuilder> theB;
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
    reco::TransientTrack TransTrack = theB->build(track);
  
    TrajectoryStateOnSurface TSOS;

    if      (sname == "OuterSurface")  TSOS = TransTrack.outermostMeasurementState();
    else if (sname == "InnerSurface")  TSOS = TransTrack.innermostMeasurementState();
    else if (sname == "ImpactPoint")   TSOS = TransTrack.impactPointState();;

    px    = TSOS.globalMomentum().x();
    py    = TSOS.globalMomentum().y();
    pz    = TSOS.globalMomentum().z();
    pt    = TSOS.globalMomentum().perp();
    phi   = TSOS.globalMomentum().phi();
    theta = TSOS.globalMomentum().theta();
    eta   = TSOS.globalMomentum().eta();

    //get the error of the kinimatic parameters
    AlgebraicSymMatrix66 errors = TSOS.cartesianError().matrix();
    double partialPterror = errors(3,3)*pow(TSOS.globalMomentum().x(),2) + errors(4,4)*pow(TSOS.globalMomentum().y(),2);
    pterror  = sqrt(partialPterror)/TSOS.globalMomentum().perp();
    pxerror  = sqrt(errors(3,3))/TSOS.globalMomentum().x();
    pyerror  = sqrt(errors(4,4))/TSOS.globalMomentum().y();
    pzerror  = sqrt(errors(5,5))/TSOS.globalMomentum().z();
    perror   = sqrt(partialPterror+errors(5,5)*pow(TSOS.globalMomentum().z(),2))/TSOS.globalMomentum().mag();
    phierror = sqrt(TSOS.curvilinearError().matrix()(2,2));
    etaerror = sqrt(TSOS.curvilinearError().matrix()(1,1))*fabs(sin(TSOS.globalMomentum().theta()));
  }
  std::map<std::string, TkParameterMEs>::iterator iPos = TkParameterMEMap.find(sname); 
  if (iPos != TkParameterMEMap.end()) {
    TkParameterMEs tkmes = iPos->second;
    
    tkmes.TrackPx->Fill(px);
    tkmes.TrackPy->Fill(py);
    tkmes.TrackPz->Fill(pz);
    tkmes.TrackPt->Fill(pt);
    
    tkmes.TrackPhi->Fill(phi);
    tkmes.TrackEta->Fill(eta);
    tkmes.TrackTheta->Fill(theta);
    tkmes.NumberOfRecHitsPerTrackVsPhi->Fill(phi, track.recHitsSize());
    tkmes.NumberOfRecHitsPerTrackVsTheta->Fill(theta, track.recHitsSize());
    tkmes.NumberOfRecHitsPerTrackVsEta->Fill(eta, track.recHitsSize());
    
    tkmes.Chi2overDoFVsTheta->Fill(theta, track.normalizedChi2());
    tkmes.Chi2overDoFVsPhi->Fill(phi, track.normalizedChi2());
    tkmes.Chi2overDoFVsEta->Fill(eta, track.normalizedChi2());

    tkmes.TrackPtErr->Fill(pterror);
    tkmes.TrackPxErr->Fill(pxerror);
    tkmes.TrackPyErr->Fill(pyerror);
    tkmes.TrackPzErr->Fill(pzerror);
    tkmes.TrackPErr->Fill(perror);
    tkmes.TrackPhiErr->Fill(phierror);
    tkmes.TrackEtaErr->Fill(etaerror);
  }
}
DEFINE_FWK_MODULE(TrackingMonitor);

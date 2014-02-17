/*
 * $Id: MuonResidualsFromTrack.cc,v 1.5 2011/10/12 23:40:24 khotilov Exp $ 
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonDT13ChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonDT2ChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonCSCChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonTrackDT13ChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonTrackDT2ChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonTrackCSCChamberResidual.h"

#include "TDecompChol.h"


void MuonResidualsFromTrack::addTrkCovMatrix(DetId chamberId, TrajectoryStateOnSurface &tsos)
{
  const AlgebraicSymMatrix55 cov55 = tsos.localError().matrix();
  TMatrixDSym cov44(4);
  // change indices from q/p,dxdz,dydz,x,y   to   x,y,dxdz,dydz
  int subs[4] = { 3, 4, 1, 2 };
  for (int i=0;i<4;i++) for (int j=0;j<4;j++)  cov44(i,j) = cov55( subs[i], subs[j] );
  m_trkCovMatrix[chamberId] = cov44;
}


MuonResidualsFromTrack::MuonResidualsFromTrack(edm::ESHandle<GlobalTrackingGeometry> globalGeometry,
                                               const Trajectory *traj,
                                               const reco::Track* trk,
                                               AlignableNavigator *navigator, double maxResidual)
  : track(trk)
{
  clear();

  std::vector<TrajectoryMeasurement> measurements = traj->measurements();
  for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin();  im != measurements.end();  ++im) {
    TrajectoryMeasurement meas = *im;
    const TransientTrackingRecHit *hit = &(*meas.recHit());
    DetId id = hit->geographicalId();

    if (hit->isValid()) {
      TrajectoryStateOnSurface tsos = m_tsoscomb(meas.forwardPredictedState(), meas.backwardPredictedState());
      if (tsos.isValid()  &&  fabs(tsos.localPosition().x() - hit->localPosition().x()) < maxResidual) {

	if (id.det() == DetId::Tracker) {
	  double xresid = tsos.localPosition().x() - hit->localPosition().x();
	  double xresiderr2 = tsos.localError().positionError().xx() + hit->localPositionError().xx();

	  m_tracker_numHits++;
	  m_tracker_chi2 += xresid * xresid / xresiderr2;

	  if (id.subdetId() == StripSubdetector::TID  ||  id.subdetId() == StripSubdetector::TEC) m_contains_TIDTEC = true;
	}

	else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
	  const DTChamberId chamberId(id.rawId());
	  const DTSuperLayerId superLayerId(id.rawId());

	  // have we seen this chamber before?
	  if (m_dt13.find(chamberId) == m_dt13.end()  &&  m_dt2.find(chamberId) == m_dt2.end()) {
	    m_chamberIds.push_back(chamberId);
	    //addTrkCovMatrix(chamberId, tsos); // only for the 1st hit
	  }

	  if (superLayerId.superlayer() == 2) {
	    if (m_dt2.find(chamberId) == m_dt2.end()) {
	      AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
	      m_dt2[chamberId] = new MuonDT2ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
	    }

	    m_dt2[chamberId]->addResidual(&tsos, hit);
	  }

	  else {
	    if (m_dt13.find(chamberId) == m_dt13.end()) {
	      AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
	      m_dt13[chamberId] = new MuonDT13ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
	    }

	    m_dt13[chamberId]->addResidual(&tsos, hit);
	  }
	}

	else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	  const CSCDetId cscDetId(id.rawId());
	  const CSCDetId chamberId(cscDetId.endcap(), cscDetId.station(), cscDetId.ring(), cscDetId.chamber());

	  // not sure why we sometimes get layer == 0
	  if (cscDetId.layer() == 0) continue;

	  // have we seen this chamber before?
	  if (m_csc.find(chamberId) == m_csc.end())
	  {
	    m_chamberIds.push_back(chamberId);
            //addTrkCovMatrix(chamberId, tsos); // only for the 1st hit
	    AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
	    m_csc[chamberId] = new MuonCSCChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
	  }

	  m_csc[chamberId]->addResidual(&tsos, hit);
	}

      } // end if track propagation is valid
    } // end if hit is valid
  } // end loop over measurments
}


MuonResidualsFromTrack::MuonResidualsFromTrack(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, const reco::Muon *mu, AlignableNavigator *navigator, double maxResidual)
  : muon(mu)
{
  clear();
  assert( muon->isTrackerMuon() && muon->innerTrack().isNonnull());
  track = muon->innerTrack().get();
  
  m_tracker_chi2 = muon->innerTrack()->chi2();
  m_tracker_numHits = muon->innerTrack()->ndof() + 5;
  m_tracker_numHits = m_tracker_numHits > 0 ? m_tracker_numHits : 0 ;
  
  /*
  for (trackingRecHit_iterator hit = muon->innerTrack()->recHitsBegin();  hit != muon->innerTrack()->recHitsEnd();  ++hit)
  {
    DetId id = (*hit)->geographicalId();
    if (id.det() == DetId::Tracker)
    {
      m_tracker_numHits++;
      if (id.subdetId() == StripSubdetector::TID  ||  id.subdetId() == StripSubdetector::TEC) m_contains_TIDTEC = true;
    }
  }
  */
  
  for (std::vector<reco::MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();  
       chamberMatch != muon->matches().end();  chamberMatch++)
  {
    if (chamberMatch->id.det() != DetId::Muon ) continue;
    
    for (std::vector<reco::MuonSegmentMatch>::const_iterator segMatch = chamberMatch->segmentMatches.begin();
         segMatch != chamberMatch->segmentMatches.end();  ++segMatch)
    {
      // select the only segment that belongs to track and is the best in station by dR
      if (! (segMatch->isMask(reco::MuonSegmentMatch::BestInStationByDR) &&
             segMatch->isMask(reco::MuonSegmentMatch::BelongsToTrackByDR)) ) continue;
      
      if (chamberMatch->id.subdetId() == MuonSubdetId::DT)
      {
        const DTChamberId chamberId(chamberMatch->id.rawId());

        DTRecSegment4DRef segmentDT = segMatch->dtSegmentRef;
        const DTRecSegment4D* segment = segmentDT.get();
        if (segment == 0)  continue;
        
        if ( segment->hasPhi()  &&  fabs(chamberMatch->x - segMatch->x) > maxResidual ) continue;
        if ( segment->hasZed()  &&  fabs(chamberMatch->y - segMatch->y) > maxResidual ) continue;
          
        // have we seen this chamber before?
        if (m_dt13.find(chamberId) == m_dt13.end()  &&  m_dt2.find(chamberId) == m_dt2.end()) {
          m_chamberIds.push_back(chamberId);
        }

        if (segment->hasZed())
        {
          if (m_dt2.find(chamberId) == m_dt2.end())
          {
            AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
            m_dt2[chamberId] = new MuonTrackDT2ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
          }
          else std::cout<<"multi segment match to tmuon: dt2  -- should not happen!"<<std::endl;
          m_dt2[chamberId]->setSegmentResidual(&(*chamberMatch), &(*segMatch));
        }
        if (segment->hasPhi())
        {
          if (m_dt13.find(chamberId) == m_dt13.end())
          {
            AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
            m_dt13[chamberId] = new MuonTrackDT13ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
          }
          else std::cout<<"multi segment match to tmuon: dt13  -- should not happen!"<<std::endl;
          m_dt13[chamberId]->setSegmentResidual(&(*chamberMatch), &(*segMatch));
        }
      }

      else if (chamberMatch->id.subdetId() == MuonSubdetId::CSC) 
      {
        const CSCDetId cscDetId(chamberMatch->id.rawId());
        const CSCDetId chamberId(cscDetId.chamberId());

        if ( fabs(chamberMatch->x - segMatch->x) > maxResidual ) continue;

        // have we seen this chamber before?
        if (m_csc.find(chamberId) == m_csc.end())
        {
          m_chamberIds.push_back(chamberId);
          AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
          m_csc[chamberId] = new MuonTrackCSCChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
        }
        else std::cout<<"multi segment match to tmuon: csc  -- should not happen!"<<std::endl;
        m_csc[chamberId]->setSegmentResidual(&(*chamberMatch), &(*segMatch));
      }

    }
  }
}


MuonResidualsFromTrack::~MuonResidualsFromTrack() 
{
  for (std::map<DetId,MuonChamberResidual*>::const_iterator residual = m_dt13.begin();  residual != m_dt13.end();  ++residual) {
    delete residual->second;
  }
  for (std::map<DetId,MuonChamberResidual*>::const_iterator residual = m_dt2.begin();  residual != m_dt2.end();  ++residual) {
    delete residual->second;
  }
  for (std::map<DetId,MuonChamberResidual*>::const_iterator residual = m_csc.begin();  residual != m_csc.end();  ++residual) {
    delete residual->second;
  }
}


void MuonResidualsFromTrack::clear()
{
  m_tracker_numHits = 0;
  m_tracker_chi2 = 0.;
  m_contains_TIDTEC = false;
  m_chamberIds.clear();
  m_dt13.clear();
  m_dt2.clear();
  m_csc.clear();
  m_trkCovMatrix.clear();
}


double MuonResidualsFromTrack::trackerRedChi2() const 
{
  if (m_tracker_numHits > 5) return m_tracker_chi2 / double(m_tracker_numHits - 5);
  else return -1.;
}


double MuonResidualsFromTrack::normalizedChi2() const 
{
  if (muon) return track->normalizedChi2();
  return trackerRedChi2();
}


MuonChamberResidual * MuonResidualsFromTrack::chamberResidual(DetId chamberId, int type)
{
  if (type == MuonChamberResidual::kDT13) {
    if (m_dt13.find(chamberId) == m_dt13.end()) return NULL;
    return m_dt13[chamberId];
  }
  else if (type == MuonChamberResidual::kDT2) {
    if (m_dt2.find(chamberId) == m_dt2.end()) return NULL;
    return m_dt2[chamberId];
  }
  else if (type == MuonChamberResidual::kCSC) {
    if (m_csc.find(chamberId) == m_csc.end()) return NULL;
    return m_csc[chamberId];
  }
  else return NULL;
}


TMatrixDSym MuonResidualsFromTrack::covMatrix(DetId chamberId)
{
  TMatrixDSym result(4);
  std::cout<<"MuonResidualsFromTrack:: cov initial:"<<std::endl;
  result.Print();
  if (m_trkCovMatrix.find(chamberId) == m_trkCovMatrix.end())
  {
    std::cout<<"MuonResidualsFromTrack:: cov does not exist!"<<std::endl;
    return result;
  }
  result = m_trkCovMatrix[chamberId];

  std::cout<<"MuonResidualsFromTrack:: cov before:"<<std::endl;
  result.Print();

  // add segment's errors in quadratures to track's covariance matrix
  double r_err;
  if (m_csc.find(chamberId) == m_csc.end())
  {
    r_err = m_csc[chamberId]->residual_error();
    result(0,0) += r_err*r_err;
    r_err = m_csc[chamberId]->resslope_error();
    result(2,2) += r_err*r_err;
  }
  if (m_dt13.find(chamberId) == m_dt13.end())
  {
    r_err = m_dt13[chamberId]->residual_error();
    result(0,0) += r_err*r_err;
    r_err = m_dt13[chamberId]->resslope_error();
    result(2,2) += r_err*r_err;
  }
  if (m_dt2.find(chamberId) == m_dt2.end())
  {
    r_err = m_dt2[chamberId]->residual_error();
    result(1,1) += r_err*r_err;
    r_err = m_dt2[chamberId]->resslope_error();
    result(3,3) += r_err*r_err;
  }
  std::cout<<"MuonResidualsFromTrack:: cov after:"<<std::endl;
  result.Print();

  return result;
}

TMatrixDSym MuonResidualsFromTrack::corrMatrix(DetId chamberId)
{
  TMatrixDSym result(4);
  TMatrixDSym cov44 = covMatrix(chamberId);

  // invert it using cholesky decomposition
  TDecompChol decomp(cov44);
  bool ok = decomp.Invert(result);
  std::cout<<"MuonResidualsFromTrack:: corr after:"<<std::endl;
  result.Print();

  if (!ok){std::cout<<"MuonResidualsFromTrack:: cov inversion failed!"<<std::endl;}
  return result;
}

TMatrixD MuonResidualsFromTrack::choleskyCorrMatrix(DetId chamberId)
{
  TMatrixD result(4,4);
  TMatrixDSym corr44 = corrMatrix(chamberId);

  // get an upper triangular matrix U such that corr = U^T * U
  TDecompChol decomp(corr44);
  bool ok = decomp.Decompose();
  result = decomp.GetU();

  std::cout<<"MuonResidualsFromTrack:: corr cholesky after:"<<std::endl;
  result.Print();

  if (!ok){std::cout<<"MuonResidualsFromTrack:: corr decomposition failed!"<<std::endl;}
  return result;
}

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"

MuonResidualsFromTrack::MuonResidualsFromTrack(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, const Trajectory *traj, AlignableNavigator *navigator, double maxResidual) {
  m_tracker_numHits = 0;
  m_tracker_chi2 = 0.;
  m_contains_TIDTEC = false;
  m_chamberIds.clear();
  m_dt13.clear();
  m_dt2.clear();
  m_csc.clear();

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
	  if (cscDetId.layer() != 0) {

	     // have we seen this chamber before?
	     if (m_csc.find(chamberId) == m_csc.end()) {
		m_chamberIds.push_back(chamberId);

		AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
		m_csc[chamberId] = new MuonCSCChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
	     }

	     m_csc[chamberId]->addResidual(&tsos, hit);

	  } // end if cscDetId.layer() != 0
	}

      } // end if track propagation is valid
    } // end if hit is valid
  } // end loop over measurments
}

MuonResidualsFromTrack::~MuonResidualsFromTrack() {
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

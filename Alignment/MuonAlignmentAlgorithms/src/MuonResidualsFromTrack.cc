#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"

MuonResidualsFromTrack::MuonResidualsFromTrack(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, const Trajectory *traj, AlignableNavigator *navigator) {
  m_tracker_numHits = 0;
  m_tracker_chi2 = 0.;
  m_contains_TIDTEC = false;
  m_chamberIds.clear();
  m_chamberResiduals.clear();

  std::vector<TrajectoryMeasurement> measurements = traj->measurements();
  for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin();  im != measurements.end();  ++im) {
    TrajectoryMeasurement meas = *im;
    const TransientTrackingRecHit *hit = &(*meas.recHit());
    DetId id = hit->geographicalId();

    if (hit->isValid()) {
      TrajectoryStateOnSurface tsos = m_tsoscomb(meas.forwardPredictedState(), meas.backwardPredictedState());
      if (tsos.isValid()) {

	if (id.det() == DetId::Tracker) {
	  double xresid = tsos.localPosition().x() - hit->localPosition().x();
	  double xresiderr2 = tsos.localError().positionError().xx() + hit->localPositionError().xx();

	  m_tracker_numHits++;
	  m_tracker_chi2 += xresid * xresid / xresiderr2;

	  if (id.subdetId() == StripSubdetector::TID  ||  id.subdetId() == StripSubdetector::TEC) m_contains_TIDTEC = true;
	}
	
	else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
	  const DTChamberId dtChamberId(id.rawId());
	  const DetId chamberId(dtChamberId);

	  if (m_chamberResiduals.find(dtChamberId) == m_chamberResiduals.end()) {
	    AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);

	    m_chamberResiduals[dtChamberId] = new MuonDTChamberResidual(globalGeometry, navigator, dtChamberId, chamberAlignable);
	    m_chamberIds.push_back(dtChamberId);
	  }
	  
	  m_chamberResiduals[dtChamberId]->addResidual(&tsos, hit);
	}

	else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	  const CSCDetId cscDetId(id.rawId());
	  const CSCDetId cscChamberId(cscDetId.endcap(), cscDetId.station(), cscDetId.ring(), cscDetId.chamber());
	  const DetId chamberId(cscChamberId);

	  if (m_chamberResiduals.find(cscChamberId) == m_chamberResiduals.end()) {
	    AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);

	    // m_chamberResiduals[cscChamberId] = new MuonCSCChamberResidual(globalGeometry, navigator, cscChamberId, chamberAlignable);
	    // m_chamberIds.push_back(cscChamberId);
	  }

	  // m_chamberResiduals[cscChamberId]->addResidual(&tsos, hit);
	}

      } // end if track projection is valid
    } // end if hit is valid
  } // end loop over measurments
}

MuonResidualsFromTrack::~MuonResidualsFromTrack() {
  for (std::map<DetId,MuonChamberResidual*>::const_iterator residual = m_chamberResiduals.begin();  residual != m_chamberResiduals.end();  ++residual) {
    delete residual->second;
  }
}

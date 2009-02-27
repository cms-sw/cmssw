#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"

MuonResidualsFromTrack::MuonResidualsFromTrack(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, const Trajectory *traj, AlignableNavigator *navigator, double maxResidual) {
  m_tracker_numHits = 0;
  m_tracker_chi2 = 0.;
  m_contains_TIDTEC = false;
  m_indexes.clear();
  m_chamberResiduals.clear();

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
	  unsigned int index = chamberId.rawId() * 2;
	  if (superLayerId.superlayer() == 2) index += 1;
	  
	  if (m_chamberResiduals.find(index) == m_chamberResiduals.end()) {
	    // const DetId chamberId_DetId(chamberId);
	    AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);  // _DetId);

	    if (superLayerId.superlayer() == 2) {
	      m_chamberResiduals[index] = new MuonDT2ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
	    }
	    else {
	      m_chamberResiduals[index] = new MuonDT13ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
	    }

	    m_indexes.push_back(index);
	  } // end if we've never seen this chamber before

	  m_chamberResiduals[index]->addResidual(&tsos, hit);
	}

	else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	  const CSCDetId cscDetId(id.rawId());
	  const CSCDetId chamberId(cscDetId.endcap(), cscDetId.station(), cscDetId.ring(), cscDetId.chamber());
	  unsigned int index = chamberId.rawId() * 2;

	  if (m_chamberResiduals.find(index) == m_chamberResiduals.end()) {
	    // const DetId chamberId_DetId(chamberId);
	    AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);  // _DetId);

	    m_chamberResiduals[index] = new MuonCSCChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);

	    m_indexes.push_back(index);
	  } // end if we've never seen this chamber before

	  m_chamberResiduals[index]->addResidual(&tsos, hit);
	}
      } // end if track projection is valid
    } // end if hit is valid
  } // end loop over measurments
}

MuonResidualsFromTrack::~MuonResidualsFromTrack() {
  for (std::map<unsigned int,MuonChamberResidual*>::const_iterator residual = m_chamberResiduals.begin();  residual != m_chamberResiduals.end();  ++residual) {
    delete residual->second;
  }
}

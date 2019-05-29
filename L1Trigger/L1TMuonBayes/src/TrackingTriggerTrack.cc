/*
 * TrackingTriggerTrack.cc
 *
 *  Created on: Jan 25, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "L1Trigger/L1TMuonBayes/interface/TrackingTriggerTrack.h"

TrackingTriggerTrack::TrackingTriggerTrack(const edm::Ptr< SimTrack >& simTrackPtr): simTrackPtr(simTrackPtr) {
  eta = simTrackPtr->momentum().eta();
  phi = simTrackPtr->momentum().phi();
  pt = simTrackPtr->momentum().pt();
  charge = simTrackPtr->type()/-13;
  if(simTrackPtr->type() == 13) //muon
    charge = -1;
  else if(simTrackPtr->type() == -13) //muon
        charge = 1;
  else {
    charge = (simTrackPtr->type() < 0 ? 1 : -1); //not necessary correct
  }
}

TrackingTriggerTrack::TrackingTriggerTrack(const edm::Ptr< TrackingParticle >& trackingParticlePtr): trackingParticlePtr(trackingParticlePtr) {
  eta = trackingParticlePtr->eta();
  phi = trackingParticlePtr->phi();
  pt = trackingParticlePtr->pt();
  charge = trackingParticlePtr->pdgId()/-13;
  if(trackingParticlePtr->pdgId() == 13) //muon
    charge = -1;
  else if(trackingParticlePtr->pdgId() == -13) //muon
        charge = 1;
  else {
    charge = (trackingParticlePtr->pdgId() < 0 ? 1 : -1); //not necessary correct
  }
}

TrackingTriggerTrack::TrackingTriggerTrack(const edm::Ptr< TTTrack< Ref_Phase2TrackerDigi_ > >& ttTrackPtr, int l1Tk_nPar): ttTrackPtr(ttTrackPtr) {
  pt   = ttTrackPtr->getMomentum(l1Tk_nPar).perp();
  eta  = ttTrackPtr->getMomentum(l1Tk_nPar).eta();
  phi  = ttTrackPtr->getMomentum(l1Tk_nPar).phi();

  charge = (ttTrackPtr->getRInv() > 0 ? 1 : -1); //ttTRack.ge //where is the charge???? TODO

  index = ttTrackPtr.key();
}

std::ostream & operator << (std::ostream &out, const TrackingTriggerTrack& ttTrack) {
  out <<"ttTrack: "
      <<" idx "<<ttTrack.index
      <<" charge "<<std::setw(2)<<ttTrack.charge
      <<" pt "<<std::setw(5)<<ttTrack.pt<<" GeV hw "<<std::setw(5)<<ttTrack.ptHw<<" bin "<<std::setw(3)<<ttTrack.ptBin<<" | "//<<std::endl;
      <<" eta "<<std::setw(5)<<ttTrack.eta<<" hw "<<std::setw(5)<<ttTrack.etaHw<<" bin "<<std::setw(3)<<ttTrack.etaBin<<" | "
      <<" phi "<<std::setw(5)<<ttTrack.phi<<" hw "<<std::setw(5)<<ttTrack.phiHw<<" "<<(ttTrack.phi * 180.0 / M_PI)<<" deg";

  return out;
}

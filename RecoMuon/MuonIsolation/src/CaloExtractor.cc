#include "RecoMuon/MuonIsolation/src/CaloExtractor.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

CaloExtractor::CaloExtractor(const ParameterSet& par)
{
  theCaloTowerCollectionLabel = par.getUntrackedParameter<string>("CaloTowerCollectionLabel");
  theThreshold_E = par.getParameter<double>("Threshold_E");
  theThreshold_H = par.getParameter<double>("Threshold_H");
  theDR_Veto_E = par.getParameter<double>("dR_Veto_E");
  theDR_Veto_H = par.getParameter<double>("dR_Veto_H");
  theDR_Max = par.getParameter<double>("dR_Max");
  vertexConstraintFlag_XY = par.getParameter<bool>("Vertex_Constraint_XY");
  vertexConstraintFlag_Z = par.getParameter<bool>("Vertex_Constraint_Z");
}


vector<MuIsoDeposit> CaloExtractor::deposits( const Event & event, 
    const EventSetup& eventSetup, const Track & muon, 
    const vector<Direction> & vetoDirections, double coneSize) const
{
  vector<MuIsoDeposit> result;
  static std::string metname = "RecoMuon/CaloExtractor";

  MuIsoDeposit edep("ECAL", muon.eta(), muon.phi() );
  MuIsoDeposit hdep("HCAL", muon.eta(), muon.phi() );
  fillDeposits(edep, hdep, muon, event, eventSetup);

  result.push_back(edep);
  result.push_back(hdep);

  return result;
}
  

void CaloExtractor::fillDeposits(MuIsoDeposit& depE
        , MuIsoDeposit& depH, const Track& mu
        , const Event& event, const EventSetup& eventSetup) const {
  Handle<CaloTowerCollection> towers;
  event.getByLabel(theCaloTowerCollectionLabel,towers);

  edm::ESHandle<CaloGeometry> caloGeom;
  eventSetup.get<IdealGeometryRecord>().get(caloGeom);

  LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
      << " >>> Muon eta phi pt: " << mu.eta() 
      << " " << mu.phi() << " " << mu.pt();

  depE.setEta(mu.eta());
  depE.setPhi(mu.phi());
  depH.setEta(mu.eta());
  depH.setPhi(mu.phi());

  CaloTowerCollection::const_iterator cal;
  for ( cal = towers->begin(); cal != towers->end(); ++cal ) {
      double deltar0 = deltaR(mu,*cal);
      if (deltar0>theDR_Max) continue;
      //LogDebug("Muon|RecoMuon|L2MuonIsolationProducer") << " >>> Calo deltaR0= " << deltar0;

      double etecal = cal->emEt();
      double ethcal = cal->hadEt();
      bool doEcal = (etecal>theThreshold_E && etecal>3*noiseEcal(*cal));
      bool doHcal = (ethcal>theThreshold_H && ethcal>3*noiseHcal(*cal));

      if ((!doEcal) && (!doHcal)) continue;

      GlobalPoint endpos = caloGeom->getPosition(cal->id());
      GlobalPoint muatcal = MuonAtCaloPosition(mu,endpos, vertexConstraintFlag_XY, vertexConstraintFlag_Z);
      double deltar = deltaR(muatcal,endpos);
      for (unsigned int i=0; i<cal->constituentsSize(); i++) {
            DetId calId = cal->constituent(i);
            endpos = caloGeom->getPosition(calId);
            muatcal = MuonAtCaloPosition(mu,endpos, vertexConstraintFlag_XY, vertexConstraintFlag_Z);
            deltar = min(deltar,deltaR(muatcal,endpos));
      }

      if (doEcal) {
            if (deltar<theDR_Veto_E) { 
                  depE.addMuonEnergy(etecal);
            } else {
                  depE.addDeposit(deltar0,etecal);
            }
      }

      if (doHcal) {
            if (deltar<theDR_Veto_H) { 
                  depH.addMuonEnergy(ethcal);
            } else {
                  depH.addDeposit(deltar0,ethcal);
            }
      }

      //LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
           //<< " >>> Muon at calo x y z: " << muatcal.x() << " " << muatcal.y() << " " << muatcal.z();
      //LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
           //<< " >>> Muon at calo eta phi: " << muatcal.eta() << " " << muatcal.phi();
      if (deltar<theDR_Veto_H) {
            LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
                 << " >>> Calo deltaR= " << deltar;
            LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
                 << " >>> Calo eta phi etHcal: " << cal->eta() << " " << cal->phi() << " " << ethcal;
            GlobalPoint hula = caloGeom->getPosition(cal->id());
            LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
                 << " >>> Calo position: x " << hula.x()
                 << " y " << hula.y()
                 << " z " << hula.z();
      }
  }

}

GlobalPoint CaloExtractor::MuonAtCaloPosition(const Track& muon, const GlobalPoint& endpos, bool fixVxy, bool fixVz) const {
      double cur = -muon.transverseCurvature();
      double phi0 = muon.phi0();
      double dca = - muon.d0();
      double theta = muon.theta();
      double dz = muon.dz();

      //LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
          //<< " Pt(GeV): " <<  muon.pt()
          //<< ", phi0 " <<  muon.phi0()
             //<< ", eta " <<  muon.eta();
      //LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
          //<< " d0 " <<  muon.d0()
          //<< ", dz " <<  muon.dz();
      //LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
          //<< " rhocal " <<  endpos.perp()
          //<< ", zcal " <<  endpos.z();

      if (fixVxy && fixVz) {
            // Note that here we assume no correlation between XY and Z projections
            // This should be a reasonable approximation for our purposes
            double errd02 = muon.covariance(muon.i_d0,muon.i_d0);
            if (pow(muon.d0(),2)<4*errd02) {
                  phi0  -= muon.d0()*muon.covariance(muon.i_d0,muon.i_phi0)
                                     /errd02;
                  cur   += muon.d0()*muon.covariance(muon.i_d0,muon.i_transverseCurvature)
                                     /errd02;
                  dca = 0;
            } 
            double errdz2 = muon.covariance(muon.i_dz,muon.i_dz);
            if (pow(muon.dz(),2)<4*errdz2) {
                  theta -= muon.dz()*muon.covariance(muon.i_dz,muon.i_theta)
                                     /errdz2;
                  dz = 0;
            } 
      } else if (fixVxy) {
            double errd02 = muon.covariance(muon.i_d0,muon.i_d0);
            if (pow(muon.d0(),2)<4*errd02) {
                  phi0  -= muon.d0()*muon.covariance(muon.i_d0,muon.i_phi0)
                                     /errd02;
                  cur   += muon.d0()*muon.covariance(muon.i_d0,muon.i_transverseCurvature)
                                     /errd02;
                  theta -= muon.d0()*muon.covariance(muon.i_d0,muon.i_theta)
                                     /errd02;
                  dz    -= muon.d0()*muon.covariance(muon.i_d0,muon.i_dz)
                                     /errd02;
                  dca = 0;
            } 
      } else if (fixVz) {
            double errdz2 = muon.covariance(muon.i_dz,muon.i_dz);
            if (pow(muon.dz(),2)<4*errdz2) {
                  theta -= muon.dz()*muon.covariance(muon.i_dz,muon.i_theta)
                                     /errdz2;
                  phi0  -= muon.dz()*muon.covariance(muon.i_dz,muon.i_phi0)
                                     /errdz2;
                  cur   += muon.dz()*muon.covariance(muon.i_dz,muon.i_transverseCurvature)
                                     /errdz2;
                  dca   += muon.dz()*muon.covariance(muon.i_dz,muon.i_d0)
                                     /errdz2;
                  dz = 0;
            } 
      }

      double sphi0 = sin(phi0);
      double cphi0 = cos(phi0);

      double xsin =  endpos.x()*sphi0 - endpos.y()*cphi0;
      double xcos =  endpos.x()*cphi0 + endpos.y()*sphi0;
      double fcdca = fabs(1-cur*dca);
      double phif = atan2( fcdca*sphi0-cur*endpos.x()
                         , fcdca*cphi0+cur*endpos.y());
      double tphif2 = tan(0.5*(phif-phi0));
      double dcaf = dca + xsin + xcos*tphif2;

      double x = endpos.x() - dcaf*sin(phif);
      double y = endpos.y() + dcaf*cos(phif);

      double deltas =  (x-muon.x())*cphi0 + (y-muon.y())*sphi0;
      double deltaphi = PhiInRange(phif-phi0);
      if (deltaphi!=0) deltas = deltas*deltaphi/sin(deltaphi);

      double z =dz;
      double tantheta = tan(theta);
      if (tantheta!=0) {
            z += deltas/tan(theta);
      } else {
            z = endpos.z();
      }

      return GlobalPoint(x,y,z);
}

double CaloExtractor::PhiInRange(const double& phi) const {
      double phiout = phi;

      if( phiout > 2*M_PI || phiout < -2*M_PI) {
            phiout = fmod( phiout, 2*M_PI);
      }
      if (phiout <= -M_PI) phiout += 2*M_PI;
      else if (phiout >  M_PI) phiout -= 2*M_PI;

      return phiout;
}

template <class T, class U>
double CaloExtractor::deltaR(const T& t, const U& u) const {
      return sqrt(pow(t.eta()-u.eta(),2) +pow(PhiInRange(t.phi()-u.phi()),2));
}

double CaloExtractor::noiseEcal(const CaloTower& tower) const {
      double noise = 0.04;
      double eta = tower.eta();
      if (fabs(eta)>1.479) noise = 0.15;
      return noise;
}

double CaloExtractor::noiseHcal(const CaloTower& tower) const {
      double noise = 0.2;
      return noise;
}

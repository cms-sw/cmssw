#include "CaloExtractor.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/Math/interface/deltaR.h"
//#include "CommonTools/Utils/interface/deltaR.h"
//#include "PhysicsTools/Utilities/interface/deltaR.h"
#include "CommonTools/Utils/interface/normalizedPhi.h"
//#include "PhysicsTools/Utilities/interface/normalizedPhi.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;
using reco::isodeposit::Direction;

CaloExtractor::CaloExtractor(const ParameterSet& par) :
  theCaloTowerCollectionLabel(par.getParameter<edm::InputTag>("CaloTowerCollectionLabel")),
  theDepositLabel(par.getUntrackedParameter<string>("DepositLabel")),
  theWeight_E(par.getParameter<double>("Weight_E")),
  theWeight_H(par.getParameter<double>("Weight_H")),
  theThreshold_E(par.getParameter<double>("Threshold_E")),
  theThreshold_H(par.getParameter<double>("Threshold_H")),
  theDR_Veto_E(par.getParameter<double>("DR_Veto_E")),
  theDR_Veto_H(par.getParameter<double>("DR_Veto_H")),
  theDR_Max(par.getParameter<double>("DR_Max")),
  vertexConstraintFlag_XY(par.getParameter<bool>("Vertex_Constraint_XY")),
  vertexConstraintFlag_Z(par.getParameter<bool>("Vertex_Constraint_Z"))
{
}

void CaloExtractor::fillVetos(const edm::Event& event, const edm::EventSetup& eventSetup, const TrackCollection& muons)
{
  theVetoCollection.clear();

  Handle<CaloTowerCollection> towers;
  event.getByLabel(theCaloTowerCollectionLabel,towers);

  edm::ESHandle<CaloGeometry> caloGeom;
  eventSetup.get<CaloGeometryRecord>().get(caloGeom);

  edm::ESHandle<MagneticField> bField;
  eventSetup.get<IdealMagneticFieldRecord>().get(bField);
  double bz = bField->inInverseGeV(GlobalPoint(0.,0.,0.)).z();

  TrackCollection::const_iterator mu;
  TrackCollection::const_iterator muEnd(muons.end());
  
  CaloTowerCollection::const_iterator cal;
  CaloTowerCollection::const_iterator calEnd(towers->end());
  
  for ( mu = muons.begin(); mu != muEnd; ++mu ) {
    for ( cal = towers->begin(); cal != calEnd; ++cal ) {
      //! make this abit faster
      double dEta = fabs(mu->eta()-cal->eta());
      if (fabs(dEta) > theDR_Max) continue;

            double deltar0 = reco::deltaR(*mu,*cal);
            if (deltar0>theDR_Max) continue;

            double etecal = cal->emEt();
            double eecal = cal->emEnergy();
            bool doEcal = theWeight_E>0 && etecal>theThreshold_E && eecal>3*noiseEcal(*cal);
            double ethcal = cal->hadEt();
            double ehcal = cal->hadEnergy();
            bool doHcal = theWeight_H>0 && ethcal>theThreshold_H && ehcal>3*noiseHcal(*cal);
            if ((!doEcal) && (!doHcal)) continue;

            DetId calId = cal->id();
            GlobalPoint endpos = caloGeom->getPosition(calId);
            GlobalPoint muatcal = MuonAtCaloPosition(*mu,bz,endpos, vertexConstraintFlag_XY, vertexConstraintFlag_Z);
            double deltar = reco::deltaR(muatcal,endpos);

            if (doEcal) {
                  if (deltar<theDR_Veto_E) theVetoCollection.push_back(calId);
            } else {
                  if (deltar<theDR_Veto_H) theVetoCollection.push_back(calId);
            }
      }
  }
     
}

IsoDeposit CaloExtractor::deposit( const Event & event, const EventSetup& eventSetup, const Track & muon) const
{
  IsoDeposit dep(muon.eta(), muon.phi() );
  LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
	  << " >>> Muon: pt " << muon.pt()
	  << " eta " << muon.eta()
	  << " phi " << muon.phi();

  Handle<CaloTowerCollection> towers;
  event.getByLabel(theCaloTowerCollectionLabel,towers);

  edm::ESHandle<CaloGeometry> caloGeom;
  eventSetup.get<CaloGeometryRecord>().get(caloGeom);

  edm::ESHandle<MagneticField> bField;
  eventSetup.get<IdealMagneticFieldRecord>().get(bField);
  double bz = bField->inInverseGeV(GlobalPoint(0.,0.,0.)).z();

  CaloTowerCollection::const_iterator cal;
  CaloTowerCollection::const_iterator calEnd(towers->end());
  for ( cal = towers->begin(); cal != calEnd; ++cal ) {
      //! make this abit faster
      double dEta = fabs(muon.eta()-cal->eta());
      if (fabs(dEta) > theDR_Max) continue;
    
      double deltar0 = reco::deltaR(muon,*cal);
      if (deltar0>theDR_Max) continue;

      double etecal = cal->emEt();
      double eecal = cal->emEnergy();
      bool doEcal = theWeight_E>0 && etecal>theThreshold_E && eecal>3*noiseEcal(*cal);
      double ethcal = cal->hadEt();
      double ehcal = cal->hadEnergy();
      bool doHcal = theWeight_H>0 && ethcal>theThreshold_H && ehcal>3*noiseHcal(*cal);
      if ((!doEcal) && (!doHcal)) continue;

      DetId calId = cal->id();
      GlobalPoint endpos = caloGeom->getPosition(calId);
      GlobalPoint muatcal = MuonAtCaloPosition(muon,bz,endpos,vertexConstraintFlag_XY, vertexConstraintFlag_Z);
      double deltar = reco::deltaR(muatcal,endpos);

      if (deltar<theDR_Veto_H) {
	      dep.setVeto(IsoDeposit::Veto(reco::isodeposit::Direction(muatcal.eta(), muatcal.phi()), theDR_Veto_H));
      }

      if (doEcal) {
            if (deltar<theDR_Veto_E) { 
                  double calodep = theWeight_E*etecal;
                  if (doHcal) calodep += theWeight_H*ethcal;
                  dep.addCandEnergy(calodep);
	            LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
	            << " >>> Calo deposit inside veto (with ECAL): deltar " << deltar
	            << " calodep " << calodep
	            << " ecaldep " << etecal
	            << " hcaldep " << ethcal
	            << " eta " << cal->eta()
	            << " phi " << cal->phi();
                  continue;
            }
      } else {
            if (deltar<theDR_Veto_H) { 
                  dep.addCandEnergy(theWeight_H*ethcal);
	            LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
	            << " >>> Calo deposit inside veto (no ECAL): deltar " << deltar
	            << " calodep " << theWeight_H*ethcal
	            << " eta " << cal->eta()
	            << " phi " << cal->phi();
                  continue;
            }
      }

      if (std::find(theVetoCollection.begin(), theVetoCollection.end()
                  , calId)!=theVetoCollection.end()) {
            LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
            << " >>> Deposits belongs to other track: deltar, etecal, ethcal= " 
            << deltar << ", " << etecal << ", " << ethcal;
            continue;
      }

      if (doEcal) {
            if (deltar>theDR_Veto_E) { 
                  double calodep = theWeight_E*etecal;
                  if (doHcal) calodep += theWeight_H*ethcal;
                  dep.addDeposit(reco::isodeposit::Direction(endpos.eta(), endpos.phi()),calodep);
	            LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
	            << " >>> Calo deposit (with ECAL): deltar " << deltar
	            << " calodep " << calodep
	            << " ecaldep " << etecal
	            << " hcaldep " << ethcal
	            << " eta " << cal->eta()
	            << " phi " << cal->phi();
            }
      } else {
            if (deltar>theDR_Veto_H) { 
	            dep.addDeposit(reco::isodeposit::Direction(endpos.eta(), endpos.phi()),theWeight_H*ethcal);
	            LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")
	            << " >>> Calo deposit (no ECAL): deltar " << deltar
	            << " calodep " << theWeight_H*ethcal
	            << " eta " << cal->eta()
	            << " phi " << cal->phi();
            }
      }
  }

  return dep;

}

GlobalPoint CaloExtractor::MuonAtCaloPosition(const Track& muon, const double bz, const GlobalPoint& endpos, bool fixVxy, bool fixVz) {
      double qoverp= muon.qoverp();
      double cur = bz*muon.charge()/muon.pt();
      double phi0 = muon.phi();
      double dca = muon.dxy();
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
            double errd02 = muon.covariance(muon.i_dxy,muon.i_dxy);
            if (pow(muon.dxy(),2)<4*errd02) {
                  phi0 -= muon.dxy()*muon.covariance(muon.i_dxy,muon.i_phi)
                                     /errd02;
                  cur -= muon.dxy()*muon.covariance(muon.i_dxy,muon.i_qoverp)
                                     /errd02 * (cur/qoverp);
                  dca = 0;
            } 
            double errdsz2 = muon.covariance(muon.i_dsz,muon.i_dsz);
            if (pow(muon.dsz(),2)<4*errdsz2) {
                  theta += muon.dsz()*muon.covariance(muon.i_dsz,muon.i_lambda)
                                     /errdsz2;
                  dz = 0;
            } 
      } else if (fixVxy) {
            double errd02 = muon.covariance(muon.i_dxy,muon.i_dxy);
            if (pow(muon.dxy(),2)<4*errd02) {
                  phi0  -= muon.dxy()*muon.covariance(muon.i_dxy,muon.i_phi)
                                     /errd02;
                  cur -= muon.dxy()*muon.covariance(muon.i_dxy,muon.i_qoverp)
                                     /errd02 * (cur/qoverp);
                  theta += muon.dxy()*muon.covariance(muon.i_dxy,muon.i_lambda)
                                     /errd02;
                  dz    -= muon.dxy()*muon.covariance(muon.i_dxy,muon.i_dsz)
                                     /errd02 * muon.p()/muon.pt();
                  dca = 0;
            } 
      } else if (fixVz) {
            double errdsz2 = muon.covariance(muon.i_dsz,muon.i_dsz);
            if (pow(muon.dsz(),2)<4*errdsz2) {
                  theta += muon.dsz()*muon.covariance(muon.i_dsz,muon.i_lambda)
                                     /errdsz2;
                  phi0  -= muon.dsz()*muon.covariance(muon.i_dsz,muon.i_phi)
                                     /errdsz2;
                  cur -= muon.dsz()*muon.covariance(muon.i_dsz,muon.i_qoverp)
                                     /errdsz2 * (cur/qoverp);
                  dca   -= muon.dsz()*muon.covariance(muon.i_dsz,muon.i_dxy)
                                     /errdsz2;
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

      double deltas =  (x-muon.vx())*cphi0 + (y-muon.vy())*sphi0;
      double deltaphi = normalizedPhi(phif-phi0);
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

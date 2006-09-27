/**  \class L2MuonIsolationProducer
 * 
 *   \author  J. Alcaraz
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L2MuonIsolationProducer/src/L2MuonIsolationProducer.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;

/// constructor with config
L2MuonIsolationProducer::L2MuonIsolationProducer(const ParameterSet& parameterSet){
  LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")<<" L2MuonIsolationProducer constructor called";

  theSACollectionLabel = parameterSet.getUntrackedParameter<string>("StandAloneCollectionLabel");
  theCaloTowerCollectionLabel = parameterSet.getUntrackedParameter<string>("CaloTowerCollectionLabel");
  theThreshold_E = parameterSet.getParameter<double>("Threshold_E");
  theThreshold_H = parameterSet.getParameter<double>("Threshold_H");
  theDR_Veto_E = parameterSet.getParameter<double>("dR_Veto_E");
  theDR_Veto_H = parameterSet.getParameter<double>("dR_Veto_H");
  theDR_Max = parameterSet.getParameter<double>("dR_Max");
  vertexConstraintFlag_XY = parameterSet.getParameter<bool>("Vertex_Constraint_XY");
  vertexConstraintFlag_Z = parameterSet.getParameter<bool>("Vertex_Constraint_Z");

  etaBounds_  = parameterSet.getParameter<std::vector<double> > ("EtaBounds");
  coneCuts_  = parameterSet.getParameter<std::vector<double> > ("ConeCuts");
  edepCuts_  = parameterSet.getParameter<std::vector<double> > ("EdepCuts");

  if (etaBounds_.size()==0) etaBounds_.push_back(999.9); // whole eta range if no input

  if (coneCuts_.size()==0) coneCuts_.push_back(0.0); // no isolation if no input
  if (coneCuts_.size()<etaBounds_.size()) {
      double conelast = coneCuts_[coneCuts_.size()-1];
      int nadd = etaBounds_.size()-coneCuts_.size();
      for (int i=0; i<nadd; i++) coneCuts_.push_back(conelast);
  }  

  if (edepCuts_.size()==0) edepCuts_.push_back(0.0); // no isolation if no input
  if (edepCuts_.size()<etaBounds_.size()) {
      double edeplast = edepCuts_[edepCuts_.size()-1];
      int nadd = etaBounds_.size()-edepCuts_.size();
      for (int i=0; i<nadd; i++) edepCuts_.push_back(edeplast);
  }  

  ecalWeight_  = parameterSet.getParameter<double> ("EcalWeight");

  produces<MuIsoDepositCollection>();
  produces<MuIsoAssociationMap>();
}
  
/// destructor
L2MuonIsolationProducer::~L2MuonIsolationProducer(){
  LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")<<" L2MuonIsolationProducer destructor called";
}

/// build deposits
void L2MuonIsolationProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "Muon|RecoMuon|L2MuonIsolationProducer";
  
  LogDebug(metname)<<" L2 Muon Isolation producing...";

  // Take the SA container
  LogDebug(metname)<<" Taking the StandAlone muons: "<<theSACollectionLabel;
  Handle<TrackCollection> tracks;
  event.getByLabel(theSACollectionLabel,tracks);

  // Find deposits and load into event
  LogDebug(metname)<<" Get energy around";
  std::auto_ptr<MuIsoDepositCollection> depCollection( new MuIsoDepositCollection());
  std::auto_ptr<MuIsoAssociationMap> depMap( new MuIsoAssociationMap());
 
  for (unsigned int i=0; i<tracks->size(); i++) {
      TrackRef tk(tracks,i);
      //LogDebug(metname) << " tketa: " << tk->eta();

      MuIsoDeposit depE("ECAL");
      MuIsoDeposit depH("HCAL");
      fillDeposits(depE, depH, *tk, event, eventSetup);
      depCollection->push_back(depE);
      depCollection->push_back(depH);

      double abseta = fabs(tk->eta());
      int ieta = etaBounds_.size()-1;
      for (unsigned int i=0; i<etaBounds_.size(); i++) {
            if (abseta<etaBounds_[i]) { ieta = i; break; }
      }
      double conesize = coneCuts_[ieta];
      double dephlt = ecalWeight_*depE.depositWithin(conesize)
                       + depH.depositWithin(conesize);
      if (dephlt<edepCuts_[ieta]) {
            depMap->insert(tk, true);
      } else {
            depMap->insert(tk, false);
      }
  }
  //LogDebug(metname) << " dep Collections: " << depCollection->size();
  event.put(depCollection);
  event.put(depMap);

  LogDebug(metname) <<" Event loaded"
		   <<"================================";
}

bool L2MuonIsolationProducer::fillDeposits(MuIsoDeposit& depE
        , MuIsoDeposit& depH, const Track& mu
        , const Event& event, const EventSetup& eventSetup) {
  Handle<CaloTowerCollection> towers;
  event.getByLabel(theCaloTowerCollectionLabel,towers);

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

  return true;
}

GlobalPoint L2MuonIsolationProducer::MuonAtCaloPosition(const Track& muon, const GlobalPoint& endpos, bool fixVxy, bool fixVz) const {
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

double L2MuonIsolationProducer::PhiInRange(const double& phi) const {
      double phiout = phi;

      if( phiout > 2*M_PI || phiout < -2*M_PI) {
            phiout = fmod( phiout, 2*M_PI);
      }
      if (phiout <= -M_PI) phiout += 2*M_PI;
      else if (phiout >  M_PI) phiout -= 2*M_PI;

      return phiout;
}

template <class T, class U>
double L2MuonIsolationProducer::deltaR(const T& t, const U& u) const {
      return sqrt(pow(t.eta()-u.eta(),2) +pow(PhiInRange(t.phi()-u.phi()),2));
}

double L2MuonIsolationProducer::noiseEcal(const CaloTower& tower) const {
      double noise = 0.04;
      double eta = caloGeom->getPosition(tower.id()).eta();
      if (fabs(eta)>1.479) noise = 0.15;
      return noise;
}

double L2MuonIsolationProducer::noiseHcal(const CaloTower& tower) const {
      double noise = 0.2;
      return noise;
}

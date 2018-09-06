#include "L1Trigger/L1TTrackMatch/interface/L1TTrackerPlusBarrelStubsMatcher.h"

L1TTrackerPlusBarrelStubsMatcher::L1TTrackerPlusBarrelStubsMatcher(const edm::ParameterSet& iConfig): 
  verbose_(iConfig.getParameter<int>("verbose"))
{
  //Create sector processors
  std::vector<int> sectors = iConfig.getParameter<std::vector<int> >("sectorsToProcess");
    for (const auto sector : sectors) {
      sectors_.push_back(L1TTrackerPlusBarrelStubsSectorProcessor(iConfig.getParameter<edm::ParameterSet>("sectorSettings"),sector));
    }
}

L1TTrackerPlusBarrelStubsMatcher::~L1TTrackerPlusBarrelStubsMatcher() {}

std::vector<l1t::L1TkMuonParticle> L1TTrackerPlusBarrelStubsMatcher::process(const TrackPtrVector& tracks,const L1MuKBMTCombinedStubRefVector& stubs) { 
  std::vector<l1t::L1TkMuonParticle> preMuons;
  for (auto& sector: sectors_) {
    std::vector<l1t::L1TkMuonParticle> tmp = sector.process(tracks,stubs);
    if (!tmp.empty())
      preMuons.insert(preMuons.end(),tmp.begin(),tmp.end());
  } 

  //Clean muons from different processors
  std::vector<l1t::L1TkMuonParticle> muons = overlapClean(preMuons);

  return muons;
}

std::vector<l1t::L1TkMuonParticle> L1TTrackerPlusBarrelStubsMatcher::overlapClean(const std::vector<l1t::L1TkMuonParticle>& preMuons) {
  //Change this with the code cleaning logic
  std::vector<l1t::L1TkMuonParticle> muonsOut;

  if (preMuons.size()>0) {
    for (const auto& muon : preMuons) {
      L1MuKBMTCombinedStubRefVector muonStubs=muon.getBarrelStubs();
    }
  }

  //Loop over muons
  if (preMuons.size()>0) {
    for (const auto& muon1 : preMuons) {
      for(const auto& muon2 : preMuons) {
        //Make intersection of muons
	const L1MuKBMTCombinedStubRefVector& muon1Stubs=muon1.getBarrelStubs();
	const L1MuKBMTCombinedStubRefVector& muon2Stubs=muon2.getBarrelStubs();
	bool muoneq=muonCheck(muon1,muon2);
	if (muoneq==false && muon1Stubs.size()>0 && muon2Stubs.size()>0) {
          L1MuKBMTCombinedStubRefVector muonInter;
	  for (const auto& stub1 : muon1.getBarrelStubs()) {
            for (const auto& stub2 : muon2.getBarrelStubs()) {
              if (stub1==stub2 && std::find(muonInter.begin(),muonInter.end(),stub1)==muonInter.end()){
                muonInter.push_back(stub1);
	      }
	    }
	  }
  
          if (muonInter.size()==0 && std::find(muonsOut.begin(),muonsOut.end(),muon1)==muonsOut.end() && std::find(muonsOut.begin(),muonsOut.end(),muon2)==muonsOut.end()) {
            muonsOut.push_back(muon1);
	    muonsOut.push_back(muon2);
	  }
	  else if (muonInter.size()>0) {
            if (muon1Stubs.size()>muon2Stubs.size() && std::find(muonsOut.begin(),muonsOut.end(),muon1)==muonsOut.end()) {
              muonsOut.push_back(muon1);
	    }
	    else if (muon2Stubs.size()>muon1Stubs.size() && std::find(muonsOut.begin(),muonsOut.end(),muon2)==muonsOut.end()) {
              muonsOut.push_back(muon2);
	    }
	    else if (muon1Stubs.size()==muon2Stubs.size()) {
	      double dPhi1=0;
	      double dPhi2=0;
	      for (const auto& stub1 : muon1Stubs) {
	        double phi1=(Geom::pi()/(2048*6))*stub1->phi();
	        double phi1T=(Geom::pi()/(2048*6))*phiProp(muon1.phi(),8192*muon1.charge()/muon1.pt(),stub1->scNum(),stub1->stNum());
	        dPhi1+=abs(deltaPhi(phi1,phi1T));
	      }
	      for (const auto& stub2 : muon2Stubs) {
		double phi2=(Geom::pi()/(2048*6))*stub2->phi();
	        double phi2T=(Geom::pi()/(2048*6))*phiProp(muon2.phi(),8192*muon2.charge()/muon2.pt(),stub2->scNum(),stub2->stNum());
	        dPhi2+=abs(deltaPhi(phi2,phi2T));
	      }
	      if (dPhi1<dPhi2 && std::find(muonsOut.begin(),muonsOut.end(),muon1)==muonsOut.end()) {
	        muonsOut.push_back(muon1);
	      }
	      else if (dPhi1>dPhi2 && std::find(muonsOut.begin(),muonsOut.end(),muon2)==muonsOut.end()) {
	        muonsOut.push_back(muon2);
	      }
	      else if (dPhi1==dPhi2 && std::find(muonsOut.begin(),muonsOut.end(),muon1)==muonsOut.end() && std::find(muonsOut.begin(),muonsOut.end(),muon2)==muonsOut.end()) {
	       muonsOut.push_back(muon1);
	      }
	    }
	  }
	}
      }
    }
  }

  //return muonsOut;
  return preMuons;
}

//Define delta phi function
int L1TTrackerPlusBarrelStubsMatcher::deltaPhi(double p1,double p2) {
  double res=p1-p2;
  while (res>Geom::pi()) {
    res-=2*Geom::pi();
  }
  while (res<-Geom::pi()) {
    res+=2*Geom::pi();
  }

  return res;
}

//Define phi propagation
int L1TTrackerPlusBarrelStubsMatcher::phiProp(int muPhi,int k,int sc,int st) {
  //Propagation constant
  double propagation_[]={1.14441,1.24939,1.31598,1.34792};
  
  //Shift phi of the track to be with respect to the sector
  double phi=muPhi-(-Geom::pi()+sc*Geom::pi()/6.0);
  if (phi>Geom::pi()) {
    phi-=2*Geom::pi();
  }
  else if (phi<-Geom::pi()) {
    phi+=2*Geom::pi();
  }

  //Convert phi to integer value and propagate
  int phiInt=phi*2048*6/Geom::pi();
  int propPhi=phiInt+propagation_[st-1]*k;

  return propPhi;
}

//Define muon check
bool L1TTrackerPlusBarrelStubsMatcher::muonCheck(const l1t::L1TkMuonParticle& muon1,const l1t::L1TkMuonParticle& muon2) {
  bool muoneq=false;
  if (muon1.eta()==muon2.eta() && muon1.phi()==muon2.phi() && muon1.pt()==muon2.pt() && muon1.charge()==muon2.charge()) {
    muoneq=true;
  }

  return muoneq;
}

#include "L1Trigger/L1TTrackMatch/interface/L1TTrackerPlusBarrelStubsSectorProcessor.h"
double pi=Geom::pi();

L1TTrackerPlusBarrelStubsSectorProcessor::L1TTrackerPlusBarrelStubsSectorProcessor(const edm::ParameterSet& iConfig, int sector): 
  verbose_(iConfig.getParameter<int>("verbose")),
  station_(iConfig.getParameter<std::vector<int> > ("stationsToProcess")),
  tol_(iConfig.getParameter<int>("tolerance")),
  phi1_(iConfig.getParameter<std::vector<double> > ("phi1")),
  phi2_(iConfig.getParameter<std::vector<double> > ("phi2")),
  propagation_(iConfig.getParameter<std::vector<double> > ("propagationConstants")),
  etaHighm2_(iConfig.getParameter<std::vector<double> > ("etaHighm2")),
  etaHighm1_(iConfig.getParameter<std::vector<double> > ("etaHighm1")),
  etaHigh0_(iConfig.getParameter<std::vector<double> > ("etaHigh0")),
  etaHigh1_(iConfig.getParameter<std::vector<double> > ("etaHigh1")),
  etaHigh2_(iConfig.getParameter<std::vector<double> > ("etaHigh2")),
  etaLowm2_(iConfig.getParameter<std::vector<double> > ("etaLowm2")),
  etaLowm1_(iConfig.getParameter<std::vector<double> > ("etaLowm1")),
  etaLow0_(iConfig.getParameter<std::vector<double> > ("etaLow0")),
  etaLow1_(iConfig.getParameter<std::vector<double> > ("etaLow1")),
  etaLow2_(iConfig.getParameter<std::vector<double> > ("etaLow2")),
  alpha_(iConfig.getParameter<std::vector<double> > ("alpha")),
  beta_(iConfig.getParameter<std::vector<double> > ("beta"))
{
  sector_=sector;
  //find the previous and the next processor
  if (sector==11) {
    previousSector_=10;
    nextSector_=0;
  } 
  else if (sector==0) {
    previousSector_=11;
    nextSector_=1;
  } 
  else {
    previousSector_=sector-1;
    nextSector_=sector+1;
  }
}

L1TTrackerPlusBarrelStubsSectorProcessor::~L1TTrackerPlusBarrelStubsSectorProcessor() {}

std::vector<l1t::L1TkMuonParticle> L1TTrackerPlusBarrelStubsSectorProcessor::process(const TrackPtrVector& tracks,const L1MuKBMTCombinedStubRefVector& stubsAll) {
  //Print statement
  //printf("Running sector processor\n");

  //Output collection
  std::vector<l1t::L1TkMuonParticle> out;
  //First thing first. Keep only the stubs on this processor
  L1MuKBMTCombinedStubRefVector stubs;

  for (const auto& stub: stubsAll) 
    if (stub->scNum()==sector_ || stub->scNum()==previousSector_ ||stub->scNum()==nextSector_)
      stubs.push_back(stub);
  
  //Next loop on tracks
  for (const auto& track : tracks) {
    //Create a muon particle from the track:
    l1t::L1TkMuonParticle::LorentzVector vec(track->getMomentum().x(),
					track->getMomentum().y(),
					track->getMomentum().z(),
					track->getMomentum().mag());
    l1t::L1TkMuonParticle muon (vec,track);
    
    //Set muon charge
    int charge=1;
    if (track->getRInv()<0)
      charge=-1;
    muon.setCharge(charge);

    int k=8192*muon.charge()/muon.pt();

    //You need some logic to see if track is in this sector here:
    //Easiest to check if muon is outside of the sector and tell the sector processor to move on
    if (sector_!=0 && (muon.phi()<=phi1_[sector_] || muon.phi()>=phi2_[sector_])) {
      continue;
    }
    if (sector_==0 && (muon.phi()<=phi1_[sector_] && muon.phi()>=phi2_[sector_])) {
      continue;
    }

    L1MuKBMTCombinedStubRefVector stubsPass; 
    for (const auto& stub : stubs) {
      //Do the matching (propagate the muon)
      int phi=stubPhi_(stub);
      int st=stub->stNum();
      int wh=stub->whNum();
      double deltaPhiTPull=-1;

      //Check if eta value of track matches the station and wheel that the stub is in
      if (wh==-2 && muon.eta()>=etaLowm2_[st-1] && muon.eta()<=etaHighm2_[st-1]) {
        deltaPhiTPull=pull_(k,phiProp_(muon.phi(),k,sector_,st)-phi,st);
      }
      else if (wh==-1 && muon.eta()>=etaLowm1_[st-1] && muon.eta()<=etaHighm1_[st-1]) {
        deltaPhiTPull=pull_(k,phiProp_(muon.phi(),k,sector_,st)-phi,st);
      }
      else if (wh==0 && muon.eta()>=etaLow0_[st-1] && muon.eta()<=etaHigh0_[st-1]) {
        deltaPhiTPull=pull_(k,phiProp_(muon.phi(),k,sector_,st)-phi,st);
      }
      else if (wh==1 && muon.eta()>=etaLow1_[st-1] && muon.eta()<=etaHigh1_[st-1]) {
        deltaPhiTPull=pull_(k,phiProp_(muon.phi(),k,sector_,st)-phi,st);
      }
      else if (wh==2 && muon.eta()>=etaLow2_[st-1] && muon.eta()<=etaHigh2_[st-1]) {
        deltaPhiTPull=pull_(k,phiProp_(muon.phi(),k,sector_,st)-phi,st);
      }
      //printf("Stub pull=%f",deltaPhiTPull);
      //printf("Muon eta=%f",muon.eta());

      if (deltaPhiTPull>=0 && deltaPhiTPull<tol_) {
        stubsPass.push_back(stub);
	//printf("Stubs passed, Pull=%f\n",deltaPhiTPull);
      }
    }

    /*
    int stubsSize=stubsPass.size();
    if (stubsSize>0) {
      printf("Stubs passed=%d\n",stubsSize);
    }
    */

    //pt phi and eta can be accesed by vec.phi(),vec.eta(),vec.pt()
    //propagate and match here 
    L1MuKBMTCombinedStubRefVector stubsFilter=select_(stubsPass,muon,k);

    //You only need to add stubs to the muons  at this stage
    //To do that just do:
    if (stubsFilter.size()>0) {
      //int size=stubsFilter.size();
      //printf("Filtered stubs=%d\n",size);
      for (const auto& stub : stubsFilter) {
        muon.addBarrelStub(stub);
      }
    }


    for (const auto& stub : muon.getBarrelStubs()) {
      printf("Stub in sector processor phi=%d\n",stub->phi());

    }

    //for now just add it
    out.push_back(muon);
  }

  return out;
}

//Define delta phi function
int L1TTrackerPlusBarrelStubsSectorProcessor::deltaPhi_(double p1,double p2) {
  double res=p1-p2;
  while (res>pi) {
    res-=2*pi;
  }
  while (res<-pi) {
    res+=2*pi;
  }
  
  return res;
}

//Define phi propagation
int L1TTrackerPlusBarrelStubsSectorProcessor::phiProp_(int muPhi,int k,int sc,int st) {
  //Shift phi of the track to be with respect to the sector
  double phi=muPhi-(-pi+sc*pi/6.0);
  if (phi>pi) {
    phi-=2*pi;
  }
  else if (phi<-pi) {
    phi+=2*pi;
  }

  //Convert phi to integer value and propagate
  int phiInt=phi*2048*6/pi;
  int propPhi=phiInt+propagation_[st-1]*k;
  
  return propPhi;
}

//Define pull function
double L1TTrackerPlusBarrelStubsSectorProcessor::pull_(int k,int dphi,int st) {
  double pullfunc=abs(dphi/sqrt(alpha_[st-1]*pow(k,2)+beta_[st-1]));
  
  return pullfunc;
}

//Get integer value for phi of the stub
int L1TTrackerPlusBarrelStubsSectorProcessor::stubPhi_(L1MuKBMTCombinedStubRef stub) {
  int phi=stub->phi();
  if (stub->scNum()==previousSector_) {
    phi-=2047;
  }
  if (stub->scNum()==nextSector_) {
    phi+=2047;
  }

  return phi;
}

//Select best stubs
L1MuKBMTCombinedStubRefVector L1TTrackerPlusBarrelStubsSectorProcessor::select_(const L1MuKBMTCombinedStubRefVector& stubsPass, l1t::L1TkMuonParticle muon,int k) {
  //printf("Selecting best stubs for matching\n");

  //Create vectors for sorting stubs
  L1MuKBMTCombinedStubRefVector stubsSelectSt1;
  L1MuKBMTCombinedStubRefVector stubsSelectSt2;
  L1MuKBMTCombinedStubRefVector stubsSelectSt3;
  L1MuKBMTCombinedStubRefVector stubsSelectSt4;
  L1MuKBMTCombinedStubRefVector stubsSelect;

  //Sort stubs by station number
  if (stubsPass.size()>0) {
    for (const auto& stub: stubsPass) {
      if (stub->stNum()==1) {
        stubsSelectSt1.push_back(stub);
      }
      if (stub->stNum()==2) {
        stubsSelectSt2.push_back(stub);
      }
      if (stub->stNum()==3) {
        stubsSelectSt3.push_back(stub);
      }
      if (stub->stNum()==4) {
        stubsSelectSt4.push_back(stub);
      }
    }
    /*
    int size=stubsPass.size();
    int size1=stubsSelectSt1.size();
    int size2=stubsSelectSt2.size();
    int size3=stubsSelectSt3.size();
    int size4=stubsSelectSt4.size();
    printf("Stubs passed=%d\nSt1=%d\nSt2=%d\nSt3=%d\nSt4=%d\n",size,size1,size2,size3,size4);
    */
  }

  //Select stub with the lowest pull value at each station
  if (stubsSelectSt1.size()>0) {
    double pullSt1=pull_(k,phiProp_(muon.phi(),k,sector_,1)-stubsSelectSt1[0]->phi(),1);
    L1MuKBMTCombinedStubRef minStubSt1;
    for (const auto& stub: stubsSelectSt1) {
      double pullStub=pull_(k,phiProp_(muon.phi(),k,sector_,1)-stub->phi(),1);
      if (pullStub<pullSt1) {
        minStubSt1=stub;
      }
    }
    stubsSelect.push_back(minStubSt1);
  }
  if (stubsSelectSt2.size()>0) {
    double pullSt2=pull_(k,phiProp_(muon.phi(),k,sector_,2)-stubsSelectSt2[0]->phi(),2);
    L1MuKBMTCombinedStubRef minStubSt2;
    for (const auto& stub: stubsSelectSt2) {
      double pullStub=pull_(k,phiProp_(muon.phi(),k,sector_,2)-stub->phi(),2);
      if (pullStub<pullSt2) {
        minStubSt2=stub;
      }
    }
    stubsSelect.push_back(minStubSt2);
  }
  if (stubsSelectSt3.size()>0) {
    double pullSt3=pull_(k,phiProp_(muon.phi(),k,sector_,3)-stubsSelectSt3[0]->phi(),3);
    L1MuKBMTCombinedStubRef minStubSt3;
    for (const auto& stub: stubsSelectSt3) {
      double pullStub=pull_(k,phiProp_(muon.phi(),k,sector_,3)-stub->phi(),3);
      if (pullStub<pullSt3) {
        minStubSt3=stub;
      }
    }
    stubsSelect.push_back(minStubSt3);
  }
  if (stubsSelectSt4.size()>0) {
    double pullSt4=pull_(k,phiProp_(muon.phi(),k,sector_,4)-stubsSelectSt4[0]->phi(),4);
    L1MuKBMTCombinedStubRef minStubSt4;
    for (const auto& stub: stubsSelectSt4) {
      double pullStub=pull_(k,phiProp_(muon.phi(),k,sector_,4)-stub->phi(),4);
      if (pullStub<pullSt4) {
        minStubSt4=stub;
      }
    }
    stubsSelect.push_back(minStubSt4);
  }
  
  if (stubsSelect.size()>0) {
    int select=stubsSelect.size();
    printf("Stubs selected=%d\n",select);
  }
  
  
  return stubsSelect;
}


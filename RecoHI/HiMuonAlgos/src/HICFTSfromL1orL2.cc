
// HeavyIonAnalysis

#include "RecoHI/HiMuonAlgos/interface/HICFTSfromL1orL2.h"
#include "DataFormats/Common/interface/RefToBase.h"
using namespace reco;
using namespace std;
//#define OK_DEBUG
//-----------------------------------------------------------------------------
// Vector of Free Trajectory State in Muon stations from L1 Global Muon Trigger
namespace cms {
vector<FreeTrajectoryState> HICFTSfromL1orL2::createFTSfromL1(vector<L1MuGMTExtendedCand>& gmt) 
{ 
// ========================================================================================
//
//  Switch on L1 muon trigger.
//
  
  vector<FreeTrajectoryState> ftsL1;

  int ngmt = gmt.size();
#ifdef DEBUG
  cout << "Number of muons found by the L1 Global Muon TRIGGER : "
       << ngmt << endl;
#endif
  if(ngmt<0) {
    return ftsL1;
  } 
  
  for ( vector<L1MuGMTExtendedCand>::const_iterator gmt_it = gmt.begin(); gmt_it != gmt.end(); gmt_it++ )
  {
   ftsL1.push_back(FTSfromL1((*gmt_it)));
  }
  return ftsL1;
} // end createFTSfromL1
//-----------------------------------------------------------------------------
// Vector of Free Trajectory State in Muon stations from L2 Muon Trigger

vector<FreeTrajectoryState> HICFTSfromL1orL2::createFTSfromL2(const RecoChargedCandidateCollection& recmuons) 
{ 
// ========================================================================================
//
//  Switch on L2 muon trigger.
//

  vector<FreeTrajectoryState> ftsL2;
//  RecQuery q(localRecAlgo);

  RecoChargedCandidateCollection::const_iterator recmuon = recmuons.begin();

//  int nrec = recmuons.size();
#ifdef DEBUG
  cout << "Number of muons found by the L2 TRIGGER : "
       << nrec << endl;
#endif
  for(recmuon=recmuons.begin(); recmuon!=recmuons.end(); recmuon++)
  {
  ftsL2.push_back(FTSfromL2((*recmuon)));
  } // endfor
  return ftsL2;
} // end createFTSfromL2


vector<FreeTrajectoryState> HICFTSfromL1orL2::createFTSfromStandAlone(const TrackCollection& recmuons) 
{ 
// ========================================================================================
//
//  Switch on L2 muon trigger.
//

  vector<FreeTrajectoryState> ftsL2;
//  RecQuery q(localRecAlgo);

  TrackCollection::const_iterator recmuon = recmuons.begin();

 // int nrec = recmuons.size();
#ifdef DEBUG
  cout << "Number of muons found by the StandAlone : "
       << nrec << endl;
#endif
  for(recmuon=recmuons.begin(); recmuon!=recmuons.end(); recmuon++)
  {
  ftsL2.push_back(FTSfromStandAlone((*recmuon)));
  } // endfor
  return ftsL2;
} // end createFTSfromStandAlone

vector<FreeTrajectoryState> HICFTSfromL1orL2::createFTSfromL2(const TrackCollection& recmuons)
{
// ========================================================================================
//
//  Switch on L2 muon trigger.
//

  vector<FreeTrajectoryState> ftsL2;
//  RecQuery q(localRecAlgo);

  TrackCollection::const_iterator recmuon = recmuons.begin();

//  int nrec = recmuons.size();
#ifdef DEBUG
  cout << "Number of muons found by the StandAlone : "
       << nrec << endl;
#endif
  for(recmuon=recmuons.begin(); recmuon!=recmuons.end(); recmuon++)
  {
  ftsL2.push_back(FTSfromStandAlone((*recmuon)));
  } // endfor
  return ftsL2;
} // end createFTSfromStandAlone



//-----------------------------------------------------------------------------
// Vector of Free Trajectory State in Muon stations from L1 Global Muon Trigger

vector<FreeTrajectoryState> HICFTSfromL1orL2::createFTSfromL1orL2(vector<L1MuGMTExtendedCand>& gmt, const RecoChargedCandidateCollection& recmuons) 
{
    double pi=4.*atan(1.);
    double twopi=8.*atan(1.);

    vector<FreeTrajectoryState> ftsL1orL2;
    vector<FreeTrajectoryState> ftsL1 = createFTSfromL1(gmt);
    vector<FreeTrajectoryState> ftsL2 = createFTSfromL2(recmuons);
//
// Clean candidates if L1 and L2 pointed to one muon. 
//
    vector<FreeTrajectoryState*>::iterator itused;    
    vector<FreeTrajectoryState*> used;
    
    for(vector<FreeTrajectoryState>::iterator tl1 = ftsL1.begin(); tl1 != ftsL1.end(); tl1++)
    { 
   //      float ptL1    =  (*tl1).parameters().momentum().perp();
         float etaL1   =  (*tl1).parameters().momentum().eta();
         float phiL1   =  (*tl1).parameters().momentum().phi();
	 if( phiL1 < 0.) phiL1 = twopi + phiL1;
    //     int chargeL1  =  (*tl1).charge();
	 int L2 = 0;  // there is no L2 muon.
	 	 
       for(vector<FreeTrajectoryState>::iterator tl2 = ftsL2.begin(); tl2 != ftsL2.end(); tl2++)
       {
         itused = find(used.begin(),used.end(),&(*tl2));
 //        float ptL2    =  (*tl2).parameters().momentum().perp();
         float etaL2   =  (*tl2).parameters().momentum().eta();
         float phiL2   =  (*tl2).parameters().momentum().phi();
	 if( phiL2 < 0.) phiL2 = twopi + phiL2;
  //       int chargeL2  =  (*tl2).charge();
         float dphi = abs(phiL1-phiL2);
	 if( dphi > pi ) dphi = twopi - dphi; 
	 float dr = sqrt((etaL1 - etaL2)*(etaL1 - etaL2)+dphi*dphi);
	 
#ifdef OK_DEBUG	
         cout<<" ===== Trigger Level 1 candidate: ptL1 "<<ptL1<<" EtaL1 "<<etaL1<<" PhiL1 "<<phiL1<<
	 " chargeL1 "<<chargeL1<<endl;
         cout<<" ===== Trigger Level 2 candidate: ptL2 "<<ptL2<<" EtaL2 "<<etaL2<<" PhiL2 "<<phiL2<<
	 " chargeL2 "<<chargeL2<<endl;
	 cout<<" abs(EtaL1 - EtaL2) "<<abs(etaL1 - etaL2)<<" dphi "<<dphi<<" dr "<<dr<<
	 " dQ "<<chargeL1 - chargeL2
	 <<" the same muon or not L2 "<<L2<<endl;
#endif	 
	 
	 if ( itused != used.end() ) {
//	    cout<<" L2 is already used in primary cycle"<<endl;
	    continue;
	 }
	 
	 
	 
	 float drmax = 0.5;
	 if( abs(etaL1) > 1.9 ) drmax = 0.6; 
	 
#ifdef OK_DEBUG
         cout<<" Drmax= "<<drmax<<endl;
#endif	  
	 L2 = 0;       
         if( dr < drmax )
	 { // it is the same muon. Take L2 trigger.
//	     if ( chargeL1 == chargeL2 ) 
//	     { // The same muon. Take L2 candidate.
	       L2 = 1;
	       ftsL1orL2.push_back((*tl2));
	       used.push_back(&(*tl2));
//	       cout<<" add L2 for same muon"<<endl;
	       break;
//	     } // endif  
//	       else 
//	     { // Probably different muons in the same cell. Try to recover on L3.
//	       ftsL1orL2.push_back((*tl1));
//	       ftsL1orL2.push_back((*tl2));
//	       cout<<" add L2 for diff muon"<<endl;
//	       used.push_back(&(*tl2));
//	     }  // endelse
	 } // endif
      } // end for L2
       if( L2 == 0 ) 
       {
//	  cout<<" add L1 for no L2 muon"<<endl;
          ftsL1orL2.push_back((*tl1)); // No L1 candidate. Take L1 candidate.
       }	  
    } // end for L1

  //  cout<<" Add the last L2 candidates that have not corresponding L1 "<<endl;
    
    if( ftsL2.size() > 0 )
    { // just add the ramains L2 candidates
    for(vector<FreeTrajectoryState>::iterator tl2 = ftsL2.begin(); tl2 != ftsL2.end(); tl2++)
    {
      itused = find(used.begin(),used.end(),&(*tl2));
      if ( itused != used.end() ) 
      {
//	 cout<<" L2 is already used in secondary cycle"<<endl;
         continue;
      }	 
      ftsL1orL2.push_back((*tl2));
    } // end for L2
    }
  //  cout<<" The number of trajectories in muon stations "<<ftsL1orL2.size()<<endl;
    return ftsL1orL2; 
}
//-----------------------------------------------------------------------------
// Vector of Free Trajectory State from L1 trigger candidate

  FreeTrajectoryState HICFTSfromL1orL2::FTSfromL1(const L1MuGMTExtendedCand& gmt){
    double pi=4.*atan(1.);
    unsigned int det = gmt.isFwd();
    double px,py,pz,x,y,z;
    float pt    =  gmt.ptValue();
    float eta   =  gmt.etaValue();
    float theta =  2*atan(exp(-eta));
    float phi   =  gmt.phiValue();
    int charge  =  gmt.charge();
    
    bool barrel = true;
    if(det) barrel = false;
//
// Take constant radius for barrel = 513 cm and constant z for endcap = 800 cm.
// hardcodded.
//
    float radius = 513.;  
    if ( !barrel ) {
    radius = 800.;
    if(eta<0.) radius=-1.*radius;
    }
    
    if (  barrel && pt < 3.5 ) pt = 3.5;
    if ( !barrel && pt < 1.0 ) pt = 1.0;
    
    
// Calculate FTS for barrel and endcap     
    
    if(barrel){
     
// barrel

    if(abs(theta-pi/2.)<0.00001){
      pz=0.;
      z=0.;
    }else{
      pz=pt/tan(theta);
      z=radius/tan(theta);
    }
     x=radius*cos(phi);
     y=radius*sin(phi);
   
    } else {
    
// endcap

    pz=pt/tan(theta);
    z=radius;
    x=z*tan(theta)*cos(phi);
    y=z*tan(theta)*sin(phi);
    } 
    
    px=pt*cos(phi);
    py=pt*sin(phi);
  
//    cout<<" CreateFts:x,y,x,px,py,pz "<<x<< " "<<y<<" "<<z<<" "<<px<<" "<<py<<" "<<pz<<endl;
    
    GlobalPoint aX(x,y,z);
    GlobalVector aP(px,py,pz);    
    GlobalTrajectoryParameters gtp(aX,aP,charge,field);
    
    AlgebraicSymMatrix m(5,0);
    m(1,1)=0.6*pt; m(2,2)=1.; m(3,3)=1.;
    m(4,4)=1.;m(5,5)=0.; 
    CurvilinearTrajectoryError cte(m);

    FreeTrajectoryState fts(gtp,cte);
    return fts;
  }
  
//-----------------------------------------------------------------------------
// Vector of Free Trajectory State from L2 trigger candidate

  FreeTrajectoryState HICFTSfromL1orL2::FTSfromL2(const RecoChargedCandidate& gmt)
  {
  
    TrackRef tk1 = gmt.get<TrackRef>();
    
    const math::XYZPoint pos0 = tk1->innerPosition();
    const math::XYZVector mom0 = tk1->innerMomentum();
    
    double pp = sqrt(mom0.x()*mom0.x()+mom0.y()*mom0.y()+mom0.z()*mom0.z());
    double pt = sqrt(mom0.x()*mom0.x()+mom0.y()*mom0.y());
    double theta = mom0.theta();
    double pz = mom0.z();
    
    GlobalVector mom(mom0.x(),mom0.y(),mom0.z());
    
    if( pt < 4.) 
    {
      pt = 4.; if (abs(pz) > 0. )  pz = pt/tan(theta);
      double corr = sqrt( pt*pt + pz*pz )/pp;
      GlobalVector mom1( corr*mom0.x(), corr*mom0.y(), corr*mom0.z() );
      mom = mom1;
    }

   // cout<<" L2::Innermost state "<<pos0<<" new momentum "<<mom<<" old momentum "<<mom0<<endl;
    
    AlgebraicSymMatrix m(5,0);
    double error;
    if( abs(mom.eta()) < 1. )
    {
     error = 0.6*mom.perp();
    }
     else
    {
     error = 0.6*abs(mom.z());
    }
    m(1,1)=0.6*mom.perp(); m(2,2)=1.; m(3,3)=1.;
    m(4,4)=1.;m(5,5)=0.; 
    CurvilinearTrajectoryError cte(m);
    GlobalPoint pos(pos0.x(),pos0.y(),pos0.z());
    
    GlobalTrajectoryParameters gtp(pos,mom,tk1->charge(), field);
    FreeTrajectoryState fts(gtp,cte);
  
  return fts;
  }
//-----------------------------------------------------------------------------
// Vector of Free Trajectory State from StanAlone candidate

  FreeTrajectoryState HICFTSfromL1orL2::FTSfromStandAlone(const Track& tk1)
  {
  
//    TrackRef tk1 = gmt.get<TrackRef>();
    
    const math::XYZPoint pos0 = tk1.innerPosition();
    const math::XYZVector mom0 = tk1.innerMomentum();
    
    double pp = sqrt(mom0.x()*mom0.x()+mom0.y()*mom0.y()+mom0.z()*mom0.z());
    double pt = sqrt(mom0.x()*mom0.x()+mom0.y()*mom0.y());
    double theta = mom0.theta();
    double pz = mom0.z();
    
    GlobalVector mom(mom0.x(),mom0.y(),mom0.z());
    
    if( pt < 4.) 
    {
      pt = 4.; if (abs(pz) > 0. )  pz = pt/tan(theta);
      double corr = sqrt( pt*pt + pz*pz )/pp;
      GlobalVector mom1( corr*mom0.x(), corr*mom0.y(), corr*mom0.z() );
      mom = mom1;
    }

    //cout<<" StandAlone::Innermost state "<<pos0<<" new momentum "<<mom<<" old momentum "<<mom0<<endl;
    
    AlgebraicSymMatrix m(5,0);
    double error;
    if( abs(mom.eta()) < 1. )
    {
     error = 0.6*mom.perp();
    }
     else
    {
     error = 0.6*abs(mom.z());
    }
    m(1,1)=0.6*mom.perp(); m(2,2)=1.; m(3,3)=1.;
    m(4,4)=1.;m(5,5)=0.; 
    CurvilinearTrajectoryError cte(m);
    GlobalPoint pos(pos0.x(),pos0.y(),pos0.z());
    
    GlobalTrajectoryParameters gtp(pos,mom,tk1.charge(), field);
    FreeTrajectoryState fts(gtp,cte);
  
  return fts;
  }

}

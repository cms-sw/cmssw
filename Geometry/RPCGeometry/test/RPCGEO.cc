// -*- C++ -*-
//
// Package:    RPCGEO
// Class:      RPCGEO
// 
/**\class RPCGEO RPCGEO.cc rpcgeo/RPCGEO/src/RPCGEO.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/91
//         Created:  Wed Sep 26 17:08:29 CEST 2007
//
//


// system include files
#include <memory>
#include <fstream>
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

class RPCGEO : public edm::one::EDAnalyzer<>
{
public:
  explicit RPCGEO(const edm::ParameterSet&);
  ~RPCGEO();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  std::ofstream detidsstream;
  std::ofstream detailstream;
};

RPCGEO::RPCGEO(const edm::ParameterSet& /*iConfig*/)
{
  detailstream.open("RPCGeometry.out"); 
  detidsstream.open("RPCDetIdLst.out");
  std::cout <<"Opening output files :: RPCGeometry.out and RPCDetIdLst.out"<< std::endl;
}

RPCGEO::~RPCGEO()
{
  detailstream.close();
  detidsstream.close();
  std::cout <<"Closing output files :: RPCGeometry.out and RPCDetIdLst.out"<< std::endl;
}

// ------------ method called to for each event  ------------
void
RPCGEO::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup)
{
   using namespace edm;

   detailstream <<" Getting the RPC Geometry"<<std::endl;
   edm::ESHandle<RPCGeometry> rpcGeo;
   iSetup.get<MuonGeometryRecord>().get(rpcGeo);

   int StripsInCMS=0;
   int counterstripsBarrel=0;
   int counterstripsEndCap=0;
   int RollsInCMS=0;
   int counterRollsBarrel=0;
   int counterRollsMB1MB2MB3=0;
   int counterRollsMB4=0;
   int RB4Wm2=0;
   int RB4Wm1=0;
   int RB4W0=0;
   int RB4W1=0;
   int RB4W2=0;
   int counterRollsEndCap=0;
   int ENDCAP[5][4];
   int ENDCAProll[5][4];
   int rollsNearDiskp3=0;	 
   int rollsNearDiskp2=0;
   float sumstripwbarrel = 0; 
   float sumstripwendcap = 0;
   float areabarrel = 0; 
   float areaendcap = 0;

   // Count chambers and rolls in the endcap
   // 4 Endcap disks with each 3 rings
   // count the chambers
   for(int i=1;i<5;i++){
     for(int j=1;j<4;j++){
       ENDCAP[i][j]=0;
     }
   }
   // count the rolls
   for(int i=1;i<5;i++){
     for(int j=1;j<4;j++){
       ENDCAProll[i][j]=0;
     }
   }

   
   for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
     // The DetId can be a chamber or a roll
     // Consider only the chambers and ask the rolls belonging to this chamber later on
     // So this is a loop on the chambers
     if( dynamic_cast< const RPCChamber* >( *it ) != 0 ){
       const RPCChamber* ch = dynamic_cast< const RPCChamber* >( *it ); 
       std::vector< const RPCRoll*> rolls = (ch->rolls());
       //detailstream<<"RPC Chamber"<<ch->id()<<std::endl;       
       if(ch->id().region()==1){	 
	 switch(ch->id().station()){
	 case 1:
	   switch(ch->id().ring()){
	   case 1:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   case 2:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   case 3:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   }
	   break;
	 case 2:
	   switch(ch->id().ring()){
	   case 1:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   case 2:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   case 3:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   }
	   break;
	 case 3:
	   switch(ch->id().ring()){
	   case 1:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   case 2:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   case 3:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   }
	   break;
	 case 4:
	   switch(ch->id().ring()){
	   case 1:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   case 2:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   case 3:
	     ++ENDCAP[ch->id().station()][ch->id().ring()];
	     break;
	   }
	   break;
	 }
       }
       // ======================================
       // Loop over the rolls inside the chamber
       // ======================================
       for(std::vector<const RPCRoll*>::const_iterator r = rolls.begin();r != rolls.end(); ++r){
	 RPCDetId rpcId = (*r)->id();
	 int stripsinthisroll=(*r)->nstrips();
	 RPCGeomServ rpcsrv(rpcId);
	 ++RollsInCMS;
	 // detailstream<<rpcId<<rpcsrv.name()<<" strips="<<stripsinthisroll<<std::endl;
	 //detailstream<<rpcId<<" - "<<rpcsrv.name()<<" - "<<rpcsrv.shortname()<<std::endl;
	 //detailstream<<rpcsrv.name()<<std::endl;

	 // start RPC Barrel
	 // ----------------
	 if (rpcId.region()==0){ 
	   //detailstream<<"Getting the RPC Topolgy"<<std::endl;
	   const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&((*r)->topology()));
	   float s1 = static_cast<float>(1)-0.5;
           float sLast = static_cast<float>(stripsinthisroll)-0.5;
	   float stripl = top_->stripLength();
	   float stripw = top_->pitch();
	   areabarrel = areabarrel + stripl*stripw*stripsinthisroll;
	   sumstripwbarrel = sumstripwbarrel+stripw*stripsinthisroll;
	   std::string name = rpcsrv.name();

	   // +++ Printout +++
	   // ++++++++++++++++
	   detidsstream<<rpcId.rawId()<<" \t "<<std::setw(24)<<name<<" \t "<<rpcId<<std::endl;
	   detailstream<<"All Info "<<rpcId.rawId()<<" = "<<std::setw(24)<<name<<" || ";
	   detailstream<<" strips: length ="<<stripl<<"[cm] & width ="<<stripw<<"[cm] & number ="<<stripsinthisroll;
	   detailstream<<" || area roll ="<<stripl*stripw<<"[cm^2] || area total barrel = "<<areabarrel<<"[cm^2]"<<std::endl;
	   // ++++++++++++++++

	   const BoundPlane & RPCSurface = (*r)->surface();
	   GlobalPoint FirstStripCenterPointInGlobal = RPCSurface.toGlobal(top_->localPosition(s1));
	   GlobalPoint LastStripCenterPointInGlobal = RPCSurface.toGlobal(top_->localPosition(sLast));
	   double rpcphiFirst = FirstStripCenterPointInGlobal.barePhi();//*180./3.141592;
	   double rpcphiLast  = LastStripCenterPointInGlobal.barePhi();//*180./3.141592;
	   //double rpcYFirst = FirstStripCenterPointInGlobal.y();
	   //double rpcYLast  = LastStripCenterPointInGlobal.y();
	   double diff=rpcphiLast-rpcphiFirst;
	   double rollphi = (rpcphiFirst+rpcphiLast)*0.5*180./3.141592;
	   double orientation=diff/fabs(diff);
	   int seg=rpcsrv.segment();

	   // ++++++++++++++++
	   // +++ Printout +++
	   // ++++++++++++++++
	   detailstream<<std::setw(45)<<name<<" ||  phi="<<rollphi<<" orientation="<<orientation<<" seg="<<seg
		    <<" phi first strip="<<rpcphiFirst<<" phi last strip="<<rpcphiLast<<" || "<<std::endl;
	   detailstream<<std::setw(45)<<name<<" ||  glob (X,Y,Z) first strip =("<<FirstStripCenterPointInGlobal.x()<<", "<<FirstStripCenterPointInGlobal.y()<<","<<FirstStripCenterPointInGlobal.z()<<")[cm]"
		    <<" glob (X,Y,Z) last strip =("<<LastStripCenterPointInGlobal.x()<<", "<<LastStripCenterPointInGlobal.y()<<","<<LastStripCenterPointInGlobal.z()<<")[cm]"
	            <<std::endl;	   
	   // ++++++++++++++++

	   ++counterRollsBarrel; 
	   if(rpcId.station()==4){
	     ++counterRollsMB4;
	       switch(rpcId.ring()){
	       case -2:
		 ++RB4Wm2;
		 break;
	       case -1:
		 ++RB4Wm1;
		 break;
	       case 0:
		 ++RB4W0;
		 break;
	       case 1:
		 ++RB4W1;
		 break;
	       case 2:
		 ++RB4W2;
		 break;
	       }
	   }
	   else ++counterRollsMB1MB2MB3;
	 }
	 // end RPC Barrel loop
	 // -------------------

	 // start RPC Endcap
	 // -------------------
	 else {
	   const TrapezoidalStripTopology* top_= dynamic_cast<const TrapezoidalStripTopology*> (&((*r)->topology()));
	   float s1 = static_cast<float>(1)-0.5;
	   float sLast = static_cast<float>(stripsinthisroll)-0.5;
	   float stripl = top_->stripLength();
	   float stripw = top_->pitch();
	   areaendcap = areaendcap + stripw*stripl*stripsinthisroll;
	   sumstripwendcap=sumstripwendcap+stripw*stripsinthisroll;
	   std::string name = rpcsrv.name();

	   // +++ Printout +++
	   // ++++++++++++++++
	   detidsstream<<rpcId.rawId()<<" \t "<<std::setw(24)<<name<<" \t "<<rpcId<<std::endl;
	   detailstream<<"All Info "<<rpcId.rawId()<<" = "<<std::setw(24)<<name<<" || ";
	   detailstream<<" strips: length ="<<stripl<<"[cm] & width ="<<stripw<<"[cm] & number ="<<stripsinthisroll;
	   detailstream<<" || area roll ="<<stripl*stripw<<"[cm^2] || area total endcap = "<<areaendcap<<"[cm^2]"<<std::endl;
	   // ++++++++++++++++

	   const BoundPlane & RPCSurface = (*r)->surface();
	   GlobalPoint FirstStripCenterPointInGlobal = RPCSurface.toGlobal(top_->localPosition(s1));
	   GlobalPoint LastStripCenterPointInGlobal = RPCSurface.toGlobal(top_->localPosition(sLast));
	   double rpcphiFirst = FirstStripCenterPointInGlobal.barePhi();//*180./3.141592;
	   double rpcphiLast  = LastStripCenterPointInGlobal.barePhi();//*180./3.141592;
	   //double rpcYFirst = FirstStripCenterPointInGlobal.y();
	   //double rpcYLast  = LastStripCenterPointInGlobal.y();
	   double diff=rpcphiLast-rpcphiFirst;
	   double rollphi = (rpcphiFirst+rpcphiLast)*0.5*180./3.141592;
	   double orientation=diff/fabs(diff);
	   int seg=rpcsrv.segment();
	   // ??? what does this mean ???
	   if(seg==19) orientation = orientation*-1;
	   // ????

	   // +++ Printout +++
	   // ++++++++++++++++
	   detailstream<<std::setw(45)<<name<<" ||  phi="<<rollphi<<" orientation="<<orientation<<" seg="<<seg
		    <<" phi first strip="<<rpcphiFirst<<" phi last strip="<<rpcphiLast<<" || "<<std::endl;
	   detailstream<<std::setw(45)<<name<<" ||  glob (X,Y,Z) first strip =("<<FirstStripCenterPointInGlobal.x()<<", "<<FirstStripCenterPointInGlobal.y()<<","<<FirstStripCenterPointInGlobal.z()<<")[cm]"
		    <<" glob (X,Y,Z) last strip =("<<LastStripCenterPointInGlobal.x()<<", "<<LastStripCenterPointInGlobal.y()<<","<<LastStripCenterPointInGlobal.z()<<")[cm]";
	   // ++++++++++++++++
	   // cscphi = 2*3.1415926536+CenterPointCSCGlobal.barePhi():cscphi=CenterPointCSCGlobal.barePhi();

	   // +++ Check the orientation +++
	   // +++++++++++++++++++++++++++++
	   bool orientation_ok = false;
	   if(rpcId.station()==1) {
	     if(rpcId.ring()==1             && orientation*rpcId.region()== 1.0) {orientation_ok=true;}
	     if(rpcId.ring()==2 && seg%2!=0 && orientation*rpcId.region()== 1.0) {orientation_ok=true;}
	     if(rpcId.ring()==2 && seg%2==0 && orientation*rpcId.region()==-1.0) {orientation_ok=true;}
	     if(rpcId.ring()==3             && orientation*rpcId.region()== 1.0) {orientation_ok=true;}
	   }
	   if(rpcId.station()==2 || rpcId.station()==4) {
	     if(orientation*rpcId.region()==-1.0) {orientation_ok=true;}
	   }
	   if(rpcId.station()==3) {
	     if(orientation*rpcId.region()==1.0) {orientation_ok=true;}
	   }
	   if(orientation_ok) detailstream<<" OK"<<std::endl;
	   else detailstream<<" WRONG!!!"<<std::endl;
	   // +++++++++++++++++++++++++++++
	     
	   ++counterRollsEndCap;
	   if(rpcId.region()==1){
	     switch(rpcId.station()){
	     case 1:
	       switch(rpcId.ring()){
	       case 1:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       case 2:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       case 3:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       }
	       break;
	     case 2:
	       switch(rpcId.ring()){
	       case 1:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       case 2:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       case 3:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       }
	       break;
	     case 3:
	       switch(rpcId.ring()){
	       case 1:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       case 2:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       case 3:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       }
	       break;
	     case 4:
	       switch(rpcId.ring()){
	       case 1:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       case 2:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       case 3:
		 ++ENDCAProll[rpcId.station()][rpcId.ring()];
		 break;
	       }
	       break;
	     }
	   }
	 }
	 // end RPC Endcap loop
	 // -------------------
       
	 // Particular Counter
	 if(rpcId.region()==1 && rpcId.station()==2 && (rpcId.sector()==3 || rpcId.sector()==4 || rpcId.sector()==5)) {
	   ++rollsNearDiskp2;
	 }
	 if(rpcId.region()==1 && rpcId.station()==3 && (rpcId.sector()==3 || rpcId.sector()==4 || rpcId.sector()==2)) {
	   ++rollsNearDiskp3;
	 }	 
	 for(int strip=1; strip<=stripsinthisroll; ++strip){
	   // LocalPoint lCentre=(*r)->centreOfStrip(strip);
	   // const BoundSurface& bSurface = (*r)->surface();
	   // GlobalPoint gCentre = bSurface.toGlobal(lCentre);
	   // detailstream<<"Strip="<<strip<<" "<<gCentre.x()<<" "<<gCentre.y()<<" "<<gCentre.z()<<std::endl;
	   ++StripsInCMS;
	   if(rpcId.region()==0) ++counterstripsBarrel;
	   else ++counterstripsEndCap; 
	 }
       }
     }
   }

   detailstream<<"Total Number of Strips in CMS="<<StripsInCMS<<std::endl;
   detailstream<<"Total Number of Rolls in CMS="<<RollsInCMS<<std::endl;
   
   detailstream<<"\n Total Number of Rolls in EndCap= "<<counterRollsEndCap<<std::endl;
   detailstream<<"Total Number of Rolls in Barrel= "<<counterRollsBarrel<<std::endl;   

   detailstream<<"\n Total Number of Strips in Barrel= "<<counterstripsBarrel<<std::endl;
   detailstream<<"Total Number of Strips in EndCap= "<<counterstripsEndCap<<std::endl;
 
   detailstream<<"\n Total Number of Rolls in MB4= "<<counterRollsMB4<<std::endl;
   detailstream<<"Total Number of Rolls in MB1,MB2,MB3= "<<counterRollsMB1MB2MB3<<std::endl;
  
   detailstream<<"RB4 in the barrel:"
	    <<"\n Wheel -2 = "<<RB4Wm2
	    <<"\n Wheel -1 = "<<RB4Wm1
	    <<"\n Wheel 0 = "<<RB4W0
	    <<"\n Wheel +1 = "<<RB4W1
	    <<"\n Wheel +2 = "<<RB4W2
	    <<std::endl;


   detailstream<<"In the Endcaps we have the follow distribution of chambers:"<<std::endl;
   for(int i=1;i<5;i++){
     for(int j=1;j<4;j++){
       detailstream<<" Station "<<i<<" Ring "<<j<<" "<<ENDCAP[i][j]<<" Chambers"<<std::endl;
     }
   }

   detailstream<<"In the Endcaps we have the follow distribution of rolls:"<<std::endl;
   for(int i=1;i<5;i++){
     for(int j=1;j<4;j++){
       detailstream<<" Station "<<i<<" Ring "<<j<<" "<<ENDCAProll[i][j]<<" Roll"<<std::endl;
     }
   }



   detailstream<<"Rolls in Near Disk 2= "<<rollsNearDiskp2<<std::endl;
   detailstream<<"Rolls in Near Disk 3= "<<rollsNearDiskp3<<std::endl;

   detailstream<<"Average Strip Width in Barrel= "<<sumstripwbarrel/counterstripsBarrel<<std::endl;
   detailstream<<"Average Strip Width in EndCap= "<<sumstripwendcap/counterstripsEndCap<<std::endl;

   detailstream<<"Expected RMS Barrel= "<<(sumstripwbarrel/counterstripsBarrel)/sqrt(12)<<std::endl;
   detailstream<<"Expected RMS EndCap= "<<(sumstripwendcap/counterstripsEndCap)/sqrt(12)<<std::endl;

   detailstream<<"Area Covered in Barrel = "<<areabarrel/10000<<"m^2"<<std::endl;
   detailstream<<"Area Covered in EndCap = "<<areaendcap/10000<<"m^2"<<std::endl;

}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCGEO);

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
// $Id: RPCGEO.cc,v 1.8 2011/10/18 13:23:18 yana Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>

//
// class decleration
//

class RPCGEO : public edm::EDAnalyzer {
   public:
      explicit RPCGEO(const edm::ParameterSet&);
      ~RPCGEO();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RPCGEO::RPCGEO(const edm::ParameterSet& /*iConfig*/){
   //now do what ever initialization is needed
}


RPCGEO::~RPCGEO()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
RPCGEO::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup)
{
   using namespace edm;

   std::cout <<" RPCGEO :: analyze :: Getting the RPC Geometry"<<std::endl;
   edm::ESHandle<RPCGeometry> rpcGeo;
   iSetup.get<MuonGeometryRecord>().get(rpcGeo);
   std::cout <<" RPCGEO :: analyze :: Got the RPC Geometry"<<std::endl;

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

   for(int i=1;i<5;i++){
     for(int j=1;j<4;j++){
       ENDCAP[i][j]=0;
     }
   }

   for(int i=1;i<5;i++){
     for(int j=1;j<4;j++){
       ENDCAProll[i][j]=0;
     }
   }

   std::cout <<" RPCGEO :: analyze :: Loop over RPC Chambers"<<std::endl;
   for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
     if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
       RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
       std::vector< const RPCRoll*> roles = (ch->rolls());
       
       //std::cout<<"RPC Chamber"<<ch->id()<<std::endl;
       
       if(ch->id().region()==1){
	 
	 switch(ch->id().station()){
	 case 1:
	   switch(ch->id().ring()){
	   case 1:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	   case 2:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	   case 3:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	   }
	   break;
	   
	 case 2:
	   switch(ch->id().ring()){
	   case 1:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	   case 2:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	   case 3:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	   }
	   break;
	   
	 case 3:
	   switch(ch->id().ring()){
	   case 1:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	     case 2:
	       ENDCAP[ch->id().station()][ch->id().ring()]++;
	       break;
	   case 3:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	   }
	   
	   break;
	 case 4:
	   
	   switch(ch->id().ring()){
	   case 1:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	   case 2:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	   case 3:
	     ENDCAP[ch->id().station()][ch->id().ring()]++;
	     break;
	   }
	   break;
	 }
       }

       std::cout <<" RPCGEO :: analyze :: Loop over RPC Rolls"<<std::endl;
       for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	 RPCDetId rpcId = (*r)->id();
	 int n_strips=(*r)->nstrips();
	 RPCGeomServ rpcsrv(rpcId);
	 
	 //std::cout<<rpcId<<rpcsrv.name()<<" strips="<<n_strips<<std::endl;
	 
	 RollsInCMS++;
	 
	 //std::cout<<rpcId<<" - "<<rpcsrv.name()<<" - "<<rpcsrv.shortname()<<std::endl;
	//std::cout<<rpcsrv.name()<<std::endl;
	

	 if (rpcId.region()==0){ 
	   //std::cout<<"Getting the RPC Topolgy"<<std::endl;
	   const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&((*r)->topology()));
	   float stripl = top_->stripLength();
	   float stripw = top_->pitch();
	   areabarrel = areabarrel + stripl*stripw*n_strips;
	   sumstripwbarrel=sumstripwbarrel+stripw*n_strips;
	   std::cout<<" All Info for "<<rpcsrv.name()<<" striplength = "<<stripl<<"[cm]  stripwidth = "<<stripw<<"[cm]  strips in this roll = "<<n_strips<<" area roll = "<<stripl*stripw*n_strips<<"[cm^2]"<<std::endl;
	   // std::cout<<" area total barrel="<<areabarrel<<std::endl;
	   counterRollsBarrel++; 
	   if(rpcId.station()==4){
	     counterRollsMB4++;
	       switch(rpcId.ring()){
	       case -2:
		 RB4Wm2++;
		 break;
	       case -1:
		 RB4Wm1++;
		 break;
	       case 0:
		 RB4W0++;
		 break;
	       case 1:
		 RB4W1++;
		 break;
	       case 2:
		 RB4W2++;
		 break;
	       }
	   }
	   else counterRollsMB1MB2MB3++;
	 }else{
	   const TrapezoidalStripTopology* top_= dynamic_cast<const TrapezoidalStripTopology*> (&((*r)->topology()));
	   float s1 = static_cast<float>(1)-0.5;
	   float sLast = static_cast<float>(n_strips)-0.5;
	   float stripl = top_->stripLength();
	   float stripw = top_->pitch();
	   areaendcap = areaendcap + stripw*stripl*n_strips;
	   sumstripwendcap=sumstripwendcap+stripw*n_strips;
	   // calculation of min and max stripwidth
	   float radius = top_->radius();
	   float radius_along_stripside = sqrt(pow(radius, 2) + pow(stripw,2));
	   // float stripangle = atan(stripw/radius);
	   // float length_along_stripside = stripw / sin(stripangle);          // Trigonometry
	   float length_along_stripside = stripl/radius*radius_along_stripside; // Thales
	   float delta_stripw = sqrt(pow(length_along_stripside,2)-pow(stripl,2));
	   
	   float stripw_min = stripw - delta_stripw;
	   float stripw_max = stripw + delta_stripw;

	   std::cout<<" All Info for "<<(int)rpcId<<" = "<<rpcsrv.name()<<" :: striplength = "<<stripl<<"[cm] stripwidth = "<<stripw<<"[cm] strips in this roll = "<<n_strips<<" area roll = "<<stripl*stripw*n_strips<<"[cm^2]"; 
	   std::cout<<" min stripwidth = "<<stripw_min<<" max stripwidth = "<<stripw_max<<std::endl;
	   // std::cout<< area total endcap="<<areaendcap<<std::endl;
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

	   if(seg==19) orientation = orientation*-1;

	   std::cout<<rpcsrv.name()<<" midlephi="<<rollphi<<" "<<orientation<<" seg="<<seg
		    <<" First.phi="<<rpcphiFirst<<" First.Y="<<FirstStripCenterPointInGlobal.y()
		    <<"  Last.phi="<<rpcphiLast<<" Last.Y="<<LastStripCenterPointInGlobal.y()
		    <<" Last.X="<<LastStripCenterPointInGlobal.x()
		    <<" Last.Z="<<LastStripCenterPointInGlobal.z();	   

	   //cscphi = 2*3.1415926536+CenterPointCSCGlobal.barePhi():cscphi=CenterPointCSCGlobal.barePhi();

	   bool ok = false;

	   if ( ( ( rpcId.station()==1 
		    && ( ( rpcId.ring()==2 && seg%2!=0 )
			 || rpcId.ring()==3 ) )
		  || rpcId.station()==3 ) 
		&& orientation*rpcId.region()==1.){
	     ok=true;
	   }
	   if ( ( ( rpcId.station()==1 && rpcId.ring()==2 && seg%2==0 ) 
		  || rpcId.station()==2 )
		&& orientation*rpcId.region()==-1. ){
	     ok=true;
	   }
// 	   if((rpcId.station()==1&&(rpcId.ring()==2&&seg%2!=0||rpcId.ring()==3)||rpcId.station()==3)
// 	      &&orientation*rpcId.region()==1.){
// 	     ok=true;
// 	   }
// 	   if((rpcId.station()==1&&rpcId.ring()==2&&seg%2==0||rpcId.station()==2)
// 	      &&orientation*rpcId.region()==-1.){
// 	     ok=true;
// 	   }

	   if(ok) std::cout<<" OK"<<std::endl;
	   else std::cout<<" WRONG!!!"<<std::endl;
	     
	   counterRollsEndCap++;

	   if(rpcId.region()==1){
	     
	     switch(rpcId.station()){
	     case 1:
	       switch(rpcId.ring()){
	       case 1:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       case 2:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       case 3:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       }
	       break;
	   
	     case 2:
	       switch(rpcId.ring()){
	       case 1:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       case 2:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       case 3:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       }
	       break;
	   
	     case 3:
	       switch(rpcId.ring()){
	       case 1:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       case 2:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       case 3:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       }
	   
	       break;
	     case 4:
	       switch(rpcId.ring()){
	       case 1:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       case 2:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       case 3:
		 ENDCAProll[rpcId.station()][rpcId.ring()]++;
		 break;
	       }
	       break;
	     }
	   }
	 }
       
	 //Particular Counter
	 if(rpcId.region()==1&&rpcId.station()==2&&(rpcId.sector()==3||rpcId.sector()==4||rpcId.sector()==5)){
	   rollsNearDiskp2++;
	 }



	 if(rpcId.region()==1&&rpcId.station()==3&&(rpcId.sector()==3||rpcId.sector()==4||rpcId.sector()==2)){
	   rollsNearDiskp3++;
	 }


	 	 
	 for(int strip=1;strip<=n_strips;++strip){
	   //LocalPoint lCentre=(*r)->centreOfStrip(strip);
	   //const BoundSurface& bSurface = (*r)->surface();
	   //GlobalPoint gCentre = bSurface.toGlobal(lCentre);
	   //std::cout<<"Strip="<<strip<<" "<<gCentre.x()<<" "<<gCentre.y()<<" "<<gCentre.z()<<std::endl;
	   StripsInCMS++;
	   if(rpcId.region()==0) counterstripsBarrel++;
	   else counterstripsEndCap++;
	   
	 }
	 
       }
     }
   }

   std::cout<<"Total Number of Strips in CMS="<<StripsInCMS<<std::endl;
   std::cout<<"Total Number of Rolls in CMS="<<RollsInCMS<<std::endl;
   
   std::cout<<"\n Total Number of Rolls in EndCap= "<<counterRollsEndCap<<std::endl;
   std::cout<<"Total Number of Rolls in Barrel= "<<counterRollsBarrel<<std::endl;   

   std::cout<<"\n Total Number of Strips in Barrel= "<<counterstripsBarrel<<std::endl;
   std::cout<<"Total Number of Strips in EndCap= "<<counterstripsEndCap<<std::endl;
 
   std::cout<<"\n Total Number of Rolls in MB4= "<<counterRollsMB4<<std::endl;
   std::cout<<"Total Number of Rolls in MB1,MB2,MB3= "<<counterRollsMB1MB2MB3<<std::endl;
  
   std::cout<<"RB4 in the barrel:"
	    <<"\n Wheel -2 = "<<RB4Wm2
	    <<"\n Wheel -1 = "<<RB4Wm1
	    <<"\n Wheel 0 = "<<RB4W0
	    <<"\n Wheel +1 = "<<RB4W1
	    <<"\n Wheel +2 = "<<RB4W2
	    <<std::endl;


   std::cout<<"In the Endcaps we have the follow distribution of chambers:"<<std::endl;
   for(int i=1;i<5;i++){
     for(int j=1;j<4;j++){
       std::cout<<" Station "<<i<<" Ring "<<j<<" "<<ENDCAP[i][j]<<" Chambers"<<std::endl;
     }
   }

   std::cout<<"In the Endcaps we have the follow distribution of rolls:"<<std::endl;
   for(int i=1;i<5;i++){
     for(int j=1;j<4;j++){
       std::cout<<" Station "<<i<<" Ring "<<j<<" "<<ENDCAProll[i][j]<<" Roll"<<std::endl;
     }
   }



   std::cout<<"Rolls in Near Disk 2= "<<rollsNearDiskp2<<std::endl;
   std::cout<<"Rolls in Near Disk 3= "<<rollsNearDiskp3<<std::endl;

   std::cout<<"Average Strip in Barrel= "<<sumstripwbarrel/counterstripsBarrel<<std::endl;
   std::cout<<"Average Strip in EndCap= "<<sumstripwendcap/counterstripsEndCap<<std::endl;

   std::cout<<"Expected RMS Barrel= "<<(sumstripwbarrel/counterstripsBarrel)/sqrt(12)<<std::endl;
   std::cout<<"Expected RMS EndCap= "<<(sumstripwendcap/counterstripsEndCap)/sqrt(12)<<std::endl;

   std::cout<<"Area Covered in Barrel = "<<areabarrel/10000<<"m^2"<<std::endl;
   std::cout<<"Area Covered in EndCap = "<<areaendcap/10000<<"m^2"<<std::endl;

}


// ------------ method called once each job just before starting event loop  ------------
void 
RPCGEO::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RPCGEO::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCGEO);

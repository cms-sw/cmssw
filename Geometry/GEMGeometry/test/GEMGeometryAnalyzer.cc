/** Derived from DTGeometryAnalyzer by Nicola Amapane
 *
 *  \author M. Maggi - INFN Bari
 */

#include <memory>
#include <fstream>
#include <FWCore/Framework/interface/Frameworkfwd.h>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

class GEMGeometryAnalyzer : public edm::EDAnalyzer {

public: 
  GEMGeometryAnalyzer( const edm::ParameterSet& pset);

  ~GEMGeometryAnalyzer();

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  
  const std::string& myName() { return myName_;}
  
private: 

  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  std::ofstream ofos;
};

GEMGeometryAnalyzer::GEMGeometryAnalyzer( const edm::ParameterSet& /*iConfig*/ )
  : dashedLineWidth_(104), dashedLine_( std::string(dashedLineWidth_, '-') ), 
    myName_( "GEMGeometryAnalyzer" ) 
{ 
  ofos.open("MytestOutput.out"); 
  std::cout <<"======================== Opening output file"<< std::endl;
}


GEMGeometryAnalyzer::~GEMGeometryAnalyzer() 
{
  ofos.close();
  std::cout <<"======================== Closing output file"<< std::endl;
}

void
GEMGeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
  edm::ESHandle<GEMGeometry> pDD;
  iSetup.get<MuonGeometryRecord>().get( pDD );     
  
  std::cout << myName() << ": Analyzer..." << std::endl;
  std::cout << "start " << dashedLine_ << std::endl;


  std::cout << " Geometry node for GEMGeom is  " << &(*pDD) << std::endl;   
  std::cout << " I have "<<pDD->detTypes().size()    << " detTypes" << std::endl;
  std::cout << " I have "<<pDD->detUnits().size()    << " detUnits" << std::endl;
  std::cout << " I have "<<pDD->dets().size()        << " dets" << std::endl;
  std::cout << " I have "<<pDD->etaPartitions().size()       << " eta partitions" << std::endl;
  //  std::cout << " I have "<<pDD->chambers().size()    << " chambers" << std::endl;

  std::cout << myName() << ": Begin iteration over geometry..." << std::endl;
  std::cout << "iter " << dashedLine_ << std::endl;

  std::cout << "\n  #     id(hex)      id(dec)                   "
    "  g(x=0)   g(y=0)   g(z=0)  g(z=-1)  g(z=+1)  Ns "
    "  phi(0)  phi(s1)  phi(sN)    dphi    dphi'      ds     off"
    "       uR       uL       lR       lL" << std::endl;

  //  const double dPi = 3.14159265358;
  //   const double radToDeg = 180. / dPi; //@@ Where to get pi from?

  std::set<GEMDetId> sids;
  std::vector<LocalPoint> vlp;
  vlp.push_back(LocalPoint(-1, 0, 0));
  vlp.push_back(LocalPoint( 0, 0, 0));
  vlp.push_back(LocalPoint( 1, 0, 0));
  vlp.push_back(LocalPoint( 0,-1, 0));
  vlp.push_back(LocalPoint( 0, 0, 0));
  vlp.push_back(LocalPoint( 0, 1, 0));
  vlp.push_back(LocalPoint( 0, 0,-1));
  vlp.push_back(LocalPoint( 0, 0, 0));
  vlp.push_back(LocalPoint( 0, 0, 1));
  /*
  int iGEMCHcount = 0;

  for(TrackingGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){

    //----------------------- GEMCHAMBER TEST -------------------------------------------------------
    
    if( dynamic_cast< GEMChamber* >( *it ) != 0 ){
      ++iGEMCHcount;
      GEMChamber* ch = dynamic_cast< GEMChamber* >( *it ); 

       
      GEMDetId detId=ch->id();
      int idRaf = detId.rawId();
      //       const GEMEtaPartition* rollRaf = ch->roll(1);
      std::cout<<"Num = "<<iGEMCHcount<<"  "<<"GEMDet = "<<idRaf<<"  Num EtaPartitions =" <<ch->nrolls()<<std::endl;
      //       "  "<<"EtaPartition 1 = "<<(rollRaf->id()).rawId()<<std::endl;

      std::vector< const GEMEtaPartition*> rollsRaf = (ch->rolls());
      for(std::vector<const GEMEtaPartition*>::iterator r = rollsRaf.begin();
	   r != rollsRaf.end(); ++r){

	if((*r)->id().region() == 0){
	  std::cout<<"GEMDetId = "<<(*r)->id().rawId()<<std::endl;
	  std::cout<<"Region = "<<(*r)->id().region()<<"  Ring = "<<(*r)->id().ring()<<"  Station = "<<(*r)->id().station()<<"  Sector = "<<(*r)->id().sector()<<"  Layer = "<<(*r)->id().layer()<<"  Subsector = "<<(*r)->id().subsector()<<"  EtaPartition = "<<(*r)->id().roll()<<std::endl;
	}
      }
    }
  */
  //_______________________________________________________________________________________________


  //      if( dynamic_cast< GEMEtaPartition* >( *it ) != 0 ){
  //        ++icount;

  //        GEMEtaPartition* roll = dynamic_cast< GEMEtaPartition*>( *it );

  //        GEMDetId detId=roll->id();
  //        int id = detId(); // or detId.rawId()

  //        const StripTopology* top_ = dynamic_cast<const StripTopology*>
  // 	 (&roll->topology());
      
  //        if (sids.find(detId) != sids.end())
  // 	 std::cout<<"VERYBAD ";
  //        std::cout << "GeomDetUnit is of type " << detId.det() << " and raw id = " << id;


  //        int iw = std::cout.width(); // save current width
  //       int ip = std::cout.precision(); // save current precision

  //        std::cout << "Parameters of roll# " << 
  // 	 std::setw( 4 ) << icount << std::endl;
  //        std::cout<<std::setw(12) << id << std::dec << std::setw(12) << id << std::dec 
  // 		 << std::setw( iw ) << detId;
       
  //        const BoundSurface& bSurface = roll->surface();
       
       
       
  //        LocalPoint  lCentre( 0., 0., 0. );
  //        GlobalPoint gCentre = bSurface.toGlobal( lCentre );
       
  //        LocalPoint  lCentre1( 0., 0., -1.);
  //        GlobalPoint gCentre1 = bSurface.toGlobal( lCentre1 );
       
  //        LocalPoint  lCentre2( 0., 0., 1.);
  //        GlobalPoint gCentre2 = bSurface.toGlobal( lCentre2 );

  //        double gx  =  gCentre.x();
  //        double gy  =  gCentre.y();
  //        double gz  =  gCentre.z();
  //        double gz1 =  gCentre1.z();
  //        double gz2 =  gCentre2.z();
  //        if ( fabs( gx )  < 1.e-06 ) gx = 0.;
  //        if ( fabs( gy )  < 1.e-06 ) gy = 0.;
  //        if ( fabs( gz )  < 1.e-06 ) gz = 0.;
  //        if ( fabs( gz1 ) < 1.e-06 ) gz1 = 0.;
  //        if ( fabs( gz2 ) < 1.e-06 ) gz2 = 0.;

  //        int now = 9;
  //        int nop = 5;
  //        std::cout << 
  // 	 std::setw( now ) << std::setprecision( nop ) << gx <<  
  // 	 std::setw( now ) << std::setprecision( nop ) << gy << 
  // 	 std::setw( now ) << std::setprecision( nop ) << gz << 
  // 	 std::setw( now ) << std::setprecision( nop ) << gz1 << 
  // 	 std::setw( now ) << std::setprecision( nop ) << gz2<<std::endl;
	 
  // Global Phi of centre of GEMEtaPartition

  //        double cphi = gCentre.phi();
  //        double cphiDeg = cphi * radToDeg;
  //        if( cphiDeg < 0. ) cphiDeg = cphiDeg + 360.;

  //        if ( fabs(cphiDeg) < 1.e-06 ) cphiDeg = 0.;
  //        int iphiDeg = static_cast<int>( cphiDeg );
  //	std::cout << "phi(0,0,0) = " << iphiDeg << " degrees" << std::endl;

  //        int nStrips = roll->nstrips();

  //        if ( (detId.region()!=0 && detId.sector() == 1 
  // 	     && detId.subsector() == 1
  // 	     && detId.roll() == 1
  // 	     && detId.station() == 1
  // 	     //&& detId.ring()==3
  // 	     && roll->type().name() == "REA1")
  // 	    ||
  // 	    (detId.region() == 0 && detId.sector() == 1
  // 	     && detId.station() == 1 && detId.layer() == 1
  // 	     && detId.roll() == 1 
  // 	     && detId.ring() == 0)
  // 	    ) {
  // 	 std::cout <<"======================== Writing output file"<< std::endl;
  // 	 ofos<<"Forward Detector "<<roll->type().name()  <<" "<<detId.region()<<" z :"<<detId<<std::endl;
  // 	 for (unsigned int i=0;i<vlp.size();i++){
  // 	   ofos<< "lp="<<vlp[i]<<" gp="<<roll->toGlobal(vlp[i]) << " pitch="<<roll->localPitch(vlp[i]);
  // 	   if ( (i+1)%3 == 0 ) {
  // 	     ofos<<" "<<std::endl; 
  // 	   }
  // 	 }
  // 	 ofos<<"            Navigating "<<std::endl;
  // 	 LocalPoint edge1 = top_->localPosition(0.);
  // 	 LocalPoint edge2 = top_->localPosition((float)nStrips);
  // 	 float lsl1 = top_->localStripLength(edge1);
  // 	 float lsl2 = top_->localStripLength(edge2);
  // 	 ofos <<" Local Point edge1 = "<<edge1<<" StripLength="<<lsl1<<std::endl; 
  // 	 ofos <<" Local Point edge1 = "<<edge2<<" StripLength="<<lsl2<<std::endl; 
  // 	 float cslength = top_->localStripLength(lCentre);
  // 	 LocalPoint s1(0.,-cslength/2.,0.);
  // 	 LocalPoint s2(0.,cslength/2.,0.);
  // 	 float base1=top_->localPitch(s1)*nStrips;
  // 	 float base2=top_->localPitch(s2)*nStrips;
  //	 LocalPoint  s11(-base1/2., -cslength/2,0.);
  //	 LocalPoint  s12(base1/2., -cslength/2,0.);
  //	 LocalPoint  s21(-base2/2., cslength/2,0.);
  //	 LocalPoint  s22(base2/2.,  cslength/2,0.);
  // 	 ofos<<  "  First Base = "<<base1<<" Second Base ="<<base2<<std::endl;
  //        }
  //        std::cout << "\nStrips =  "<<std::setw( 4 ) << nStrips<<"\n";
  //        for(int is=0;is<nStrips;is++){
  // 	 std::cout <<"s="<<std::setw(3)<<is+1<<" pos="
  // 		   <<roll->centreOfStrip(is+1);
  // 	 if ((is+1)%5==0){ 
  // 	   float str=is;
  // 	   std::cout <<"s="<<std::setw(6)<<str<<" pos="
  // 		     <<roll->centreOfStrip(str)<<" gpos="<<
  // 	     roll->toGlobal(roll->centreOfStrip(str));
  // 	   std::cout <<std::endl;
  // 	 }
  //}
  //std::cout <<std::endl;
       
  //       double cstrip1  = layer->centerOfStrip(1).phi();
  //       double cstripN  = layer->centerOfStrip(nStrips).phi();

  //       double phiwid   = layer->geometry()->stripPhiPitch();
  //       double stripwid = layer->geometry()->stripPitch();
  //       double stripoff = layer->geometry()->stripOffset();
  //       double phidif   = fabs(cstrip1 - cstripN);
  
  //       // May have one strip at 180-epsilon and other at -180+epsilon
  //       // If so the raw difference is 360-(phi extent of chamber)
  //       // Want to reset that to (phi extent of chamber):
  //       if ( phidif > dPi ) phidif = fabs(phidif - 2.*dPi);
  //       double phiwid_check = phidif/double(nStrips-1);

  //       // Clean up some stupid floating decimal aesthetics
  //       cstrip1 = cstrip1 * radToDeg;
  //       if ( cstrip1 < 0. ) cstrip1 = cstrip1 + 360.;
  //       if ( fabs( cstrip1 ) < 1.e-06 ) cstrip1 = 0.;
  //       cstripN = cstripN * radToDeg;
  //       if ( cstripN < 0. ) cstripN = cstrip1 + 360.;
  //       if ( fabs( cstripN ) < 1.e-06 ) cstripN = 0.;

  //       if ( fabs( stripoff ) < 1.e-06 ) stripoff = 0.;

  //       now = 9;
  //       nop = 4;
   //       std::cout 
  // 	<< std::setw( now ) << std::setprecision( nop ) << cphiDeg
  // 	<< std::setw( now ) << std::setprecision( nop ) << cstrip1
  // 	<< std::setw( now ) << std::setprecision( nop ) << cstripN
  // 	<< std::setw( now ) << std::setprecision( nop ) << phiwid 
  // 	<< std::setw( now ) << std::setprecision( nop ) << phiwid_check 
  // 	<< std::setw( now ) << std::setprecision( nop ) << stripwid
  // 	<< std::setw( now ) << std::setprecision( nop ) << stripoff ;
  //          << std::setw(8) << (layer->getOwner())->sector() ; //@@ No sector yet!

  //       // Layer geometry:  layer corner phi's...

  //       std::vector<float> parameters = layer->geometry()->parameters();
  //       // these parameters are half-lengths, due to GEANT
  //       float bottomEdge = parameters[0];
  //       float topEdge    = parameters[1];
  //       float thickness  = parameters[2];
  //       float apothem    = parameters[3];

  //       // first the face nearest the interaction
  //       // get the other face by using positive thickness
  //       LocalPoint upperRightLocal(topEdge, apothem, -thickness);
  //       LocalPoint upperLeftLocal(-topEdge, apothem, -thickness);
  //       LocalPoint lowerRightLocal(bottomEdge, -apothem, -thickness);
  //       LocalPoint lowerLeftLocal(-bottomEdge, -apothem, -thickness);

  //       GlobalPoint upperRightGlobal = bSurface.toGlobal(upperRightLocal);
  //       GlobalPoint upperLeftGlobal  = bSurface.toGlobal(upperLeftLocal);
  //       GlobalPoint lowerRightGlobal = bSurface.toGlobal(lowerRightLocal);
  //       GlobalPoint lowerLeftGlobal  = bSurface.toGlobal(lowerLeftLocal);
  
  //       float uRGp = radToDeg * upperRightGlobal.phi();
  //       float uLGp = radToDeg * upperLeftGlobal.phi();
  //       float lRGp = radToDeg * lowerRightGlobal.phi();
  //       float lLGp = radToDeg * lowerLeftGlobal.phi();
  
      
  //       now = 9;
  //       std::cout 
  // 	<< std::setw( now ) << uRGp
  // 	<< std::setw( now ) << uLGp
  // 	<< std::setw( now ) << lRGp
  // 	<< std::setw( now ) << lLGp 
  // 	<< std::endl;
      
  //       // Reset the values we changed
  //       std::cout << std::setprecision( ip ) << std::setw( iw );
  
  //        sids.insert(detId);
  //      }  
  //    }
  
  //   //   for (TrackingGeometry::DetTypeContainer::const_iterator it = pDD->detTypes().begin(); it != pDD->detTypes().end(); it ++){
  //   //   }

   std::cout << dashedLine_ << " end" << std::endl;
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(GEMGeometryAnalyzer);

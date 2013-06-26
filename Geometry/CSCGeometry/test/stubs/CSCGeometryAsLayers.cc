#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <DataFormats/GeometryVector/interface/Pi.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <DataFormats/GeometryVector/interface/GlobalPoint.h>

#include <string>
#include <cmath>
#include <iomanip> // for setw() etc.
#include <vector>

class CSCGeometryAsLayers : public edm::EDAnalyzer {

   public:
 
     explicit CSCGeometryAsLayers( const edm::ParameterSet& );
      ~CSCGeometryAsLayers();

      virtual void analyze( const edm::Event&, const edm::EventSetup& );
 
      const std::string& myName() { return myName_;}

   private: 

      const int dashedLineWidth_;
      const std::string dashedLine_;
      const std::string myName_;
};

CSCGeometryAsLayers::CSCGeometryAsLayers( const edm::ParameterSet& iConfig )
 : dashedLineWidth_(194), dashedLine_( std::string(dashedLineWidth_, '-') ), 
  myName_( "CSCGeometryAsLayers" )
{
}


CSCGeometryAsLayers::~CSCGeometryAsLayers()
{
}

void
 CSCGeometryAsLayers::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   const double dPi = Geom::pi();
   const double radToDeg = 180. / dPi;

   std::cout << myName() << ": Analyzer..." << std::endl;
   std::cout << "start " << dashedLine_ << std::endl;
   std::cout << "pi = " << dPi << ", radToDeg = " << radToDeg << std::endl;

   edm::ESHandle<CSCGeometry> pgeom;
   iSetup.get<MuonGeometryRecord>().get( pgeom );     
   std::cout << " Geometry node for CSCGeom is  " << &(*pgeom) << std::endl;   
   std::cout << " I have "<<pgeom->dets().size() << " detectors" << std::endl;
   std::cout << " I have "<<pgeom->detTypes().size() << " types" << "\n" << std::endl;

   std::cout << myName() << ": Begin iteration over geometry..." << std::endl;

   std::vector<CSCLayer*> vl = pgeom->layers();
   std::cout << "No. of layers stored = " << vl.size() << std::endl;

   std::cout << "\n  #     id(dec)      id(oct)                   "
     "  g(x=0)   g(y=0)   g(z=0)  g(z=-1)  g(z=+1)  Ns "
     "  phi(0)  phi(s1)  phi(sN)    dphi    dphi'      ds     off"
     "       uR       uL       lR       lL" << std::endl;
   std::cout << dashedLine_ << std::endl;

   int icount = 0;

   for( std::vector<CSCLayer*>::const_iterator it = vl.begin(); it != vl.end(); ++it ){

      const CSCLayer* layer = *it;
     
      if( layer ){
        ++icount;

        DetId detId = layer->geographicalId();
        int id = detId(); // or detId.rawId()

	// There's going to be a lot of messing with field width (and precision) so
	// save input values...
        int iw = std::cout.width(); // save current width
        int ip = std::cout.precision(); // save current precision

	std::cout <<
           std::setw( 4 ) << icount << 
  	   std::setw(12) << id << std::oct << std::setw(12) << id << std::dec << std::setw( iw ) <<
           "   E" << CSCDetId::endcap(id) << 
           " S" << CSCDetId::station(id) << 
           " R" << CSCDetId::ring(id) << 
           " C" << std::setw( 2 ) << CSCDetId::chamber(id) << std::setw( iw ) << 
           " L" << CSCDetId::layer(id);

	// What's its surface?
	// The surface knows how to transform local <-> global

	const Surface& bSurface = layer->surface();

	// Check global coordinates of centre of CSCLayer, and how
	// local z direction relates to global z direction

        LocalPoint  lCentre( 0., 0., 0. );
        GlobalPoint gCentre = bSurface.toGlobal( lCentre );

        LocalPoint  lCentre1( 0., 0., -1.);
        GlobalPoint gCentre1 = bSurface.toGlobal( lCentre1 );

        LocalPoint  lCentre2( 0., 0., 1.);
        GlobalPoint gCentre2 = bSurface.toGlobal( lCentre2 );

        double gx  =  gCentre.x();
        double gy  =  gCentre.y();
        double gz  =  gCentre.z();
        double gz1 =  gCentre1.z();
        double gz2 =  gCentre2.z();
        if ( fabs( gx )  < 1.e-06 ) gx = 0.;
        if ( fabs( gy )  < 1.e-06 ) gy = 0.;
        if ( fabs( gz )  < 1.e-06 ) gz = 0.;
        if ( fabs( gz1 ) < 1.e-06 ) gz1 = 0.;
        if ( fabs( gz2 ) < 1.e-06 ) gz2 = 0.;

	int now = 9;
	int nop = 5;
	std::cout << 
	    std::setw( now ) << std::setprecision( nop ) << gx <<  
            std::setw( now ) << std::setprecision( nop ) << gy << 
            std::setw( now ) << std::setprecision( nop ) << gz << 
	    std::setw( now ) << std::setprecision( nop ) << gz1 << 
            std::setw( now ) << std::setprecision( nop ) << gz2;

	// Global Phi of centre of CSCLayer

	//@@ CARE The following attempted conversion to degrees can be easily
	// subverted by GeometryVector/Phi.h enforcing its range convention!
	// Either a) use a separate local double before scaling...
	//        double cphi = gCentre.phi();
	//        double cphiDeg = cphi * radToDeg;
	// or b) use Phi's degree conversion...
        double cphiDeg = gCentre.phi().degrees();

	// I want to display in range 0 to 360

        // Handle some occasional ugly precision problems around zero
        if ( fabs(cphiDeg) < 1.e-06 ) {
          cphiDeg = 0.;
	}
        else if ( cphiDeg < 0. ) {
          cphiDeg += 360.;
	}
        else if ( cphiDeg >= 360. ) {
	  std::cout << "WARNING: resetting phi= " << cphiDeg << " to zero." << std::endl;
          cphiDeg = 0.;
	}

	//        int iphiDeg = static_cast<int>( cphiDeg );
	//	std::cout << "phi(0,0,0) = " << iphiDeg << " degrees" << std::endl;

        int nStrips = layer->geometry()->numberOfStrips();
        std::cout << std::setw( 4 ) << nStrips;

        double cstrip1  = layer->centerOfStrip(1).phi();
        double cstripN  = layer->centerOfStrip(nStrips).phi();

        double phiwid   = layer->geometry()->stripPhiPitch();
        double stripwid = layer->geometry()->stripPitch();
        double stripoff = layer->geometry()->stripOffset();
        double phidif   = fabs(cstrip1 - cstripN);

        // May have one strip at 180-epsilon and other at -180+epsilon
        // If so the raw difference is 360-(phi extent of chamber)
        // Want to reset that to (phi extent of chamber):
        if ( phidif > dPi ) phidif = fabs(phidif - 2.*dPi);
        double phiwid_check = phidif/double(nStrips-1);

	// Clean up some stupid floating decimal aesthetics
	cstrip1 = cstrip1 * radToDeg;
	if ( fabs( cstrip1 ) < 1.e-06 ) cstrip1 = 0.;
	else if ( cstrip1 < 0. ) cstrip1 += 360.;

	cstripN = cstripN * radToDeg;
	if ( fabs( cstripN ) < 1.e-06 ) cstripN = 0.;
	else if ( cstripN < 0. ) cstripN += 360.;

	if ( fabs( stripoff ) < 1.e-06 ) stripoff = 0.;

	now = 9;
	nop = 4;
        std::cout 
          << std::setw( now ) << std::setprecision( nop ) << cphiDeg
	  << std::setw( now ) << std::setprecision( nop ) << cstrip1
          << std::setw( now ) << std::setprecision( nop ) << cstripN
          << std::setw( now ) << std::setprecision( nop ) << phiwid 
          << std::setw( now ) << std::setprecision( nop ) << phiwid_check 
          << std::setw( now ) << std::setprecision( nop ) << stripwid
	  << std::setw( now ) << std::setprecision( nop ) << stripoff ;
	  //          << std::setw(8) << (layer->getOwner())->sector() ; //@@ No sector yet!

        // Layer geometry:  layer corner phi's...

	std::array<const float, 4> const & parameters = layer->geometry()->parameters();
        // these parameters are half-lengths, due to GEANT
        float hBottomEdge = parameters[0];
        float hTopEdge    = parameters[1];
        float hThickness  = parameters[2];
        float hApothem    = parameters[3];

        // first the face nearest the interaction
        // get the other face by using positive hThickness
        LocalPoint upperRightLocal(hTopEdge, hApothem, -hThickness);
        LocalPoint upperLeftLocal(-hTopEdge, hApothem, -hThickness);
        LocalPoint lowerRightLocal(hBottomEdge, -hApothem, -hThickness);
        LocalPoint lowerLeftLocal(-hBottomEdge, -hApothem, -hThickness);
 
        GlobalPoint upperRightGlobal = bSurface.toGlobal(upperRightLocal);
        GlobalPoint upperLeftGlobal  = bSurface.toGlobal(upperLeftLocal);
        GlobalPoint lowerRightGlobal = bSurface.toGlobal(lowerRightLocal);
        GlobalPoint lowerLeftGlobal  = bSurface.toGlobal(lowerLeftLocal);

        float uRGp = upperRightGlobal.phi().degrees();
        float uLGp = upperLeftGlobal.phi().degrees();
        float lRGp = lowerRightGlobal.phi().degrees();
        float lLGp = lowerLeftGlobal.phi().degrees();

        now = 9;
        std::cout 
         << std::setw( now ) << uRGp
         << std::setw( now ) << uLGp
         << std::setw( now ) << lRGp
         << std::setw( now ) << lLGp 
         << std::endl;
        
	// Reset the values we changed
	std::cout << std::setprecision( ip ) << std::setw( iw );
  
    }
    else {
      std::cout << "WEIRD ERROR: a null CSCLayer* " << std::endl;
    }
  }	

   std::cout << dashedLine_ << " end" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCGeometryAsLayers);

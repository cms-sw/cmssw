// LAST UPDATED 16.09.2009 ptc

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <DataFormats/GeometryVector/interface/Pi.h>
#include <DataFormats/GeometryVector/interface/GlobalPoint.h>

#include <string>
#include <cmath>
#include <iomanip> // for setw() etc.
#include <vector>

class CSCGeometryOfWires : public edm::EDAnalyzer {

   public:
 
     explicit CSCGeometryOfWires( const edm::ParameterSet& );
      ~CSCGeometryOfWires();

      virtual void analyze( const edm::Event&, const edm::EventSetup& );
 
      const std::string& myName() { return myName_;}

   private: 

      const int dashedLineWidth_;
      const std::string dashedLine_;
      const std::string myName_;
};

CSCGeometryOfWires::CSCGeometryOfWires( const edm::ParameterSet& iConfig )
 : dashedLineWidth_(101), dashedLine_( std::string(dashedLineWidth_, '-') ), 
  myName_( "CSCGeometryOfWires" )
{
}


CSCGeometryOfWires::~CSCGeometryOfWires()
{
}

void
 CSCGeometryOfWires::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   const double dPi = Geom::pi();
   const double radToDeg = 180. / dPi;

   std::cout << myName() << ": Analyzer..." << std::endl;
   std::cout << "start " << dashedLine_ << std::endl;

   edm::ESHandle<CSCGeometry> pDD;
   iSetup.get<MuonGeometryRecord>().get( pDD );     
   std::cout << " Geometry node for CSCGeometry is  " << &(*pDD) << std::endl;   
   std::cout << " I have "<<pDD->detTypes().size()    << " detTypes" << std::endl;
   std::cout << " I have "<<pDD->detUnits().size()    << " detUnits" << std::endl;
   std::cout << " I have "<<pDD->dets().size()        << " dets" << std::endl;
   std::cout << " I have "<<pDD->layers().size()      << " layers" << std::endl;
   std::cout << " I have "<<pDD->chambers().size()    << " chambers" << std::endl;


   std::cout << myName() << ": Begin iteration over geometry..." << std::endl;
   std::cout << "iter " << dashedLine_ << std::endl;

   int icount = 0;

   // Check the DetUnits
   for( CSCGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); ++it ){


     // Do we really have a CSC layer?

     CSCLayer* layer = dynamic_cast<CSCLayer*>( *it );
     
      if( layer ) {
        ++icount;

	// What's its DetId?

        DetId detId = layer->geographicalId();
        int id = detId(); // or detId.rawId()

	//	std::cout << "GeomDetUnit is of type " << detId.det() << " and raw id = " << id << std::endl;

 	std::cout << "\n" << "Parameters of layer# " << icount <<
	     " id= " << id << " = " << std::oct << id << std::dec << " (octal) " << 
             "   E" << CSCDetId::endcap(id) << 
             " S" << CSCDetId::station(id) << 
             " R" << CSCDetId::ring(id) << 
             " C" << CSCDetId::chamber(id) << 
 	     " L" << CSCDetId::layer(id) << " are:" << std::endl;

        const CSCLayerGeometry* geom = layer->geometry();
        std::cout << *geom;

        const CSCStripTopology* cst = geom->topology();
        std::cout << "\n" << *cst;

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

        int nStrips = geom->numberOfStrips();
	//        std::cout << std::setw( 4 ) << nStrips;

        double cstrip1  = layer->centerOfStrip(1).phi();
        double cstripN  = layer->centerOfStrip(nStrips).phi();

        double phiwid   = geom->stripPhiPitch();
        double stripwid = geom->stripPitch();
        double stripoff = geom->stripOffset();
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

        // Layer geometry:  layer corner phi's... OF ALUMINUM FRAME

	std::array<const float, 4> const & parameters = geom->parameters();
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

        LocalPoint upperEdgeOnY( 0., hApothem );
        LocalPoint lowerEdgeOnY( 0., -hApothem );
        LocalPoint leftEdgeOnX( -(hTopEdge+hBottomEdge)/2., 0. );
        LocalPoint rightEdgeOnX( (hTopEdge+hBottomEdge)/2., 0. );

 
        GlobalPoint upperRightGlobal = bSurface.toGlobal(upperRightLocal);
        GlobalPoint upperLeftGlobal  = bSurface.toGlobal(upperLeftLocal);
        GlobalPoint lowerRightGlobal = bSurface.toGlobal(lowerRightLocal);
        GlobalPoint lowerLeftGlobal  = bSurface.toGlobal(lowerLeftLocal);

        float uRGp = upperRightGlobal.phi().degrees();
        float uLGp = upperLeftGlobal.phi().degrees();
        float lRGp = lowerRightGlobal.phi().degrees();
        float lLGp = lowerLeftGlobal.phi().degrees();


	std::cout << "\nCHAMBER FRAME corners in local coordinates: \n UR " <<
	  upperRightLocal << "\n UL " << upperLeftLocal << "\n LR " <<
	  lowerRightLocal << "\n LL " << lowerLeftLocal << std::endl;

	std::cout << "CHAMBER FRAME corners in global coords: \n UR " << 
	  upperRightGlobal << "\n UL " << upperLeftGlobal << "\n LR " <<
	  lowerRightGlobal << "\n LL " << lowerLeftGlobal << 
	  "\n   phi: UR " << uRGp << " UL " << uLGp << " LR " <<
	  lRGp << " LL " << lLGp << std::endl;


	float ang = nStrips * phiwid;
	float ctoi = cst->centreToIntersection();

	std::cout << "\nSTRIP PLANE:" << std::endl;
	std::cout << "============" << std::endl;
	std::cout << "centreToIntersection, R = " << ctoi << std::endl;
	std::cout << "local y of centre of strip plane, yOff = " << cst->yCentreOfStripPlane() << std::endl;
	std::cout << "originToIntersection, R-yOff = " << cst->originToIntersection() << std::endl;
	std::cout << "extent of strip plane in local y = " << cst->yExtentOfStripPlane() << std::endl;
	std::cout << "no. of strips = " << nStrips << std::endl;

	// Local coordinates of ends of strips
        MeasurementPoint mp1m(0.5, -0.5); // strip 1, -y
	MeasurementPoint mp1p(0.5,  0.5); // strip 1, +y
	MeasurementPoint mp2m(static_cast<float>(nStrips)-0.5, -0.5); // strip N, -y
	MeasurementPoint mp2p(static_cast<float>(nStrips)-0.5,  0.5); // strip N, +y
	LocalPoint lp1m = geom->topology()->localPosition(mp1m);
	LocalPoint lp1p = geom->topology()->localPosition(mp1p);
	LocalPoint lp2m = geom->topology()->localPosition(mp2m);
	LocalPoint lp2p = geom->topology()->localPosition(mp2p);

	std::cout << "Strip 1 local coords " << lp1m << " " << lp1p << std::endl;
	std::cout << "Strip " << nStrips <<" local coords " << lp2m << " " << lp2p << std::endl;

	std::cout << "angular width of one strip = " << phiwid << " rads " << std::endl;
	std::cout << "angle subtended by layer, A = nstrips x stripPhiPitch = " << ang << " rads = " <<
	  ang * radToDeg  << " deg" << std::endl;
	std::cout << "phi (clockwise from local y axis) of one edge = " << cst->phiOfOneEdge() << " rads " << std::endl;

	std::cout << "phi width check: (centre strip N - centre strip 1)/(nstrips-1) = " << phiwid_check << std::endl;

	std::cout << "Check how well approximations work: " << std::endl;
	std::cout << "[T+B = " << hTopEdge+hBottomEdge <<
	  "] compared with [2R*tan(A/2) = " << 
	  2.*ctoi*tan(ang/2.) << "]" << std::endl;
	std::cout << "[T-B = " << hTopEdge-hBottomEdge <<
	  "] compared with [2a*tan(A/2) = " << 
	  2.*hApothem*tan(ang/2.) << "]" << std::endl;
	std::cout << "[R = " << ctoi <<
	  "] compared with [0.5*(T+B)/tan(A/2) = " << 
	  0.5 *(hTopEdge+hBottomEdge)/tan(ang/2.) << "]" << std::endl;

	std::cout << "Approximations to where strips intersect: " << std::endl;
	std::cout << "RST: match y=0, oi = (hT+hB)/2 / tan(A/2) = " << 
	  0.5*(hTopEdge+hBottomEdge) / tan(0.5*nStrips*phiwid) << std::endl;
	std::cout << "RST: match top, oi = hT / tan(A/2) = " << 
	  hTopEdge / tan(0.5*nStrips*phiwid) << std::endl;
	std::cout << "TST: oi = hA*(hT+hB)/(hT-hB) = " << 
	  hApothem * (hTopEdge+hBottomEdge)/(hTopEdge-hBottomEdge) << std::endl;

	std::cout << "\nstrip offset = " << stripoff << std::endl;

	std::cout << "\nlocal(0,0,-1) = global " << gCentre1 << std::endl;
	std::cout << "local(0,0)    = global " << gCentre << std::endl;
	std::cout << "local(0,0,+1) = global " << gCentre2 << std::endl;
   

	// CSCLG::stripAngle(int strip)
	std::cout << "CSCLG Angle of strip 1 = " << 
	  geom->stripAngle(1) * radToDeg << " deg " << std::endl;
	std::cout << "CSCLG Angle of strip " << nStrips/2 << " = " <<
	  geom->stripAngle( nStrips/2 ) * radToDeg << " deg " << std::endl;
	std::cout << "CSCLG Angle of strip " << nStrips << " = " <<
	  geom->stripAngle( nStrips ) * radToDeg << " deg " << std::endl;

	// CSCST::stripAngle(float strip) Yes this one's float the CSCLG is int
	std::cout << "CSCST Angle of centre of strip 1 = " << 
	  cst->stripAngle(0.5) * radToDeg << " deg " << std::endl;
	std::cout << "CSCST Angle of centre of strip " << nStrips/2 << " = " <<
	  cst->stripAngle( nStrips/2 -0.5 ) * radToDeg << " deg " << std::endl;
	std::cout << "CSCST Angle of centre of strip " << nStrips << " = " <<
	  cst->stripAngle( nStrips -0.5 ) * radToDeg << " deg " << std::endl;

	std::cout << "Local x of strip 1 on x axis = " << 
	  geom->xOfStrip(1, 0.) << std::endl;
	std::cout << "Local x of strip " << nStrips/2 << " on x axis = " <<
	  geom->xOfStrip( nStrips/2, 0. ) << std::endl;
	std::cout << "Local x of strip " << nStrips << " on x axis = " <<
	  geom->xOfStrip( nStrips, 0. ) << std::endl;

	std::cout << "Local x of strip 1 at upper edge = " << 
	  geom->xOfStrip(1, hApothem) << std::endl;
	std::cout << "Local x of strip " << nStrips/2 << " at upper edge = " <<
	  geom->xOfStrip( nStrips/2, hApothem ) << std::endl;
	std::cout << "Local x of strip " << nStrips << " at upper edge = " <<
	  geom->xOfStrip( nStrips, hApothem ) << std::endl;

	std::cout << "Local x of strip 1 at lower edge = " << 
	  geom->xOfStrip(1, -hApothem) << std::endl;
	std::cout << "Local x of strip " << nStrips/2 << " at lower edge = " <<
	  geom->xOfStrip( nStrips/2, -hApothem ) << std::endl;
	std::cout << "Local x of strip " << nStrips << " at lower edge = " <<
	  geom->xOfStrip( nStrips, -hApothem ) << std::endl;

	std::cout << "Strip width           = " << stripwid << std::endl;
	std::cout << "Strip pitch at middle = " << geom->stripPitch() << std::endl;
	std::cout << "Strip pitch at (0,0)  = " << geom->stripPitch( lCentre ) << std::endl;

	std::cout << "Strip pitch at UR     = " << geom->stripPitch( upperRightLocal ) << std::endl;
	std::cout << "Strip pitch at UL     = " << geom->stripPitch( upperLeftLocal ) << std::endl;
	std::cout << "Strip pitch at LL     = " << geom->stripPitch( lowerLeftLocal ) << std::endl;
	std::cout << "Strip pitch at LR     = " << geom->stripPitch( lowerRightLocal ) << std::endl;

	std::cout << "Strip pitch at upper edge on y axis = " << geom->stripPitch( upperEdgeOnY ) << std::endl;
	std::cout << "Strip pitch at lower edge on y axis = " << geom->stripPitch( lowerEdgeOnY ) << std::endl;
	std::cout << "Strip pitch at left edge on x axis  = " << geom->stripPitch( leftEdgeOnX ) << std::endl;
	std::cout << "Strip pitch at right edge on x axis = " << geom->stripPitch( rightEdgeOnX ) << std::endl;

	// Check input to nearestStrip()
	std::cout << "Strip units for (0,0) =                " << cst->strip( lCentre ) << std::endl;
	std::cout << "Strip units for upper edge on y axis = " << cst->strip( upperEdgeOnY ) << std::endl;
	std::cout << "Strip units for lower edge on y axis = " << cst->strip( lowerEdgeOnY ) << std::endl;
	std::cout << "Strip units for left edge on x axis  = " << cst->strip( leftEdgeOnX ) << std::endl;
	std::cout << "Strip units for right edge on x axis = " << cst->strip( rightEdgeOnX ) << std::endl;

	std::cout << "Nearest strip to (0,0) =                " << geom->nearestStrip( lCentre ) << std::endl;
	std::cout << "Nearest strip to upper edge on y axis = " << geom->nearestStrip( upperEdgeOnY ) << std::endl;
	std::cout << "Nearest strip to lower edge on y axis = " << geom->nearestStrip( lowerEdgeOnY ) << std::endl;
	std::cout << "Nearest strip to left edge on x axis  = " << geom->nearestStrip( leftEdgeOnX ) << std::endl;
	std::cout << "Nearest strip to right edge on x axis = " << geom->nearestStrip( rightEdgeOnX ) << std::endl;

	int iNUR =  geom->nearestStrip( upperRightLocal );
	int iNUL =  geom->nearestStrip( upperLeftLocal );
	int iNLR =  geom->nearestStrip( lowerRightLocal );
	int iNLL =  geom->nearestStrip( lowerLeftLocal );

	int jNUR =  geom->nearestWire( upperRightLocal );
	int jNUL =  geom->nearestWire( upperLeftLocal );
	int jNLR =  geom->nearestWire( lowerRightLocal );
	int jNLL =  geom->nearestWire( lowerLeftLocal );

	std::cout << "Calculated no. of strips across top    = " << (iNUR-iNUL+1) << std::endl;
	std::cout << "Calculated no. of strips across bottom = " << (iNLR-iNLL+1) << std::endl;

	std::cout << "Nearest strip, wire to UR = " << iNUR << ", " << jNUR << std::endl;
	std::cout << "Nearest strip, wire to UL = " << iNUL << ", " << jNUL << std::endl;
	std::cout << "Nearest strip, wire to LR = " << iNLR << ", " << jNLR << std::endl;
	std::cout << "Nearest strip, wire to LL = " << iNLL << ", " << jNLL << std::endl;

	std::cout << "yOfWire(" << jNUR << " , +hTopEdge ) = " << 
	  geom->yOfWire( static_cast<float>(jNUR), hTopEdge ) << std::endl;
	std::cout << "yOfWire(" << jNUL << " , -hTopEdge ) = " << 
	  geom->yOfWire( static_cast<float>(jNUL), -hTopEdge ) << std::endl;
	std::cout << "yOfWire(" << jNLR << " , hBottomEdge ) = " << 
	  geom->yOfWire( static_cast<float>(jNLR), hBottomEdge ) << std::endl;
	std::cout << "yOfWire(" << jNLL << " , -hBottomEdge ) = " << 
	  geom->yOfWire( static_cast<float>(jNLL), -hBottomEdge ) << std::endl;

	std::cout << "Examine global phi of strip at top and bottom of chamber frame:" << std::endl;
	float phi_1_c = (layer->centerOfStrip(1)).phi();
	float phi_n_c = (layer->centerOfStrip(nStrips)).phi();
	float phi_c_c = (layer->centerOfStrip(nStrips/2)).phi();
	float x_1_t = geom->xOfStrip(1, hApothem); // x of strip 1 at top edge
	float x_1_b = geom->xOfStrip(1, -hApothem); // x of strip 1 at bottom edge
	float x_n_t = geom->xOfStrip(nStrips, hApothem); // x of strip n at top edge
	float x_n_b = geom->xOfStrip(nStrips, -hApothem); // x of strip n at bottom edge
	float x_c_t = geom->xOfStrip(nStrips/2, hApothem); // x of strip n/2 at top edge
	float x_c_b = geom->xOfStrip(nStrips/2, -hApothem); // x of strip n/2 at bottom edge
	GlobalPoint g_1_t = layer->toGlobal( LocalPoint(x_1_t, hApothem, 0.) );
	GlobalPoint g_1_b = layer->toGlobal( LocalPoint(x_1_b, -hApothem, 0.) );
	GlobalPoint g_n_t = layer->toGlobal( LocalPoint(x_n_t, hApothem, 0.) );
	GlobalPoint g_n_b = layer->toGlobal( LocalPoint(x_n_b, -hApothem, 0.) );
	GlobalPoint g_c_t = layer->toGlobal( LocalPoint(x_c_t, 0., 0.) );
	GlobalPoint g_c_b = layer->toGlobal( LocalPoint(x_c_b, 0., 0.) );
	float phi_1_t = g_1_t.phi();
	float phi_1_b = g_1_b.phi();
	float phi_n_t = g_n_t.phi();
	float phi_n_b = g_n_b.phi();
	float phi_c_t = g_c_t.phi();
	float phi_c_b = g_c_b.phi();
	std::cout << " strip  1 top: " << phi_1_t << " centre: " << phi_1_c << " bottom: " << phi_1_b << " top-bottom: " << phi_1_t-phi_1_b << std::endl;
	std::cout << " strip " << nStrips/2 << " top: " << phi_c_t << " centre: " << phi_c_c << " bottom: " << phi_c_b << " top-bottom: " << phi_c_t-phi_c_b << std::endl;
	std::cout << " strip " << nStrips << " top: " << phi_n_t << " centre: " << phi_n_c << " bottom: " << phi_n_b << " top-bottom: " << phi_n_t-phi_n_b << std::endl;


	int nwg = geom->numberOfWireGroups();
	int nw = geom->numberOfWires();

	std::cout << "\nWIRE PLANE:" << std::endl;
	std::cout << "===========" << std::endl;
	std::cout << "wireSpacing = " << geom->wirePitch() << " cm " << std::endl;
    	std::cout << "wireAngle = " << geom->wireAngle() << " rads " << std::endl;
	std::cout << "no. of wires = " << nw << std::endl;
	std::cout << "no. of wire groups = " << nwg << std::endl;
	std::cout << "no. of wires in wg 1 = " << geom->numberOfWiresPerGroup( 1 ) << std::endl;
	std::cout << "no. of wires in wg " << nwg << " = " << geom->numberOfWiresPerGroup( nwg ) << std::endl;
	std::cout << "wire group containing wire 1 = " << geom->wireGroup( 1 ) << std::endl;
	std::cout << "wire group containing wire " << nw  << " = " << geom->wireGroup( nw ) << std::endl;
	std::cout << "length of wg 1 = " << geom->lengthOfWireGroup( 1 ) << std::endl;
	std::cout << "length of wg " << nwg << " = " << geom->lengthOfWireGroup( nwg ) << std::endl;
	std::cout << "middle wire of wg 1 = " << geom->middleWireOfGroup( 1 ) << std::endl;
	std::cout << "middle wire of wg " << nwg << " = " << geom->middleWireOfGroup( nwg ) << std::endl;
	std::cout << "y of wg 1 at x=0 is " << geom->yOfWireGroup( 1 ) << std::endl;
	std::cout << "y of wg " << nwg << " at x=0 is " << geom->yOfWireGroup( nwg ) << std::endl;
	std::cout << "centre of wg 1 is " << geom->localCenterOfWireGroup( 1 ) << std::endl;
	std::cout << "centre of wg " << nwg << " is " << geom->localCenterOfWireGroup( nwg ) << std::endl;
    	std::cout << "y of wire 1 at x=0 = " << geom->wireTopology()->yOfWire(1., 0.) << std::endl;
    	std::cout << "y of wire " << nw << " at x=0 = " << geom->wireTopology()->yOfWire(static_cast<float>(nw), 0.) << std::endl;

	std::cout << "narrow width of wire plane = " << geom->wireTopology()->narrowWidthOfPlane() << std::endl;
	std::cout << "wide width   of wire plane = " << geom->wireTopology()->wideWidthOfPlane() << std::endl;
	std::cout << "y extent     of wire plane = " << geom->wireTopology()->lengthOfPlane() << std::endl;
	std::cout << "perp extent  of wire plane = " << geom->wireTopology()->extentOfWirePlane() << std::endl;

	std::pair< LocalPoint, LocalPoint > lw1 =
	  geom->wireTopology()->wireEnds( geom->middleWireOfGroup( 1 ) );
	std::cout << "local coords of ends of central wire of wire group 1 = " <<
	  "(" << lw1.first.x() << ", " << lw1.first.y() << "), " <<
	  "(" << lw1.second.x() << ", " << lw1.second.y() << ")" << std::endl;

	std::pair< LocalPoint, LocalPoint > lw2 =
  	  geom->wireTopology()->wireEnds( geom->middleWireOfGroup( nwg ) );
	std::cout << "local coords of ends of central wire of wire group " << nwg << " = " <<
	  "(" << lw2.first.x() << ", " << lw2.first.y() << "), " <<
	  "(" << lw2.second.x() << ", " << lw2.second.y() << ")" << std::endl;


	// Check idToDetUnit
	const GeomDetUnit * gdu = pDD->idToDetUnit(detId);
	assert(gdu==layer);
	// Check idToDet
	const GeomDet * gd = pDD->idToDet(detId);
	assert(gd==layer);
    }
    else {
      std::cout << "Could not dynamic_cast to a CSCLayer* " << std::endl;
    }
   }
   std::cout << dashedLine_ << " end" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCGeometryOfWires);

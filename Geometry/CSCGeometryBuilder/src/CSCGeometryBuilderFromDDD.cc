#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include <Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CSCGeometry/src/CSCWireGroupPackage.h>
#include <Geometry/CommonDetUnit/interface/TrackingGeometry.h>
#include <Geometry/MuonNumbering/interface/CSCNumberingScheme.h>
#include <Geometry/MuonNumbering/interface/MuonBaseNumber.h>
#include <Geometry/MuonNumbering/interface/MuonDDDNumbering.h>
#include <Geometry/Surface/interface/BoundPlane.h>
#include <Geometry/Surface/interface/TrapezoidalPlaneBounds.h>
#include <Geometry/Vector/interface/Basic3DVector.h>

#include <iostream>
#include <iomanip>

CSCGeometryBuilderFromDDD::CSCGeometryBuilderFromDDD(){}


CSCGeometryBuilderFromDDD::~CSCGeometryBuilderFromDDD(){}


CSCGeometry* CSCGeometryBuilderFromDDD::build(const DDCompactView* cview){

  try {
    std::string attribute = "MuStructure";      // could come from outside
    std::string value     = "MuonEndcapCSC";    // could come from outside
    DDValue muval(attribute, value, 0.0);

    // Asking for a specific section of the MuStructure

    DDSpecificsFilter filter;
    filter.setCriteria(muval, // name & value of a variable 
		       DDSpecificsFilter::equals,
		       DDSpecificsFilter::AND, 
		       true, // compare strings otherwise doubles
		       true // use merged-specifics or simple-specifics
		       );

    DDFilteredView fview( *cview );
    fview.addFilter(filter);

    bool doSubDets = fview.firstChild();
    doSubDets      = fview.firstChild(); // and again?!
    return buildEndcaps( &fview ); 
  }

  catch (const DDException & e ) {
    std::cerr << "DDD Exception: something went wrong during XML parsing!" << std::endl
	      << "  Message: " << e << std::endl
	      << "  Terminating execution ... " << std::endl;
    throw;	       
  }
  catch (const std::exception & e) {
    std::cerr << "An unexpected exception occured: " << e.what() << std::endl;
    throw;
  }
  catch (...) {
    std::cerr << "An unexpected exception occured!" << std::endl
	      << "  Terminating execution ... " << std::endl;
    std::unexpected();	         
  }

}

CSCGeometry* CSCGeometryBuilderFromDDD::buildEndcaps( DDFilteredView* fv ){

  bool doAll(true);

  CSCGeometry* theGeometry = new CSCGeometry;

  while (doAll) {
    std::string upstr = "upar";
    std::vector<const DDsvalues_type *> spec = fv->specifics();
    std::vector<const DDsvalues_type *>::const_iterator spit = spec.begin();

    std::vector<double> uparvals;

    // Package up the wire group info as it's decoded
    CSCWireGroupPackage wg; 

    //std::cout << "size of spec=" << spec.size() << std::endl;
    for (;  spit != spec.end(); spit++) {
      std::map< unsigned int, DDValue >::const_iterator it = (**spit).begin();
      for (;  it != (**spit).end(); it++) {
	//std::cout << "it->second.name()=" << it->second.name() << std::endl;  
	if (it->second.name() == upstr) {
	  uparvals = it->second.doubles();
	  //std::cout << "found upars " << std::endl;
	} else if (it->second.name() == "NumWiresPerGrp") {
	  //numWiresInGroup = it->second.doubles();
	  for ( size_t i = 0 ; i < it->second.doubles().size(); i++) {
	    wg.wiresInEachGroup.push_back( int( it->second.doubles()[i] ) );
	  }
	  //std::cout << "found upars " << std::endl;
	} else if ( it->second.name() == "NumGroups" ) {
	  //numGroups = it->second.doubles();
	  for ( size_t i = 0 ; i < it->second.doubles().size(); i++) {
	    wg.consecutiveGroups.push_back( int( it->second.doubles()[i] ) );
	  }
	} else if ( it->second.name() == "WireSpacing" ) {
	  wg.wireSpacing = it->second.doubles()[0];
	} else if ( it->second.name() == "CenterPinToFirstWire" ) {
	  wg.alignmentPinToFirstWire = it->second.doubles()[0]; 
	} else if ( it->second.name() == "TotNumWireGroups" ) {
	  wg.numberOfGroups = int(it->second.doubles()[0]);
	}
      }
    }

    std::vector<float> fpar;
    std::vector<double> dpar = fv->logicalPart().solid().parameters();
    
    if ( debugV ) {
      std::cout << myName << ": fill fpar..." << std::endl;
      std::cout << myName << ": dpars are... " << 
          dpar[4]/cm << ", " << dpar[8]/cm << ", " << 
          dpar[3]/cm << ", " << dpar[0]/cm << std::endl;
    }

    fpar.push_back( static_cast<float>( dpar[4]/cm) );
    fpar.push_back( static_cast<float>( dpar[8]/cm ) ); 
    fpar.push_back( static_cast<float>( dpar[3]/cm ) ); 
    fpar.push_back( static_cast<float>( dpar[0]/cm ) ); 

    if ( debugV ) std::cout << myName << ": fill gtran..." << std::endl;

    std::vector<float> gtran;
    DDTranslation tran = fv->translation();
    for (size_t i = 0; i < 3; i++) {
      gtran.push_back( (float) 1.0 *  tran[i] / cm );
    }

    if ( debugV ) std::cout << myName << ": fill grmat..." << std::endl;

    std::vector<float> grmat( 9 ); // set dim so can use [.] to fill
    size_t rotindex = 0;
    for (size_t i = 0; i < 3; i++) {
      rotindex = i;
      HepRotation::HepRotation_row r = fv->rotation()[i];
      size_t j = 0;
      for (; j < 3; j++) {
	grmat[rotindex] = (float) 1.0 *  r[j];
	rotindex += 3;
      }
    }

    if ( debugV ) std::cout << myName << ": fill fupar..." << std::endl;

    std::vector<float> fupar;
    for (size_t i = 0; i < uparvals.size(); i++)
      fupar.push_back( static_cast<float>( uparvals[i] ) );

    // MuonNumbering numbering wraps the subdetector hierarchy labelling

    if ( debugV ) std::cout << myName << ": create numbering scheme..." << std::endl;

    MuonDDDNumbering mdn;
    MuonBaseNumber mbn = mdn.geoHistoryToBaseNumber(fv->geoHistory());
    CSCNumberingScheme mens;

    if ( debugV ) std::cout << myName << ": find detid..." << std::endl;

    int detid = mens.baseNumberToUnitNumber( mbn ); //@@ FIXME perhaps should return CSCDetId itself?

    if ( debugV ) {
      std::cout << myName << ": detid for this layer is " << detid << 
	", octal " << std::oct << detid << ", hex " << std::hex << detid << std::dec << std::endl;
      std::cout << myName << ": looking for wire group info for layer " << std::endl;
      std::cout << "E" << CSCDetId::endcap(detid) << 
             " S" << CSCDetId::station(detid) << 
	     " R" << CSCDetId::ring(detid) <<
             " C" << CSCDetId::chamber(detid) <<
             " L" << CSCDetId::layer(detid) << std::endl 
	          << fv->geoHistory() << std::endl;
	    
      if ( wg.numberOfGroups != 0 ) {
	std::cout << "fv->geoHistory:      = " << fv->geoHistory() << std::endl;
	std::cout << "TotNumWireGroups     = " << wg.numberOfGroups << std::endl;
	std::cout << "WireSpacing          = " << wg.wireSpacing << std::endl;
	std::cout << "CenterPinToFirstWire = " << wg.alignmentPinToFirstWire << std::endl;
	std::cout << "wg.consecutiveGroups.size() = " << wg.consecutiveGroups.size() << std::endl;
	std::cout << "wg.wiresInEachGroup.size() = " << wg.wiresInEachGroup.size() << std::endl;
	std::cout << "NumGroups\tWiresInGroup" << std::endl;
	for (size_t i = 0; i < wg.consecutiveGroups.size(); i++) {
	  std::cout << wg.consecutiveGroups[i] << "\t\t" << wg.wiresInEachGroup[i] << std::endl;
	}
      } else {
	std::cout << myName << ":DDD is MISSING SpecPars for wire groups" << std::endl;
      }
      std::cout << myName << ": end of wire group info. " << std::endl;
    }

    this->buildLayer (theGeometry, detid, fpar, fupar, gtran, grmat, wg );

    doAll = fv->next();
  }

  return theGeometry;  
}

void CSCGeometryBuilderFromDDD::buildLayer (  
	CSCGeometry* theGeometry,         // the geometry container
	int    detid,                     // packed index from CSCDetId
        const std::vector<float>& fpar,   // volume parameetrs
	const std::vector<float>& fupar,  // user parameters
	const std::vector<float>& gtran,  // translation vector
	const std::vector<float>& grmat,  // rotation matric
        const CSCWireGroupPackage& wg     // wire group info
	) {

  if ( debugV ) std::cout << "\n\n" << myName << ": entering buildLayer" << std::endl;
  int jend   = CSCDetId::endcap( detid );
  int jstat  = CSCDetId::station( detid );
  int jring  = CSCDetId::ring( detid );
  int jch    = CSCDetId::chamber( detid );
  int jlay   = CSCDetId::layer( detid );

  CSCDetId layerId = CSCDetId( detid );
  CSCDetId chamberId = CSCDetId( jend, jstat, jring, jch, 0 );

  //   Geometrical shape parameters for Endcap Muon for active gas volumes MWij
  //   cf. the hardware nomenclature for Endcap Muon chambers is MEi/j

  //  In the Endcap Muon system, a chamber (a cathode strip chamber) is composed of 6 'layers'.
  //  A 'layer' is a gas volume between two planes of cathode strips,
  //  with a plane of anode wires midway between.

  //@@ Magic Number... to find center of chamber from position of Layer 1
  // Note that par[2] is the gas-volume half-thickness, not the physical unit half-thickness;
  // It is either 0.28cm (ME11) or 0.476cm (all others), so that 'full gas gap' is 5.6mm or 9.52mm.
  // But what we need here is the physical layer thickness which is gas volume + honeycomb...
  // Overall, this is about an inch, or nearer 2.56 or 2.57cm. I take 2.56cm, which is also
  // consistent with GEANT z-positions of layers (layer 1 to layer 6 is about 12.8cm)

  const float hFiveLayerThickness = 6.4; // (cm) 2.5 layer thicknesses along z
  
  const size_t kNpar = 4;
  if ( fpar.size() != kNpar ) 
    std::cerr << "mu_end_make_geom: Error, expected npar=" 
	      << kNpar << ", found npar=" << fpar.size() << std::endl;

  if ( debugV ) {
    std::cout << myName << ":  E" << jend << " S" << jstat << " R" << jring <<
      " C" << jch << " L" << jlay << std::endl;
    std::cout << "npar=" << fpar.size() << " par[0]=" << fpar[0] 
         << " par[1]=" << fpar[1] << " par[2]=" << fpar[2] << " par[3]=" << fpar[3] << std::endl;
    std::cout << "gtran[0,1,2]=" << gtran[0] << " " << gtran[1] << " " << gtran[2] << std::endl;
    std::cout << "grmat[0-8]=" << grmat[0] << " " << grmat[1] << " " << grmat[2] << " "
         << grmat[3] << " " << grmat[4] << " " << grmat[5] << " "
         << grmat[6] << " " << grmat[7] << " " << grmat[8] << std::endl;
    std::cout << "nupar=" << fupar.size() << " upar[0]=" << fupar[0] << std::endl;
  }

   // offset translates centre of Layer 1 to centre of Chamber
   // Since Layer 1 is nearest IP in station 1,2 but Layer 6 is
   // nearest in station 3,4, we need to make it negative sometimes...
  float offset = hFiveLayerThickness;
  if ( (jend==1 && jstat>2 ) || ( jend==2 && jstat<3 ) ) offset=-offset;

      //@@ The z of a ring is not well-defined but isn't useful anyway.
      // The trouble is that the chambers in a ring overlap so there is no
      // easy way to calculate the ring z from the Layer 1 z of first chamber
      // constructed in that ring.
  //    float ringZ = gtran[2] + offset; //@@ Wrong, but no easy solution.
  //    float layerRMin = sqrt(gtran[0]*gtran[0] + gtran[1]*gtran[1]) - par[3]; // gtran gives center
  //    float layerRMax = layerRMin + 2*par[3];


  Pointer2Chamber chamber = theGeometry->getChamber( chamberId );
  if ( chamber ){
  }
  else { // this chamber not yet built/stored
  
   // Get or build ChamberSpecs for this chamber
    if ( debugV ) std::cout << myName << ": CSCChamberSpecs::build requested." << std::endl;
    int chamberType = CSCChamberSpecs::whatChamberType( jstat, jring );
    CSCChamberSpecs* aSpecs = CSCChamberSpecs::lookUp( chamberType );
    if ( aSpecs == 0 ) aSpecs = CSCChamberSpecs::build( chamberType, fpar, fupar, wg );

   // Build a Transformation out of GEANT gtran and grmat...
   // These are used to transform a point in the local reference frame
   // of a subdetector to the global frame of CMS by
   //         (grmat)^(-1)*local + (gtran)
   // The corresponding transformation from global to local is
   //         (grmat)*(global - gtran)
 
    //    BoundSurface::RotationType aRot( grmat );
    BoundSurface::RotationType aRot( grmat[0], grmat[1], grmat[2], 
                                     grmat[3], grmat[4], grmat[5],
                                     grmat[6], grmat[7], grmat[8] );

      // This rotation we get from GEANT takes the detector face as the x-z plane.
      // For consistency with the rest of CMS we would like the detector
      // face to be the local x-y plane.
      // On top of this the -z endcap has LH local coordinates, since it
      // is built in CMSIM/GEANT3 as a *reflection* of +z.
      // So we need to rotate, and in -z flip local x.

      // aRot.rotateAxes will transform aRot so that it becomes
      // applicable to the new local coordinates: detector face in x-y plane
      // looking out along z, in either endcap.

      // The interface for rotateAxes specifies 'new' X,Y,Z but the
      // implementation in fact deals with them as the 'old'. Confusing!

    Basic3DVector<float> oldX( 1., 0.,  0. );
    Basic3DVector<float> oldY( 0., 0., -1. );
    Basic3DVector<float> oldZ( 0., 1.,  0. );

    if ( gtran[2]<0. ) oldX *= -1; 

    aRot.rotateAxes(oldX, oldY, oldZ);
      
      // Choose the odd-layer geometry, just for its plane bounds.
    TrapezoidalPlaneBounds* bounds = 
	new TrapezoidalPlaneBounds(*aSpecs->oddLayerGeometry( jend ) );
      // Centre of chamber in z is 2.5 layers from centre of layer 1...
    Surface::PositionType aVec( gtran[0], gtran[1], gtran[2]+offset ); 
    BoundPlane * plane = new BoundPlane(aVec, aRot, bounds); 
      // bounds has been passed to BoundPlane, which clones it, so we can delete it here
    delete bounds;

    Pointer2Chamber aChamber( new CSCChamber( plane, chamberId, aSpecs ) );
    theGeometry->addChamber( chamberId, aChamber ); 
    chamber = aChamber;

    if ( debugV )  std::cout << myName << ": E" << jend << " S" << jstat 
	           << " R" << jring << " C" << jch << " L" << jlay 
                   << " gtran[2]=" << gtran[2]
		   << " offset=" << offset << " par[2]=" << fpar[2]
		   << " chamber adr=" << chamber << std::endl;
  }
   
  const CSCLayer* cLayer = dynamic_cast<const CSCLayer*> (theGeometry->idToDet( layerId ) );
  if ( cLayer == 0 ) {
    const CSCChamberSpecs* aSpecs = chamber->specs();
    const CSCLayerGeometry* aGeom = 
        (jlay%2 != 0) ? aSpecs->oddLayerGeometry( jend ) : 
                        aSpecs->evenLayerGeometry( jend );

    // Make the BoundPlane 

    //@@ FIXME when chamber has a surface can retrieve rotation rather than re-creating it
    //    BoundSurface::RotationType chamberRotation( grmat[0], grmat[1], grmat[2], 
    //                                     grmat[3], grmat[4], grmat[5],
    //                                     grmat[6], grmat[7], grmat[8] );

    //    Basic3DVector<float> oldX( 1., 0.,  0. );
    //    Basic3DVector<float> oldY( 0., 0., -1. );
    //    Basic3DVector<float> oldZ( 0., 1.,  0. );
    //    if ( gtran[2]<0. ) oldX *= -1; 
    //    chamberRotation.rotateAxes(oldX, oldY, oldZ);

    BoundSurface::RotationType chamberRotation = chamber->surface().rotation();

      // Put the layer at the correct Z
    BoundPlane::PositionType layerPosition( gtran[0], gtran[1], gtran[2] );
    TrapezoidalPlaneBounds* bounds = new TrapezoidalPlaneBounds( *aGeom );
    BoundPlane * aPlane = new BoundPlane(layerPosition, chamberRotation, bounds);
    delete bounds;

    CSCLayer* aLayer = new CSCLayer( aPlane, layerId, chamber, aGeom );
    chamber->addComponent(jlay, aLayer); 
    theGeometry->addDet( aLayer );
    theGeometry->addDetId( aLayer->geographicalId() );
    theGeometry->addDetType( const_cast<GeomDetType*>( &(aLayer->type()) ) ); //@@ FIXME drop const_cast asap!
  }
  else {
    std::cerr << myName << ": ERROR, layer " << jlay <<
            " for chamber = " << ( chamber->cscId() ) <<
            " already exists: layer address=" << cLayer <<
            " chamber address=" << chamber << std::endl;
    //    if ( !logger().debugOut ) muEndDumpAddress( 1, emu, jend, jstat, jring, jch, jlay );

  }         
  //  if ( logger().debugOut ) muEndDumpAddress( 9, emu, jend, jstat, jring, jch, jlay );
}

const std::string CSCGeometryBuilderFromDDD::myName = "CSCGeometryBuilderFromDDD";
bool CSCGeometryBuilderFromDDD::debugV = false;


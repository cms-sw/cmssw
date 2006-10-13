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
#include <Geometry/MuonNumbering/interface/MuonDDDConstants.h>
#include <Geometry/Surface/interface/BoundPlane.h>
#include <Geometry/Surface/interface/TrapezoidalPlaneBounds.h>
#include <Geometry/Vector/interface/Basic3DVector.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>
#include <iomanip>

CSCGeometryBuilderFromDDD::CSCGeometryBuilderFromDDD() : myName("CSCGeometryBuilderFromDDD"){}


CSCGeometryBuilderFromDDD::~CSCGeometryBuilderFromDDD(){}


CSCGeometry* CSCGeometryBuilderFromDDD::build(const DDCompactView* cview, const MuonDDDConstants& muonConstants){

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
    return buildEndcaps( &fview, muonConstants ); 
  }

  catch (const DDException & e ) {
    edm::LogError("DDD") << "something went wrong during XML parsing!" << std::endl
	      << "  Message: " << e << std::endl
	      << "  Terminating execution ... " << std::endl;
    throw;	       
  }
  catch (const std::exception & e) {
    edm::LogError("CSC") << "An unexpected exception occured: " << e.what() << std::endl;
    throw;
  }
  catch (...) {
    edm::LogError("CSC") << "An unexpected unnamed exception occured!" << std::endl
	      << "  Terminating execution ... " << std::endl;
    std::unexpected();	         
  }

}

CSCGeometry* CSCGeometryBuilderFromDDD::buildEndcaps( DDFilteredView* fv, const MuonDDDConstants& muonConstants ){

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
    
    LogDebug("CSC") << ": fill fpar..." << "\n";
    LogDebug("CSC") << ": dpars are... " << 
          dpar[4]/cm << ", " << dpar[8]/cm << ", " << 
          dpar[3]/cm << ", " << dpar[0]/cm << "\n";


    fpar.push_back( static_cast<float>( dpar[4]/cm) );
    fpar.push_back( static_cast<float>( dpar[8]/cm ) ); 
    fpar.push_back( static_cast<float>( dpar[3]/cm ) ); 
    fpar.push_back( static_cast<float>( dpar[0]/cm ) ); 

    LogDebug("CSC") << ": fill gtran..." << "\n";

    std::vector<float> gtran;
    DDTranslation tran = fv->translation();
    for (size_t i = 0; i < 3; i++) {
      gtran.push_back( (float) 1.0 *  tran[i] / cm );
    }

    LogDebug("CSC") << ": fill grmat..." << "\n";

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

    LogDebug("CSC") << ": fill fupar..." << "\n";

    std::vector<float> fupar;
    for (size_t i = 0; i < uparvals.size(); i++)
      fupar.push_back( static_cast<float>( uparvals[i] ) );

    // MuonNumbering numbering wraps the subdetector hierarchy labelling

    LogDebug("CSC") << ": create numbering scheme..." << "\n";

    MuonDDDNumbering mdn(muonConstants);
    MuonBaseNumber mbn = mdn.geoHistoryToBaseNumber(fv->geoHistory());
    CSCNumberingScheme mens(muonConstants);

    LogDebug("CSC") << ": find detid..." << "\n";

    int detid = mens.baseNumberToUnitNumber( mbn ); //@@ FIXME perhaps should return CSCDetId itself?


      LogDebug("CSC") << ": detid for this layer is " << detid << 
	", octal " << std::oct << detid << ", hex " << std::hex << detid << std::dec << "\n";
      LogDebug("CSC") << ": looking for wire group info for layer " << "\n";
      LogDebug("CSC") << "E" << CSCDetId::endcap(detid) << 
             " S" << CSCDetId::station(detid) << 
	     " R" << CSCDetId::ring(detid) <<
             " C" << CSCDetId::chamber(detid) <<
  	     " L" << CSCDetId::layer(detid) << "\n";
	//			   << fv->geoHistory() << "\n";
	    
      if ( wg.numberOfGroups != 0 ) {
	LogDebug("CSC") << "fv->geoHistory:      = " << fv->geoHistory() << "\n";
	LogDebug("CSC") << "TotNumWireGroups     = " << wg.numberOfGroups << "\n";
	LogDebug("CSC") << "WireSpacing          = " << wg.wireSpacing << "\n";
	LogDebug("CSC") << "CenterPinToFirstWire = " << wg.alignmentPinToFirstWire << "\n";
	LogDebug("CSC") << "wg.consecutiveGroups.size() = " << wg.consecutiveGroups.size() << "\n";
	LogDebug("CSC") << "wg.wiresInEachGroup.size() = " << wg.wiresInEachGroup.size() << "\n";
	LogDebug("CSC") << "NumGroups\tWiresInGroup" << "\n";
	for (size_t i = 0; i < wg.consecutiveGroups.size(); i++) {
	  LogDebug("CSC") << wg.consecutiveGroups[i] << "\t\t" << wg.wiresInEachGroup[i] << "\n";
	}
      } else {
	LogDebug("CSC") << ":DDD is MISSING SpecPars for wire groups" << "\n";
      }
      LogDebug("CSC")<< ": end of wire group info. " << "\n";


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

  LogDebug("CSC") << ": entering buildLayer" << "\n";
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

  const float layerThickness = 2.56; // (cm) effective thickness of a chamber
  const float centreChamberToFirstLayer = layerThickness * 2.5; // (cm) distance between centre of chamber and first layer
  
  const size_t kNpar = 4;
  if ( fpar.size() != kNpar ) 
    edm::LogError("CSC") << "Error, expected npar=" 
	      << kNpar << ", found npar=" << fpar.size() << std::endl;

  LogDebug("CSC") << ":  E" << jend << " S" << jstat << " R" << jring <<
    " C" << jch << " L" << jlay << "\n";
  LogDebug("CSC") << "npar=" << fpar.size() << " par[0]=" << fpar[0] 
		  << " par[1]=" << fpar[1] << " par[2]=" << fpar[2] << " par[3]=" << fpar[3] << "\n";
  LogDebug("CSC") << "gtran[0,1,2]=" << gtran[0] << " " << gtran[1] << " " << gtran[2] << "\n";
  LogDebug("CSC") << "grmat[0-8]=" << grmat[0] << " " << grmat[1] << " " << grmat[2] << " "
         << grmat[3] << " " << grmat[4] << " " << grmat[5] << " "
		  << grmat[6] << " " << grmat[7] << " " << grmat[8] << "\n";
  LogDebug("CSC") << "nupar=" << fupar.size() << " upar[0]=" << fupar[0] << "\n";


   // offset translates centre of Layer 1 to centre of Chamber
   // Since Layer 1 is nearest IP in station 1,2 but Layer 6 is
   // nearest in station 3,4, we need to make it negative sometimes...
  float offset = centreChamberToFirstLayer;
  if ( (jend==1 && jstat>2 ) || ( jend==2 && jstat<3 ) ) offset=-offset;

  CSCChamber* chamber = const_cast<CSCChamber*>(theGeometry->chamber( chamberId ));
  if ( chamber ){
  }
  else { // this chamber not yet built/stored
  
   // Get or build ChamberSpecs for this chamber
    LogDebug("CSC") << ": CSCChamberSpecs::build requested." << "\n";
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
      
      // Find parameters of Trapezoidal Plane Bounds of a layer in the chamber
    std::vector<float> pars = (*aSpecs->oddLayerGeometry( jend ) ).parameters();
    float hChamberThickness = layerThickness * 3.; // 6 layers so half-thickness is 3
      // Now create TPB for chamber 
      // N.B. apothem is 4th in pars but 3rd in ctor (!)
    TrapezoidalPlaneBounds* bounds = 
      new TrapezoidalPlaneBounds( pars[0], pars[1], pars[3], hChamberThickness ); 
      // Centre of chamber in z is 2.5 layers from centre of layer 1...
    Surface::PositionType aVec( gtran[0], gtran[1], gtran[2]+offset ); 
    BoundPlane * plane = new BoundPlane(aVec, aRot, bounds); 
      // bounds has been passed to BoundPlane, which clones it, so we can delete it here
    delete bounds;

    CSCChamber*  aChamber = new CSCChamber( plane, chamberId, aSpecs );
    theGeometry->addChamber( aChamber ); 
    chamber = aChamber;

    LogDebug("CSC") << myName << ": Create chamber E" << jend << " S" << jstat 
	            << " R" << jring << " C" << jch 
                    << " z=" << gtran[2]+offset
		    << " hThick=" << hChamberThickness
		    << " adr=" << chamber << "\n";
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

    LogDebug("CSC") << myName << ": Create layer E" << jend << " S" << jstat 
	            << " R" << jring << " C" << jch << " L" << jlay 
                    << " z=" << gtran[2]
		    << " hThick=" << fpar[2]
		    << " adr=" << aLayer << " layerGeom adr=" << aGeom << "\n";

    chamber->addComponent(jlay, aLayer); 
    theGeometry->addLayer( aLayer );
  }
  else {
    edm::LogError("CSC") << ": ERROR, layer " << jlay <<
            " for chamber = " << ( chamber->id() ) <<
            " already exists: layer address=" << cLayer <<
      " chamber address=" << chamber << "\n";
  }         

}




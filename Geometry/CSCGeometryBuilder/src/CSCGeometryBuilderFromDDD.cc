#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/CLHEPToROOTMath.h>

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
#include <DataFormats/GeometrySurface/interface/BoundPlane.h>
#include <DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h>
#include <DataFormats/GeometryVector/interface/Basic3DVector.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <CLHEP/Units/SystemOfUnits.h>

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
    edm::LogError(myName) << "An unexpected exception occured: " << e.what() << std::endl;
    throw;
  }
  catch (...) {
    edm::LogError(myName) << "An unexpected unnamed exception occured!" << std::endl
	      << "  Terminating execution ... " << std::endl;
    std::unexpected();	         
  }

}

CSCGeometry* CSCGeometryBuilderFromDDD::buildEndcaps( DDFilteredView* fv, const MuonDDDConstants& muonConstants ){

  bool doAll(true);

  CSCGeometry* theGeometry = new CSCGeometry;

  // Here we're reading the cscSpecs.xml file

  int noOfAnonParams = 0;

  while (doAll) {
    std::string upstr = "upar";
    std::vector<const DDsvalues_type *> spec = fv->specifics();
    std::vector<const DDsvalues_type *>::const_iterator spit = spec.begin();

    std::vector<double> uparvals;

    // Package up the wire group info as it's decoded
    CSCWireGroupPackage wg; 

    //std::cout << "size of spec=" << spec.size() << std::endl;
    for (;  spit != spec.end(); spit++) {
      DDsvalues_type::const_iterator it = (**spit).begin();
      for (;  it != (**spit).end(); it++) {
	//std::cout << "it->second.name()=" << it->second.name() << std::endl;  
	if (it->second.name() == upstr) {
	  uparvals = it->second.doubles();
	  //std::cout << "found upars " << std::endl;
	} else if (it->second.name() == "NoOfAnonParams") {
	  noOfAnonParams = static_cast<int>( it->second.doubles()[0] );
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
	} else if ( it->second.name() == "AlignmentPinToFirstWire" ) {
	  wg.alignmentPinToFirstWire = it->second.doubles()[0]; 
	} else if ( it->second.name() == "TotNumWireGroups" ) {
	  wg.numberOfGroups = int(it->second.doubles()[0]);
	} else if ( it->second.name() == "LengthOfFirstWire" ) {
	  wg.narrowWidthOfWirePlane = it->second.doubles()[0];
	} else if ( it->second.name() == "LengthOfLastWire" ) {
	  wg.wideWidthOfWirePlane = it->second.doubles()[0];
	} else if ( it->second.name() == "RadialExtentOfWirePlane" ) {
	  wg.lengthOfWirePlane = it->second.doubles()[0];
	}
      }
    }

    std::vector<float> fpar;
    std::vector<double> dpar = fv->logicalPart().solid().parameters();
    
    LogTrace(myName) << myName  << ": noOfAnonParams=" << noOfAnonParams;

    LogTrace(myName) << myName  << ": fill fpar...";
    LogTrace(myName) << myName  << ": dpars are... " << 
          dpar[4]/cm << ", " << dpar[8]/cm << ", " << 
      dpar[3]/cm << ", " << dpar[0]/cm;


    fpar.push_back( static_cast<float>( dpar[4]/cm) );
    fpar.push_back( static_cast<float>( dpar[8]/cm ) ); 
    fpar.push_back( static_cast<float>( dpar[3]/cm ) ); 
    fpar.push_back( static_cast<float>( dpar[0]/cm ) ); 

    LogTrace(myName) << myName  << ": fill gtran...";

    std::vector<float> gtran( 3 );
// MEC: first pass conversion to ROOT::Math take or modify as you will..
     gtran[0] = (float) 1.0 * (fv->translation().X() / cm);
     gtran[1] = (float) 1.0 * (fv->translation().Y() / cm);
     gtran[2] = (float) 1.0 * (fv->translation().Z() / cm);

// MEC: Other option on final pass ROOT::Math 
    //    std::vector<double> dblgtran( 3 );
    //    fv->translation().GetCoordinates(dblgtran.begin(), dblgtran.end());
    //    size_t gtind = 0;
    //    for (std::vector<double>::const_iterator dit= dblgtran.begin(); dit != dblgtran.end(); ++dit) {
    //      gtran[gtind++]=( (float) 1.0 * (*dit) );
    //    }

    LogTrace(myName) << myName  << ": fill grmat...";

    std::vector<float> grmat( 9 ); // set dim so can use [.] to fill
    // MEC: ROOT::Math conversion
     std::vector<float> trm(9);
     fv->rotation().GetComponents(trm.begin(), trm.end());
     size_t rotindex = 0;
     for (size_t i = 0; i < 9; ++i) {
       grmat[i] = (float) 1.0 * trm[rotindex];
       rotindex = rotindex + 3;
       if ( (i+1) % 3 == 0 ) {
	 rotindex = (i+1) / 3;
       }
     }

    LogTrace(myName) << myName  << ": fill fupar...";

    std::vector<float> fupar;
    for (size_t i = 0; i < uparvals.size(); i++)
      fupar.push_back( static_cast<float>( uparvals[i] ) ); //@ FIXME perhaps should keep as double!

    // MuonNumbering numbering wraps the subdetector hierarchy labelling

    LogTrace(myName) << myName  << ": create numbering scheme...";

    MuonDDDNumbering mdn(muonConstants);
    MuonBaseNumber mbn = mdn.geoHistoryToBaseNumber(fv->geoHistory());
    CSCNumberingScheme mens(muonConstants);

    LogTrace(myName) << myName  << ": find detid...";

    int id = mens.baseNumberToUnitNumber( mbn ); //@@ FIXME perhaps should return CSCDetId itself?

      LogTrace(myName) << myName  << ": raw id for this detector is " << id << 
	", octal " << std::oct << id << ", hex " << std::hex << id << std::dec;
      LogTrace(myName) << myName  << ": looking for wire group info for layer " <<
              "E" << CSCDetId::endcap(id) << 
             " S" << CSCDetId::station(id) << 
	     " R" << CSCDetId::ring(id) <<
             " C" << CSCDetId::chamber(id) <<
  	     " L" << CSCDetId::layer(id);
	    
      if ( wg.numberOfGroups != 0 ) {
	LogTrace(myName) << myName  << ": fv->geoHistory:      = " << fv->geoHistory() ;
	LogTrace(myName) << myName  << ": TotNumWireGroups     = " << wg.numberOfGroups ;
	LogTrace(myName) << myName  << ": WireSpacing          = " << wg.wireSpacing ;
	LogTrace(myName) << myName  << ": AlignmentPinToFirstWire = " << wg.alignmentPinToFirstWire ;
	LogTrace(myName) << myName  << ": Narrow width of wire plane = " << wg.narrowWidthOfWirePlane ;
	LogTrace(myName) << myName  << ": Wide width of wire plane = " << wg.wideWidthOfWirePlane ;
	LogTrace(myName) << myName  << ": Length in y of wire plane = " << wg.lengthOfWirePlane ;
	LogTrace(myName) << myName  << ": wg.consecutiveGroups.size() = " << wg.consecutiveGroups.size() ;
	LogTrace(myName) << myName  << ": wg.wiresInEachGroup.size() = " << wg.wiresInEachGroup.size() ;
	LogTrace(myName) << myName  << ": \tNumGroups\tWiresInGroup" ;
	for (size_t i = 0; i < wg.consecutiveGroups.size(); i++) {
	  LogTrace(myName) << myName  << " \t" << wg.consecutiveGroups[i] << "\t\t" << wg.wiresInEachGroup[i] ;
	}
      } else {
	LogTrace(myName) << myName  << ": DDD is MISSING SpecPars for wire groups" ;
      }
      LogTrace(myName) << myName << ": end of wire group info. " ;

      CSCDetId detid = CSCDetId( id );
      int jendcap  = detid.endcap();
      int jstation = detid.station();
      int jring    = detid.ring();
      int jchamber = detid.chamber();

      if ( jstation==1 && jring==1 ) {
	// set up params for ME1a and ME1b and call buildChamber *for each*
	// Both get the full ME11 dimensions

	// detid is for ME11 and that's what we're using for ME1b in the software
        buildChamber (theGeometry, detid, fpar, fupar, gtran, grmat, wg ); // ME1b

        // No. of anonymous parameters per chamber type should be read from cscSpecs file...
	// Only required for ME11 splitting into ME1a and ME1b values,
        // If it isn't seen may as well try to get further but this value will depend
        // on structure of the file so may not even match! 
        const int kNoOfAnonParams = 35;
        if ( noOfAnonParams == 0 ) { noOfAnonParams = kNoOfAnonParams; } // in case it wasn't seen

	std::copy( fupar.begin()+noOfAnonParams, fupar.end(), fupar.begin() ); // copy ME1a params from back to the front
        CSCDetId detid1a = CSCDetId( jendcap, 1, 4, jchamber, 0 ); // reset to ME1A
        buildChamber (theGeometry, detid1a, fpar, fupar, gtran, grmat, wg ); // ME1a

      }
      else {
        buildChamber (theGeometry, detid, fpar, fupar, gtran, grmat, wg );
      }

    doAll = fv->next();
  }

  return theGeometry;  
}

void CSCGeometryBuilderFromDDD::buildChamber (  
	CSCGeometry* theGeometry,         // the geometry container
	CSCDetId chamberId,               // the DetId for this chamber
        const std::vector<float>& fpar,   // volume parameters hB, hT. hD, hH
	const std::vector<float>& fupar,  // user parameters
	const std::vector<float>& gtran,  // translation vector
	const std::vector<float>& grmat,  // rotation matrix
        const CSCWireGroupPackage& wg     // wire group info
	) {

  LogTrace(myName) << myName  << ": entering buildChamber" ;

  int jend   = chamberId.endcap();
  int jstat  = chamberId.station();
  int jring  = chamberId.ring();
  int jch    = chamberId.chamber();
  int jlay   = chamberId.layer();

  if (jlay != 0 ) edm::LogWarning(myName) << "Error! CSCGeometryBuilderFromDDD was fed layer id = " << jlay << "\n";

  const size_t kNpar = 4;
  if ( fpar.size() != kNpar ) 
    edm::LogError(myName) << "Error, expected npar=" 
	      << kNpar << ", found npar=" << fpar.size() << std::endl;

  LogTrace(myName) << myName  << ":  E" << jend << " S" << jstat << " R" << jring <<
    " C" << jch << " L" << jlay ;
  LogTrace(myName) << myName  << ": npar=" << fpar.size() << " hB=" << fpar[0] 
		  << " hT=" << fpar[1] << " hD=" << fpar[2] << " hH=" << fpar[3] ;
  LogTrace(myName) << myName  << ": gtran[0,1,2]=" << gtran[0] << " " << gtran[1] << " " << gtran[2] ;
  LogTrace(myName) << myName  << ": grmat[0-8]=" << grmat[0] << " " << grmat[1] << " " << grmat[2] << " "
         << grmat[3] << " " << grmat[4] << " " << grmat[5] << " "
		  << grmat[6] << " " << grmat[7] << " " << grmat[8] ;
  LogTrace(myName) << myName  << ": nupar=" << fupar.size() << " upar[0]=" << fupar[0]
		   << " upar[" << fupar.size()-1 << "]=" << fupar[fupar.size()-1];


  CSCChamber* chamber = const_cast<CSCChamber*>(theGeometry->chamber( chamberId ));
  if ( chamber ){
  }
  else { // this chamber not yet built/stored
  
    LogTrace(myName) << myName <<": CSCChamberSpecs::build requested for ME" << jstat << jring ;
    int chamberType = CSCChamberSpecs::whatChamberType( jstat, jring );
    CSCChamberSpecs* aSpecs = CSCChamberSpecs::lookUp( chamberType );
    if ( aSpecs == 0 ) aSpecs = CSCChamberSpecs::build( chamberType, fpar, fupar, wg );

   // Build a Transformation out of GEANT gtran and grmat...
   // These are used to transform a point in the local reference frame
   // of a subdetector to the global frame of CMS by
   //         (grmat)^(-1)*local + (gtran)
   // The corresponding transformation from global to local is
   //         (grmat)*(global - gtran)
 
    BoundSurface::RotationType aRot( grmat[0], grmat[1], grmat[2], 
                                     grmat[3], grmat[4], grmat[5],
                                     grmat[6], grmat[7], grmat[8] );

   // This rotation from GEANT considers the detector face as the x-z plane.
   // We want this to be the local x-y plane.
   // Furthermore, the -z_global endcap has LH local coordinates, since it is built
   // in GEANT as a *reflection* of the +z_global endcap.
   // So we need to rotate, and in -z flip local x.

   // aRot.rotateAxes will transform aRot in place so that it becomes
   // applicable to the new local coordinates: detector face in x-y plane
   // looking out along z, in either endcap.

   // The interface for rotateAxes specifies 'new' X,Y,Z but the
   // implementation deals with them as the 'old'.

    Basic3DVector<float> oldX( 1., 0.,  0. );
    Basic3DVector<float> oldY( 0., 0., -1. );
    Basic3DVector<float> oldZ( 0., 1.,  0. );

    if ( gtran[2]<0. ) oldX *= -1; 

    aRot.rotateAxes( oldX, oldY, oldZ );
      
   // Need to know z of layers w.r.t to z of centre of chamber. 

    float frameThickness     = fupar[31]/10.; // mm -> cm
    float gapThickness       = fupar[32]/10.; // mm -> cm
    float panelThickness     = fupar[33]/10.; // mm -> cm
    float distAverageAGVtoAF = fupar[34]/10.; // mm -> cm

    float layerThickness = gapThickness; // consider the layer to be the gas gap
    float layerSeparation = gapThickness + panelThickness; // centre-to-centre of neighbouring layers

    float chamberThickness = 7.*panelThickness + 6.*gapThickness + 2.*frameThickness ; // chamber frame thickness
    float hChamberThickness = chamberThickness/2.; // @@ should match value returned from DDD directly

    // distAverageAGVtoAF is offset between centre of chamber (AF) and (L1+L6)/2 (average AGVs) 
    float centreChamberToFirstLayer = 2.5 * layerSeparation + distAverageAGVtoAF ; // local z wrt chamber centre
   
   // Now z of wires in layer 1 = z_s1 = centreChamberToFirstLayer;; // layer 1 is at most +ve local z
   //     z of wires in layer N = z_sN = z_s1 - (N-1)*layerSeparation; 
   //     z of strips  in layer N = z_sN = z_sN + gapThickness/2.; @@ BEWARE: need to check if it should be '-gapThickness/2' !

   // Set dimensions of trapezoidal chamber volume 
   // N.B. apothem is 4th in fpar but 3rd in ctor 

    // hChamberThickness and fpar[2] should be the same - but using the above value at least shows
    // how chamber structure works

    //    TrapezoidalPlaneBounds* bounds =  new TrapezoidalPlaneBounds( fpar[0], fpar[1], fpar[3], fpar[2] ); 
    TrapezoidalPlaneBounds* bounds =  new TrapezoidalPlaneBounds( fpar[0], fpar[1], fpar[3], hChamberThickness ); 

   // Centre of chamber in z is specified in DDD
    Surface::PositionType aVec( gtran[0], gtran[1], gtran[2] ); 

    BoundPlane::BoundPlanePointer plane = BoundPlane::build(aVec, aRot, bounds); 
    delete bounds; // bounds cloned by BoundPlane, so we can delete it

    CSCChamber* chamber = new CSCChamber( plane, chamberId, aSpecs );
    theGeometry->addChamber( chamber ); 

    LogTrace(myName) << myName << ": Create chamber E" << jend << " S" << jstat 
 	             << " R" << jring << " C" << jch 
                     << " z=" << gtran[2]
		     << " t/2=" << fpar[2] << " (DDD) or " << hChamberThickness 
		     << " (specs) adr=" << chamber ;

    // Create the component layers of this chamber   
    // We're taking the z as the z of the wire plane within the layer (middle of gas gap)

    // Specify global z of layer by offsetting from centre of chamber: since layer 1 
    // is nearest to IP in stations 1/2 but layer 6 is nearest in stations 3/4, 
    // we need to adjust sign of offset appropriately...
    int localZwrtGlobalZ = +1;
    if ( (jend==1 && jstat<3 ) || ( jend==2 && jstat>2 ) ) localZwrtGlobalZ = -1;

    LogTrace(myName) << myName << ": layerSeparation=" << layerSeparation << ", distAverageAGVtoAF=" 
                     << distAverageAGVtoAF << ", centreChamberToFirstLayer=" 
		     << centreChamberToFirstLayer << ", localZwrtGlobalZ=" << localZwrtGlobalZ
		     << ", gtran[2]=" << gtran[2] ;

    for ( short j = 1; j <= 6; ++j ) {

      CSCDetId layerId = CSCDetId( jend, jstat, jring, jch, j );

      // extra-careful check that we haven't already built this layer
      const CSCLayer* cLayer = dynamic_cast<const CSCLayer*> (theGeometry->idToDet( layerId ) );

      if ( cLayer == 0 ) {

	// build the layer - need the chamber's specs and an appropriate layer-geometry
         const CSCChamberSpecs* aSpecs = chamber->specs();
         const CSCLayerGeometry* geom = 
                    (j%2 != 0) ? aSpecs->oddLayerGeometry( jend ) : 
                                 aSpecs->evenLayerGeometry( jend );

        // Build appropriate BoundPlane, based on parent chamber, with gas gap as thickness

	// centre of chamber is at global z = gtran[2]
	float zlayer = gtran[2] + localZwrtGlobalZ*( centreChamberToFirstLayer - (j-1)*layerSeparation );

        BoundSurface::RotationType chamberRotation = chamber->surface().rotation();
        BoundPlane::PositionType layerPosition( gtran[0], gtran[1], zlayer );
        TrapezoidalPlaneBounds* bounds = new TrapezoidalPlaneBounds( *geom );
	std::vector<float> dims = bounds->parameters(); // returns hb, ht, d, a
        dims[2] = layerThickness/2.; // half-thickness required and note it is 3rd value in vector
        delete bounds;        
        bounds = new TrapezoidalPlaneBounds( dims[0], dims[1], dims[3], dims[2] );
        BoundPlane::BoundPlanePointer plane = BoundPlane::build(layerPosition, chamberRotation, bounds);
	delete bounds;

        CSCLayer* layer = new CSCLayer( plane, layerId, chamber, geom );

        LogTrace(myName) << myName << ": Create layer E" << jend << " S" << jstat 
	            << " R" << jring << " C" << jch << " L" << j
                    << " z=" << zlayer
			 << " t=" << layerThickness << " or " << layer->surface().bounds().thickness()
		    << " adr=" << layer << " layerGeom adr=" << geom ;

        chamber->addComponent(j, layer); 
        theGeometry->addLayer( layer );
      }
      else {
        edm::LogError(myName) << ": ERROR, layer " << j <<
            " for chamber = " << ( chamber->id() ) <<
            " already exists: layer address=" << cLayer <<
            " chamber address=" << chamber << "\n";
      }

    } // layer construction within chamber
  } // chamber construction
}




#include "CSCGeometryBuilder.h"

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <DataFormats/DetId/interface/DetId.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <Geometry/CSCGeometry/src/CSCWireGroupPackage.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <vector>

CSCGeometryBuilder::CSCGeometryBuilder() : myName("CSCGeometryBuilder"){}


CSCGeometryBuilder::~CSCGeometryBuilder(){}


void CSCGeometryBuilder::build( boost::shared_ptr<CSCGeometry> theGeometry
				, const RecoIdealGeometry& rig
				, const CSCRecoDigiParameters& cscpars ) {

  std::vector<float> fpar;
  std::vector<float> gtran;
  std::vector<float> grmat;
  std::vector<float> fupar;
  std::vector<double>::const_iterator it, endIt;
  const std::vector<DetId>& detids(rig.detIds());

  for ( size_t idt = 0; idt < detids.size(); ++idt) {
    CSCDetId detid = CSCDetId( detids[idt] );
    int jstation = detid.station();
    int jring    = detid.ring();

    endIt = rig.shapeEnd(idt);
    fpar.clear();
    for ( it = rig.shapeStart(idt); it != endIt; ++it) {
      fpar.push_back( (float)(*it) );
    }

    gtran.clear();
    endIt = rig.tranEnd(idt);
    for ( it = rig.tranStart(idt); it != endIt; ++it ) {
      gtran.push_back((float)(*it));
    }
    grmat.clear();
    endIt = rig.rotEnd(idt);
    for ( it = rig.rotStart(idt) ; it != endIt; ++it ) {
      grmat.push_back((float)(*it));
    }

    // get the chamber type from existing info
    int chamberType = CSCChamberSpecs::whatChamberType( jstation, jring );
    size_t cs = 0;
    assert ( cscpars.pChamberType.size() != 0 );
    while (cs < cscpars.pChamberType.size() && chamberType != cscpars.pChamberType[cs]) {
      ++cs;
    }
    assert ( cs != cscpars.pChamberType.size() );
      
    // check the existence of the specs for this type WHY? Remove it...
    size_t fu, numfuPars;
    CSCWireGroupPackage wg;
    fu = cscpars.pUserParOffset[cs];
    numfuPars = fu + 1 + size_t(cscpars.pfupars[fu]);

    // upars from db are now uparvals + wg info so we need to unwrap only part here first...
    LogTrace(myName) << myName << ": I think I have " << cscpars.pUserParSize[cs] << " values in pfupars (uparvals)." << std::endl;
    LogTrace(myName) << myName << ": For fupar I will start at " << cscpars.pUserParOffset[cs] + 1 
		     << " in pfupars and go to " << numfuPars << "." << std::endl;
      for ( ++fu; fu < numfuPars; ++fu ) {
	LogTrace(myName) << myName << ": pfupars[" << fu << "]=" << cscpars.pfupars[fu] << std::endl;
	fupar.push_back(cscpars.pfupars[fu]);
      }
    // now, we need to start from "here" at fu to go on and build wg...
    wg.wireSpacing = cscpars.pfupars[fu++];
    wg.alignmentPinToFirstWire = cscpars.pfupars[fu++];
    wg.numberOfGroups = int(cscpars.pfupars[fu++]);
    wg.narrowWidthOfWirePlane = cscpars.pfupars[fu++];
    wg.wideWidthOfWirePlane = cscpars.pfupars[fu++];
    wg.lengthOfWirePlane = cscpars.pfupars[fu++];
    size_t numgrp = static_cast<size_t>(cscpars.pfupars[fu]);
    size_t maxFu = fu + 1 + numgrp;
    fu++;
    for ( ;fu < maxFu; ++fu ) {
      wg.wiresInEachGroup.push_back(int(cscpars.pfupars[fu]));
    } 
    maxFu = fu + numgrp;
    for ( ;fu < maxFu; ++fu ) {
      wg.consecutiveGroups.push_back(int(cscpars.pfupars[fu]));
    } 
          
    if ( wg.numberOfGroups != 0 ) {
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
      
    // Are we going to apply centre-to-intersection offsets, even if values exist in the specs file?
    if ( !theGeometry->centreTIOffsets() ) fupar[30] = 0.;  // reset to zero if flagged 'off'
      
    buildChamber (theGeometry, detid, fpar, fupar, gtran, grmat, wg ); //, cscpars.pWGPs[cs] );
    fupar.clear();
  }
}

void CSCGeometryBuilder::buildChamber (  
				       boost::shared_ptr<CSCGeometry> theGeometry // the geometry container
				       , CSCDetId chamberId                         // the DetId for this chamber
				       , const std::vector<float>& fpar           // volume parameters hB, hT. hD, hH	
				       , const std::vector<float>& fupar          // user parameters 
				       , const std::vector<float>& gtran          // translation vector	
				       , const std::vector<float>& grmat          // rotation matrix
				       , const CSCWireGroupPackage& wg            // wire group info
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


  const CSCChamber* chamber = theGeometry->chamber( chamberId );
  if ( chamber ){
  }
  else { // this chamber not yet built/stored
  
    LogTrace(myName) << myName <<": CSCChamberSpecs::build requested for ME" << jstat << jring ;
     int chamberType = CSCChamberSpecs::whatChamberType( jstat, jring );
     const CSCChamberSpecs* aSpecs = theGeometry->findSpecs( chamberType );
    if ( fupar.size() != 0 && aSpecs == 0 ) {
      // make new one:
      aSpecs = theGeometry->buildSpecs (chamberType, fpar, fupar, wg);
    } else if ( fupar.size() == 0 && aSpecs == 0 ) {
      edm::LogError(myName) << "SHOULD BE THROW? Error, wg and/or fupar size are 0 BUT this Chamber Spec has not been built!";
    }

   // Build a Transformation out of GEANT gtran and grmat...
   // These are used to transform a point in the local reference frame
   // of a subdetector to the global frame of CMS by
   //         (grmat)^(-1)*local + (gtran)
   // The corresponding transformation from global to local is
   //         (grmat)*(global - gtran)
 
    Surface::RotationType aRot( grmat[0], grmat[1], grmat[2], 
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
    float zAverageAGVtoAF    = fupar[34]/10.; // mm -> cm

    float layerThickness = gapThickness; // consider the layer to be the gas gap
    float layerSeparation = gapThickness + panelThickness; // centre-to-centre of neighbouring layers

    float chamberThickness = 7.*panelThickness + 6.*gapThickness + 2.*frameThickness ; // chamber frame thickness
    float hChamberThickness = chamberThickness/2.; // @@ should match value returned from DDD directly

    // distAverageAGVtoAF is offset between centre of chamber (AF) and (L1+L6)/2 (average AGVs) 
    // where AF = AluminumFrame and AGV=ActiveGasVolume (volume names in DDD).
    // It is signed based on global z values: zc - (zl1+zl6)/2
   
    // Local z values w.r.t. AF...
    //     z of wires in layer 1 = z_w1 = +/- zAverageAGVtoAF + 2.5*layerSeparation; // layer 1 is at most +ve local z
    // The sign in '+/-' depends on relative directions of local and global z. 
    // It is '-' if they are the same direction, and '+' if opposite directions.
    //     z of wires in layer N   = z_wN = z_w1 - (N-1)*layerSeparation; 
    //     z of strips in layer N  = z_sN = z_wN + gapThickness/2.; @@ BEWARE: need to check if it should be '-gapThickness/2' !

    // Set dimensions of trapezoidal chamber volume 
    // N.B. apothem is 4th in fpar but 3rd in ctor 

    // hChamberThickness and fpar[2] should be the same - but using the above value at least shows
    // how chamber structure works

    TrapezoidalPlaneBounds* bounds =  new TrapezoidalPlaneBounds( fpar[0], fpar[1], fpar[3], hChamberThickness ); 

   // Centre of chamber in z is specified in DDD
    Surface::PositionType aVec( gtran[0], gtran[1], gtran[2] ); 

    Plane::PlanePointer plane = Plane::build(aVec, aRot, bounds); 

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
    int globalZ = +1;
    if ( jend == 2 ) globalZ = -1;

    LogTrace(myName) << myName << ": layerSeparation=" << layerSeparation
                     << ", zAF-zAverageAGV="  << zAverageAGVtoAF
                     << ", localZwrtGlobalZ=" << localZwrtGlobalZ
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
        float zlayer = gtran[2] - globalZ*zAverageAGVtoAF + localZwrtGlobalZ*(3.5-j)*layerSeparation;

        Surface::RotationType chamberRotation = chamber->surface().rotation();
        Surface::PositionType layerPosition( gtran[0], gtran[1], zlayer );
	std::array<const float, 4> const & dims = geom->parameters(); // returns hb, ht, d, a
        // dims[2] = layerThickness/2.; // half-thickness required and note it is 3rd value in vector
        TrapezoidalPlaneBounds* bounds = new TrapezoidalPlaneBounds( dims[0], dims[1], dims[3], layerThickness/2. );
        Plane::PlanePointer plane = Plane::build(layerPosition, chamberRotation, bounds);

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

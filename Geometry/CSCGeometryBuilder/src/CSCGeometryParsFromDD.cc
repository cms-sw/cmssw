#include "CSCGeometryParsFromDD.h"

#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/MuonNumbering/interface/CSCNumberingScheme.h>
#include <Geometry/MuonNumbering/interface/MuonBaseNumber.h>
#include <Geometry/MuonNumbering/interface/MuonDDDNumbering.h>
#include <Geometry/MuonNumbering/interface/MuonDDDConstants.h>

#include <Geometry/CSCGeometry/src/CSCWireGroupPackage.h>
#include <CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h>
#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

CSCGeometryParsFromDD::CSCGeometryParsFromDD() : myName("CSCGeometryParsFromDD") { }

CSCGeometryParsFromDD::~CSCGeometryParsFromDD() { }

bool CSCGeometryParsFromDD::build( const DDCompactView* cview
				   , const MuonDDDConstants& muonConstants
				   , RecoIdealGeometry& rig
				   , CSCRecoDigiParameters& rdp
				   ) {
  std::string attribute = "MuStructure";      // could come from outside
  std::string value     = "MuonEndcapCSC";    // could come from outside
  DDValue muval(attribute, value, 0.0);

  // Asking for a specific section of the MuStructure

  DDSpecificsFilter filter;
  filter.setCriteria(muval, // name & value of a variable 
		     DDCompOp::equals,
		     DDLogOp::AND, 
		     true, // compare strings otherwise doubles
		     true // use merged-specifics or simple-specifics
		     );

  DDFilteredView fv( *cview );
  fv.addFilter(filter);

  bool doSubDets = fv.firstChild();

  if (!doSubDets) {
    edm::LogError("CSCGeometryParsFromDD") << "Can not proceed, no CSC parts found with the filter.  The current node is: " << fv.logicalPart().toString();
    return false;
  }
  int noOfAnonParams = 0;
  std::vector<const DDsvalues_type *> spec = fv.specifics();
  std::vector<const DDsvalues_type *>::const_iterator spit = spec.begin();
  std::vector<double> uparvals;
  std::vector<double> fpar;
  std::vector<double> dpar;
  std::vector<double> gtran( 3 );
  std::vector<double> grmat( 9 );
  std::vector<double> trm ( 9 );
  while (doSubDets) {
    spec = fv.specifics();
    spit = spec.begin();

    // get numbering information early for possible speed up of code.

    LogTrace(myName) << myName  << ": create numbering scheme...";

    MuonDDDNumbering mdn(muonConstants);
    MuonBaseNumber mbn = mdn.geoHistoryToBaseNumber(fv.geoHistory());
    CSCNumberingScheme mens(muonConstants);

    LogTrace(myName) << myName  << ": find detid...";

    int id = mens.baseNumberToUnitNumber( mbn ); //@@ FIXME perhaps should return CSCDetId itself?

    LogTrace(myName) << myName  << ": raw id for this detector is " << id << 
      ", octal " << std::oct << id << ", hex " << std::hex << id << std::dec;

    CSCDetId detid = CSCDetId( id );
    int jendcap  = detid.endcap();
    int jstation = detid.station();
    int jring    = detid.ring();
    int jchamber = detid.chamber();
    int jlayer   = detid.layer();

    // Package up the wire group info as it's decoded
    CSCWireGroupPackage wg; 
    uparvals.clear();
    LogDebug(myName) << "size of spec=" << spec.size();

    // if the specs are made no need to get all this crap!
    int chamberType = CSCChamberSpecs::whatChamberType( jstation, jring );
    LogDebug(myName) << "Chamber Type: " << chamberType;
    size_t ct = 0;
    bool chSpecsAlreadyExist = false;
    for ( ; ct < rdp.pChamberType.size() ; ++ct ) {
      if ( chamberType == rdp.pChamberType[ct] ) {
	break;
      }
    }
    if ( ct < rdp.pChamberType.size() && rdp.pChamberType[ct] == chamberType ) {
      // it was found, therefore no need to load all the intermediate crap from DD.
      LogDebug(myName) << "already found a " << chamberType << " at index " << ct;
      chSpecsAlreadyExist = true;
    } else {
      for (;  spit != spec.end(); spit++) {
	DDsvalues_type::const_iterator it = (**spit).begin();
	for (;  it != (**spit).end(); it++) {
	  LogDebug(myName) << "it->second.name()=" << it->second.name();  
	  if (it->second.name() == "upar") {
	    uparvals.push_back(it->second.doubles().size());
	    for ( size_t i = 0; i < it->second.doubles().size(); ++i) {
	      uparvals.push_back(it->second.doubles()[i]);
	    }
	    LogDebug(myName) << "found upars ";
	  } else if (it->second.name() == "NoOfAnonParams") {
	    noOfAnonParams = static_cast<int>( it->second.doubles()[0] );
	  } else if (it->second.name() == "NumWiresPerGrp") {
	    //numWiresInGroup = it->second.doubles();
	    for ( size_t i = 0 ; i < it->second.doubles().size(); i++) {
	      wg.wiresInEachGroup.push_back( int( it->second.doubles()[i] ) );
	    }
	    LogDebug(myName) << "found upars " << std::endl;
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
      
      /** crap: using a constructed wg to deconstruct it and put it in db... alternative?
	  use temporary (not wg!) storage.
	  
	  format as inserted is best documented by the actualy push_back statements below.
	  
	  fupar size now becomes origSize+6+wg.wiresInEachGroup.size()+wg.consecutiveGroups.size()
      **/
      uparvals.push_back( wg.wireSpacing );
      uparvals.push_back( wg.alignmentPinToFirstWire );
      uparvals.push_back( wg.numberOfGroups );
      uparvals.push_back( wg.narrowWidthOfWirePlane );
      uparvals.push_back( wg.wideWidthOfWirePlane );
      uparvals.push_back( wg.lengthOfWirePlane );
      uparvals.push_back( wg.wiresInEachGroup.size() ); 
      for (CSCWireGroupPackage::Container::const_iterator it = wg.wiresInEachGroup.begin();
	   it != wg.wiresInEachGroup.end(); ++it) {
	uparvals.push_back(*it);
      }
      for (CSCWireGroupPackage::Container::const_iterator it = wg.consecutiveGroups.begin();
	   it != wg.consecutiveGroups.end(); ++it) {
	uparvals.push_back(*it);
      }
      /** end crap **/
    }
    fpar.clear();
    //    dpar = fv.logicalPart().solid().parameters();
    if ( fv.logicalPart().solid().shape() == ddsubtraction ) {
      const DDSubtraction& first = fv.logicalPart().solid();
      const DDSubtraction& second = first.solidA();
      const DDSolid& third = second.solidA();
      dpar = third.parameters();
    } else {
      dpar = fv.logicalPart().solid().parameters();
    }

    LogTrace(myName) << myName  << ": noOfAnonParams=" << noOfAnonParams;
    LogTrace(myName) << myName  << ": fill fpar...";
    LogTrace(myName) << myName  << ": dpars are... " << 
      dpar[4]/cm << ", " << dpar[8]/cm << ", " << 
      dpar[3]/cm << ", " << dpar[0]/cm;

    fpar.push_back( ( dpar[4]/cm) );
    fpar.push_back( ( dpar[8]/cm ) ); 
    fpar.push_back( ( dpar[3]/cm ) ); 
    fpar.push_back( ( dpar[0]/cm ) ); 

    LogTrace(myName) << myName  << ": fill gtran...";

    gtran[0] = (float) 1.0 * (fv.translation().X() / cm);
    gtran[1] = (float) 1.0 * (fv.translation().Y() / cm);
    gtran[2] = (float) 1.0 * (fv.translation().Z() / cm);

    LogTrace(myName) << myName << ": gtran[0]=" << gtran[0] << ", gtran[1]=" <<
      gtran[1] << ", gtran[2]=" << gtran[2];

    LogTrace(myName) << myName  << ": fill grmat...";

    fv.rotation().GetComponents(trm.begin(), trm.end());
    size_t rotindex = 0;
    for (size_t i = 0; i < 9; ++i) {
      grmat[i] = (float) 1.0 * trm[rotindex];
      rotindex = rotindex + 3;
      if ( (i+1) % 3 == 0 ) {
	rotindex = (i+1) / 3;
      }
    }
    LogTrace(myName) << myName  << ": looking for wire group info for layer " <<
      "E" << CSCDetId::endcap(id) << 
      " S" << CSCDetId::station(id) << 
      " R" << CSCDetId::ring(id) <<
      " C" << CSCDetId::chamber(id) <<
      " L" << CSCDetId::layer(id);
          
    if ( wg.numberOfGroups != 0 ) {
      LogTrace(myName) << myName  << ": fv.geoHistory:      = " << fv.geoHistory() ;
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

    LogTrace(myName) << myName << ":_z_ E" << jendcap << " S" << jstation << " R" << jring <<
      " C" << jchamber << " L" << jlayer <<
      " gx=" << gtran[0] << ", gy=" << gtran[1] << ", gz=" << gtran[2] <<
      " thickness=" << fpar[2]*2.;

    if ( jlayer == 0 ) { // Can only build chambers if we're filtering them

      LogTrace(myName) << myName  << ":_z_ frame=" << uparvals[31]/10. <<
        " gap=" << uparvals[32]/10. << " panel=" << uparvals[33]/10. << " offset=" << uparvals[34]/10.;

    if ( jstation==1 && jring==1 ) {
      // set up params for ME1a and ME1b and call buildChamber *for each*
      // Both get the full ME11 dimensions

      // detid is for ME11 and that's what we're using for ME1b in the software

      rig.insert( id, gtran, grmat, fpar );
      if ( !chSpecsAlreadyExist ) {
	//	rdp.pCSCDetIds.push_back(CSCDetId(id));
	LogDebug(myName) << " inserting chamber type " << chamberType << std::endl;
	rdp.pChamberType.push_back(chamberType);
	rdp.pUserParOffset.push_back(rdp.pfupars.size());
	rdp.pUserParSize.push_back(uparvals.size());
	std::copy ( uparvals.begin(), uparvals.end(), std::back_inserter(rdp.pfupars));
      }

      // No. of anonymous parameters per chamber type should be read from cscSpecs file...
      // Only required for ME11 splitting into ME1a and ME1b values,
      // If it isn't seen may as well try to get further but this value will depend
      // on structure of the file so may not even match! 
      const int kNoOfAnonParams = 35;
      if ( noOfAnonParams == 0 ) { noOfAnonParams = kNoOfAnonParams; } // in case it wasn't seen
      
      std::copy( uparvals.begin()+noOfAnonParams+1, uparvals.begin()+(2*noOfAnonParams)+2, uparvals.begin()+1 ); // copy ME1a params from back to the front
      
      CSCDetId detid1a = CSCDetId( jendcap, 1, 4, jchamber, 0 ); // reset to ME1A
      rig.insert( detid1a.rawId(), gtran, grmat, fpar );
      int chtypeA = CSCChamberSpecs::whatChamberType( 1, 4 );
      ct = 0;
      for ( ; ct < rdp.pChamberType.size() ; ++ct ) {
	if ( chtypeA == rdp.pChamberType[ct] ) {
	  break;
	}
      }
      if ( ct < rdp.pChamberType.size() && rdp.pChamberType[ct] == chtypeA ) {
	// then its in already, don't put it
	LogDebug(myName) << "found chamber type " << chtypeA << " so don't put it in! ";
      } else {
	//	rdp.pCSCDetIds.push_back(detid1a);
	LogDebug(myName) << " inserting chamber type " << chtypeA;
	rdp.pChamberType.push_back(chtypeA);
	rdp.pUserParOffset.push_back(rdp.pfupars.size());
	rdp.pUserParSize.push_back(uparvals.size());
	std::copy ( uparvals.begin(), uparvals.end(), std::back_inserter(rdp.pfupars));
      }
      
    }
    else {
      rig.insert( id, gtran, grmat, fpar );
      if ( !chSpecsAlreadyExist ) {
	//   rdp.pCSCDetIds.push_back(CSCDetId(id));
	LogDebug(myName) << " inserting chamber type " << chamberType;
	rdp.pChamberType.push_back(chamberType);
	rdp.pUserParOffset.push_back(rdp.pfupars.size());
	rdp.pUserParSize.push_back(uparvals.size());
	std::copy ( uparvals.begin(), uparvals.end(), std::back_inserter(rdp.pfupars));
      }
    }

    } // filtering chambers.

    doSubDets = fv.next();
  }
  return true;
}

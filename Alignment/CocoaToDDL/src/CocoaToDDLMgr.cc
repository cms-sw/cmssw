//   COCOA class implementation file
//Id:  CocoaToDDLMgr.cc
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaToDDL/interface/CocoaToDDLMgr.h"
#include "Alignment/CocoaToDDL/interface/UnitConverter.h"
#define UC(val,category) UnitConverter(val,category).ucstring()

#include "Alignment/CocoaDDLObjects/interface/CocoaMaterialElementary.h"
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeTubs.h"

#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaModel/interface/Measurement.h"



CocoaToDDLMgr* CocoaToDDLMgr::instance = 0;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CocoaToDDLMgr* CocoaToDDLMgr::getInstance()
{
  if(!instance) {
    instance = new CocoaToDDLMgr;
  }
  return instance;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeDDDFile( ALIstring filename) 
{
  //---- Write header
  writeHeader( filename );

  //---- Write materials
  writeMaterials();

  //---- Write solids
  writeSolids();

  //---- Write logical volumes
  writeLogicalVolumes();

  //---- Write physical volumes
  writePhysicalVolumes();

  //---- Write rotations
  writeRotations();

  //---- Write SpecPar's
  writeSpecPars();

  newPartPost( filename, "" ); 


}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeHeader( ALIstring filename)
{
  newPartPre( filename ); 
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeMaterials()
{
  newSectPre_ma("");
  auto &optolist = Model::OptOList();
  for(auto ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue;
    CocoaMaterialElementary* mat = (*ite)->getMaterial();
    //-    std::cout << " mat of opto " << (*ite)->name() << " = " << mat->getName() << std::endl;
    if( mat ) {
      if( !materialIsRepeated( mat ) ) ma( mat );
    }
  }

  newSectPost_ma("");
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeSolids()
{
  newSectPre_so("");
  
  auto &optolist = Model::OptOList();
  for(auto ite = optolist.begin(); ite != optolist.end(); ite++ ){
    bool alreadyWritten = false;
    for(auto ite2 = optolist.begin(); ite2 != ite; ite2++ ){
      if( (*ite)->shortName() == (*ite2)->shortName() ) {
	alreadyWritten = true;
      }
    }
std::cout << " CocoaToDDLMgr::writeSolids() " << alreadyWritten << *ite;
std::cout << (*ite)->name() << std::endl;
    if( !alreadyWritten ) so( *ite );
  }
  
  newSectPost_so("");

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeLogicalVolumes()
{
  newSectPre_lv("");
  
  auto &optolist = Model::OptOList();
  for(auto ite = optolist.begin(); ite != optolist.end(); ite++ ){
    bool alreadyWritten = false;
    for(auto ite2 = optolist.begin(); ite2 != ite; ite2++ ){
      if( (*ite)->shortName() == (*ite2)->shortName() ) {
	alreadyWritten = true;
      }
    }
    if( !alreadyWritten ) lv( *ite );
  }
  
  newSectPost_lv("");

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writePhysicalVolumes()
{
  newSectPre_pv("");
  
  auto &optolist = Model::OptOList();
  for(auto ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue;
    pv( *ite );
  }
  
  newSectPost_pv("");

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeRotations()
{
  newSectPre_ro("");
  std::vector<CLHEP::HepRotation>::const_iterator ite;
  int nc = 0;
  for( ite = theRotationList.begin(); ite != theRotationList.end(); ite++) {
    //-  std::cout << nc << " rot size " <<  theRotationList.size() << std::endl;
    ro( *ite, nc );
    nc++;
  }    
  newSectPost_ro("");

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeSpecPars()
{
  newSectPre_specPar("");
  
  auto &optolist = Model::OptOList();
  for(auto ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue;
    specPar( *ite );
  }
  
  writeSpecParsCocoa();

  //---- Write Measurements's
  measurementsAsSpecPars();


  newSectPost_specPar("");


}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newPartPre(std::string name)
{
   filename_=name;
   file_.open(filename_.c_str()); 
   file_.precision(8);
   file_ << "<?xml version=\"1.0\"?>" << std::endl;

   // all files get schema references and namespaces.
   file_ << "<DDDefinition xmlns=\"http://www.cern.ch/cms/DDL\""
         << " xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\""
	 << " xsi:schemaLocation=\"http://www.cern.ch/cms/DDL ../../DDLSchema/DDLSchema.xsd\">"
	 << std::endl << std::endl;

   #ifdef gdebug
     cout << "part-pre:" << name << std::endl;
   #endif  
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newPartPost(std::string name, std::string extension)
{
   file_ << std::endl << "</DDDefinition>" << std::endl;
   file_.close();
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPre_ma(std::string name)
{
#ifdef gdebug
  std::cout << " sect-mat-pre:" << name << '-' << std::endl;
#endif
   newSectPre(filename_,std::string("MaterialSection"));
}    

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::ma(CocoaMaterialElementary* ma)
{
  theMaterialList.push_back( ma );

#ifdef gdebug
  cout << "  ma:" << ma->getName() << std::endl;
#endif
  
  ALIfloat density       = ma->getDensity();///g*cm3;
  
  // start tag
  file_ << "  <ElementaryMaterial";
  ALIstring cSymbol = ma->getSymbol();
  
  // name attribute
  file_ << " name=\"" << ma->getName() << "\"";
  
  // put out common attributes
  //  file_ << " density=\"" << UnitConverter(density,"Volumic Mass") << "\"";
  file_ << " density=\"" << UC(density,"Volumic Mass") << "\"";
  file_ << " symbol=\"" << ma->getSymbol() << "\"";
  
  
  // finish last two attributes and end material element
  file_ << " atomicWeight=\"" << (ma->getA()) << "*g/mole\""
	<< " atomicNumber=\"" << ma->getZ() << "\""
	<< "/>" << std::endl;
  
} 

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPost_ma(std::string name)
{
   #ifdef gdebug
    cout << " sect-mat-post:" << name << '-' << std::endl;
   #endif
   newSectPost("MaterialSection");
}    


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPre_so(std::string name)
{
   #ifdef gdebug
    cout << " sect-so-pre:" << name << '-' << std::endl;
   #endif
   newSectPre(filename_,std::string("SolidSection"));
}    

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::so(OpticalObject * opto) 
{
 std::cout << " CocoaToDDLMgr::so( " << opto;
std::cout << " " << opto->shortName() << std::endl;

  std::string name = opto->shortName();

  if( opto->type() == "system" ){
    //    file_ << " <Box name=\"" << name << "\"";
    file_ << " <Box name=\"" << opto->name() << "\"";
    file_ << " dx=\"10.*m" 
	  << "\" dy=\"10.*m" 
	  << "\" dz=\"10.*m" 
	  << "\"/>" << std::endl;
    return;
  }

  CocoaSolidShape* so = opto->getSolidShape();

 std::cout << " CocoaToDDLMgr::so( so " << so << std::endl;  
std::string solidType = so->getType();

  if (solidType == "Box")
    {
      file_ << " <" << solidType << " name=\"" << name << "\"";
      CocoaSolidShapeBox * sb = dynamic_cast<CocoaSolidShapeBox*>(so);
      file_ << " dx=\"" << UC(sb->getXHalfLength(),"Length")
            << "\" dy=\"" << UC(sb->getYHalfLength(),"Length") 
	    << "\" dz=\"" << UC(sb->getZHalfLength(),"Length")
	    << "\"/>" << std::endl;
    }
  else if (solidType == "Tubs")
    {
      CocoaSolidShapeTubs * tu = dynamic_cast < CocoaSolidShapeTubs * > (so);
      file_ << " <" << solidType 
            << " name=\""     << name                                                    << "\""
	    << " rMin=\""     << UC(tu->getInnerRadius(),"Length")   << "\""
	    << " rMax=\""     << UC(tu->getOuterRadius(),"Length")   << "\""
	    << " dz=\""       << UC(tu->getZHalfLength(),"Length")   << "\""
	    << " startPhi=\"" << UC(tu->getStartPhiAngle(),"Angle") << "\""
	    << " deltaPhi=\"" << UC(tu->getDeltaPhiAngle(),"Angle") << "\""
	    << "/>" << std::endl;
    }
  /*  else if (solidType == "Cons")
    {
      G4Cons * cn = dynamic_cast < G4Cons * > (so);
      file_ << " <" << solidType 
            << " name=\""     << name                                              << "\""
	    << " dz=\""       << UC(cn->getZHalfLength(),"Length")       << "\""
	    << " rMin1=\""    << UC(cn->getInnerRadiusMinusZ(),"Length") << "\"" 
	    << " rMax1=\""    << UC(cn->getOuterRadiusMinusZ(),"Length") << "\""
	    << " rMin2=\""    << UC(cn->getInnerRadiusPlusZ(),"Length")  << "\""	    
	    << " rMax2=\""    << UC(cn->getOuterRadiusPlusZ(),"Length")  << "\""
            << " startPhi=\"" << UC(cn->getStartPhiAngle(),"Angle")     << "\""
	    << " deltaPhi=\"" << UC(cn->getDeltaPhiAngle(),"Angle")    << "\""
	  //<< " lengthUnit=\"mm\" angleUnit=\"degree\"/>" 
	    << " />" << std::endl;
    }
  else if (solidType == "Polycone")
    {
      G4Polycone * pc = dynamic_cast < G4Polycone * > (so);
      file_ << " <Polycone name=\"" << name<< "\"";
      bool isOpen = pc->IsOpen();
      G4int numRZCorner = (dynamic_cast < G4Polycone * > (so))->getNumRZCorner();
      
      file_ << " startPhi=\"" << UC(pc->getStartPhi(),"Angle") << "\""
	    //<< " deltaPhi=\"" << UC(fabs((pc->getEndPhi()/deg - pc->getStartPhi()/deg))*deg,"Angle")   << "\"" 
	    //<< " deltaPhi=\"" << UC(pc->getEndPhi(),"Angle")   << "\"" 
	    << " deltaPhi=\"" << UC(pc->original_parameters->Opening_angle,"Angle")   << "\""
	    //<< " angleUnit=\"degree\">" 
	    << " >" << std::endl;

      G4PolyconeSideRZ rz;
      
      //liendl: FIXME put a switch which decides whether RZ or Rmin,Rmax,Z types should
      //              by generated ....
      //outPolyZSections(rz, (dynamic_cast < G4Polycone * > (so)), numRZCorner);
      G4double * zVal;
      G4double * rmin;
      G4double * rmax;
      G4int zPlanes;
      zPlanes = pc->original_parameters->Num_z_planes;
      zVal = pc->original_parameters->Z_values;
      rmin = pc->original_parameters->Rmin;
      rmax = pc->original_parameters->Rmax;
      outPolySections(zPlanes, zVal, rmin, rmax);
      file_ << " </Polycone> " << std::endl;
    }
  else if (solidType == "Polyhedra")
    {
      //      bool isOpen = (dynamic_cast < G4Polyhedra * > (so))->IsOpen();
      G4Polyhedra * ph = (dynamic_cast < G4Polyhedra * > (so));
      G4int numRZCorner = ph->getNumRZCorner();
 
      file_ << " <Polyhedra name=\"" << name<< "\""
            << " numSide=\"" << ph->getNumSide() << "\""
            << " startPhi=\"" << UC(ph->getStartPhi(),"Angle") << "\""
	    //<< " deltaPhi=\""   <<  UC(fabs((ph->getEndPhi()/deg - ph->getStartPhi()/deg))*deg,"Angle")   << "\""
	    << " deltaPhi=\"" << UC(ph->original_parameters->Opening_angle,"Angle")   << "\"" 
	    << " >" << std::endl;

      G4PolyhedraSideRZ rz;
      //liendl: FIXME put a switch which decides whether RZ or Rmin,Rmax,Z types should
      //              by generated ....
      // outPolyZSections(rz, (dynamic_cast < G4Polyhedra * > (so)), numRZCorner);
      G4double * zVal;
      G4double * rmin;
      G4double * rmax;
      // convertRad of ctor of G4Polyhedra(..) ....
      G4double strangeG3G4Factor = cos(0.5*ph->original_parameters->Opening_angle/G4double(ph->getNumSide()));
      G4int zPlanes;
      zPlanes = ph->original_parameters->Num_z_planes;
      zVal = ph->original_parameters->Z_values;
      rmin = ph->original_parameters->Rmin;
      rmax = ph->original_parameters->Rmax;
      for (int i=0; i<zPlanes;++i) {
       *(rmin+i) = *(rmin+i) * strangeG3G4Factor;
       *(rmax+i) = *(rmax+i) * strangeG3G4Factor;
      }
      outPolySections(zPlanes, zVal, rmin, rmax);

      file_ << " </Polyhedra>" << std::endl;
    }
  else if (solidType == "Trapezoid")
    {
      // DDL fields
      // ALP1, ALP2, Bl1, Bl2, Dz, H1, H2, Phi, Thet, TL1, TL2, lengthUnit, angleUnit	 
      // Phi and Theta are !NOT!optional.
       G4Trap * trp = dynamic_cast < G4Trap * > (so);
       G4ThreeVector symAxis(trp->getSymAxis());
       double theta, phi;
       theta = symAxis.theta();
       phi = symAxis.phi();

      file_ << " <" << solidType 
	    << " name=\"" << name                                                       << "\""
	    << " dz=\""   << UC(trp->getZHalfLength(),"Length")      << "\""
	    << " alp1=\"" << UC(atan(trp->getTanAlpha1()/rad),"Angle") << "\"" 
	    << " bl1=\""  << UC(trp->getXHalfLength1(),"Length")     << "\""
	    << " tl1=\""  << UC(trp->getXHalfLength2(),"Length")     << "\""	    
	    << " h1=\""   << UC(trp->getYHalfLength1(),"Length")    << "\""
	    << " alp2=\"" << UC(atan(trp->getTanAlpha2()/rad),"Angle") << "\""
	    << " bl2=\""  << UC(trp->getXHalfLength3(),"Length")     << "\""
	    << " tl2=\""  << UC(trp->getXHalfLength4(),"Length")    << "\""	    
	    << " h2=\""   << UC(trp->getYHalfLength2(),"Length")     << "\""
	    << " phi=\""  << UC(phi,"Angle") << "\""
	    << " theta=\"" << UC(theta,"Angle") << "\"" 
	    << " />" << std::endl ;
    }
  else if (solidType == "Trd1")
    {
      G4Trd * tr = dynamic_cast < G4Trd * > (so);
      file_ << " <" << solidType  
	    << " name=\"" << name                                                  << "\""
            << " dz=\""   << UC(tr->getZHalfLength(),"Length")  << "\""
	    << " dy1=\""  << UC(tr->getYHalfLength1(),"Length") << "\""
            << " dy2=\""  << UC(tr->getYHalfLength2(),"Length")<< "\"" 
	    << " dx1=\""  << UC(tr->getXHalfLength1(),"Length") << "\""
	    << " dx2=\""  << UC(tr->getXHalfLength2(),"Length") << "\""
	    //<< " lengthUnit=\"mm\"/>" 
	    << " />" << std::endl;
    }
  */  
  else
    {
     std::cerr << " <!-- NOT HANDLED: " << solidType << " name=\"" << name<< "\""
	    << ">" << std::endl
            << " </" << solidType << "> -->" << std::endl;
    std::exception();
    }
    
}      

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPost_so(std::string name)
{
   #ifdef gdebug
    cout << " sect-so-post:" << name << '-' << std::endl;
   #endif
   newSectPost("SolidSection");
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPre_lv(std::string name)
{
   #ifdef gdebug
    cout << " sect-lv-pre:" << name << '-'  << std::endl;
   #endif
   newSectPre(filename_,std::string("LogicalPartSection"));
}    

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::lv(OpticalObject * opto)
{ 
  std::string name = opto->shortName();
  std::string rSolid = opto->shortName();
  std::string sensitive = "unspecified";

  if( opto->type() == "system" ){
    file_ << " <LogicalPart name=\"" 
	  <<  name << "\" category=\"" << sensitive << "\">" << std::endl
	  << "  <rSolid name=\"" << rSolid << "\"/>" << std::endl
	  << "  <rMaterial name=\"Hydrogen\"" 
	  << "/>" << std::endl
	  << " </LogicalPart>" << std::endl;			
    return;
  }

#ifdef gdebug_v
  cout << "xml:lv " << opto->name() << std::endl;
#endif 
  file_ << " <LogicalPart name=\"" 
	<<  name << "\" category=\"" << sensitive << "\">" << std::endl
	<< "  <rSolid name=\"" << rSolid << "\"/>" << std::endl
	<< "  <rMaterial name=\"" << opto->getMaterial()->getName() << "\"" 
	<< "/>" << std::endl
	<< " </LogicalPart>" << std::endl;			
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPost_lv(std::string name)
{
   #ifdef gdebug
    cout << " sect-lv-post:" << name << '-'<< std::endl;
   #endif 
   newSectPost("LogicalPartSection");
}    


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPre_pv(std::string name)
{
   #ifdef gdebug
     cout << " sect-pv-pre:" << name << '-' << std::endl;
   #endif
   newSectPre(filename_,std::string("PosPartSection"));
}    

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::pv(OpticalObject * opto)
{
   #ifdef gdebug_v
    cout << "  pv:" << opto->name() 
	 << ':' << opto->parent()->name() << std::endl;
   #endif

    //   file_ << " <PosPart copyNumber=\"" << pv->GetCopyNo() << "\">" << std::endl; 
   file_ << " <PosPart copyNumber=\"" << "1" << "\">" << std::endl; 
   file_ << "   <rParent name=\"";

   //t   if (file!=filename_) file_ << file << ":";
   
   file_ << opto->parent()->shortName() << "\"/>" << std::endl;

   file_ << "   <rChild name=\""; 
   //t  if (file_d != filename_)  file_<<  file_d << ":"; 	
   file_ << opto->shortName();	
   file_ << "\"/>" << std::endl;
   
   int rotNumber = buildRotationNumber( opto );
   //CocoaDDLRotation* rot = buildRotationNotRepeated( opto );
   
   if( rotNumber != -1 ) file_ << "  <rRotation name=\"R" << rotNumber << "\"/>" << std::endl;

   CLHEP::Hep3Vector t =  opto->centreLocal();
   if(t != CLHEP::Hep3Vector()) { //if (0,0,0) write nothing
     const CLHEP::Hep3Vector t = opto->centreLocal();

     file_ << "  <Translation x=\"" <<  UC(t[0],"Length") << "\""
           <<               " y=\"" << UC(t[1],"Length") << "\""
	   <<               " z=\"" << UC(t[2],"Length")<< "\" />"
           << std::endl;
   }	      	 	 
    
   file_ << " </PosPart>" << std::endl;	 
		  	 
}   

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPost_pv(std::string name)
{
   #ifdef gdebug
    cout << " sect-pv-post:" << name << '-' << std::endl;
   #endif
   newSectPost("PosPartSection");
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPre_ro(std::string name)
{
   newSectPre(filename_,std::string("RotationSection"));
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// handlers reflections and rotations ...
void CocoaToDDLMgr::ro(const CLHEP::HepRotation& ro, int n)
{
   
   CLHEP::HepRotation roinv = inverseOf(ro);
  //-    G4ThreeVector v(1.,1.,1.);
  //-     G4ThreeVector a;
  //-   a = (*ro)*v;
  bool identity = false;
  ALIstring tag = " <Rotation name=\"R";
  identity=roinv.isIdentity();  
  
  //---- DDD convention is to use the inverse matrix, COCOA is the direct one!!!
  if (! identity) {     
    file_ << tag << n << "\"";
    file_ << " phiX=\""   << UC(roinv.phiX(),"Angle")   << "\""
	  << " thetaX=\"" << UC(roinv.thetaX(),"Angle") << "\""
	  << " phiY=\""   << UC(roinv.phiY(),"Angle")   << "\""
	  << " thetaY=\"" << UC(roinv.thetaY(),"Angle") << "\""
	  << " phiZ=\""   << UC(roinv.phiZ(),"Angle")   << "\""
	  << " thetaZ=\"" << UC(roinv.thetaZ(),"Angle") << "\""
      //<< " angleUnit=\"degree\"/>" 
	  << " />" << std::endl;
  }	     
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPost_ro(std::string name)
{
   newSectPost("RotationSection");
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPre_specPar(std::string name)
{
   #ifdef gdebug
    cout << " sect-lv-pre:" << name << '-'  << std::endl;
   #endif
    //-   newSectPre(filename_,std::string("SpecParSection"));
   file_ << "<SpecParSection label=\"" << filename_ << "\" eval=\"true\">" << std::endl;

}    


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::specPar(OpticalObject * opto)
{
  file_ << " <SpecPar name=\"" << opto->name() << "_PARAMS\">" << std::endl;
  file_ << "   <PartSelector path=\"/" << opto->name() << "\"/> " << std::endl;
  file_ << "   <Parameter name=\"cocoa_type\""  << " value=\"" << opto->type() << "\"   eval=\"false\" /> " << std::endl;
  file_ << "   <Parameter name=\"cmssw_ID\""  << " value=\"" << opto->getCmsswID() << "\" /> " << std::endl;

  const std::vector< Entry* > coord = opto->CoordinateEntryList();
  for( int ii=3; ii<6; ii++ ){
    Entry* ent = coord[ii];
    file_ << "   <Parameter name=\"" << ent->name()+std::string("_value") << "\" value=\"";
    file_ << UC(ent->value(),"Angle");
    file_ << "\" /> " << std::endl;
  }
  for( int ii=0; ii<6; ii++ ){
    Entry* ent = coord[ii];
    file_ << "   <Parameter name=\"" << ent->name()+std::string("_sigma") << "\" value=\"";
    if( ii < 3 ){
      file_ << UC(ent->sigma(),"Length");
    }else {
      file_ << UC(ent->sigma(),"Angle");
    }
    file_ << "\" /> " << std::endl;
    file_ << "   <Parameter name=\"" << ent->name()+std::string("_quality") << "\" value=\"" << ent->quality() << "\" /> " << std::endl;
  }
  
  const std::vector< Entry* > extraEnt = opto->ExtraEntryList();
  for( ALIuint ii=0; ii<extraEnt.size(); ii++ ){
    Entry* ent = extraEnt[ii]; 
    file_ << "   <Parameter name=\"extra_entry\" value=\"" << ent->name() << "\"  eval=\"false\" /> " << std::endl;
    file_ << "   <Parameter name=\"dimType\" value=\"" << ent->type() << "\"  eval=\"false\" /> " << std::endl;
    file_ << "   <Parameter name=\"value\" value=\"";
    if( ent->type() == "nodim" ) {
      file_ << ent->value();
    }else if( ent->type() == "length" ) {
      file_ << UC(ent->value(),"Length");
    }else if( ent->type() == "angle" ) {
      file_ << UC(ent->value(),"Angle");
    }
    file_ << "\"  eval=\"true\" /> " << std::endl;

    file_ << "   <Parameter name=\"sigma\" value=\"";
    if( ent->type() == "nodim" ) {
      file_ << ent->sigma();
    }else if( ent->type() == "length" ) {
      file_ << UC(ent->sigma(),"Length");
    }else if( ent->type() == "angle" ) {
      file_ << UC(ent->sigma(),"Angle");
    }
    file_ << "\"  eval=\"true\" /> " << std::endl;

    file_ << "   <Parameter name=\"quality\" value=\"" << ent->quality() << "\"  eval=\"true\" /> " << std::endl;
  }

  file_ << " </SpecPar>" << std::endl;
  
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::measurementsAsSpecPars()
{
  
  std::vector< Measurement* > measlist = Model::MeasurementList();
  std::vector< Measurement* >::iterator mite;
  std::vector<ALIstring>::iterator site;
  std::multimap<OpticalObject*,Measurement*> optoMeasMap;
  for( mite = measlist.begin(); mite != measlist.end(); mite++ ) {
    auto &optolist = (*mite)->OptOList();
    OpticalObject* opto = optolist[optolist.size()-1];
    optoMeasMap.insert( std::multimap<OpticalObject*,Measurement*>::value_type(opto, *mite) );
  }

  typedef std::multimap<OpticalObject*,Measurement*>::const_iterator itemom;
  itemom omite;
  std::pair<itemom, itemom > omitep;
  itemom omite2, omite3;

  for( omite = optoMeasMap.begin(); omite != optoMeasMap.end(); omite++ ){
    omitep = optoMeasMap.equal_range( (*omite).first );
    if( omite != optoMeasMap.begin() && (*omite).first == (*omite3).first ) continue; // check that it is not the same OptO than previous one
    omite3 = omite;
    for( omite2 = omitep.first; omite2 != omitep.second; omite2++ ){
      OpticalObject* opto = (*(omite2)).first;
      Measurement* meas = (*(omite2)).second;
      std::vector<ALIstring> namelist = meas->OptONameList();
      if( omite2 == omitep.first ){
	file_ << " <SpecPar name=\"" << meas->name() << "_MEASUREMENT\">" << std::endl;
	file_ << "   <PartSelector path=\"/" << opto->name() << "\"/> " << std::endl;
      }

      file_ << "   <Parameter name=\"" << std::string("meas_name") << "\" value=\"" << meas->name() << "\"  eval=\"false\" /> " << std::endl;
      file_ << "   <Parameter name=\"" << std::string("meas_type") << "\" value=\"" << meas->type() << "\"  eval=\"false\" /> " << std::endl;
      for( site = namelist.begin(); site != namelist.end(); site++ ){     
	file_ << "   <Parameter name=\"" << std::string("meas_object_name_")+meas->name() << "\" value=\"" << (*site) << "\"  eval=\"false\" /> " << std::endl;
      }
      for( ALIuint ii = 0; ii < meas->dim(); ii++ ){
	file_ << "   <Parameter name=\"" << std::string("meas_value_name_")+meas->name() << "\" value=\"" << meas->valueType(ii) << "\"  eval=\"false\" /> " << std::endl;
	file_ << "   <Parameter name=\"" << std::string("meas_value_")+meas->name() << "\" value=\"" << meas->value(ii) << "\"  eval=\"true\" /> " << std::endl;
	file_ << "   <Parameter name=\"" << std::string("meas_sigma_")+meas->name() << "\" value=\"" << meas->sigma(ii) << "\"  eval=\"true\" /> " << std::endl;
	file_ << "   <Parameter name=\"" << std::string("meas_is_simulated_value_")+meas->name() << "\" value=\"" << meas->valueIsSimulated(ii) << "\"  eval=\"true\" /> " << std::endl;
      }
      
    }
    file_ << " </SpecPar>" << std::endl;
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeSpecParsCocoa()
{
  file_ << "<!--    Define volumes as COCOA objects --> " << std::endl
	<< "  <SpecPar name=\"COCOA\"> " << std::endl;

  auto &optolist = Model::OptOList();
  for(auto ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue;
    file_ << "    <PartSelector path=\"/" << (*ite)->name() << "\"/> " << std::endl;
  }
   
  file_ << "   <String name=\"COCOA\" value=\"COCOA\"/> " << std::endl
	<< "  </SpecPar> " << std::endl;

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPost_specPar(std::string name)
{
   newSectPost("SpecParSection");
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPre(std::string name, std::string type)
{
   file_ << "<" << type << " label=\"" << name << "\">" << std::endl;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::newSectPost(std::string name)
{
   file_ << "</" << name << ">" << std::endl << std::endl;
}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool CocoaToDDLMgr::materialIsRepeated( CocoaMaterialElementary* ma )
{
  ALIbool isRepeated = false;
  std::vector<CocoaMaterialElementary*>::const_iterator ite;

  for(ite = theMaterialList.begin(); ite != theMaterialList.end(); ite++ ){
    if( *(*ite) == *ma ){
      isRepeated = true;
      break;
    }
  }

  return isRepeated;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
std::string CocoaToDDLMgr::scrubString(const std::string& s)
{
  std::string::const_iterator ampat;
  static const std::string amp = "_"; //"&amp;";
  std::string ret = "";
  for (ampat = s.begin(); ampat !=  s.end(); ampat++)
    {
     if (*ampat == '&')
       ret = ret + amp;
     else if (*ampat == '/')
       ret = ret + ";";
     else if (*ampat == ':')
       ret = ret + '_';
     else
       ret = ret + *ampat;
    }
  // this works when used alone.  when in this file it fails.  i don't know why.
  //for (ampat = s.begin(); ampat != s.end(); ampat++)
  //  {
  //    if (*ampat == '&'){
  //	s.insert(ampat+1, amp.begin(), amp.end());
  //    }
  //  }
  //  replace(s.begin(), s.end(), '/', ';');
  //return s;
  //cout << "AMP: " << ret << endl;
  return ret;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIint CocoaToDDLMgr::buildRotationNumber( OpticalObject* opto )
{
  ALIint rotnum = -1;

  if(opto->rmLocal().isIdentity() ) return rotnum;

  std::vector<CLHEP::HepRotation>::const_iterator ite;

  int nc = 0;
  for( ite = theRotationList.begin(); ite != theRotationList.end(); ite++) {
    if( (*ite) == opto->rmLocal() ) {
      rotnum = nc;
      break;
    }
    nc++;
  }

  if( rotnum == -1 ) {
    theRotationList.push_back( opto->rmLocal() );
    rotnum = theRotationList.size()-1;
  }

  return rotnum;

}


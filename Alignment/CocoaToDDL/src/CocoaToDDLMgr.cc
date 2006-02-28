//   COCOA class implementation file
//Id:  CocoaToDDLMgr.cc
//
//   History: v1.0 
//   Pedro Arce


#include "OpticalAlignment/CocoaToDDL/interface/CocoaToDDLMgr.h"
#include "OpticalAlignment/CocoaToDDL/interface/UnitConverter.h"
#define UC(val,category) UnitConverter(val,category).ucstring()

#include "OpticalAlignment/CocoaDDLObjects/interface/CocoaMaterialElementary.h"
#include "OpticalAlignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"
#include "OpticalAlignment/CocoaDDLObjects/interface/CocoaSolidShapeTubs.h"

#include "OpticalAlignment/CocoaModel/interface/Model.h"
#include "OpticalAlignment/CocoaModel/interface/OpticalObject.h"
#include "OpticalAlignment/CocoaModel/interface/Entry.h"

#include "CLHEP/Units/SystemOfUnits.h"


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
  static std::vector< OpticalObject* > optolist = Model::OptOList();
  static std::vector< OpticalObject* >::const_iterator ite;
  for(ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue;
    CocoaMaterialElementary* mat = (*ite)->getMaterial();
    //-    std::cout << " mat of opto " << (*ite)->name() << " = " << mat->getName() << std::endl;
    if( !materialIsRepeated( mat ) ) ma( mat );
  }

  newSectPost_ma("");
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeSolids()
{
  newSectPre_so("");
  
  static std::vector< OpticalObject* > optolist = Model::OptOList();
  static std::vector< OpticalObject* >::const_iterator ite;
  for(ite = optolist.begin(); ite != optolist.end(); ite++ ){
    so( *ite );
  }
  
  newSectPost_so("");

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeLogicalVolumes()
{
  newSectPre_lv("");
  
  static std::vector< OpticalObject* > optolist = Model::OptOList();
  static std::vector< OpticalObject* >::const_iterator ite;
  for(ite = optolist.begin(); ite != optolist.end(); ite++ ){
    //each OpticalObject is a distinct logical volume. This is so because two OptO's of the same type may have different optical properties and this optical properties are SpecPart's. And to have different values of an SpecPart they have to be different logical volumes
    lv( *ite );
  }
  
  newSectPost_lv("");

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writePhysicalVolumes()
{
  newSectPre_pv("");
  
  static std::vector< OpticalObject* > optolist = Model::OptOList();
  static std::vector< OpticalObject* >::const_iterator ite;
  for(ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue;
    pv( *ite );
  }
  
  newSectPost_pv("");

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeRotations()
{
  newSectPre_ro("");
  std::vector<HepRotation>::const_iterator ite;
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
  
  static std::vector< OpticalObject* > optolist = Model::OptOList();
  static std::vector< OpticalObject* >::const_iterator ite;
  for(ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue;
    specPar( *ite );
  }
  
  writeSpecParsCocoa();

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
  
  ALIfloat density             = ma->getDensity();///g*cm3;
  
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
  file_ << " atomicWeight=\"" << (ma->getA())/(g/mole) << "*g/mole\""
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
  std::string name = scrubString(opto->name());

  if( opto->type() == "system" ){
    //    file_ << " <Box name=\"" << name << "\"";
    file_ << " <Box name=\"OCMS\"";
    file_ << " dx=\"0.*m" 
	  << "\" dy=\"0.*m" 
	  << "\" dz=\"0.*m" 
	  << "\"/>" << std::endl;
    return;
  }

  CocoaSolidShape* so = opto->getSolidShape();

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
  std::string name = opto->name();
  name = scrubString(name);
  std::string rSolid = opto->name();
  rSolid = scrubString(rSolid);
  std::string sensitive = "unspecified";

  if( opto->type() == "system" ){
    file_ << " <LogicalPart name=\"" 
	  <<  name << "\" category=\"" << sensitive << "\">" << std::endl
	  << "  <rSolid name=\"" << rSolid << "\"/>" << std::endl
	  << "  <rMaterial name=\"materials: NONE\"" 
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
	<< "  <rMaterial name=\"materials:" << opto->getMaterial()->getName() << "\"" 
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
   newSectPre(filename_,std::string("PostsPartSection"));
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
   
   file_ << scrubString( opto->parent()->name()) << "\"/>" << std::endl;

   file_ << "   <rChild name=\""; 
   //t  if (file_d != filename_)  file_<<  file_d << ":"; 	
   file_ << scrubString(opto->name()) ;	
   file_ << "\"/>" << std::endl;
   
   int rotNumber = buildRotationNumber( opto );
   //CocoaDDLRotation* rot = buildRotationNotRepeated( opto );
   
   file_ << "  <rRotation name=\"R" << rotNumber << "\"/>" << std::endl;

   Hep3Vector t =  opto->centreGlob();
   if(t != Hep3Vector()) { //if (0,0,0) write nothing
     const Hep3Vector t = opto->centreGlob();
     file_ << "  <Translation x=\"" <<  UC(t[0],"Length") << "\""
           <<                " y=\"" << UC(t[1],"Length") << "\""
	   <<                " z=\"" << UC(t[2],"Length")<< "\" />"
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
void CocoaToDDLMgr::ro(const HepRotation& ro, int n)
{
   
   HepRotation roinv = inverseOf(ro);
  //-    G4ThreeVector v(1.,1.,1.);
  //-     G4ThreeVector a;
  //-   a = (*ro)*v;
  bool identity = false;
  ALIstring tag = " <Rotation name=\"R";
  identity=roinv.isIdentity();  
  
  //  if (! identity) {     
    file_ << tag << n << "\"";
    file_ << " phiX=\""   << UC(roinv.phiX(),"Angle")   << "\""
	  << " thetaX=\"" << UC(roinv.thetaX(),"Angle") << "\""
	  << " phiY=\""   << UC(roinv.phiY(),"Angle")   << "\""
	  << " thetaY=\"" << UC(roinv.thetaY(),"Angle") << "\""
	  << " phiZ=\""   << UC(roinv.phiZ(),"Angle")   << "\""
	  << " thetaZ=\"" << UC(roinv.thetaZ(),"Angle") << "\""
      //<< " angleUnit=\"degree\"/>" 
	  << " />" << std::endl;
    //  }	     
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
   newSectPre(filename_,std::string("SpecParSection"));
}    


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::specPar(OpticalObject * opto)
{
  
  file_ << " <SpecPar name=\"" << opto->name() << "\" />" << std::endl;
  const std::vector< Entry* > coord = opto->CoordinateEntryList();
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
  for( int ii=0; ii<extraEnt.size(); ii++ ){
    Entry* ent = extraEnt[ii];    file_ << "   <Parameter name=\"OptiProp_name=" << ent->name() << "\" value=\"0.\" /> " << std::endl;
    file_ << "   <Parameter name=\"OptiProp_" << ent->name() + std::string("_dimType=") + ent->type() << "\" value=\"0.\" /> " << std::endl;
    file_ << "   <Parameter name=\"OptiProp_" << ent->name() + std::string("_value") << "\" value=\"";
    if( ent->type() == "nodim" ) {
      file_ << ent->value();
    }else if( ent->type() == "length" ) {
      file_ << UC(ent->value(),"Length");
    }else if( ent->type() == "angle" ) {
      file_ << UC(ent->value(),"Angle");
    }
    file_ << "\" /> " << std::endl;

    file_ << "   <Parameter name=\"OptiProp_" << ent->name() + std::string("_sigma") << "\" value=\"";
    if( ent->type() == "nodim" ) {
      file_ << ent->sigma();
    }else if( ent->type() == "length" ) {
      file_ << UC(ent->sigma(),"Length");
    }else if( ent->type() == "angle" ) {
      file_ << UC(ent->sigma(),"Angle");
    }
    file_ << "\" /> " << std::endl;

    file_ << "   <Parameter name=\"OptiProp_" << ent->name() + std::string("_quality") << "\" value=\"" << ent->quality() << "\" /> " << std::endl;
  }

  file_ << " </SpecPar>" << std::endl;
  
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void CocoaToDDLMgr::writeSpecParsCocoa()
{
  file_ << "<!--    Define volumes as COCOA objects --> " << std::endl
	<< "  <SpecPar name=\"COCOA\"> " << std::endl;

  static std::vector< OpticalObject* > optolist = Model::OptOList();
  static std::vector< OpticalObject* >::const_iterator ite;
  for(ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue;
    file_ << "    <PartSelector path=\"//" << (*ite)->name() << "\"/> " << std::endl;
  }
   
  file_ << "   <Parameter name=\"COCOA\" value=\"COCOA\"/> " << std::endl
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

  std::vector<HepRotation>::const_iterator ite;

  int nc = 0;
  for( ite = theRotationList.begin(); ite != theRotationList.end(); ite++) {
    if( (*ite) == opto->rmGlob() ) {
      rotnum = nc;
      break;
    }
    nc++;
  }

  if( rotnum == -1 ) {
    theRotationList.push_back( opto->rmGlob() );
    rotnum = theRotationList.size()-1;
  }

  return rotnum;

}


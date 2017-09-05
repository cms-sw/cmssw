#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromDDD10EtaPart.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <algorithm>
#include <boost/lexical_cast.hpp>

ME0GeometryBuilderFromDDD10EtaPart::ME0GeometryBuilderFromDDD10EtaPart()
{ }

ME0GeometryBuilderFromDDD10EtaPart::~ME0GeometryBuilderFromDDD10EtaPart() 
{ }


ME0Geometry* ME0GeometryBuilderFromDDD10EtaPart::build(const DDCompactView* cview, const MuonDDDConstants& muonConstants)
{

  std::string attribute = "MuStructure";
  std::string value     = "MuonEndCapME0";

  // Asking only for the MuonME0's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fview(*cview,filter);

  return this->buildGeometry(fview, muonConstants);
}


ME0Geometry* ME0GeometryBuilderFromDDD10EtaPart::buildGeometry(DDFilteredView& fv, const MuonDDDConstants& muonConstants)
{

  ME0Geometry* geometry = new ME0Geometry();

  LogTrace("ME0GeometryBuilderFromDDD") <<"Building the geometry service";
  LogTrace("ME0GeometryBuilderFromDDD") <<"About to run through the ME0 structure\n" 
					<<"Top level logical part: "
					<<fv.logicalPart().name().name();

  // ==========================================
  // ===  Test to understand the structure  ===
  // ========================================== 
  #ifdef EDM_ML_DEBUG
  bool testChambers = fv.firstChild();
  LogTrace("ME0GeometryBuilderFromDDD") << "doChamber = fv.firstChild() = " << testChambers;
  // ----------------------------------------------------------------------------------------------------------------------------------------------
  while (testChambers) {
    // to etapartitions
    LogTrace("ME0GeometryBuilderFromDDD")<<"to layer "<<fv.firstChild(); // commented out in case only looping over sensitive volumes
    LogTrace("ME0GeometryBuilderFromDDD")<<"to etapt "<<fv.firstChild(); // commented out in case only looping over sensitive volumes
    MuonDDDNumbering mdddnum(muonConstants);
    ME0NumberingScheme me0Num(muonConstants);
    int rawId = me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
    ME0DetId detId = ME0DetId(rawId);
    ME0DetId detIdCh = detId.chamberId();
    // back to chambers
    LogTrace("ME0GeometryBuilderFromDDD")<<"back to layer "<<fv.parent(); // commented out in case only looping over sensitive volumes
    LogTrace("ME0GeometryBuilderFromDDD")<<"back to chamb "<<fv.parent(); // commented out in case only looping over sensitive volumes
    // ok lets get started ...                             
    LogTrace("ME0GeometryBuilderFromDDD") << "In DoChambers Loop :: ME0DetId "<<detId<<" = "<<detId.rawId()<<" (which belongs to ME0Chamber "<<detIdCh<<" = "<<detIdCh.rawId()<<")";
    LogTrace("ME0GeometryBuilderFromDDD") << "Second level logical part: " << fv.logicalPart().name().name();
    DDBooleanSolid solid2 = (DDBooleanSolid)(fv.logicalPart().solid());
    std::vector<double> dpar2  = solid2.parameters();
    std::stringstream parameters2;
    for(unsigned int i=0; i<dpar2.size(); ++i) {
      parameters2 << " dpar["<<i<<"]="<< dpar2[i]/10 << "cm ";
    }
    LogTrace("ME0GeometryBuilderFromDDD") << "Second level parameters: vector with size = "<<dpar2.size()<<" and elements "<<parameters2.str();
    // from GEM
    // DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
    // std::vector<double> dpar = solid.solidA().parameters();
    /*
    if(solid2.solidA()) {
      std::vector<double> dpar2a = solid2.solidA().parameters();
      std::stringstream parameters2a;
      for(unsigned int i=0; i<dpar2a.size(); ++i) {
	parameters2a << " dpara["<<i<<"]="<< dpar2a[i]/10 << "cm ";
      }
      LogTrace("ME0GeometryBuilderFromDDD") << "Second level parameters: vector with size = "<<dpar2a.size()<<" and elements "<<parameters2.str();
    }
    if(solid2.solidB()) {
      std::vector<double> dpar2b = solid2.solidB().parameters();
      std::stringstream parameters2b;
      for(unsigned int i=0; i<dpar2b.size(); ++i) {
	parameters2b << " dparb["<<i<<"]="<< dpar2b[i]/10 << "cm ";
      }
      LogTrace("ME0GeometryBuilderFromDDD") << "Second level parameters: vector with size = "<<dpar2b.size()<<" and elements "<<parameters2.str();
    }
    */
    bool doLayers = fv.firstChild();
    // --------------------------------------------------------------------------------------------------------------------------------------------
    LogTrace("ME0GeometryBuilderFromDDD") << "doLayer = fv.firstChild() = " << doLayers;
    while (doLayers) {
      // to etapartitions
      LogTrace("ME0GeometryBuilderFromDDD")<<"to etapt "<<fv.firstChild(); // commented out in case only looping over sensitive volumes
      MuonDDDNumbering mdddnum(muonConstants);
      ME0NumberingScheme me0Num(muonConstants);
      int rawId = me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
      ME0DetId detId = ME0DetId(rawId);
      ME0DetId detIdLa = detId.layerId();
      // back to layers
      LogTrace("ME0GeometryBuilderFromDDD")<<"back to layer "<<fv.parent(); // commented out in case only looping over sensitive volumes
      LogTrace("ME0GeometryBuilderFromDDD") << "In DoLayers Loop :: ME0DetId "<<detId<<" = "<<detId.rawId()<<" (which belongs to ME0Layer "<<detIdLa<<" = "<<detIdLa.rawId()<<")";
      LogTrace("ME0GeometryBuilderFromDDD") << "Third level logical part: " << fv.logicalPart().name().name();
      DDBooleanSolid solid3 = (DDBooleanSolid)(fv.logicalPart().solid());
      std::vector<double> dpar3 = solid3.parameters();
      std::stringstream parameters3;
      for(unsigned int i=0; i<dpar3.size(); ++i) {
	parameters3 << " dpar["<<i<<"]="<< dpar3[i]/10 << "cm ";
      }
      LogTrace("ME0GeometryBuilderFromDDD") << "Third level parameters: vector with size = "<<dpar3.size()<<" and elements "<<parameters3.str();
      bool doEtaParts = fv.firstChild(); 
      // --------------------------------------------------------------------------------------------------------------------------------------------
      LogTrace("ME0GeometryBuilderFromDDD") << "doEtaPart = fv.firstChild() = " << doEtaParts;
      while (doEtaParts) {
	LogTrace("ME0GeometryBuilderFromDDD") << "In DoEtaParts Loop :: ME0DetId "<<detId<<" = "<<detId.rawId();
	LogTrace("ME0GeometryBuilderFromDDD") << "Fourth level logical part: " << fv.logicalPart().name().name();
	DDBooleanSolid solid4 = (DDBooleanSolid)(fv.logicalPart().solid());
	std::vector<double> dpar4 = solid4.parameters();
	std::stringstream parameters4;
	for(unsigned int i=0; i<dpar4.size(); ++i) {
	  parameters4 << " dpar["<<i<<"]="<< dpar4[i]/10 << "cm ";
	}
	LogTrace("ME0GeometryBuilderFromDDD") << "Fourth level parameters: vector with size = "<<dpar4.size()<<" and elements "<<parameters4.str();
	// --------------------------------------------------------------------------------------------------------------------------------------------
	doEtaParts = fv.nextSibling();
	LogTrace("ME0GeometryBuilderFromDDD") << "doEtaPart = fv.nextSibling() = " << doEtaParts;
      }
      fv.parent(); // commented out in case only looping over sensitive volumes
      LogTrace("ME0GeometryBuilderFromDDD") << "went back to parent :: name = "<<fv.logicalPart().name().name()<<" will now ask for nextSibling";
      doLayers = fv.nextSibling();
      LogTrace("ME0GeometryBuilderFromDDD") << "doLayer = fv.nextSibling() = " << doLayers;
    }
    fv.parent(); // commented out in case only looping over sensitive volumes
    LogTrace("ME0GeometryBuilderFromDDD") << "went back to parent :: name = "<<fv.logicalPart().name().name()<<" will now ask for nextSibling";
    testChambers = fv.nextSibling();
    LogTrace("ME0GeometryBuilderFromDDD") << "doChamber = fv.nextSibling() = " << testChambers;
  }
  fv.parent();
  #endif

 
  // ==========================================
  // === Here the Real ME0 Geometry Builder ===
  // ==========================================
  bool doChambers = fv.firstChild();
  while (doChambers) {
    // to etapartitions and back again to pick up DetId
    fv.firstChild();
    fv.firstChild();
    MuonDDDNumbering mdddnum(muonConstants);
    ME0NumberingScheme me0Num(muonConstants);
    int rawId = me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
    ME0DetId detId = ME0DetId(rawId);
    ME0DetId detIdCh = detId.chamberId();
    fv.parent();
    fv.parent();

    // build chamber 
    ME0Chamber *me0Chamber = buildChamber(fv, detIdCh);
    geometry->add(me0Chamber);

    // loop over layers of the chamber
    bool doLayers = fv.firstChild();
    while (doLayers) {
      // to etapartitions and back again to pick up DetId
      fv.firstChild();
      MuonDDDNumbering mdddnum(muonConstants);
      ME0NumberingScheme me0Num(muonConstants);
      int rawId = me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
      ME0DetId detId = ME0DetId(rawId);
      ME0DetId detIdLa = detId.layerId();
      fv.parent();

      // build layer
      ME0Layer *me0Layer = buildLayer(fv, detIdLa);
      me0Chamber->add(me0Layer);
      geometry->add(me0Layer);


      // loop over etapartitions of the layer
      bool doEtaParts = fv.firstChild(); 
      while (doEtaParts) {
	// pick up DetId
	MuonDDDNumbering mdddnum(muonConstants);
	ME0NumberingScheme me0Num(muonConstants);
	int rawId = me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
	ME0DetId detId = ME0DetId(rawId);

	// build etapartition
	ME0EtaPartition *etaPart = buildEtaPartition(fv, detId);
	me0Layer->add(etaPart);
	geometry->add(etaPart);

	doEtaParts = fv.nextSibling();
      }
      fv.parent();
      doLayers = fv.nextSibling();
    }
    fv.parent();
    doChambers = fv.nextSibling();
  }

  return geometry;
}

ME0Chamber* ME0GeometryBuilderFromDDD10EtaPart::buildChamber(DDFilteredView& fv, ME0DetId detId) const {
  LogTrace("ME0GeometryBuilderFromDDD") << "buildChamber "<<fv.logicalPart().name().name() <<" "<< detId <<std::endl;
  
  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  // std::vector<double> dpar = solid.solidA().parameters();
  std::vector<double> dpar = solid.parameters(); 
  double L  = dpar[0]/cm;// length is along local Y
  double T  = dpar[3]/cm;// thickness is long local Z
  double b  = dpar[4]/cm;// bottom width is along local X
  double B  = dpar[8]/cm;// top width is along local X
  // hardcoded :: double b = 21.9859, B = 52.7261, L = 87.1678, T = 12.9;

  #ifdef EDM_ML_DEBUG  
  LogTrace("ME0GeometryBuilderFromDDD") << " name of logical part = "<<fv.logicalPart().name().name()<<std::endl;
  LogTrace("ME0GeometryBuilderFromDDD") << " dpar is vector with size = "<<dpar.size()<<std::endl;
  for(unsigned int i=0; i<dpar.size(); ++i) {
    LogTrace("ME0GeometryBuilderFromDDD") << " dpar ["<<i<<"] = "<< dpar[i]/10 << " cm "<<std::endl;
  }
  LogTrace("ME0GeometryBuilderFromDDD") << "size  b: "<< b << "cm, B: " << B << "cm,  L: " << L << "cm, T: " << T <<"cm "<<std::endl;
  #endif

  bool isOdd = false; // detId.chamber()%2;
  ME0BoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(b,B,L,T), isOdd ));
  ME0Chamber* chamber = new ME0Chamber(detId.chamberId(), surf);
  return chamber;
}

ME0Layer* ME0GeometryBuilderFromDDD10EtaPart::buildLayer(DDFilteredView& fv, ME0DetId detId) const {
  LogTrace("ME0GeometryBuilderFromDDD") << "buildLayer "<<fv.logicalPart().name().name() <<" "<< detId <<std::endl;
  
  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  // std::vector<double> dpar = solid.solidA().parameters();
  std::vector<double> dpar = solid.parameters();
  double L = dpar[0]/cm;// length is along local Y
  double t = dpar[3]/cm;// thickness is long local Z
  double b = dpar[4]/cm;// bottom width is along local X
  double B = dpar[8]/cm;// top width is along local X
  // dpar = solid.solidB().parameters();
  // dz += dpar[3]/cm;     // layer thickness --- to be checked !!! layer thickness should be same as eta part thickness
  // hardcoded :: double b = 21.9859, B = 52.7261, L = 87.1678, t = 0.4;

  #ifdef EDM_ML_DEBUG
  LogTrace("ME0GeometryBuilderFromDDD") << " name of logical part = "<<fv.logicalPart().name().name()<<std::endl;
  LogTrace("ME0GeometryBuilderFromDDD") << " dpar is vector with size = "<<dpar.size()<<std::endl;
  for(unsigned int i=0; i<dpar.size(); ++i) {
    LogTrace("ME0GeometryBuilderFromDDD") << " dpar ["<<i<<"] = "<< dpar[i]/10 << " cm "<<std::endl;
  }
  LogTrace("ME0GeometryBuilderFromDDD") << "size  b: "<< b << "cm, B: " << B << "cm,  L: " << L << "cm, t: " << t <<"cm "<<std::endl;
  #endif

  bool isOdd = false; // detId.chamber()%2;
  ME0BoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(b,B,L,t), isOdd ));
  ME0Layer* layer = new ME0Layer(detId.layerId(), surf);
  return layer;
}

ME0EtaPartition* ME0GeometryBuilderFromDDD10EtaPart::buildEtaPartition(DDFilteredView& fv, ME0DetId detId) const {
  LogTrace("ME0GeometryBuilderFromDDD") << "buildEtaPartition "<<fv.logicalPart().name().name() <<" "<< detId <<std::endl;
  
  // EtaPartition specific parameter (nstrips and npads) 
  DDValue numbOfStrips("nStrips");
  DDValue numbOfPads("nPads");
  std::vector<const DDsvalues_type* > specs(fv.specifics());
  std::vector<const DDsvalues_type* >::iterator is = specs.begin();
  double nStrips = 0., nPads = 0.;
  for (;is != specs.end(); is++){
    if (DDfetch( *is, numbOfStrips)) nStrips = numbOfStrips.doubles()[0];
    if (DDfetch( *is, numbOfPads))   nPads = numbOfPads.doubles()[0];
  }
  LogTrace("ME0GeometryBuilderFromDDD") 
    << ((nStrips == 0. ) ? ("No nStrips found!!") : ("Number of strips: " + boost::lexical_cast<std::string>(nStrips))); 
  LogTrace("ME0GeometryBuilderFromDDD") 
    << ((nPads == 0. ) ? ("No nPads found!!") : ("Number of pads: " + boost::lexical_cast<std::string>(nPads)));
  
  // EtaPartition specific parameter (size) 
  std::vector<double> dpar = fv.logicalPart().solid().parameters();
  double b = dpar[4]/cm; // half bottom edge
  double B = dpar[8]/cm; // half top edge
  double L = dpar[0]/cm; // half apothem
  double t = dpar[3]/cm; // half thickness
  
  #ifdef EDM_ML_DEBUG
  LogTrace("ME0GeometryBuilderFromDDD") << " name of logical part = "<<fv.logicalPart().name().name()<<std::endl;
  LogTrace("ME0GeometryBuilderFromDDD") << " dpar is vector with size = "<<dpar.size()<<std::endl;
  for(unsigned int i=0; i<dpar.size(); ++i) {
    LogTrace("ME0GeometryBuilderFromDDD") << " dpar ["<<i<<"] = "<< dpar[i]/10 << " cm "<<std::endl;
  }
  LogTrace("ME0GeometryBuilderFromDDD") << "size  b: "<< b << "cm, B: " << B << "cm,  L: " << L << "cm, t: " << t <<"cm "<<std::endl;
  #endif

  std::vector<float> pars;
  pars.emplace_back(b); 
  pars.emplace_back(B); 
  pars.emplace_back(L); 
  pars.emplace_back(nStrips);
  pars.emplace_back(nPads);
  
  bool isOdd = false; // detId.chamber()%2; // this gives the opportunity (in future) to change the face of the chamber (electronics facing IP or electronics away from IP)
  ME0BoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(b, B, L, t), isOdd ));
  std::string name = fv.logicalPart().name().name();
  ME0EtaPartitionSpecs* e_p_specs = new ME0EtaPartitionSpecs(GeomDetEnumerators::ME0, name, pars);
  
  ME0EtaPartition* etaPartition = new ME0EtaPartition(detId, surf, e_p_specs);
  return etaPartition;
}

ME0GeometryBuilderFromDDD10EtaPart::ME0BoundPlane
ME0GeometryBuilderFromDDD10EtaPart::boundPlane(const DDFilteredView& fv,
                                      Bounds* bounds, bool isOddChamber) const {
  // extract the position
  const DDTranslation & trans(fv.translation());
  const Surface::PositionType posResult(float(trans.x()/cm),
                                        float(trans.y()/cm),
                                        float(trans.z()/cm));

  // now the rotation
  //  DDRotationMatrix tmp = fv.rotation(); 
  // === DDD uses 'active' rotations - see CLHEP user guide === 
  //     ORCA uses 'passive' rotation.                          
  //     'active' and 'passive' rotations are inverse to each other
  //  DDRotationMatrix tmp = fv.rotation();                        
  const DDRotationMatrix& rotation = fv.rotation();//REMOVED .Inverse();  
  DD3Vector x, y, z;
  rotation.GetComponents(x,y,z);
  // LogTrace("GEMGeometryBuilderFromDDD") << "translation: "<< fv.translation() << std::endl;
  // LogTrace("GEMGeometryBuilderFromDDD") << "rotation   : "<< fv.rotation() << std::endl;   
  // LogTrace("GEMGeometryBuilderFromDDD") << "INVERSE rotation manually: \n"                 
  //        << x.X() << ", " << x.Y() << ", " << x.Z() << std::endl                           
  //        << y.X() << ", " << y.Y() << ", " << y.Z() << std::endl                          
  //        << z.X() << ", " << z.Y() << ", " << z.Z() << std::endl;                         

  Surface::RotationType rotResult(float(x.X()),float(x.Y()),float(x.Z()),
                                  float(y.X()),float(y.Y()),float(y.Z()),
                                  float(z.X()),float(z.Y()),float(z.Z()));

  //Change of axes for the forward
  Basic3DVector<float> newX(1.,0.,0.);
  Basic3DVector<float> newY(0.,0.,1.);
  Basic3DVector<float> newZ(0.,1.,0.);
  newY *= -1;

  rotResult.rotateAxes(newX, newY, newZ);

  return ME0BoundPlane( new BoundPlane( posResult, rotResult, bounds));
}


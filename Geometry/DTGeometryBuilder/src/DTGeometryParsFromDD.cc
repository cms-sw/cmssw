/** \file
 *
 *  \author Stefano Lacaprara  <lacaprara@pd.infn.it>  INFN LNL
 */

#include <Geometry/DTGeometryBuilder/src/DTGeometryParsFromDD.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/DTGeometry/interface/DTChamber.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>

#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include <string>

using namespace std;

#include <string>

using namespace std;

DTGeometryParsFromDD::DTGeometryParsFromDD() {}

DTGeometryParsFromDD::~DTGeometryParsFromDD(){}


void DTGeometryParsFromDD::build(const DDCompactView* cview,
                                 const MuonDDDConstants& muonConstants,
                                 RecoIdealGeometry& rig) {
  //  cout << "DTGeometryParsFromDD::build" << endl;
  //   static const string t0 = "DTGeometryParsFromDD::build";
  //   TimeMe timer(t0,true);

  std::string attribute = "MuStructure"; 
  std::string value     = "MuonBarrelDT";
  DDValue val(attribute, value, 0.0);

  // Asking only for the Muon DTs
  DDSpecificsFilter filter;
  filter.setCriteria(val,  // name & value of a variable 
		     DDCompOp::matches,
		     DDLogOp::AND, 
		     true, // compare strings otherwise doubles
		     true  // use merged-specifics or simple-specifics
		     );
  DDFilteredView fview(*cview);
  fview.addFilter(filter);
  buildGeometry(fview, muonConstants, rig);
  //cout << "RecoIdealGeometry " << rig.size() << endl;
}


void DTGeometryParsFromDD::buildGeometry(DDFilteredView& fv,
                                         const MuonDDDConstants& muonConstants,
                                         RecoIdealGeometry& rig) const {
  // static const string t0 = "DTGeometryParsFromDD::buildGeometry";
  // TimeMe timer(t0,true);

  bool doChamber = fv.firstChild();

  // Loop on chambers
  int ChamCounter=0;
  while (doChamber){
    ChamCounter++;
    DDValue val("Type");
    const DDsvalues_type params(fv.mergedSpecifics());
    string type;
    if (DDfetch(&params,val)) type = val.strings()[0];
    // FIXME
    val=DDValue("FEPos");
    string FEPos;
    if (DDfetch(&params,val)) FEPos = val.strings()[0];
    insertChamber(fv,type, muonConstants,rig);

    // Loop on SLs
    bool doSL = fv.firstChild();
    int SLCounter=0;
    while (doSL) {
      SLCounter++;
      insertSuperLayer(fv, type, muonConstants,rig);

      bool doL = fv.firstChild();
      int LCounter=0;
      // Loop on SLs
      while (doL) {
        LCounter++;
        insertLayer(fv, type, muonConstants, rig);

        // fv.parent();
        doL = fv.nextSibling(); // go to next layer
      } // layers

      fv.parent();
      doSL = fv.nextSibling(); // go to next SL
    } // sls

    fv.parent();
    doChamber = fv.nextSibling(); // go to next chamber
  } // chambers
}

void DTGeometryParsFromDD::insertChamber(DDFilteredView& fv,
                                         const string& type,
                                         const MuonDDDConstants& muonConstants,
                                         RecoIdealGeometry& rig) const {
  MuonDDDNumbering mdddnum (muonConstants);
  DTNumberingScheme dtnum (muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTChamberId detId(rawid);
  //cout << "inserting Chamber " << detId << endl;

  // Chamber specific parameter (size) 
  vector<double> par;
  par.push_back(DTChamberTag);
  vector<double> size= extractParameters(fv);
  par.insert(par.end(), size.begin(), size.end());

  ///SL the definition of length, width, thickness depends on the local reference frame of the Det
  // width is along local X
  // length is along local Y
  // thickness is long local Z

  PosRotPair posRot(plane(fv));

  rig.insert(rawid, posRot.first, posRot.second, par);
}

void DTGeometryParsFromDD::insertSuperLayer(DDFilteredView& fv,
                                            const std::string& type, 
                                            const MuonDDDConstants& muonConstants,
                                            RecoIdealGeometry& rig) const {

  MuonDDDNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTSuperLayerId slId(rawid);
  //cout << "inserting SuperLayer " << slId << endl;

  // Slayer specific parameter (size)
  vector<double> par;
  par.push_back(DTSuperLayerTag);
  vector<double> size= extractParameters(fv);
  par.insert(par.end(), size.begin(), size.end());

  // Ok this is the slayer position...
  PosRotPair posRot(plane(fv));

  rig.insert(slId, posRot.first, posRot.second, par);
}

void DTGeometryParsFromDD::insertLayer(DDFilteredView& fv,
                                       const std::string& type,
                                       const MuonDDDConstants& muonConstants,
                                       RecoIdealGeometry& rig) const {

  MuonDDDNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTLayerId layId(rawid);
  //cout << "inserting Layer " << layId << endl;

  // Layer specific parameter (size)
  vector<double> par;
  par.push_back(DTLayerTag);
  vector<double> size= extractParameters(fv);
  par.insert(par.end(), size.begin(), size.end());

  // Loop on wires
  bool doWire = fv.firstChild();
  int WCounter=0;
  int firstWire=fv.copyno();
  //float wireLength = par[1]/cm;
  while (doWire) {
    WCounter++;
    doWire = fv.nextSibling(); // next wire
  }
  vector<double> sensSize= extractParameters(fv);
  //int lastWire=fv.copyno();
  par.push_back(firstWire);
  par.push_back(WCounter);
  par.push_back(sensSize[1]);
  fv.parent();

  PosRotPair posRot(plane(fv));

  rig.insert(layId, posRot.first, posRot.second, par);

}

vector<double> 
DTGeometryParsFromDD::extractParameters(DDFilteredView& fv) const {
  vector<double> par;
  if (fv.logicalPart().solid().shape() != ddbox) {
    DDBooleanSolid bs(fv.logicalPart().solid());
    DDSolid A = bs.solidA();
    while (A.shape() != ddbox) {
      DDBooleanSolid bs(A);
      A = bs.solidA();
    }
    par=A.parameters();
  } else {
    par = fv.logicalPart().solid().parameters();
  }
  return par;
}

DTGeometryParsFromDD::PosRotPair
DTGeometryParsFromDD::plane(const DDFilteredView& fv) const {
  // extract the position
  const DDTranslation & trans(fv.translation());

  std::vector<double> gtran( 3 );
  gtran[0] = (float) 1.0 * (trans.x() / cm);
  gtran[1] = (float) 1.0 * (trans.y() / cm);
  gtran[2] = (float) 1.0 * (trans.z() / cm);

  // now the rotation
  //  DDRotationMatrix tmp = fv.rotation();
  // === DDD uses 'active' rotations - see CLHEP user guide ===
  //     ORCA uses 'passive' rotation. 
  //     'active' and 'passive' rotations are inverse to each other
  //  DDRotationMatrix tmp = fv.rotation();
  DDRotationMatrix rotation = fv.rotation();//REMOVED .Inverse();
  DD3Vector x, y, z;
  rotation.GetComponents(x,y,z);
//   std::cout << "INVERSE rotation by its own operator: "<< fv.rotation() << std::endl;
//   std::cout << "INVERSE rotation manually: "
// 	    << x.X() << ", " << x.Y() << ", " << x.Z() << std::endl
// 	    << y.X() << ", " << y.Y() << ", " << y.Z() << std::endl
// 	    << z.X() << ", " << z.Y() << ", " << z.Z() << std::endl;

  std::vector<double> grmat( 9 );
  grmat[0] = (float) 1.0 * x.X();
  grmat[1] = (float) 1.0 * x.Y();
  grmat[2] = (float) 1.0 * x.Z();

  grmat[3] = (float) 1.0 * y.X();
  grmat[4] = (float) 1.0 * y.Y();
  grmat[5] = (float) 1.0 * y.Z();

  grmat[6] = (float) 1.0 * z.X();
  grmat[7] = (float) 1.0 * z.Y();
  grmat[8] = (float) 1.0 * z.Z();

//   std::cout << "rotation by its own operator: "<< tmp << std::endl;
//   DD3Vector tx, ty,tz;
//   tmp.GetComponents(tx, ty, tz);
//   std::cout << "rotation manually: "
// 	    << tx.X() << ", " << tx.Y() << ", " << tx.Z() << std::endl
// 	    << ty.X() << ", " << ty.Y() << ", " << ty.Z() << std::endl
// 	    << tz.X() << ", " << tz.Y() << ", " << tz.Z() << std::endl;

  return pair<std::vector<double>, std::vector<double> >(gtran, grmat);
}

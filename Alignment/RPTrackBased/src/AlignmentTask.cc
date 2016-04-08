/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "Alignment/RPTrackBased/interface/AlignmentTask.h"
#include "Alignment/RPTrackBased/interface/AlignmentConstraint.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "DataFormats/TotemRPDetId/interface/TotemRPDetId.h"

using namespace std;
using namespace edm;



AlignmentTask::AlignmentTask(const ParameterSet& ps) :
  resolveShR(ps.getParameter<bool>("resolveShR")),
  resolveShZ(ps.getParameter<bool>("resolveShZ")),
  resolveRotZ(ps.getParameter<bool>("resolveRotZ")),
  resolveRPShZ(ps.getParameter<bool>("resolveRPShZ")),
  useExtendedRotZConstraint(ps.getParameter<bool>("useExtendedRotZConstraint")),
  useZeroThetaRotZConstraint(ps.getParameter<bool>("useZeroThetaRotZConstraint")),
  useExtendedShZConstraints(ps.getParameter<bool>("useExtendedShZConstraints")),
  useExtendedRPShZConstraint(ps.getParameter<bool>("useExtendedRPShZConstraint")),
  oneRotZPerPot(ps.getParameter<bool>("oneRotZPerPot")),
  homogeneousConstraints(ps.getParameterSet("homogeneousConstraints")),
  fixedDetectorsConstraints(ps.getParameterSet("fixedDetectorsConstraints"))
{
  if (resolveShZ && resolveRPShZ)
    throw cms::Exception("AlignmentTask::AlignmentTask") << "resolveShZ and resolveRPShZ cannot be both set to True.";

  if (resolveShR) quantityClasses.push_back(qcShR);
  if (resolveShZ) quantityClasses.push_back(qcShZ);
  if (resolveRPShZ) quantityClasses.push_back(qcRPShZ);
  if (resolveRotZ) quantityClasses.push_back(qcRotZ);
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::BuildGeometry(const vector<unsigned int> &RPIds,
  const std::vector<unsigned int> excludePlanes, const TotemRPGeometry *input, double z0, AlignmentGeometry &geometry)
{
  geometry.clear();
  geometry.z0 = z0; 

  for (vector<unsigned int>::const_iterator rp = RPIds.begin(); rp != RPIds.end(); ++rp) {
    set<unsigned int> DetIds = input->DetsInRP(*rp);
    for (set<unsigned int>::iterator det = DetIds.begin(); det != DetIds.end(); ++det) {
      unsigned int rawId = TotemRPDetId::decToRawId(*det);

      CLHEP::Hep3Vector d = input->LocalToGlobalDirection(rawId, CLHEP::Hep3Vector(0., 1., 0.));
      DDTranslation c = input->GetDetector(rawId)->translation();
      double z = c.z() - z0;

      unsigned int rpNum = ((*det) / 10) % 10;
      unsigned int detNum = (*det) % 10;
      bool isU = (detNum % 2 != 0);
      if (rpNum == 2 || rpNum == 3)
        isU = !isU;

      bool exclude = false;
      for (auto p : excludePlanes)
      {
        if (p == *det)
        {
          exclude = true;
          break;
        }
      }

      if (exclude)
        continue;

      geometry.Insert(*det, DetGeometry(z, d.x(), d.y(), c.x(), c.y(), isU));
    }
  }

  // set matrix and rpMatrix indeces
  unsigned int index = 0;
  unsigned int rpIndex = 0;
  signed int lastRP = -1;
  for (AlignmentGeometry::iterator it = geometry.begin(); it != geometry.end(); ++it, ++index) {
    it->second.matrixIndex = index;
    signed int rp = it->first / 10;
    if (lastRP > 0 && lastRP != rp)
      rpIndex++; 
    lastRP = rp;
    it->second.rpMatrixIndex = rpIndex;
  }
}

//----------------------------------------------------------------------------------------------------

string AlignmentTask::QuantityClassTag(QuantityClass qc)
{
  switch (qc) {
    case qcShR: return "ShR"; 
    case qcShZ: return "ShZ"; 
    case qcRPShZ: return "RPShZ"; 
    case qcRotZ: return "RotZ"; 
  }

  throw cms::Exception("AlignmentTask::QuantityClassTag") << "Unknown quantity class " << qc << ".";
}

//----------------------------------------------------------------------------------------------------

unsigned int AlignmentTask::QuantitiesOfClass(QuantityClass qc)
{
  return (qc == qcRPShZ) ? geometry.RPs() : geometry.Detectors();
}

//----------------------------------------------------------------------------------------------------

unsigned int AlignmentTask::ConstraintsForClass(QuantityClass qc)
{
  switch (qc) {
    case qcShR: return 4; 
    case qcShZ: return (useExtendedShZConstraints) ? 4 : 2;
    case qcRPShZ: return (useExtendedRPShZConstraint) ? 2 : 1; 
    case qcRotZ:
      if (oneRotZPerPot)
        return 9*geometry.RPs() + 1;
      else {
        unsigned int count = (useZeroThetaRotZConstraint) ? 2 : 1;
        if (useExtendedRotZConstraint)
          count *= 2;
        return count;
      }
  }

  throw cms::Exception("AlignmentTask::ConstraintsForClass") << "Unknown quantity class " << qc << ".";
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::BuildHomogeneousConstraints(vector<AlignmentConstraint> &constraints)
{
  for (unsigned int cl = 0; cl < quantityClasses.size(); cl++) {
    unsigned int size = ConstraintsForClass(quantityClasses[cl]);
    const string &tag = QuantityClassTag(quantityClasses[cl]);
    
    // just one basic constraint
    if (oneRotZPerPot && quantityClasses[cl] == qcRotZ)
      size = 1;

    // get constraint values
    char buf[20];
    sprintf(buf, "%s_values", tag.c_str());
    vector<double> values(homogeneousConstraints.getParameter< vector<double> >(buf));
    if (values.size() < size)
      throw cms::Exception("AlignmentTask::BuildHomogeneousConstraints") <<
        "Invalid number of constraint values for " << tag << ". Given " << values.size() <<
        ", expected " << size << ".";

    for (unsigned int j = 0; j < size; j++) {
      // prepare a constraint with coefficient vectors
      AlignmentConstraint ac;
      ac.forClass = quantityClasses[cl];
      for (unsigned int i = 0; i < quantityClasses.size(); i++) {
        ac.coef[quantityClasses[i]].ResizeTo(QuantitiesOfClass(quantityClasses[i]));
        ac.coef[quantityClasses[i]].Zero();
      }
      ac.val = values[j];
      ac.extended = false;

      unsigned int indeces = QuantitiesOfClass(quantityClasses[cl]);

      for (unsigned int idx = 0; idx < indeces; ++idx) {
        double &coef = ac.coef[quantityClasses[cl]][idx];
        const DetGeometry &dt = (quantityClasses[cl] == qcRPShZ) ? 
          geometry.FindFirstByRPMatrixIndex(idx)->second : geometry.FindByMatrixIndex(idx)->second;
        double sc = -dt.dx*dt.sy + dt.dy*dt.sx;

        switch (quantityClasses[cl]) {
          case qcShR:
            switch (j) {
              case 0: ac.name = "ShR: z*dx"; coef = dt.z * dt.dx; break;
              case 1: ac.name = "ShR: dx"; coef = dt.dx; break;
              case 2: ac.name = "ShR: z*dy"; coef = dt.z * dt.dy; break;
              case 3: ac.name = "ShR: dy"; coef = dt.dy; break;
            }
            break;

          case qcShZ:
            switch (j) {
              case 0: ac.name = "ShZ: z"; coef = dt.z; break;
              case 1: ac.name = "ShZ: 1"; coef = 1.; break;
              case 2: ac.name = "ShZ: z for V-det"; coef = (dt.isU) ? 0. : dt.z; ac.extended = true; break;
              case 3: ac.name = "ShZ: 1 for V-det"; coef = (dt.isU) ? 0. : 1.; ac.extended = true; break;
            }
            break;
        
          case qcRPShZ:
            switch (j) {
              case 0: ac.name = "RPShZ: 1"; coef = 1.; break;
              case 1: ac.name = "RPShZ: z"; coef = dt.z; ac.extended = true; break;
            }
            break;
          
          case qcRotZ:
            unsigned int je = j;
            if (!useExtendedRotZConstraint)
              je *= 2;
            switch (je) {
              case 0: ac.name = "RotZ: 1 all det"; coef = 1.; break;
              case 1: ac.name = "RotZ: 1 V-det"; coef = (dt.isU) ? 0. : 1.; ac.extended = true; break;
              case 2: ac.name = "RotZ: z all det"; coef = dt.z; ac.extended = true; break;
              case 3: ac.name = "RotZ: z V-det"; coef = (dt.isU) ? 0. : dt.z; ac.extended = true; break;
            }
            ac.coef[qcShR][idx] = sc * ac.coef[qcRotZ][idx];
            break;
        }
      }

      constraints.push_back(ac);
    } 
    
    // only 1 rot_z per RP
    if (oneRotZPerPot && quantityClasses[cl] == qcRotZ)
      BuildOneRotZPerPotConstraints(constraints);
  }
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::BuildFixedDetectorsConstraints(vector<AlignmentConstraint> &constraints)
{
  for (unsigned int cl = 0; cl < quantityClasses.size(); cl++) {
    const string &tag = QuantityClassTag(quantityClasses[cl]);

    unsigned int basicNumber = 0;
    switch (quantityClasses[cl]) {
      case qcShR: basicNumber = 4; break;
      case qcRotZ: basicNumber = 1; break;
      case qcShZ: basicNumber = 2; break;
      case qcRPShZ: basicNumber = 1; break;
    }

    const ParameterSet &classSettings = fixedDetectorsConstraints.getParameterSet(tag.c_str());
    vector<unsigned int> ids(classSettings.getParameter< vector<unsigned int> >("ids"));
    vector<double> values(classSettings.getParameter< vector<double> >("values"));

    if (ids.size() != values.size())
      throw cms::Exception("AlignmentTask::BuildFixedDetectorsConstraints") << 
        "Different number of constraint ids and values for " << tag << ".";
    
    unsigned int size = ConstraintsForClass(quantityClasses[cl]);
    
    // just one basic constraint
    if (oneRotZPerPot && quantityClasses[cl] == qcRotZ)
      size = 1;
    
    if (ids.size() < size)
      throw cms::Exception("AlignmentTask::BuildFixedDetectorsConstraints") << 
        "Too few constrainted ids for " << tag << ". Given " << ids.size() <<
        ", while " << size << " expected.";
    
    for (unsigned int j = 0; j < size; j++) {
      AlignmentConstraint ac;
      ac.forClass = quantityClasses[cl];
      for (unsigned int i = 0; i < quantityClasses.size(); i++) {
        ac.coef[quantityClasses[i]].ResizeTo(QuantitiesOfClass(quantityClasses[i]));
        ac.coef[quantityClasses[i]].Zero();
      }
      ac.val = values[j] * 1E-3;
      ac.extended = (j >= basicNumber);

      char buf[40];
      if (quantityClasses[cl] == qcRPShZ)
        sprintf(buf, "%s: fixed RP %4u", tag.c_str(), ids[j]/10);
      else
        sprintf(buf, "%s: fixed plane %4u", tag.c_str(), ids[j]);
      ac.name = buf;

      // is the detector in geometry?
      if (!geometry.ValidSensorId(ids[j]))
        throw cms::Exception("AlignmentTask::BuildFixedDetectorsConstraints") <<
          "Detector with id " << ids[j] << " is not in the geometry.";

      unsigned int idx = (quantityClasses[cl] == qcRPShZ) ? geometry[ids[j]].rpMatrixIndex : geometry[ids[j]].matrixIndex;
      ac.coef[quantityClasses[cl]][idx] = 1.;

      constraints.push_back(ac);
    }
    
    if (oneRotZPerPot && quantityClasses[cl] == qcRotZ)
        BuildOneRotZPerPotConstraints(constraints);
  }
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::BuildOneRotZPerPotConstraints(std::vector<AlignmentConstraint> &constraints)
{
/*
  AlignmentConstraint ac;
  ac.forClass = qcRotZ;
  for (unsigned int i = 0; i < quantityClasses.size(); i++) {
    ac.coef[quantityClasses[i]].ResizeTo(QuantitiesOfClass(quantityClasses[i]));
    ac.coef[quantityClasses[i]].Zero();
  }
  ac.val = 0;
  ac.name = "RotZ: global";
  ac.extended = false;
  AlignmentGeometry::iterator it = geometry.begin();
  ac.coef[qcRotZ][geometry[it->first].matrixIndex] = 1.;
  constraints.push_back(ac);
*/

  // geometry is sorted by the detector number
  unsigned int prev_rp = 12345;
  for (AlignmentGeometry::iterator it = geometry.begin(); it != geometry.end(); ++it) {
    // do not mix different pots
    if (it->first / 10 != prev_rp) {
      prev_rp = it->first / 10;
      continue;
    }

    AlignmentConstraint ac;
    ac.forClass = qcRotZ;
    for (unsigned int i = 0; i < quantityClasses.size(); i++) {
      ac.coef[quantityClasses[i]].ResizeTo(QuantitiesOfClass(quantityClasses[i]));
      ac.coef[quantityClasses[i]].Zero();
    }
    ac.val = 0;
    ac.extended = true;
    ac.coef[qcRotZ][geometry[it->first].matrixIndex] = 1.;
    ac.coef[qcRotZ][geometry[it->first-1].matrixIndex] = -1.;

    char buf[40];
    sprintf(buf, "RotZ: %u = %u", it->first, it->first-1);
    ac.name = buf;

    constraints.push_back(ac);
  }
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::BuildOfficialConstraints(vector<AlignmentConstraint> &constraints)
{
  for (unsigned int cl = 0; cl < quantityClasses.size(); cl++) {
    // for both shr and rotz, there is one constraint per unit per projection
    unsigned int size = 0;
    if (quantityClasses[cl] == qcShR)
      size = 4;
    if (quantityClasses[cl] == qcRotZ)
      size = 4;
    if (size == 0)
      continue;

    for (unsigned int j = 0; j < size; j++) {
      // prepare a constraint with coefficient vectors
      AlignmentConstraint ac;
      ac.forClass = quantityClasses[cl];
      for (unsigned int i = 0; i < quantityClasses.size(); i++) {
        ac.coef[quantityClasses[i]].ResizeTo(QuantitiesOfClass(quantityClasses[i]));
        ac.coef[quantityClasses[i]].Zero();
      }
      ac.val = 0.;
      ac.extended = false;

      unsigned int indeces = QuantitiesOfClass(quantityClasses[cl]);

      for (unsigned int idx = 0; idx < indeces; ++idx) {
        double &coef = ac.coef[quantityClasses[cl]][idx];
        unsigned int id = geometry.FindByMatrixIndex(idx)->first;
        const DetGeometry &dt = geometry.FindByMatrixIndex(idx)->second;
//        double sc = -dt.dx*dt.sy + dt.dy*dt.sx;

        unsigned rp = id / 10;
//        double sign = 0.;
//        if (rp % 10 == 0 || rp % 10 == 4)
//          sign = +1.;
//        if (rp % 10 == 1 || rp % 10 == 5)
//          sign = -1.;

        bool st200 = ((rp%100) / 10 == 2);
        bool farUnit = ((id/10) % 10 > 2);
        bool hor = (rp % 10 == 2 || rp % 10 == 3);

        switch (quantityClasses[cl]) {
          case qcShR:
            switch (j) {
              case 0: ac.name = "ShR: V, near";
                coef = (st200 && !farUnit && !hor) ? dt.dx : 0.;
                break;
              case 1: ac.name = "ShR: U, near";
                coef = (st200 && !farUnit && !hor) ? dt.dy : 0.;
                break;
              case 2: ac.name = "ShR: V, far";
                coef = (st200 && farUnit && !hor) ? dt.dx : 0.;
                break;
              case 3: ac.name = "ShR: U, far";
                coef = (st200 && farUnit && !hor) ? dt.dy : 0.;
                break;
            }
            break;
          
          case qcRotZ:
            switch (j) {
              case 0: ac.name = "RotZ: V, near";
                if (st200 && !farUnit && !dt.isU)
                  ac.coef[qcRotZ][idx] = 1.;
                break;
              case 1: ac.name = "RotZ: U, near";
                if (st200 && !farUnit && dt.isU)
                  ac.coef[qcRotZ][idx] = 1.;
                break;
              case 2: ac.name = "RotZ: V, far";
                if (st200 && farUnit && !dt.isU)
                  ac.coef[qcRotZ][idx] = 1.;
                break;
              case 3: ac.name = "RotZ: U, far";
                if (st200 && farUnit && dt.isU)
                  ac.coef[qcRotZ][idx] = 1.;
                break;
            }

            ac.extended = (j > 0);
            //ac.coef[qcShR][idx] = sc * ac.coef[qcRotZ][idx];
            break;

          case qcShZ:
            break;
          case qcRPShZ:
            break;
        }
      }

      constraints.push_back(ac);
    } 
  }
}


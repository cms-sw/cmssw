/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#include "CalibPPS/AlignmentRelative/interface/AlignmentTask.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentConstraint.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include <algorithm>

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

AlignmentTask::AlignmentTask(const ParameterSet &ps)
    : resolveShR(ps.getParameter<bool>("resolveShR")),
      resolveShZ(ps.getParameter<bool>("resolveShZ")),
      resolveRotZ(ps.getParameter<bool>("resolveRotZ")),

      oneRotZPerPot(ps.getParameter<bool>("oneRotZPerPot")),
      useEqualMeanUMeanVRotZConstraints(ps.getParameter<bool>("useEqualMeanUMeanVRotZConstraints")),

      fixedDetectorsConstraints(ps.getParameterSet("fixedDetectorsConstraints")),
      standardConstraints(ps.getParameterSet("standardConstraints")) {
  if (resolveShR) {
    quantityClasses.push_back(qcShR1);
    quantityClasses.push_back(qcShR2);
  }

  if (resolveShZ) {
    quantityClasses.push_back(qcShZ);
  }

  if (resolveRotZ) {
    quantityClasses.push_back(qcRotZ);
  }
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::buildGeometry(const vector<unsigned int> &rpDecIds,
                                  const vector<unsigned int> &excludedSensors,
                                  const CTPPSGeometry *input,
                                  double z0,
                                  AlignmentGeometry &geometry) {
  geometry.z0 = z0;

  // traverse full known geometry
  for (auto it = input->beginSensor(); it != input->endSensor(); ++it) {
    // skip excluded sensors
    if (find(excludedSensors.begin(), excludedSensors.end(), it->first) != excludedSensors.end())
      continue;

    // is RP selected?
    const CTPPSDetId detId(it->first);
    const unsigned int rpDecId = 100 * detId.arm() + 10 * detId.station() + detId.rp();
    if (find(rpDecIds.begin(), rpDecIds.end(), rpDecId) == rpDecIds.end())
      continue;

    // extract geometry data
    CTPPSGeometry::Vector c = input->localToGlobal(detId, CTPPSGeometry::Vector(0., 0., 0.));
    CTPPSGeometry::Vector d1 = input->localToGlobal(detId, CTPPSGeometry::Vector(1., 0., 0.)) - c;
    CTPPSGeometry::Vector d2 = input->localToGlobal(detId, CTPPSGeometry::Vector(0., 1., 0.)) - c;

    // for strips: is it U plane?
    bool isU = false;
    if (detId.subdetId() == CTPPSDetId::sdTrackingStrip) {
      TotemRPDetId stripDetId(it->first);
      unsigned int rpNum = stripDetId.rp();
      unsigned int plNum = stripDetId.plane();
      isU = (plNum % 2 != 0);
      if (rpNum == 2 || rpNum == 3)
        isU = !isU;
    }

    DetGeometry dg(c.z() - z0, c.x(), c.y(), isU);
    dg.setDirection(1, d1.x(), d1.y(), d1.z());
    dg.setDirection(2, d2.x(), d2.y(), d2.z());
    geometry.insert(it->first, dg);
  }
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::buildIndexMaps() {
  // remove old mapping
  mapMeasurementIndeces.clear();
  mapQuantityIndeces.clear();

  // loop over all classes
  for (const auto &qcl : quantityClasses) {
    // create entry for this class
    mapMeasurementIndeces[qcl];

    // loop over all sensors
    unsigned int idxMeas = 0;
    unsigned int idxQuan = 0;
    for (const auto &git : geometry.getSensorMap()) {
      const unsigned int detId = git.first;
      const unsigned int subdetId = CTPPSDetId(git.first).subdetId();

      // update measurement map
      if (qcl == qcShR1) {
        if (subdetId == CTPPSDetId::sdTimingDiamond)
          mapMeasurementIndeces[qcl][{detId, 1}] = idxMeas++;
        if (subdetId == CTPPSDetId::sdTrackingPixel)
          mapMeasurementIndeces[qcl][{detId, 1}] = idxMeas++;
      }

      if (qcl == qcShR2) {
        if (subdetId == CTPPSDetId::sdTrackingStrip)
          mapMeasurementIndeces[qcl][{detId, 2}] = idxMeas++;
        if (subdetId == CTPPSDetId::sdTrackingPixel)
          mapMeasurementIndeces[qcl][{detId, 2}] = idxMeas++;
      }

      if (qcl == qcShZ) {
        if (subdetId == CTPPSDetId::sdTrackingStrip)
          mapMeasurementIndeces[qcl][{detId, 2}] = idxMeas++;
        if (subdetId == CTPPSDetId::sdTrackingPixel)
          mapMeasurementIndeces[qcl][{detId, 1}] = idxMeas++;
        if (subdetId == CTPPSDetId::sdTrackingPixel)
          mapMeasurementIndeces[qcl][{detId, 2}] = idxMeas++;
        if (subdetId == CTPPSDetId::sdTimingDiamond)
          mapMeasurementIndeces[qcl][{detId, 1}] = idxMeas++;
      }

      if (qcl == qcRotZ) {
        if (subdetId == CTPPSDetId::sdTrackingStrip)
          mapMeasurementIndeces[qcl][{detId, 2}] = idxMeas++;
        if (subdetId == CTPPSDetId::sdTrackingPixel)
          mapMeasurementIndeces[qcl][{detId, 1}] = idxMeas++;
        if (subdetId == CTPPSDetId::sdTrackingPixel)
          mapMeasurementIndeces[qcl][{detId, 2}] = idxMeas++;
        if (subdetId == CTPPSDetId::sdTimingDiamond)
          mapMeasurementIndeces[qcl][{detId, 1}] = idxMeas++;
      }

      // update quantity map
      if (qcl == qcShR1) {
        if (subdetId == CTPPSDetId::sdTimingDiamond)
          mapQuantityIndeces[qcl][detId] = idxQuan++;
        if (subdetId == CTPPSDetId::sdTrackingPixel)
          mapQuantityIndeces[qcl][detId] = idxQuan++;
      }

      if (qcl == qcShR2) {
        if (subdetId == CTPPSDetId::sdTrackingStrip)
          mapQuantityIndeces[qcl][detId] = idxQuan++;
        if (subdetId == CTPPSDetId::sdTrackingPixel)
          mapQuantityIndeces[qcl][detId] = idxQuan++;
      }

      if (qcl == qcShZ) {
        if (subdetId == CTPPSDetId::sdTrackingStrip)
          mapQuantityIndeces[qcl][detId] = idxQuan++;
        if (subdetId == CTPPSDetId::sdTimingDiamond)
          mapQuantityIndeces[qcl][detId] = idxQuan++;
        if (subdetId == CTPPSDetId::sdTrackingPixel)
          mapQuantityIndeces[qcl][detId] = idxQuan++;
      }

      if (qcl == qcRotZ) {
        if (subdetId == CTPPSDetId::sdTrackingStrip)
          mapQuantityIndeces[qcl][detId] = idxQuan++;
        if (subdetId == CTPPSDetId::sdTimingDiamond)
          mapQuantityIndeces[qcl][detId] = idxQuan++;
        if (subdetId == CTPPSDetId::sdTrackingPixel)
          mapQuantityIndeces[qcl][detId] = idxQuan++;
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

signed int AlignmentTask::getMeasurementIndex(QuantityClass cl, unsigned int detId, unsigned int dirIdx) const {
  auto clit = mapMeasurementIndeces.find(cl);
  if (clit == mapMeasurementIndeces.end())
    return -1;

  auto it = clit->second.find({detId, dirIdx});
  if (it == clit->second.end())
    return -1;

  return it->second;
}

//----------------------------------------------------------------------------------------------------

signed int AlignmentTask::getQuantityIndex(QuantityClass cl, unsigned int detId) const {
  auto clit = mapQuantityIndeces.find(cl);
  if (clit == mapQuantityIndeces.end())
    return -1;

  auto it = clit->second.find(detId);
  if (it == clit->second.end())
    return -1;

  return it->second;
}

//----------------------------------------------------------------------------------------------------

string AlignmentTask::quantityClassTag(QuantityClass qc) const {
  switch (qc) {
    case qcShR1:
      return "ShR1";
    case qcShR2:
      return "ShR2";
    case qcShZ:
      return "ShZ";
    case qcRotZ:
      return "RotZ";
  }

  throw cms::Exception("PPS") << "Unknown quantity class " << qc << ".";
}

//----------------------------------------------------------------------------------------------------

unsigned int AlignmentTask::measurementsOfClass(QuantityClass qc) const {
  auto it = mapMeasurementIndeces.find(qc);
  if (it == mapMeasurementIndeces.end())
    return 0;
  else
    return it->second.size();
}

//----------------------------------------------------------------------------------------------------

unsigned int AlignmentTask::quantitiesOfClass(QuantityClass qc) const {
  auto it = mapQuantityIndeces.find(qc);
  if (it == mapQuantityIndeces.end())
    return 0;
  else
    return it->second.size();
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::buildFixedDetectorsConstraints(vector<AlignmentConstraint> &constraints) const {
  for (auto &quantityClass : quantityClasses) {
    // get input
    const string &tag = quantityClassTag(quantityClass);

    const ParameterSet &classSettings = fixedDetectorsConstraints.getParameterSet(tag.c_str());
    vector<unsigned int> ids(classSettings.getParameter<vector<unsigned int>>("ids"));
    vector<double> values(classSettings.getParameter<vector<double>>("values"));

    if (ids.size() != values.size())
      throw cms::Exception("PPS") << "Different number of constraint ids and values for " << tag << ".";

    // determine number of constraints
    unsigned int size = ids.size();

    // just one basic constraint
    if (oneRotZPerPot && quantityClass == qcRotZ) {
      if (size > 1)
        size = 1;
    }

    // build constraints
    for (unsigned int j = 0; j < size; j++) {
      // prepare empty constraint
      AlignmentConstraint ac;

      for (auto &qcit : quantityClasses) {
        ac.coef[qcit].ResizeTo(quantitiesOfClass(qcit));
        ac.coef[qcit].Zero();
      }

      // set constraint name
      char buf[40];
      sprintf(buf, "%s: fixed plane %4u", tag.c_str(), ids[j]);
      ac.name = buf;

      // get quantity index
      signed int qIndex = getQuantityIndex(quantityClass, ids[j]);
      if (qIndex < 0)
        throw cms::Exception("AlignmentTask::BuildFixedDetectorsConstraints")
            << "Quantity index for class " << quantityClass << " and id " << ids[j] << " is " << qIndex;

      // set constraint coefficient and value
      ac.coef[quantityClass][qIndex] = 1.;
      ac.val = values[j] * 1E-3;

      // save constraint
      constraints.push_back(ac);
    }

    if (oneRotZPerPot && quantityClass == qcRotZ)
      buildOneRotZPerPotConstraints(constraints);
  }
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::buildStandardConstraints(vector<AlignmentConstraint> &constraints) const {
  const vector<unsigned int> &decUnitIds = standardConstraints.getParameter<vector<unsigned int>>("units");

  // count planes in RPs
  map<unsigned int, unsigned int> planesPerPot;
  for (const auto &it : geometry.getSensorMap()) {
    CTPPSDetId detId(it.first);
    planesPerPot[detId.rpId()]++;
  }

  // ShR constraints
  if (resolveShR) {
    for (const auto &decUnitId : decUnitIds) {
      // prepare empty constraints
      AlignmentConstraint ac_X;
      for (auto &qcit : quantityClasses) {
        ac_X.coef[qcit].ResizeTo(quantitiesOfClass(qcit));
        ac_X.coef[qcit].Zero();
      }
      ac_X.val = 0;

      AlignmentConstraint ac_Y(ac_X);

      // set constraint names
      char buf[50];
      sprintf(buf, "ShR: unit %u, MeanX=0", decUnitId);
      ac_X.name = buf;
      sprintf(buf, "ShR: unit %u, MeanY=0", decUnitId);
      ac_Y.name = buf;

      // traverse geometry
      for (const auto &git : geometry.getSensorMap()) {
        // stop is sensor not in the selected arm
        CTPPSDetId senId(git.first);
        unsigned int senDecUnit = senId.arm() * 100 + senId.station() * 10;
        if (senId.rp() > 2)
          senDecUnit += 1;

        if (senDecUnit != decUnitId)
          continue;

        // fill constraint for strip sensors
        if (senId.subdetId() == CTPPSDetId::sdTrackingStrip) {
          signed int qIndex = getQuantityIndex(qcShR2, git.first);
          if (qIndex < 0)
            throw cms::Exception("AlignmentTask::BuildStandardConstraints")
                << "Cannot get quantity index for class " << qcShR2 << " and sensor id " << git.first << ".";

          // determine weight
          const double weight = 1. / planesPerPot[senId.rpId()];

          // set constraint coefficients
          ac_X.coef[qcShR2][qIndex] = git.second.getDirectionData(2).dx * weight;
          ac_Y.coef[qcShR2][qIndex] = git.second.getDirectionData(2).dy * weight;
        }

        // fill constraint for pixel sensors
        if (senId.subdetId() == CTPPSDetId::sdTrackingPixel) {
          // get quantity indeces
          const signed int qIndex1 = getQuantityIndex(qcShR1, git.first);
          if (qIndex1 < 0)
            throw cms::Exception("AlignmentTask::BuildStandardConstraints")
                << "Cannot get quantity index for class " << qcShR1 << " and sensor id " << git.first << ".";

          const signed int qIndex2 = getQuantityIndex(qcShR2, git.first);
          if (qIndex2 < 0)
            throw cms::Exception("AlignmentTask::BuildStandardConstraints")
                << "Cannot get quantity index for class " << qcShR2 << " and sensor id " << git.first << ".";

          // determine weight (two constraints per plane)
          const double weight = 0.5 / planesPerPot[senId.rpId()];

          // get geometry
          const double d1x = git.second.getDirectionData(1).dx;
          const double d1y = git.second.getDirectionData(1).dy;
          const double d2x = git.second.getDirectionData(2).dx;
          const double d2y = git.second.getDirectionData(2).dy;

          // calculate coefficients, by inversion of this matrix relation
          //  [ s1 ] = [ d1x  d1y ] * [ de x ]
          //  [ s2 ]   [ d2x  d2y ]   [ de y ]
          const double D = d1x * d2y - d1y * d2x;
          const double coef_x_s1 = +d2y / D;
          const double coef_y_s1 = -d2x / D;
          const double coef_x_s2 = -d1y / D;
          const double coef_y_s2 = +d1x / D;

          // set constraint coefficients
          ac_X.coef[qcShR1][qIndex1] = coef_x_s1 * weight;
          ac_Y.coef[qcShR1][qIndex1] = coef_y_s1 * weight;
          ac_X.coef[qcShR2][qIndex2] = coef_x_s2 * weight;
          ac_Y.coef[qcShR2][qIndex2] = coef_y_s2 * weight;
        }
      }

      // add constraints
      constraints.push_back(ac_X);
      constraints.push_back(ac_Y);
    }
  }

  // RotZ constraints
  if (resolveRotZ) {
    for (const auto &decUnitId : decUnitIds) {
      // prepare empty constraints
      AlignmentConstraint ac;
      for (unsigned int i = 0; i < quantityClasses.size(); i++) {
        ac.coef[quantityClasses[i]].ResizeTo(quantitiesOfClass(quantityClasses[i]));
        ac.coef[quantityClasses[i]].Zero();
      }
      ac.val = 0;

      char buf[50];
      sprintf(buf, "RotZ: unit %u, Mean=0", decUnitId);
      ac.name = buf;

      // traverse geometry
      for (const auto &git : geometry.getSensorMap()) {
        // stop is sensor not in the selected arm
        CTPPSDetId senId(git.first);
        unsigned int senDecUnit = senId.arm() * 100 + senId.station() * 10;
        if (senId.rp() > 2)
          senDecUnit += 1;

        if (senDecUnit != decUnitId)
          continue;

        // determine weight
        const double weight = 1. / planesPerPot[senId.rpId()];

        // set coefficient
        signed int qIndex = getQuantityIndex(qcRotZ, git.first);
        ac.coef[qcRotZ][qIndex] = weight;
      }

      constraints.push_back(ac);
    }
  }

  if (resolveRotZ && oneRotZPerPot)
    buildOneRotZPerPotConstraints(constraints);

  if (resolveRotZ && useEqualMeanUMeanVRotZConstraints)
    buildEqualMeanUMeanVRotZConstraints(constraints);
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::buildOneRotZPerPotConstraints(std::vector<AlignmentConstraint> &constraints) const {
  // build map rp id --> sensor ids
  map<unsigned int, vector<unsigned int>> m;
  for (const auto &p : geometry.getSensorMap()) {
    CTPPSDetId detId(p.first);
    CTPPSDetId rpId = detId.rpId();
    unsigned int decRPId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();
    m[decRPId].push_back(p.first);
  }

  // traverse all RPs
  for (const auto &p : m) {
    // build n_planes-1 constraints
    unsigned int prev_detId = 0;
    for (const auto &detId : p.second) {
      if (prev_detId != 0) {
        AlignmentConstraint ac;

        char buf[100];
        sprintf(buf, "RotZ: RP %u, plane %u = plane %u", p.first, prev_detId, detId);
        ac.name = buf;

        ac.val = 0;

        for (auto &qcit : quantityClasses) {
          ac.coef[qcit].ResizeTo(quantitiesOfClass(qcit));
          ac.coef[qcit].Zero();
        }

        signed int qIdx1 = getQuantityIndex(qcRotZ, prev_detId);
        signed int qIdx2 = getQuantityIndex(qcRotZ, detId);

        ac.coef[qcRotZ][qIdx1] = +1.;
        ac.coef[qcRotZ][qIdx2] = -1.;

        constraints.push_back(ac);
      }

      prev_detId = detId;
    }
  }
}

//----------------------------------------------------------------------------------------------------

void AlignmentTask::buildEqualMeanUMeanVRotZConstraints(vector<AlignmentConstraint> &constraints) const {
  // build map strip rp id --> pair( vector of U planes, vector of V planes )
  map<unsigned int, pair<vector<unsigned int>, vector<unsigned int>>> m;
  for (const auto &p : geometry.getSensorMap()) {
    CTPPSDetId detId(p.first);

    if (detId.subdetId() != CTPPSDetId::sdTrackingStrip)
      continue;

    CTPPSDetId rpId = detId.rpId();
    unsigned int decRPId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    if (p.second.isU)
      m[decRPId].first.push_back(p.first);
    else
      m[decRPId].second.push_back(p.first);
  }

  // loop over RPs
  for (const auto &p : m) {
    AlignmentConstraint ac;

    char buf[100];
    sprintf(buf, "RotZ: RP %u, MeanU = MeanV", p.first);
    ac.name = buf;

    ac.val = 0;

    for (auto &qcit : quantityClasses) {
      ac.coef[qcit].ResizeTo(quantitiesOfClass(qcit));
      ac.coef[qcit].Zero();
    }

    for (const string &proj : {"U", "V"}) {
      const auto &planes = (proj == "U") ? p.second.first : p.second.second;
      const double c = ((proj == "U") ? -1. : +1.) / planes.size();

      for (const auto &plane : planes) {
        signed int qIdx = getQuantityIndex(qcRotZ, plane);
        ac.coef[qcRotZ][qIdx] = c;

        TotemRPDetId plId(plane);
      }
    }

    constraints.push_back(ac);
  }
}

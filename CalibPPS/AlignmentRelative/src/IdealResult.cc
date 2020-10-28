/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#include "CalibPPS/AlignmentRelative/interface/IdealResult.h"

#include "CalibPPS/AlignmentRelative/interface/AlignmentTask.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"

#include "TMatrixD.h"
#include "TVectorD.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

IdealResult::IdealResult(const edm::ParameterSet &gps, AlignmentTask *_t) : AlignmentAlgorithm(gps, _t) {}

//----------------------------------------------------------------------------------------------------

void IdealResult::begin(const CTPPSGeometry *geometryReal, const CTPPSGeometry *geometryMisaligned) {
  gReal = geometryReal;
  gMisaligned = geometryMisaligned;
}
//----------------------------------------------------------------------------------------------------

unsigned int IdealResult::solve(const std::vector<AlignmentConstraint> &constraints,
                                std::map<unsigned int, AlignmentResult> &results,
                                TDirectory *dir) {
  /* 
  STRATEGY:

  1) Determine true misalignment as misalign_geometry - real_geometry.

  2) Represent the true misalignment as the "chi" vector (cf. Jan algorithm), denoted chi_tr.

  3) Find the expected result of track-based alignment, subject to the given constraints. Formally this corresponds to finding
       chi_exp = chi_tr + I * Lambda
     where the I matrix contains the "inaccessible alignment modes" as columns and Lambda is a vector of parameters. The constraints are given by
       C^T * chi_exp = V
     Since the problem may be generally overconstrained, it is better to formulate it as search for Lambda which minimises
       |C^ * chi_exp - V|^2
     This gives
       Lambda = (A^T * A)^-1 * A^T * b,  A = C^T * I,  b = V - C^T * chi_tr
  */

  results.clear();

  // determine dimension and offsets
  unsigned int dim = 0;
  vector<unsigned int> offsets;
  map<unsigned int, unsigned int> offset_map;
  for (unsigned int qci = 0; qci < task->quantityClasses.size(); qci++) {
    offsets.push_back(dim);
    offset_map[task->quantityClasses[qci]] = dim;
    dim += task->quantitiesOfClass(task->quantityClasses[qci]);
  }

  // collect true misalignments
  TVectorD chi_tr(dim);

  for (const auto &dit : task->geometry.getSensorMap()) {
    unsigned int detId = dit.first;

    const DetGeomDesc *real = gReal->sensor(detId);
    const DetGeomDesc *misal = gMisaligned->sensor(detId);

    // extract shift
    const auto shift = misal->translation() - real->translation();

    // extract rotation around z
    const auto rotation = misal->rotation() * real->rotation().Inverse();

    double r_xx, r_xy, r_xz;
    double r_yx, r_yy, r_yz;
    double r_zx, r_zy, r_zz;
    rotation.GetComponents(r_xx, r_xy, r_xz, r_yx, r_yy, r_yz, r_zx, r_zy, r_zz);

    if (std::abs(r_zz - 1.) > 1E-5)
      throw cms::Exception("PPS") << "IdealResult::Solve: only rotations about z are supported.";

    double rot_z = atan2(r_yx, r_xx);

    const auto &geom = task->geometry.get(detId);

    for (unsigned int qci = 0; qci < task->quantityClasses.size(); ++qci) {
      const auto &qc = task->quantityClasses[qci];
      signed int idx = task->getQuantityIndex(qc, detId);
      if (idx < 0)
        continue;

      double v = 0.;

      if (qc == AlignmentTask::qcShR1) {
        const auto &d = geom.getDirectionData(1);
        v = shift.x() * d.dx + shift.y() * d.dy + shift.z() * d.dz;
      }

      if (qc == AlignmentTask::qcShR2) {
        const auto &d = geom.getDirectionData(2);
        v = shift.x() * d.dx + shift.y() * d.dy + shift.z() * d.dz;
      }

      if (qc == AlignmentTask::qcRotZ)
        v = rot_z;

      chi_tr(offsets[qci] + idx) = v;
    }
  }

  // build list of "inaccessible" modes
  vector<TVectorD> inaccessibleModes;

  if (task->resolveShR) {
    TVectorD fm_ShX_gl(dim);
    fm_ShX_gl.Zero();
    TVectorD fm_ShX_lp(dim);
    fm_ShX_lp.Zero();
    TVectorD fm_ShY_gl(dim);
    fm_ShY_gl.Zero();
    TVectorD fm_ShY_lp(dim);
    fm_ShY_lp.Zero();

    for (const auto &sp : task->geometry.getSensorMap()) {
      CTPPSDetId senId(sp.first);
      const auto &geom = sp.second;

      if (senId.subdetId() == CTPPSDetId::sdTrackingStrip) {
        signed int qIndex = task->getQuantityIndex(AlignmentTask::qcShR2, senId);

        const double d2x = geom.getDirectionData(2).dx;
        const double d2y = geom.getDirectionData(2).dy;

        const auto &offset2 = offset_map[AlignmentTask::qcShR2];
        fm_ShX_gl(offset2 + qIndex) = d2x;
        fm_ShX_lp(offset2 + qIndex) = d2x * geom.z;
        fm_ShY_gl(offset2 + qIndex) = d2y;
        fm_ShY_lp(offset2 + qIndex) = d2y * geom.z;
      }

      if (senId.subdetId() == CTPPSDetId::sdTrackingPixel) {
        const signed int qIndex1 = task->getQuantityIndex(AlignmentTask::qcShR1, senId);
        const signed int qIndex2 = task->getQuantityIndex(AlignmentTask::qcShR2, senId);

        const double d1x = geom.getDirectionData(1).dx;
        const double d1y = geom.getDirectionData(1).dy;
        const double d2x = geom.getDirectionData(2).dx;
        const double d2y = geom.getDirectionData(2).dy;

        const auto &offset1 = offset_map[AlignmentTask::qcShR1];
        fm_ShX_gl(offset1 + qIndex1) = d1x;
        fm_ShX_lp(offset1 + qIndex1) = d1x * geom.z;
        fm_ShY_gl(offset1 + qIndex1) = d1y;
        fm_ShY_lp(offset1 + qIndex1) = d1y * geom.z;

        const auto &offset2 = offset_map[AlignmentTask::qcShR2];
        fm_ShX_gl(offset2 + qIndex2) = d2x;
        fm_ShX_lp(offset2 + qIndex2) = d2x * geom.z;
        fm_ShY_gl(offset2 + qIndex2) = d2y;
        fm_ShY_lp(offset2 + qIndex2) = d2y * geom.z;
      }
    }

    inaccessibleModes.push_back(fm_ShX_gl);
    inaccessibleModes.push_back(fm_ShX_lp);
    inaccessibleModes.push_back(fm_ShY_gl);
    inaccessibleModes.push_back(fm_ShY_lp);
  }

  if (task->resolveRotZ) {
    TVectorD fm_RotZ_gl(dim);
    fm_RotZ_gl.Zero();
    TVectorD fm_RotZ_lp(dim);
    fm_RotZ_lp.Zero();

    for (const auto &sp : task->geometry.getSensorMap()) {
      CTPPSDetId senId(sp.first);
      const auto &geom = sp.second;

      for (int m = 0; m < 2; ++m) {
        double rho = 0.;
        TVectorD *fmp = nullptr;
        if (m == 0) {
          rho = 1.;
          fmp = &fm_RotZ_gl;
        }
        if (m == 1) {
          rho = geom.z;
          fmp = &fm_RotZ_lp;
        }
        TVectorD &fm = *fmp;

        const signed int qIndex = task->getQuantityIndex(AlignmentTask::qcRotZ, senId);
        const auto &offset = offset_map[AlignmentTask::qcRotZ];
        fm(offset + qIndex) = rho;

        const double as_x = -rho * geom.sy;
        const double as_y = +rho * geom.sx;

        if (senId.subdetId() == CTPPSDetId::sdTrackingStrip) {
          const double d2x = geom.getDirectionData(2).dx;
          const double d2y = geom.getDirectionData(2).dy;

          const signed int qIndex2 = task->getQuantityIndex(AlignmentTask::qcShR2, senId);
          const auto &offset2 = offset_map[AlignmentTask::qcShR2];
          fm(offset2 + qIndex2) = d2x * as_x + d2y * as_y;
        }

        if (senId.subdetId() == CTPPSDetId::sdTrackingPixel) {
          const double d1x = geom.getDirectionData(1).dx;
          const double d1y = geom.getDirectionData(1).dy;
          const double d2x = geom.getDirectionData(2).dx;
          const double d2y = geom.getDirectionData(2).dy;

          const signed int qIndex1 = task->getQuantityIndex(AlignmentTask::qcShR1, senId);
          const auto &offset1 = offset_map[AlignmentTask::qcShR1];
          fm(offset1 + qIndex1) = d1x * as_x + d1y * as_y;

          const signed int qIndex2 = task->getQuantityIndex(AlignmentTask::qcShR2, senId);
          const auto &offset2 = offset_map[AlignmentTask::qcShR2];
          fm(offset2 + qIndex2) = d2x * as_x + d2y * as_y;
        }
      }
    }

    inaccessibleModes.push_back(fm_RotZ_gl);
    inaccessibleModes.push_back(fm_RotZ_lp);
  }

  // build matrices and vectors
  TMatrixD C(dim, constraints.size());
  TVectorD V(constraints.size());
  for (unsigned int i = 0; i < constraints.size(); i++) {
    V(i) = constraints[i].val;

    unsigned int offset = 0;
    for (auto &quantityClass : task->quantityClasses) {
      const TVectorD &cv = constraints[i].coef.find(quantityClass)->second;
      for (int k = 0; k < cv.GetNrows(); k++) {
        C[offset][i] = cv[k];
        offset++;
      }
    }
  }

  TMatrixD I(dim, inaccessibleModes.size());
  for (unsigned int i = 0; i < inaccessibleModes.size(); ++i) {
    for (int j = 0; j < inaccessibleModes[i].GetNrows(); ++j)
      I(j, i) = inaccessibleModes[i](j);
  }

  // determine expected track-based alignment result
  TMatrixD CT(TMatrixD::kTransposed, C);
  TMatrixD CTI(CT * I);

  const TMatrixD &A = CTI;
  TMatrixD AT(TMatrixD::kTransposed, A);
  TMatrixD ATA(AT * A);
  TMatrixD ATA_inv(TMatrixD::kInverted, ATA);

  TVectorD b = V - CT * chi_tr;

  TVectorD La(ATA_inv * AT * b);

  TVectorD chi_exp(chi_tr + I * La);

  // save result
  for (const auto &dit : task->geometry.getSensorMap()) {
    AlignmentResult r;

    for (unsigned int qci = 0; qci < task->quantityClasses.size(); ++qci) {
      const auto &qc = task->quantityClasses[qci];
      const auto idx = task->getQuantityIndex(qc, dit.first);
      if (idx < 0)
        continue;

      const auto &v = chi_exp(offsets[qci] + idx);

      switch (qc) {
        case AlignmentTask::qcShR1:
          r.setShR1(v);
          break;
        case AlignmentTask::qcShR2:
          r.setShR2(v);
          break;
        case AlignmentTask::qcShZ:
          r.setShZ(v);
          break;
        case AlignmentTask::qcRotZ:
          r.setRotZ(v);
          break;
      }
    }

    results[dit.first] = r;
  }

  return 0;
}

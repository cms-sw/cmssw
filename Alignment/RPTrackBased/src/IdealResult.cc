/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/


#include "Alignment/RPTrackBased/interface/IdealResult.h"
#include "Alignment/RPTrackBased/interface/AlignmentTask.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Alignment/RPTrackBased/interface/MatrixTools.h"

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TGraph.h"
#include "TFile.h"
#include "Math/GenVector/EulerAngles.h"

using namespace std;

//#define DEBUG


IdealResult::IdealResult(const edm::ParameterSet& gps, AlignmentTask *_t) : AlignmentAlgorithm(gps, _t)
{
  const edm::ParameterSet &ps = gps.getParameterSet("IdealResult");
  useExtendedConstraints = ps.getParameter<bool>("useExtendedConstraints"); // TODO: use the same parameters as Jan algorithm
}

//----------------------------------------------------------------------------------------------------

void IdealResult::Begin(const edm::EventSetup& iSetup)
{
  iSetup.get<VeryForwardRealGeometryRecord>().get(gReal);
  iSetup.get<VeryForwardMisalignedGeometryRecord>().get(gMisaligned);
}

//----------------------------------------------------------------------------------------------------

vector<SingularMode> IdealResult::Analyze()
{
  vector<SingularMode> dummy;
  return dummy;
}

//----------------------------------------------------------------------------------------------------

unsigned int IdealResult::Solve(const std::vector<AlignmentConstraint> &constraints,
  RPAlignmentCorrections &result, TDirectory *dir)
{
  printf(">> IdealResult::Solve\n\tvalues in mm and rad\n");
  result.Clear();

#if 0
  // extract full correction vector
  TVectorD *Fc = new TVectorD[task->quantityClasses.size()];
  unsigned int dim = 0;
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    Fc[i].ResizeTo((task->quantityClasses[i] == AlignmentTask::qcRPShZ) ? task->geometry.RPs() : task->geometry.Detectors());
    Fc[i].Zero();
    dim += Fc[i].GetNrows();
  }
#endif

  const unsigned int D = task->geometry.Detectors();
  const unsigned int R = task->geometry.RPs();
  TVectorD F_ShX(D);
  TVectorD F_ShY(D);
  TVectorD F_ShZ(D);
  TVectorD F_RPShZ(R);
  TVectorD F_RotZ(D);
  
  TVectorD ca_x(D);
  TVectorD ca_y(D);

  TVectorD da_x(D);
  TVectorD da_y(D);

  // collect true misalignments
  for (AlignmentGeometry::const_iterator dit = task->geometry.begin(); dit != task->geometry.end(); ++dit) {
      unsigned int rawId = TotemRPDetId::decToRawId(dit->first);
      
      DetGeomDesc *real = gReal->GetDetector(rawId);
      DetGeomDesc *misal = gMisaligned->GetDetector(rawId);

      // resolve shift
      DDTranslation shift = misal->translation() - real->translation();

      // resolve rotation around z
      DDRotationMatrix rotation = misal->rotation() * real->rotation().Inverse();
      ROOT::Math::EulerAngles ea(rotation);
      if (fabs(ea.Theta()) > 1E-5 )
        throw cms::Exception("StraightTrackAlignmentIdealResult::Calculate") << 
          "Don't know how to handle general rotations yet. Here's the troublesome rotation: " << rotation << endl;
      double rot_z = ea.Phi() + ea.Psi();
      rot_z -= floor((rot_z - M_PI) / 2. / M_PI)* 2. * M_PI;
      rot_z = -rot_z;   // sign/convention incompatibility between EulerAngles and RPAlignmentCorrection classes

      const unsigned int mi = dit->second.matrixIndex;
      const unsigned int rmi = dit->second.rpMatrixIndex;
      F_ShX[mi] = shift.x();
      F_ShY[mi] = shift.y();
      F_ShZ[mi] = shift.z();
      F_RPShZ[rmi] = shift.z();
      F_RotZ[mi] = rot_z;
      
      ca_x[mi] = real->translation().x();
      ca_y[mi] = real->translation().y();

#ifdef DEBUG
      printf("%2u %u | %+.2E %+.2E %+.2E | %+.8E %+.8E %+.2E\n", mi, rmi, F_ShX[mi], 
        F_ShY[mi], F_ShZ[mi], ea.Phi(), ea.Psi(), F_RotZ[mi]);
#endif

      // actual (misaligned) readout direction
      CLHEP::Hep3Vector da = gMisaligned->LocalToGlobalDirection(rawId, CLHEP::Hep3Vector(0., 1., 0.));
      da_x[mi] = da.x();
      da_y[mi] = da.y();
      // TODO: check z component - it shall be 0

#if 0
      // fill the vectors
      for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
        switch (task->quantityClasses[i]) {
          case AlignmentTask::qcShR: Fc[i][dit->second.matrixIndex] = sh_r; break;
          case AlignmentTask::qcShZ: Fc[i][dit->second.matrixIndex] = sh_z; break;
          case AlignmentTask::qcRPShZ: Fc[i][dit->second.rpMatrixIndex] = sh_z; break; // TODO taking avarage would be more "correct" that sh_z of the last plane
          case AlignmentTask::qcRotZ: Fc[i][dit->second.matrixIndex] = rot_z; break;
        }
      }
#endif
  }

#if 0
  // build full F vector
  TVectorD F(dim);
  unsigned int offset = 0;
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    F.SetSub(offset, Fc[i]);
    offset += Fc[i].GetNrows();
  }
#endif

  // remove extended constraints, if desired
  // separate rotation constraints
  vector<AlignmentConstraint> con_rot, con_nrot;
  for (unsigned int i = 0; i < constraints.size(); i++) {
    if (!useExtendedConstraints && constraints[i].extended)
      continue;

    if (constraints[i].forClass == AlignmentTask::qcRotZ)
      con_rot.push_back(constraints[i]);
    else
      con_nrot.push_back(constraints[i]);
  }

  // assumed singular modes = homogeneous constraints
  vector<AlignmentConstraint> asModes;
  task->BuildHomogeneousConstraints(asModes);

  if (asModes.size() != constraints.size()) {
    printf("ERROR in IdealResult::Solve > Number of constraints (%lu) is different that the number of assumed singular modes (%lu).\n",
      constraints.size(), asModes.size());
    return 1;
  }

  // separate rotation constraints for assumed singular modes
  vector<AlignmentConstraint> asModes_rot, asModes_nrot;
  for (unsigned int i = 0; i < asModes.size(); i++) {
    if (!useExtendedConstraints && asModes[i].extended)
      continue;

    if (asModes[i].forClass == AlignmentTask::qcRotZ)
      asModes_rot.push_back(asModes[i]);
    else
      asModes_nrot.push_back(asModes[i]);
  }

  // treat rotations
  TMatrixD Cr(task->QuantitiesOfClass(AlignmentTask::qcRotZ), con_rot.size());
  TMatrixD Er(task->QuantitiesOfClass(AlignmentTask::qcRotZ), con_rot.size());
  TVectorD Vr(con_rot.size());

  for (unsigned int c = 0; c < con_rot.size(); c++) {
    for (unsigned int i = 0; i < task->QuantitiesOfClass(AlignmentTask::qcRotZ); i++) {
      Cr(i, c) = con_rot[c].coef[AlignmentTask::qcRotZ][i];
      Vr(c) = con_rot[c].val;
      Er(i, c) = asModes_rot[c].coef[AlignmentTask::qcRotZ][i];
    }
  }

#ifdef DEBUG
  printf("* rotations\n");
  Print(Er, "Er");
  Print(Cr, "Cr");
#endif

  TMatrixD CrT(TMatrixD::kTransposed, Cr);
  TMatrixD CrTErI(Cr, TMatrixD::kTransposeMult, Er);
  const TVectorD &Fr = F_RotZ;
  TVectorD Fr_fil(Fr);
  TVectorD Fr_sol(Fr);
  TVectorD Vr_fil(con_rot.size());

  if (Cr.GetNcols() > 0) {
    CrTErI.Invert();
    Vr_fil = CrTErI * CrT * Fr;
    Fr_fil -= Er * Vr_fil; 
    Fr_sol = Fr_fil + Er * CrTErI * Vr; 
  }

#ifdef DEBUG
  printf("\n* constraint values:\n    constraint                       Vr_fil    Vr\n");
  for (int i = 0; i < Vr_fil.GetNrows(); i++)
    printf("    %2u (%25s)  %+10.2E  %+10.2E\n", i, con_rot[i].name.c_str(), Vr_fil[i], Vr[i]);

  printf("\n* Fr vectors (in um and mrad):\n    idx    F (full)        F_fil    F_sol        F_sol - F\n");
  for (int i = 0; i < Fr.GetNrows(); i++) {
    printf("    %2u   %+10.2f   %+10.2f   %+10.2f   %+10.2f\n", i, Fr[i]*1E3, Fr_fil[i]*1E3, Fr_sol[i]*1E3, (Fr_sol[i] - Fr[i])*1E3);
  }
#endif

  // calculate (rotation corrected) shift in readout direction
  TVectorD F_ShR(D);
  for (unsigned int i = 0; i < D; i++) {
    F_ShR[i] = da_x[i]*F_ShX[i] + da_y[i]*F_ShY[i];
    double drho = Fr_sol[i] - Fr[i];
    double ImC = 1. - cos(drho);
    double S = -sin(drho);
    double correction = da_x[i] * (ImC * ca_x[i] + S * ca_y[i]) + da_y[i] * (ImC * ca_y[i] - S * ca_x[i]);
#ifdef DEBUG
    printf("%i | %+.2E %+.2E | %+.2E %+.2E\n", i, da_x[i], da_y[i], F_ShR[i], correction);
#endif
    F_ShR[i] += correction;
  }

  // build vector of misalignments other than rotations 
  unsigned int dim = 0;
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    if (task->quantityClasses[i] == task->qcRotZ)
      continue;
    dim += task->QuantitiesOfClass(task->quantityClasses[i]);
  }
  printf("dim = %i\n", dim);
 
  TVectorD Fnr(dim);
  unsigned int offset = 0;
  map<unsigned int, unsigned int> offsets;
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    TVectorD *Fc = NULL;

    switch (task->quantityClasses[i]) {
      case AlignmentTask::qcShR: Fc = &F_ShR; break;
      case AlignmentTask::qcShZ: Fc = &F_ShZ; break;
      case AlignmentTask::qcRPShZ: Fc = &F_RPShZ; break;
      case AlignmentTask::qcRotZ: continue;
    }

    if (!Fc)
      continue;
    
    offsets[i] = offset;
    Fnr.SetSub(offset, *Fc);
    offset += Fc->GetNrows();
  }

  // build C matrix and vector of (imposed) constraint values
  TMatrixD Cnr(dim, con_nrot.size());
  TVectorD Vnr(con_nrot.size());
  for (unsigned int i = 0; i < con_nrot.size(); i++) {
    offset = 0;
    Vnr[i] = con_nrot[i].val;
    for (unsigned int j = 0; j < task->quantityClasses.size(); j++) {
      if (task->quantityClasses[j] == task->qcRotZ)
        continue;

      const TVectorD &cv = con_nrot[i].coef.find(task->quantityClasses[j])->second;
      for (int k = 0; k < cv.GetNrows(); k++) {
        Cnr[offset][i] = cv[k];
        offset++;
      }
    }
  }
  
  // build E matrix (assumed singular vectors of S)
  TMatrixD Enr(dim, asModes_nrot.size());
  for (unsigned int i = 0; i < asModes_nrot.size(); i++) {
    offset = 0;
    for (unsigned int j = 0; j < task->quantityClasses.size(); j++) {
      if (task->quantityClasses[j] == task->qcRotZ)
        continue;

      const TVectorD &hcv = asModes_nrot[i].coef.find(task->quantityClasses[j])->second;
      for (int k = 0; k < hcv.GetNrows(); k++) {
        Enr[offset][i] = hcv[k];
        offset++;
      }
    }
  }
  
#ifdef DEBUG
  printf("* shifts\n");
  Print(Cnr, "Cnr");
  Print(Enr, "Enr");
#endif

  // build projector to ideal result, calculate constraint values
  TMatrixD CnrT(TMatrixD::kTransposed, Cnr);
  TMatrixD CnrTEnrI(Cnr, TMatrixD::kTransposeMult, Enr);
  TVectorD Fnr_fil(Fnr);
  TVectorD Fnr_sol(Fnr);
  TVectorD Vnr_fil(con_nrot.size());

  if (Cnr.GetNcols() > 0) {
    CnrTEnrI.Invert();
    Vnr_fil = CnrTEnrI * CnrT * Fnr; // 
    Fnr_fil -= Enr * Vnr_fil;  // 
    Fnr_sol = Fnr_fil + Enr * CnrTEnrI * Vnr; 
  }
  
#ifdef DEBUG
  printf("\n* constraint values:\n    constraint                       Vnr_fil    Vnr\n");
  for (int i = 0; i < Vnr_fil.GetNrows(); i++)
    printf("    %2u (%25s)  %+10.2E  %+10.2E\n", i, constraints[i].name.c_str(), Vnr_fil[i], Vnr[i]);

  printf("\n* Fnr vectors (in um and mrad):\n    idx    F (full)        F_fil    F_sol        F_sol - F\n");
  for (int i = 0; i < Fnr.GetNrows(); i++) {
    printf("    %2u   %+10.2f   %+10.2f   %+10.2f   %+10.2f\n", i, Fnr[i]*1E3, Fnr_fil[i]*1E3, Fnr_sol[i]*1E3, (Fnr_sol[i] - Fnr[i])*1E3);
  }
#endif
  
  // save result
  for (AlignmentGeometry::const_iterator dit = task->geometry.begin(); dit != task->geometry.end(); ++dit) {
    RPAlignmentCorrection r;

    for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
      unsigned idx = (task->quantityClasses[i] != AlignmentTask::qcRPShZ) ? dit->second.matrixIndex : dit->second.rpMatrixIndex;
      const double &v = (task->quantityClasses[i] != AlignmentTask::qcRotZ) ? Fnr_sol[offsets[i] + idx] : Fr_sol[idx];
      switch (task->quantityClasses[i]) {
        case AlignmentTask::qcShR: r.SetTranslationR(v); break;
        case AlignmentTask::qcShZ: r.SetTranslationZ(v); break;
        case AlignmentTask::qcRPShZ: r.SetTranslationZ(v); break;
        case AlignmentTask::qcRotZ: r.SetRotationZ(v); break;
      }
    }

    result.SetSensorCorrection(dit->first, r);
  }

#if 0
  // save graphs
  string graphFileName = "idealGraphs.root";
  if (!graphFileName.empty()) {
    TFile *f = new TFile(graphFileName.c_str(), "recreate");

    TGraph** gUV = new TGraph*[offsets.size()];
    TGraph** gU = new TGraph*[offsets.size()];
    TGraph** gV = new TGraph*[offsets.size()];
    for (unsigned int i = 0; i < offsets.size(); i++) {
      const string &qcLabel  = AlignmentTask::QuantityClassTag(task->quantityClasses[i]);

      char buf[50];
      sprintf(buf, "%s_uv", qcLabel.c_str()); gUV[i] = new TGraph(); gUV[i]->SetName(buf);
      sprintf(buf, "%s_u", qcLabel.c_str()); gU[i] = new TGraph(); gU[i]->SetName(buf);
      sprintf(buf, "%s_v", qcLabel.c_str()); gV[i] = new TGraph(); gV[i]->SetName(buf);
    }

    for (AlignmentGeometry::const_iterator dit = task->geometry.begin(); dit != task->geometry.end(); ++dit) {
      for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
        unsigned idx = (task->quantityClasses[i] != AlignmentTask::qcRPShZ) ? dit->second.matrixIndex : dit->second.rpMatrixIndex;
        unsigned int fi = offsets[i] + idx;
        const double &v = F_sol[fi];
        const double &z = dit->second.z;
        gUV[i]->SetPoint(gUV[i]->GetN(), z, v);
        if (dit->second.isU)
          gU[i]->SetPoint(gU[i]->GetN(), z, v);
        else
          gV[i]->SetPoint(gV[i]->GetN(), z, v);
      }
    }

    for (unsigned int i = 0; i < offsets.size(); i++) {
      gUV[i]->Write();
      gU[i]->Write();
      gV[i]->Write();
    }

    delete f;
    delete [] gUV;
    delete [] gU;
    delete [] gV;
  }
#endif

  return 0;
}


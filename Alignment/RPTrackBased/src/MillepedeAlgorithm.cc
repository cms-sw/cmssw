/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/RPTrackBased/interface/MillepedeAlgorithm.h"
#include "Alignment/RPTrackBased/interface/AlignmentTask.h"

#include "TMatrixDSymEigen.h"
//#include "TDecompSVD.h"

#include <cmath>

using namespace std;
using namespace edm;

MillepedeAlgorithm::MillepedeAlgorithm(const edm::ParameterSet& ps, AlignmentTask *_t) :
  AlignmentAlgorithm(ps, _t),
  workingDir(ps.getParameterSet("MillepedeAlgorithm").getParameter<string>("workingDir")),
  mille(NULL)
{
  if (task->resolveRPShZ)
    throw cms::Exception("MillepedeAlgorithm::MillepedeAlgorithm") << "RP shifts in z not yet implemented";
}

//----------------------------------------------------------------------------------------------------

MillepedeAlgorithm::~MillepedeAlgorithm()
{
}

//----------------------------------------------------------------------------------------------------

void MillepedeAlgorithm::Begin(const edm::EventSetup&)
{
  string dataFile = workingDir + "/mp.input";
   mille = new Mille(dataFile.c_str());
}

//----------------------------------------------------------------------------------------------------

void MillepedeAlgorithm::Feed(const HitCollection &selection, const LocalTrackFit &trackFit,
  const LocalTrackFit&)
{
  if (verbosity > 9)
    printf(">> MillepedeAlgorithm::Feed\n");

  // prepare fit - make z0 compatible
  double hax = trackFit.ax;
  double hay = trackFit.ay;
  double hbx = trackFit.bx + trackFit.ax * (task->geometry.z0 - trackFit.z0);
  double hby = trackFit.by + trackFit.ay * (task->geometry.z0 - trackFit.z0);

  for (HitCollection::const_iterator it = selection.begin(); it != selection.end(); ++it) {
    unsigned int id = it->id;
    DetGeometry &d = task->geometry[id];

    double hx = hax * d.z + hbx;  // in mm
    double hy = hay * d.z + hby;
    double C = d.dx, S = d.dy;
    double m = it->position + d.s;  // in mm

//    float derLc[4] = {d.z*C, C, d.z*S, S};
    float derLc[4];
    derLc[0] = d.z*C;
    derLc[1] = C;
	derLc[2] = d.z*S;
	derLc[3] = S;

    float derGl[3] = { 0., 0., 0. };
    int label[3] = { 0, 0, 0 };

    double cf_shr = -1.;
    double cf_shz = hax*C + hay*S;
    double cf_rotz = (hx - d.sx)*(-S) + (hy - d.sy)*C;

    unsigned int idx = 0;
    if (task->resolveShR) { derGl[idx] = cf_shr; label[idx] = AlignmentTask::qcShR*10000 + id; idx++; }
    if (task->resolveShZ) { derGl[idx] = cf_shz; label[idx] = AlignmentTask::qcShZ*10000 + id; idx++; }
    if (task->resolveRotZ) { derGl[idx] = cf_rotz; label[idx] = AlignmentTask::qcRotZ*10000 + id; idx++; }
    // TODO RPShZ

    mille->mille(4, derLc, idx, derGl, label, m, it->sigma);
  }

  mille->end();
}

//----------------------------------------------------------------------------------------------------

vector<SingularMode> MillepedeAlgorithm::Analyze()
{
  // close mille file
  delete mille;
  mille = NULL;

  vector<SingularMode> sm;
  return sm;
}

//----------------------------------------------------------------------------------------------------

unsigned int MillepedeAlgorithm::Solve(const std::vector<AlignmentConstraint> &constraints,
  RPAlignmentCorrections &result, TDirectory *dir)
{
  printf(">> MillepedeAlgorithm::Solve\n");
  result.Clear();

  // go to working directory
  char cwd[200];
  getcwd(cwd, 200);
  chdir(workingDir.c_str());

  // create steer file
  FILE *f;
  f = fopen("mp.steer", "w");
  fprintf(f, "Cfiles\n");
  fprintf(f, "mp.input\n\n");

  for (unsigned int i = 0; i < constraints.size(); ++i) {
    fprintf(f, "Constraint %E\n", constraints[i].val);

    for (unsigned int j = 0; j < task->geometry.size(); ++j) {
      unsigned int id = task->geometry.MatrixIndexToDetId(j);
      if (task->resolveShR) fprintf(f, "%u %E\n", AlignmentTask::qcShR*10000 + id, constraints[i].coef.find(AlignmentTask::qcShR)->second[j]);
      if (task->resolveShZ) fprintf(f, "%u %E\n", AlignmentTask::qcShZ*10000 + id, constraints[i].coef.find(AlignmentTask::qcShZ)->second[j]);
      if (task->resolveRotZ) fprintf(f, "%u %E\n", AlignmentTask::qcRotZ*10000 + id, constraints[i].coef.find(AlignmentTask::qcRotZ)->second[j]);
      // TODO RPShZ
    }

    fprintf(f, "\n");
  }

  fprintf(f, "method diagonalization 5 0.1\n");
  fprintf(f, "end\n");

  fclose(f);

  // run pede
  int ret = system("pede mp.steer > mp.dump");
  if (ret)
    throw cms::Exception("StraightTrackAlignment") << ">> Problems with running Pede." << endl;


  // read pede results
  fstream ff("millepede.res");
  string line;
  getline(ff, line);

  while (!ff.eof() && !ff.fail()) {
    unsigned int label;
    float value;
    
    getline(ff, line);
    istringstream iss(line);
    iss >> label >> value;
    if (!iss.fail()) {
      unsigned int det = label % 10000, type = label / 10000;
      switch (type) {
        case AlignmentTask::qcShR: result.GetSensorCorrection(det).SetTranslationR(value); break;
        case AlignmentTask::qcShZ: result.GetSensorCorrection(det).SetTranslationZ(value); break;
        case AlignmentTask::qcRotZ: result.GetSensorCorrection(det).SetRotationZ(value); break;
                                    // TODO RPShZ is missing
      }
    }
  }

  ff.close();

  // read pede eigenvalues
  ff.clear();
  ff.open("mp.dump");

  map<int, double> eigVal;

  while (!ff.eof() && !ff.fail()) {
    getline(ff, line);
    if (line.find("Eigenvector") == string::npos)
      continue;

    istringstream iss(line);
    string s1, s2, s3;
    int idx;
    double value;
    iss >> s1 >> idx >> s2 >> s3 >> value;
    if (!iss.fail())
      eigVal[idx] = value;
  }

  ff.close();

  // print out
  printf("\n* Millepede eigenvalues (CS)\n");
  for (map<int, double>::iterator it = eigVal.begin(); it != eigVal.end(); ++it)
      printf("\t%i\t%+.2E\n", it->first, it->second);

  // return to original directory
  chdir(cwd);

  // success
  return 0;
}

//----------------------------------------------------------------------------------------------------

void MillepedeAlgorithm::End()
{
  // TODO: remove temporary files
}


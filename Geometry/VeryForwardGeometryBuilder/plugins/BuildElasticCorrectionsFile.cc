/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/RPAlignmentCorrectionsDataSequence.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "DataFormats/TotemRPDetId/interface/TotemRPDetId.h"

#include <string>

class BuildElasticCorrectionsFile : public edm::one::EDAnalyzer<>
{
  public:
    BuildElasticCorrectionsFile(const edm::ParameterSet &ps); 
    ~BuildElasticCorrectionsFile() {}

  private:
    std::string inputFileName, outputFileName;

    virtual void beginRun(edm::Run const&, edm::EventSetup const&);
    virtual void analyze(const edm::Event &e, const edm::EventSetup &es) {}
    virtual void endJob() {}

    void ProcessOneStation(unsigned int id, double N_a, double N_b, double N_c,
      double F_a, double F_b, double F_c, RPAlignmentCorrectionsData &corr, const TotemRPGeometry &geom);

    void ProcessOnePot(unsigned int id, double a, double b, double c,
      RPAlignmentCorrectionsData &corr, TotemRPGeometry const &geom);
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

BuildElasticCorrectionsFile::BuildElasticCorrectionsFile(const ParameterSet &ps) :
  inputFileName(ps.getParameter<string>("inputFileName")),
  outputFileName(ps.getParameter<string>("outputFileName"))
{
}

//----------------------------------------------------------------------------------------------------

void BuildElasticCorrectionsFile::beginRun(edm::Run const&, edm::EventSetup const& es)
{
  // get geometry (including pre-elastic alignments)
  ESHandle<TotemRPGeometry> geom;
  es.get<VeryForwardRealGeometryRecord>().get(geom);

  // open input file
  FILE *inF = fopen(inputFileName.c_str(), "r");
  if (!inF)
    throw cms::Exception("BuildElasticCorrectionsFile") << "Can't open file `" << inputFileName << "'." << endl;

  // prepare output
  RPAlignmentCorrectionsDataSequence sequence;

  // process input data
  while (!feof(inF)) {
    unsigned long from, to;
    float L_F_a, L_F_b, L_F_c;
    float L_N_a, L_N_b, L_N_c;
    float R_N_a, R_N_b, R_N_c;
    float R_F_a, R_F_b, R_F_c;

    int count = fscanf(inF, "%lu,%lu,%E,%E,%E,%E,%E,%E,%E,%E,%E,%E,%E,%E", &from, &to,
      &L_F_a, &L_F_b, &L_F_c, &L_N_a, &L_N_b, &L_N_c,
      &R_N_a, &R_N_b, &R_N_c, &R_F_a, &R_F_b, &R_F_c);

    if (count >=0 && count != 14)
      throw cms::Exception("BuildElasticCorrectionsFile") << "Only " << count << " numbers in a row." << endl;

    if (count != 14)
      continue;

    RPAlignmentCorrectionsData corr;

    ProcessOneStation( 2, L_N_a*1E-3, L_N_b*1E-3, L_N_c*1E-3, L_F_a*1E-3, L_F_b*1E-3, L_F_c*1E-3, corr, *geom);
    ProcessOneStation(12, R_N_a*1E-3, R_N_b*1E-3, R_N_c*1E-3, R_F_a*1E-3, R_F_b*1E-3, R_F_c*1E-3, corr, *geom);

    sequence.Insert(from, to, corr);
  }

  fclose(inF);

  // save output
  sequence.WriteXMLFile(outputFileName, false, false, false, true, false, true);

#if 0
  // get parameters
  double z1 = ps.getUntrackedParameter<double>("z1");
  double z2 = ps.getUntrackedParameter<double>("z2");
  double de_x1 = ps.getUntrackedParameter<double>("de_x1");
  double de_x2 = ps.getUntrackedParameter<double>("de_x2");
  double de_y1 = ps.getUntrackedParameter<double>("de_y1");
  double de_y2 = ps.getUntrackedParameter<double>("de_y2");
  double de_rho1 = ps.getUntrackedParameter<double>("de_rho1");
  double de_rho2 = ps.getUntrackedParameter<double>("de_rho2");

  // calculate slopes and intercepts
  double a_x = (de_x2 - de_x1) / (z2 - z1), b_x = de_x1 - a_x * z1;
  double a_y = (de_y2 - de_y1) / (z2 - z1), b_y = de_y1 - a_y * z1;
  double a_rho = (de_rho2 - de_rho1) / (z2 - z1), b_rho = de_rho1 - a_rho * z1;

  if (z1 == z2)
    throw cms::Exception("z1 equals z2");
  
  // prepare output - expand input to sensor level
  RPAlignmentCorrectionsData output;
  for (RPAlignmentCorrectionsData::mapType::const_iterator it = input.GetSensorMap().begin();
      it != input.GetSensorMap().end(); ++it) { 
    unsigned int rawId = TotemRPDetId::decToRawId(it->first);
    CLHEP::Hep3Vector d = geom->LocalToGlobalDirection(rawId, CLHEP::Hep3Vector(0., 1., 0.));

    RPAlignmentCorrectionsData ac = input.GetFullSensorCorrection(it->first);
    ac.XYTranslationToReadout(d.x(), d.y());
    output.SetSensorCorrection(it->first, ac);
  }

  // apply singular-mode change
  printf("\tID      shift in x    shift in y    rotation about z\n");
  for (RPAlignmentCorrectionsData::mapType::const_iterator it = output.GetSensorMap().begin();
      it != output.GetSensorMap().end(); ++it) { 
    unsigned int rawId = TotemRPDetId::decToRawId(it->first);
    CLHEP::Hep3Vector d = geom->LocalToGlobalDirection(rawId, CLHEP::Hep3Vector(0., 1., 0.));
    double dx = d.x(), dy = d.y();
    CLHEP::Hep3Vector c = geom->GetDetTranslation(rawId);
    double cx = c.x(), cy = c.y(), z = c.z();

    double de_x = a_x * z + b_x;
    double de_y = a_y * z + b_y;
    double de_rho = a_rho * z + b_rho;

    printf("\t%u %+10.1f um %+10.1f um %+10.1f mrad\n", it->first, de_x*1E3, de_y*1E3, de_rho*1E3);
    //printf("\t\tcx=%e, cy=%E | dx=%E, dy=%E\n", cx, cy, dx, dy);

    double inc_s = +(dx*de_x + dy*de_y) - de_rho * (-dy*(cx + de_x) + dx*(cy + de_y));
    double inc_rho = de_rho;
    //printf("\t\t %E, %E\n", inc_s, inc_rho);
    
    RPAlignmentCorrectionsData &ac = output.GetSensorCorrection(it->first);
    ac.SetTranslationR(ac.sh_r() + inc_s, ac.sh_r_e());
    ac.SetRotationZ(ac.rot_z() + inc_rho, ac.rot_z_e());
    ac.ReadoutTranslationToXY(dx, dy);
  }

  // factorize alignments and write output
  vector<unsigned int> rps;
  unsigned int last_rp = 123456;
  for (RPAlignmentCorrectionsData::mapType::const_iterator it = input.GetSensorMap().begin();
      it != input.GetSensorMap().end(); ++it) {
      unsigned int rp = it->first/10;
      if (last_rp != rp) {
        rps.push_back(rp);
        last_rp = rp;
      }
  }
  AlignmentGeometry alGeom;
  AlignmentTask::BuildGeometry(rps, geom.product(), 0., alGeom);
  RPAlignmentCorrectionsData expanded, factored;
  output.FactorRPFromSensorCorrections(expanded, factored, alGeom);
  RPAlignmentCorrectionsMethods.WriteXMLFile(factored, ps.getUntrackedParameter<string>("outputFile"));


  // constants
  double z0 = ps.getUntrackedParameter("z0", 0.);
  

  // transverse shifts
  const ParameterSet &ps_shift_xy = ps.getParameterSet("shift_xy");
  bool perform = ps_shift_xy.getUntrackedParameter<bool>("perform");
  if (perform) {
    // process user input
    bool add = ps_shift_xy.getUntrackedParameter<bool>("add");

    double z_x1 = ps_shift_xy.getUntrackedParameter<double>("z_x1");
    double z_x2 = ps_shift_xy.getUntrackedParameter<double>("z_x2");
    double z_y1 = ps_shift_xy.getUntrackedParameter<double>("z_y1");
    double z_y2 = ps_shift_xy.getUntrackedParameter<double>("z_y2");
    double v_x1 = ps_shift_xy.getUntrackedParameter<double>("v_x1");
    double v_x2 = ps_shift_xy.getUntrackedParameter<double>("v_x2");
    double v_y1 = ps_shift_xy.getUntrackedParameter<double>("v_y1");
    double v_y2 = ps_shift_xy.getUntrackedParameter<double>("v_y2");

    double ax = (v_x2 - v_x1) / (z_x2 - z_x1);
    double ay = (v_y2 - v_y1) / (z_y2 - z_y1);
    double bx = v_x1 - ax*z_x1;
    double by = v_y1 - ay*z_y1;
    
    // determine the current coefficients to the singular modes
    unsigned int size = output.GetSensorMap().size();
    TMatrixD A(size, 4);
    TVectorD M(size);
    unsigned idx = 0;
    for (RPAlignmentCorrectionsData::mapType::const_iterator it = output.GetSensorMap().begin();
        it != output.GetSensorMap().end(); ++it) {
      unsigned int rawId = TotemRPDetId::decToRawId(it->first);

      CLHEP::Hep3Vector d = geom->LocalToGlobalDirection(rawId, CLHEP::Hep3Vector(0., 1., 0.));
      DDTranslation c = geom->GetDetector(rawId)->translation();
      double z = c.z() - z0;
      
      RPAlignmentCorrectionData &ac = output.GetSensorCorrection(it->first);
      double sh_r = ac.sh_r();
      
      A(idx, 0) = d.x()*z;
      A(idx, 1) = d.x();
      A(idx, 2) = d.y()*z;
      A(idx, 3) = d.y();
      M(idx) = sh_r;
      idx++;
    }


#endif
}

//----------------------------------------------------------------------------------------------------

void BuildElasticCorrectionsFile::ProcessOneStation(unsigned int id, double N_a, double N_b, double N_c,
      double F_a, double F_b, double F_c, RPAlignmentCorrectionsData &corr, const TotemRPGeometry &geom)
{
  
  ProcessOnePot(id*10 + 0, N_a, N_b, N_c, corr, geom);
  ProcessOnePot(id*10 + 1, N_a, N_b, N_c, corr, geom);

  // TODO: horizontals 

  ProcessOnePot(id*10 + 4, F_a, F_b, F_c, corr, geom);
  ProcessOnePot(id*10 + 5, F_a, F_b, F_c, corr, geom);
}

//----------------------------------------------------------------------------------------------------

void BuildElasticCorrectionsFile::ProcessOnePot(unsigned int rpId, double a, double b, double c,
  RPAlignmentCorrectionsData &corr, const TotemRPGeometry &geom)
{
  // distances in mm, angles in rad
  
  for (unsigned int i = 0; i < 10; i++) {
    unsigned int symId = rpId*10 + i;
    unsigned int rawId = TotemRPDetId::decToRawId(symId);
    CLHEP::Hep3Vector dc = geom.GetDetTranslation(rawId);

    double de_x = (cos(a) - 1.) * dc.x() - sin(a) * dc.y();
    double de_y = sin(a) * dc.x() + (cos(a) - 1.) * dc.y();
  
    double sh_x = de_x - b;
    double sh_y = de_y - c;
    double rot_z = a;

    corr.SetSensorCorrection(symId, RPAlignmentCorrectionData(sh_x, sh_y, 0., rot_z));
  }
}


DEFINE_FWK_MODULE(BuildElasticCorrectionsFile);


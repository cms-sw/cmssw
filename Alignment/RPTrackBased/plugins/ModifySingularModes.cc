/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Alignment/RPDataFormats/interface/RPAlignmentCorrections.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "DataFormats/TotemRPDetId/interface/TotemRPDetId.h"
#include "Alignment/RPTrackBased/interface/AlignmentTask.h"

#include "TMatrixD.h"
#include "TVectorD.h"

/**
 *\brief Modifies the alignment modes unconstrained by the track-based alignment.
 **/
class ModifySingularModes : public edm::EDAnalyzer
{
  public:
    ModifySingularModes(const edm::ParameterSet &ps); 
    ~ModifySingularModes() {}

  private:
    edm::ParameterSet ps;

    virtual void beginRun(edm::Run const&, edm::EventSetup const&);
    virtual void analyze(const edm::Event &e, const edm::EventSetup &es) {}
    virtual void endJob() {}
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

ModifySingularModes::ModifySingularModes(const ParameterSet &_ps) : ps(_ps)
{
}

//----------------------------------------------------------------------------------------------------

void ModifySingularModes::beginRun(edm::Run const&, edm::EventSetup const& es)
{
  printf(">> ModifySingularModes::beginRun\n");

  // get input alignments
  RPAlignmentCorrections input(ps.getUntrackedParameter<string>("inputFile"));
  //ESHandle<RPAlignmentCorrections> input;
  //es.get<VeryForwardRealGeometryRecord>().get(input);
  
  // get (base) geometry
  ESHandle<TotemRPGeometry> geom;
  es.get<VeryForwardRealGeometryRecord>().get(geom);

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
  RPAlignmentCorrections output;
  for (RPAlignmentCorrections::mapType::const_iterator it = input.GetSensorMap().begin();
      it != input.GetSensorMap().end(); ++it) { 
    unsigned int rawId = TotemRPDetId::decToRawId(it->first);
    CLHEP::Hep3Vector d = geom->LocalToGlobalDirection(rawId, CLHEP::Hep3Vector(0., 1., 0.));

    RPAlignmentCorrection ac = input.GetFullSensorCorrection(it->first);
    ac.XYTranslationToReadout(d.x(), d.y());
    output.SetSensorCorrection(it->first, ac);
  }

  // apply singular-mode change
  printf("\tID      shift in x    shift in y    rotation about z\n");
  for (RPAlignmentCorrections::mapType::const_iterator it = output.GetSensorMap().begin();
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
    
    RPAlignmentCorrection &ac = output.GetSensorCorrection(it->first);
    ac.SetTranslationR(ac.sh_r() + inc_s, ac.sh_r_e());
    ac.SetRotationZ(ac.rot_z() + inc_rho, ac.rot_z_e());
    ac.ReadoutTranslationToXY(dx, dy);
  }

  // factorize alignments and write output
  vector<unsigned int> rps;
  unsigned int last_rp = 123456;
  for (RPAlignmentCorrections::mapType::const_iterator it = input.GetSensorMap().begin();
      it != input.GetSensorMap().end(); ++it) {
      unsigned int rp = it->first/10;
      if (last_rp != rp) {
        rps.push_back(rp);
        last_rp = rp;
      }
  }
  AlignmentGeometry alGeom;
  vector<unsigned int> excludePlanes;
  AlignmentTask::BuildGeometry(rps, excludePlanes, geom.product(), 0., alGeom);
  RPAlignmentCorrections expanded, factored;
  output.FactorRPFromSensorCorrections(expanded, factored, alGeom);
  factored.WriteXMLFile(ps.getUntrackedParameter<string>("outputFile"));


#if 0
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
    for (RPAlignmentCorrections::mapType::const_iterator it = output.GetSensorMap().begin();
        it != output.GetSensorMap().end(); ++it) {
      unsigned int rawId = TotemRPDetId::decToRawId(it->first);

      CLHEP::Hep3Vector d = geom->LocalToGlobalDirection(rawId, CLHEP::Hep3Vector(0., 1., 0.));
      DDTranslation c = geom->GetDetector(rawId)->translation();
      double z = c.z() - z0;
      
      RPAlignmentCorrection &ac = output.GetSensorCorrection(it->first);
      double sh_r = ac.sh_r();
      
      A(idx, 0) = d.x()*z;
      A(idx, 1) = d.x();
      A(idx, 2) = d.y()*z;
      A(idx, 3) = d.y();
      M(idx) = sh_r;
      idx++;
    }

    TMatrixD AT(4, size);
    AT.Transpose(A);

    TMatrixD ATA(4, 4);
    ATA = AT * A;
    TMatrixD ATAi(4, 4);
    ATAi = ATA.Invert();

    TVectorD th(4);
    th = ATAi * AT * M;

    printf(">> fit: th0=%E, th1=%E, th2=%E, th3=%E\n", th[0], th[1], th[2], th[3]);

    printf(">> user: ax=%E, bx=%E, ay=%E, by=%E\n", ax, bx, ay, by);

    if (!add) {
      ax -= th[0];
      bx -= th[1];
      ay -= th[2];
      by -= th[3];
    }
    
    printf(">> user (- fit): ax=%E, bx=%E, ay=%E, by=%E\n", ax, bx, ay, by);
    
    // build new alignment corrections
    idx = 0;
    for (RPAlignmentCorrections::mapType::const_iterator it = output.GetSensorMap().begin();
        it != output.GetSensorMap().end(); ++it) {
      RPAlignmentCorrection &rc = output.GetSensorCorrection(it->first);
      double corr = A(idx, 0)*ax + A(idx, 1)*bx + A(idx, 2)*ay + A(idx, 3)*by;
      printf("* %3u, %4u, %E\n", idx, it->first, corr);
      rc.SetTranslationR(rc.sh_r() + corr);
      rc.ReadoutTranslationToXY(A(idx, 1), A(idx, 3));
      idx++;
    }
  }

  // shift in z
  // TODO
  const ParameterSet &ps_shift_z = ps.getParameterSet("shift_z");

  // RP shift in z
  // TODO?
  
  // rot about z
  const ParameterSet &ps_rot_z = ps.getParameterSet("rot_z");
  perform = rot_z.getUntrackedParameter<bool>("perform");
  if (perform) {

  }

#endif
}

DEFINE_FWK_MODULE(ModifySingularModes);


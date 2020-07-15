#include "VolumeGridTester.h"
#include "MagneticField/Interpolation/interface/MFGrid.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <map>

using namespace std;

namespace {
  float eps(float x, int ulp) {  // move x by ulp times the float numerical precision
    return x + std::numeric_limits<float>::epsilon() * std::fabs(x) * ulp;
  }
}  // namespace

bool VolumeGridTester::testInside() const {
  //   static string lastName("firstcall");
  //   if (lastName == volume_->name) return true; // skip multiple calls
  //   else lastName = volume_->name;

  const MFGrid* grid = dynamic_cast<const MFGrid*>(magProvider_);
  if (grid == nullptr) {
    cout << "VolumeGridTester: magProvider is not a MFGrid3D, cannot test it..." << endl << "expected ";
    return false;
  }

  bool result = true;
  const double tolerance = 0.03;

  //   cout << "The volume position is " << volume_->position() << endl;
  //   cout << "Is the volume position inside the volume? " << volume_->inside(volume_->position(), tolerance) << endl;

  Dimensions sizes = grid->dimensions();
  //   cout << "Grid has " << 3 << " dimensions "
  //        << " number of nodes is " << sizes.w << " " << sizes.h << " " << sizes.d << endl;

  size_t dumpCount = 0;
  for (int j = 0; j < sizes.h; j++) {
    for (int k = 0; k < sizes.d; k++) {
      for (int i = 0; i < sizes.w; i++) {
        MFGrid::LocalPoint lp = grid->nodePosition(i, j, k);
        // Check that grid point is inside its own volume
        if (!volume_->inside(lp, tolerance)) {
          result = false;
          if (++dumpCount < 2)
            dumpProblem(lp, tolerance);
          else
            return result;
        }

        // Check that points within numerical accuracy to grid boundary points are discoverable
        if (field_ != nullptr &&
            ((j == 0 || j == sizes.h - 1) || (k == 0 || k == sizes.d - 1) || (i == 0 || i == sizes.w - 1))) {
          GlobalPoint gp = volume_->toGlobal(lp);

          for (int nulp = -5; nulp <= 5; ++nulp) {
            result &= testFind(GlobalPoint(eps(gp.x(), nulp), gp.y(), gp.z()));
            result &= testFind(GlobalPoint(gp.x(), eps(gp.y(), nulp), gp.z()));
            result &= testFind(GlobalPoint(gp.x(), gp.y(), eps(gp.z(), nulp)));

            if (result == false)
              return result;
          }
        }
      }
    }
  }
  return result;
}

void VolumeGridTester::dumpProblem(const MFGrid::LocalPoint& lp, double tolerance) const {
  MFGrid::GlobalPoint gp(volume_->toGlobal(lp));
  cout << "ERROR: VolumeGridTester: Grid point " << lp << " (local) " << gp << " (global) " << gp.perp() << " "
       << gp.phi() << " (R,phi global) is not in its own volume!" << endl;

  const vector<VolumeSide>& faces = volume_->faces();
  for (vector<VolumeSide>::const_iterator v = faces.begin(); v != faces.end(); v++) {
    cout << " Volume face has position " << v->surface().position() << " side " << (int)v->surfaceSide() << " rotation "
         << endl
         << v->surface().rotation() << endl;

    Surface::Side side = v->surface().side(gp, tolerance);
    if (side != v->surfaceSide() && side != SurfaceOrientation::onSurface) {
      cout << " Wrong side: " << (int)side << " local position in surface frame " << v->surface().toLocal(gp) << endl;
    } else
      cout << " Correct side: " << (int)side << endl;
  }
}

bool VolumeGridTester::testFind(GlobalPoint gp) const {
  if (field_->isDefined(gp)) {
    MagVolume const* vol = field_->findVolume(gp);
    if (vol == nullptr) {
      cout << "ERROR: VolumeGridTester: No volume found! Global point: " << setprecision(8) << gp
           << " , at R= " << gp.perp() << ", Z= " << gp.z() << endl;
      return false;
    }
  }
  return true;
}

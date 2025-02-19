#include "VolumeGridTester.h"
#include "MagneticField/Interpolation/interface/MFGrid.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include <iostream>
#include <string>
#include <map>


using namespace std;


bool VolumeGridTester::testInside() const
{

//   static string lastName("firstcall");
//   if (lastName == volume_->name) return true; // skip multiple calls
//   else lastName = volume_->name;


  const MFGrid * grid = dynamic_cast<const MFGrid *>(magProvider_);
  if (grid == 0) {
    cout << "VolumeGridTester: magProvider is not a MFGrid3D, cannot test it..." << endl
	 << "expected ";
    return false;
  }

  bool result = true;
  const double tolerance = 0.03;
  
  cout << "The volume position is " << volume_->position() << endl;
  cout << "Is the volume position inside the volume? " 
       << volume_->inside( volume_->position(), tolerance) <<endl;

  Dimensions sizes = grid->dimensions();
  cout << "Grid has " << 3 << " dimensions " 
       << " number of nodes is " << sizes.w << " " << sizes.h << " " << sizes.d << endl;

  size_t dumpCount = 0;
  for (int j=0; j < sizes.h; j++) {
    for (int k=0; k < sizes.d; k++) {
      for (int i=0; i < sizes.w; i++) {
	MFGrid::LocalPoint lp = grid->nodePosition( i, j, k);
	if (! volume_->inside(lp, tolerance)) {
	  result = false;
	  if (++dumpCount < 2) dumpProblem( lp, tolerance);
	  else return result;
	}
      }
    }
  }
  return result;
}

void VolumeGridTester::dumpProblem( const MFGrid::LocalPoint& lp, double tolerance) const
{
  MFGrid::GlobalPoint gp( volume_->toGlobal(lp));
  cout << "Point " << lp << " (local) " 
       << gp << " (global) " << gp.perp() << " " << gp.phi()
       << " (R,phi global) not in volume!" << endl;

  const vector<VolumeSide>& faces = volume_->faces();
  for (vector<VolumeSide>::const_iterator v=faces.begin(); v!=faces.end(); v++) {
    cout << "Volume face has position " << v->surface().position() 
 	 << " side " << (int) v->surfaceSide() << " rotation " << endl
 	 << v->surface().rotation() << endl;

    Surface::Side side = v->surface().side( gp, tolerance);
    if ( side != v->surfaceSide() && side != SurfaceOrientation::onSurface) {
      cout << "Wrong side: " << (int) side 
	   << " local position in surface frame " << v->surface().toLocal(gp) << endl;
    }
    else cout << "Correct side: " << (int) side << endl;
  }
}

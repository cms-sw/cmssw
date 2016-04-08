#include "Math/GenVector/RotationZYX.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/PositionVector3D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/Vector3D.h"

#include <iostream>

using namespace std;

void rotation_test()
{
	ROOT::Math::RotationZYX rot(0., 0., 0.1);

	ROOT::Math::Rotation3D rotM(rot);
	double xx, xy, xz, yx, yy, yz, zx, zy, zz;
	rotM.GetComponents(xx, xy, xz, yx, yy, yz, zx, zy, zz);
	printf("%+.3f\t%+.3f\t%+.3f\n", xx, xy, xz);
	printf("%+.3f\t%+.3f\t%+.3f\n", yx, yy, yz);
	printf("%+.3f\t%+.3f\t%+.3f\n", zx, zy, zz);

	cout << rot << endl;
	
	cout << rotM << endl;

	ROOT::Math::XYZVector v(0., 1., 0.);

	cout << v << endl;
	cout << (rot * v) << endl;
}

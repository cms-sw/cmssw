#include "DataFormats/Math/interface/Angle0to2pi.h"

#include <iostream>
#include <math.h>
#include <ctime>
#include <chrono>


using namespace angle0to2pi;
using namespace std;
using namespace std::chrono;


template <class valType>
static int iterationTest(valType increm) {
	Angle0to2pi<valType> ang1 = 0.;
	const int iters = 123456789;
	steady_clock::time_point startTime = steady_clock::now();
	for (int cnt = 0; cnt < iters; ++cnt) {
		ang1 += increm;
	}
	steady_clock::time_point endTime = steady_clock::now();
	cout << "Angle0to2pi is  now " << ang1 << endl;
	duration<double> time_span = duration_cast<duration<double>>(endTime - startTime);
	cout << "Time diff is  " << time_span.count() << endl;
	valType plainAng = 0.;
	startTime = steady_clock::now();
	for (int cnt = 0; cnt < iters; ++cnt) {
		plainAng = make0to2pi(increm + plainAng);
	}
	endTime = steady_clock::now();
	cout << "plainAng  is  now " << plainAng << endl;
	duration<double> time_span2 = duration_cast<duration<double>>(endTime - startTime);
	cout << "Time diff is  " << time_span2.count() << endl;
	cout << "Ratio is " << (time_span.count() / time_span2.count()) << endl;
	if (ang1 != plainAng) {
		cout << "Angles should have come out the same but ang1 = " << ang1;
		cout << " and plainAng = " << plainAng << ". Test failure." << endl;
		return (1);
	}
	return (0);
}


int main() {
	{
		double angle = 3.3_pi;
		angle = make0to2pi(angle);
		if (angle < 0. || angle >= 2._pi) {
			cout << "Angle should be from 0-2pi but it is out of range = " << angle << endl;
			return (1);
		}
	}
	{
		long double angle = -3.3_pi;
		angle = make0to2pi(angle);
		if (angle < 0. || angle >= 2._pi) {
			cout << "Angle should be from 0-2pi but it is out of range = " << angle << endl;
			return (1);
		}
	}
	{ // Test with long double
		Angle0to2pi<long double> ang1(15._pi);
		cout << "Angle0to2pi that started as 15pi is " << ang1 << endl;
		cout << "In degrees " << ang1.degrees() << endl;
		constexpr long double testval = 14.999_pi;
		Angle0to2pi<long double> ang2 = ang1 - testval;
		ang1 -= testval;
		if  (ang1 != ang2) {
			cout << "angle1 = " << ang1 << " and angle2 = " << ang2;
			cout << " should be equal but they are not. Test failure." << endl;
			return (1);
		}
		if  (! ang1.nearZero()) {
			cout << "angle = " << ang1 << " should be considered nearly 0 but it is not.";
			cout << " Test failure." << endl;
			return (1);
		}
	}
	{ // Test with double
		Angle0to2pi<double> ang1(15._pi);
		cout << "Angle0to2pi that started as 15pi is " << ang1 << endl;
		cout << "In degrees " << ang1.degrees() << endl;
		constexpr double testval = 14.999_pi;
		Angle0to2pi<double> ang2 = ang1 - testval;
		ang1 -= testval;
		if  (ang1 != ang2) {
			cout << "angle1 = " << ang1 << " and angle2 = " << ang2;
			cout << " should be equal but they are not. Test failure." << endl;
			return (1);
		}
	}
	cout << "Test repeated large increment\n";
	if (iterationTest<float>(7.77_pi) == 1)
		return (1);
	cout << "Test repeated large decrement\n";
	if (iterationTest<double>(-7.77_pi) == 1)
		return (1);
	cout << "Test repeated small increment\n";
	if (iterationTest<long double>(1._deg) == 1)
		return (1);
	cout << "Test repeated small decrement\n";
	if (iterationTest<double>(-1._deg) == 1)
		return (1);
   return (0);
}


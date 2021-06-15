#include "DataFormats/GeometryVector/interface/Phi.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <chrono>

using namespace angle0to2pi;
using namespace Geom;
using namespace std;
using namespace reco;
using namespace std::chrono;

template <class valType>
inline constexpr valType useReduceRange(valType angle) {
  constexpr valType twoPi = 2._pi;
  angle = reduceRange(angle);
  if (angle < 0.)
    angle += twoPi;
  return angle;
}

template <class valType>
inline constexpr valType simpleMake0to2pi(valType angle) {
  constexpr valType twoPi = 2._pi;

  angle = fmod(angle, twoPi);
  if (angle < 0.)
    angle += twoPi;
  return angle;
}

template <class valType>
static int testSmall() {  // Test with long double
  Phi<valType, ZeroTo2pi> ang1(15.3_pi);
  cout << "Phi that started as 15.3 pi is " << ang1 << endl;
  cout << "In degrees " << ang1.degrees() << endl;
  constexpr valType testval = 15.2999_pi;
  Phi<valType, ZeroTo2pi> ang2 = ang1 - testval;
  ang1 -= testval;
  if (ang1 != ang2) {
    cout << "angle1 = " << ang1 << " and angle2 = " << ang2;
    cout << " should be equal but they are not. Test failure." << endl;
    return (1);
  }
  if (ang1.nearZero()) {
    cout << "angle = " << ang1 << " close enough to zero to be considered zero." << endl;
  } else {
    cout << "angle = " << ang1 << " should be considered nearly 0 but it is not.";
    cout << " Test failure." << endl;
    return (1);
  }
  return (0);
}

template <class valType>
static int iterationTest(valType increm) {
  Phi<valType, ZeroTo2pi> ang1 = 0.;
  const int iters = 1000 * 1000;
  steady_clock::time_point startTime = steady_clock::now();
  for (int cnt = 0; cnt < iters; ++cnt) {
    ang1 += increm;
  }
  steady_clock::time_point endTime = steady_clock::now();
  cout << "Phi after " << iters << " iterations is " << ang1 << endl;
  duration<double> time_span = duration_cast<duration<double>>(endTime - startTime);
  cout << "Time diff is  " << time_span.count() << endl;
  valType plainAng = 0.;
  startTime = steady_clock::now();
  for (int cnt = 0; cnt < iters; ++cnt) {
    plainAng = make0To2pi(increm + plainAng);
  }
  endTime = steady_clock::now();
  cout << "plainAng  is  now " << plainAng << endl;
  cout << "Base-type variable after " << iters << " iterations is " << plainAng << endl;
  duration<double> time_span2 = duration_cast<duration<double>>(endTime - startTime);
  cout << "Time diff is  " << time_span2.count() << endl;
  cout << "Ratio of class/base-type CPU time is " << (time_span.count() / time_span2.count()) << endl;
  if (ang1 != plainAng) {
    cout << "Angles should have come out the same but ang1 = " << ang1;
    cout << " and plainAng = " << plainAng << ". Test failure." << endl;
    return (1);
  }
  return (0);
}

template <class valType>
static int iter3Test(valType increm) {
  // const int iters = 1234567899;
  const int iters = 1000 * 1000;
  valType ang1 = 0.;
  steady_clock::time_point startTime = steady_clock::now();
  for (int cnt = 0; cnt < iters; ++cnt) {
    ang1 = make0To2pi(increm + ang1);
  }
  steady_clock::time_point endTime = steady_clock::now();
  cout << "Fast version after " << iters << " iterations is " << ang1 << endl;
  duration<double> time_span = duration_cast<duration<double>>(endTime - startTime);
  cout << "Time diff is  " << time_span.count() << endl;
  valType plainAng = 0.;
  startTime = steady_clock::now();
  for (int cnt = 0; cnt < iters; ++cnt) {
    plainAng = simpleMake0to2pi(increm + plainAng);
  }
  endTime = steady_clock::now();
  cout << "Simple version after " << iters << " iterations is " << plainAng << endl;
  duration<double> time_span2 = duration_cast<duration<double>>(endTime - startTime);
  cout << "Time diff is  " << time_span2.count() << endl;
  plainAng = 0.;
  startTime = steady_clock::now();
  for (int cnt = 0; cnt < iters; ++cnt) {
    plainAng = useReduceRange(increm + plainAng);
  }
  endTime = steady_clock::now();
  cout << "ReduceRange after " << iters << " iterations is " << plainAng << endl;
  time_span2 = duration_cast<duration<double>>(endTime - startTime);
  cout << "Time diff is  " << time_span2.count() << endl;
  return (0);
}

int main() {
  cout << "long pi   = " << std::setprecision(32) << M_PIl << endl;
  cout << "double pi = " << std::setprecision(32) << M_PI << endl;
  cout << "pi difference = " << M_PIl - M_PI << endl;
  Phi<double, ZeroTo2pi> testval2{39.3_pi};
  cout << "testval2 initialized from 39.3pi, should be 0to2pi = " << testval2 << endl;
  Phi<double, ZeroTo2pi> testval = 39.3_pi;
  cout << "Sizes of Phi<double> and double = " << setprecision(16) << sizeof(testval) << ", " << sizeof(double) << endl;
  {
    Phi<double, ZeroTo2pi> angle = 3.3_pi;
    if (!angle.nearEqual(1.3_pi)) {
      cout << "Angle should be from 0-2pi but it is out of range = " << angle << endl;
      return (1);
    }
  }
  double getval = testval > 0. ? static_cast<float>(testval) : 3.f;
  cout << "getval should be 39.3pi = " << getval << endl;
  {
    Phi<long double, ZeroTo2pi> angle = -3.3_pi;
    if (!angle.nearEqual(0.7_pi)) {
      cout << "Angle should be from 0-2pi but it is out of range = " << angle << endl;
      return (1);
    }
  }
  // Test operations
  Phi<double, ZeroTo2pi> phi1, phi2;
  phi1 = 0.25_pi;
  phi2 = 1._pi / 6.;
  cout << "pi/4 + pi/6 = " << phi1 + phi2 << endl;
  cout << "pi/4 - pi/6 = " << phi1 - phi2 << endl;
  cout << "pi/4 * pi/6 = " << phi1 * phi2 << endl;
  cout << "pi/4 / pi/6 = " << phi1 / phi2 << endl;

  Phi<double, ZeroTo2pi> phi3{3.2_pi};
  cout << "Phi0To2pi started at 3.2pi but reduced to = " << phi3 << endl;
  phi3 += 1.9_pi;
  cout << "Phi0To2pi add 1.9pi = " << phi3 << endl;
  phi3 -= 8.9_pi;
  cout << "Phi0To2pi subtract 8.9pi = " << phi3 << endl;
  Phi<double, ZeroTo2pi> phi4{2.2_pi};
  phi3 = -phi4;
  phi4 = -30._pi;
  cout << "Phi0To2pi set to -2.2pi = " << phi3 << endl;
  cout << "Phi0To2pi set to -30.pi = " << phi4 << endl;
  phi4 = 2._pi;
  cout << "Phi0To2pi set to 2.pi = " << phi4 << endl;
  Phi0To2pi<float> phi5{-3._pi};
  cout << "Phi0To2pi set to -3.pi = " << phi5 << endl;

  cout << "Test with float\n";
  if (testSmall<float>() == 1)
    return (1);
  cout << "Test with double\n";
  if (testSmall<double>() == 1)
    return (1);
  cout << "Test with long double\n";
  if (testSmall<long double>() == 1)
    return (1);
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

  // long double smallincr = 1.39_deg;
  long double smallincr = 1._deg;
  long double bigincr = 7.77_pi;

  cout << "** Use double arithmetic **\n";
  cout << "Test 3 versions small decr\n";
  if (iter3Test<double>(-smallincr) == 1)
    return (1);
  cout << "Test 3 versions small incr\n";
  if (iter3Test<double>(smallincr) == 1)
    return (1);
  cout << "Test 3 versions big decre\n";
  if (iter3Test<double>(-bigincr) == 1)
    return (1);
  cout << "Test 3 versions big incr\n";
  if (iter3Test<double>(bigincr) == 1)
    return (1);
  cout << "** Use long double arithmetic **\n";
  cout << "Test 3 versions small decr\n";
  if (iter3Test<long double>(-smallincr) == 1)
    return (1);
  cout << "Test 3 versions small incr\n";
  if (iter3Test<long double>(smallincr) == 1)
    return (1);
  cout << "Test 3 versions big decre\n";
  if (iter3Test<long double>(-bigincr) == 1)
    return (1);
  cout << "Test 3 versions big incr\n";
  if (iter3Test<long double>(bigincr) == 1)
    return (1);

  return (0);
}

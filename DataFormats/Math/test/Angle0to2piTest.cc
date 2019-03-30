#include "DataFormats/Math/interface/Angle0to2pi.h"

#include <iostream>
#include <iomanip>
#include <math.h>
#include <ctime>
#include <chrono>


using namespace angle0to2pi;
using namespace std;
using namespace reco;
using namespace std::chrono;


template <class valType>
inline constexpr valType useReduceRange(valType angle) {
  constexpr valType twoPi = 2._pi;
  angle = reduceRange(angle);
  if (angle < 0.) angle += twoPi;
  return angle;
}

template <class valType>
inline constexpr valType simpleMake0to2pi(valType angle)
{
    constexpr valType twoPi = 2._pi;
    
    angle = fmod(angle, twoPi);
    if (angle < 0.) angle += twoPi;
      return angle;
}

template <class valType>
static int testSmall()
{ // Test with long double
  Angle0to2pi<valType> ang1(15.3_pi);
  cout << "Angle0to2pi that started as 15.3 pi is " << ang1 << endl;
  cout << "In degrees " << ang1.degrees() << endl;
  constexpr valType testval = 15.2999_pi;
  Angle0to2pi<valType> ang2 = ang1 - testval;
  ang1 -= testval;
  if  (ang1 != ang2) {
    cout << "angle1 = " << ang1 << " and angle2 = " << ang2;
    cout << " should be equal but they are not. Test failure." << endl;
    return (1);
  }
  if  (ang1.nearZero()) {
    cout << "angle = " << ang1 << " close enough to zero to be considered zero." << endl;
  } else  {
    cout << "angle = " << ang1 << " should be considered nearly 0 but it is not.";
    cout << " Test failure." << endl;
    return (1);
  }
  return (0);
}


template <class valType>
static int iterationTest(valType increm) {
  Angle0to2pi<valType> ang1 = 0.;
  const int iters = 123456789;
  steady_clock::time_point startTime = steady_clock::now();
  for (int cnt = 0; cnt < iters; ++cnt) {
    ang1 += increm;
  }
  steady_clock::time_point endTime = steady_clock::now();
  cout << "Angle0to2pi after "<< iters << " iterations is " << ang1 << endl;
  duration<double> time_span = duration_cast<duration<double>>(endTime - startTime);
  cout << "Time diff is  " << time_span.count() << endl;
  valType plainAng = 0.;
  startTime = steady_clock::now();
  for (int cnt = 0; cnt < iters; ++cnt) {
    plainAng = make0to2pi(increm + plainAng);
  }
  endTime = steady_clock::now();
  cout << "plainAng  is  now " << plainAng << endl;
  cout << "Base-type variable after "<< iters << " iterations is " << plainAng << endl;
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
  const int iters = 1234567980;
  valType ang1 = 0.;
  steady_clock::time_point startTime = steady_clock::now();
  for (int cnt = 0; cnt < iters; ++cnt) {
    ang1 = make0to2pi(increm + ang1);
  }
  steady_clock::time_point endTime = steady_clock::now();
  cout << "Fast version after "<< iters << " iterations is " << ang1 << endl;
  duration<double> time_span = duration_cast<duration<double>>(endTime - startTime);
  cout << "Time diff is  " << time_span.count() << endl;
  valType plainAng = 0.;
  startTime = steady_clock::now();
  for (int cnt = 0; cnt < iters; ++cnt) {
    plainAng = simpleMake0to2pi(increm + plainAng);
  }
  endTime = steady_clock::now();
  cout << "Simple version after "<< iters << " iterations is " << plainAng << endl;
  duration<double> time_span2 = duration_cast<duration<double>>(endTime - startTime);
  cout << "Time diff is  " << time_span2.count() << endl;
  plainAng = 0.;
  startTime = steady_clock::now();
  for (int cnt = 0; cnt < iters; ++cnt) {
    plainAng = useReduceRange(increm + plainAng);
  }
  endTime = steady_clock::now();
  cout << "ReduceRange after "<< iters << " iterations is " << plainAng << endl;
  time_span2 = duration_cast<duration<double>>(endTime - startTime);
  cout << "Time diff is  " << time_span2.count() << endl;
  return (0);
}


int main() {
  cout << "long pi   = " << std::setprecision(32) << M_PIl << endl;
  cout << "double pi = " << std::setprecision(32) << M_PI << endl;
  cout << "pi difference = " << M_PIl - M_PI << endl;
  Angle0to2pi<double> testval = 39.3_pi;
  cout << "Sizes of Angle0to2pi<double> and double = " << setprecision(16) << sizeof(testval) << ", " << sizeof(double) << endl;
  {
    Angle0to2pi<double> angle = 3.3_pi;
    if (! angle.nearEqual(1.3_pi)) {
      cout << "Angle should be from 0-2pi but it is out of range = " << angle << endl;
      return (1);
    }
  }
  {
    Angle0to2pi<long double> angle = -3.3_pi;
    if (! angle.nearEqual(0.7_pi)) {
      cout << "Angle should be from 0-2pi but it is out of range = " << angle << endl;
      return (1);
    }
  }
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

   // Tetst operations
   Angle0to2pi<double> phi1, phi2;
   phi1 = 0.25_pi;
   phi2 = 1._pi / 6.;
   cout << "pi/4 + pi/6 = " << phi1 + phi2 << endl;
   cout << "pi/4 - pi/6 = " << phi1 - phi2 << endl;
   cout << "pi/4 * pi/6 = " << phi1 * phi2 << endl;
   cout << "pi/4 / pi/6 = " << phi1 / phi2 << endl;
}


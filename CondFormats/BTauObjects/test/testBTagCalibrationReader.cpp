#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>

#include "../src/headers.h"

static bool eq(float a, float b)
{
  float d = a-b;
  float d2 = d*d;
  return (d2 < 1e-8);
}

int main()
{
  using namespace std;

  string csv1(
    "0, comb, central, 0, 0, 1, 0, 1, 0, 999, \"2*x\" \n"
    "0, comb, central, 0, 0, 1, 1, 2, 0, 999, \"2*x\" \n"
    "0, comb, central, 0, 1, 2, 0, 1, 0, 999, \"-2*x\" \n"
    "0, comb, central, 0, 1, 2, 1, 2, 0, 999, \"-2*x\" \n"
    "3, comb, central, 0, 0, 1, 0, 1, 2, 3, \"2*x\" \n"
    "3, comb, central, 0, -1, 0, 0, 1, 2, 3, \"-2*x\" \n"
  );
  stringstream csv1Stream(csv1);
  BTagCalibration bc1("csv");
  bc1.readCSV(csv1Stream);

  // test pt-dependent function
  BTagCalibrationReader bcr1(&bc1, BTagEntry::OP_LOOSE);
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 3.0, 1.5), 0.));  // out of range
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 1.5, 3.0), 0.));  // out of range
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 0.5, 0.5), 1.));
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 0.5, 1.5), 3.));
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 1.5, 0.5), -1.));
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 1.5, 1.5), -3.));
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, -1.5, 1.5), -3.));  // abseta

  // test discr-dependent function
  BTagCalibrationReader bcr2(&bc1, BTagEntry::OP_RESHAPING);
  assert (eq(bcr2.eval(BTagEntry::FLAV_B, 0.5, 0.5, 1.0), 0.));
  assert (eq(bcr2.eval(BTagEntry::FLAV_B, 0.5, 0.5, 4.0), 0.));
  assert (eq(bcr2.eval(BTagEntry::FLAV_B, 0.5, 0.5, 2.5), 5.));
  assert (eq(bcr2.eval(BTagEntry::FLAV_B, -0.5, 0.5, 2.5), -5.));  // no abseta

  return 0;
}





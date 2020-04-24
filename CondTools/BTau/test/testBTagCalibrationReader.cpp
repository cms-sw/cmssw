#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>

#include "CondFormats/BTauObjects/interface/BTagEntry.h"
#include "CondFormats/BTauObjects/interface/BTagCalibration.h"
#include "CondTools/BTau/interface/BTagCalibrationReader.h"


static bool eq(float a, float b, float prec=1e-8)
{
  float d = a-b;
  float d2 = d*d;
  return (d2 < prec);
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
    "1, test, central, 1, -2, 2, 50, 500, 0, 999, \"2*x\" \n"
    "1, test, up,      1, -2, 2, 50, 500, 0, 999, \"2.1*x\" \n"
    "1, test, down,    1, -2, 2, 50, 500, 0, 999, \"1.9*x\" \n"
  );
  stringstream csv1Stream(csv1);
  BTagCalibration bc1("csv");
  bc1.readCSV(csv1Stream);

  // test pt-dependent function
  BTagCalibrationReader bcr1(BTagEntry::OP_LOOSE);
  bcr1.load(bc1, BTagEntry::FLAV_B, "comb");
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 3.0, 1.5), 0.));  // out of range
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 1.5, 3.0), 0.));  // out of range
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 0.5, 0.5), 1.));
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 0.5, 1.5), 3.));
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 1.5, 0.5), -1.));
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, 1.5, 1.5), -3.));
  assert (eq(bcr1.eval(BTagEntry::FLAV_B, -1.5, 1.5), -3.));  // abseta

  // test discr-dependent function
  BTagCalibrationReader bcr2(BTagEntry::OP_RESHAPING);
  bcr2.load(bc1, BTagEntry::FLAV_B, "comb");
  assert (eq(bcr2.eval(BTagEntry::FLAV_B, 0.5, 0.5, 1.0), 0.));
  assert (eq(bcr2.eval(BTagEntry::FLAV_B, 0.5, 0.5, 4.0), 0.));
  assert (eq(bcr2.eval(BTagEntry::FLAV_B, 0.5, 0.5, 2.5), 5.));
  assert (eq(bcr2.eval(BTagEntry::FLAV_B, -0.5, 0.5, 2.5), -5.));  // no abseta

  // test auto bounds
  BTagCalibrationReader bcr3(BTagEntry::OP_MEDIUM, "central", {"up", "down"});
  bcr3.load(bc1, BTagEntry::FLAV_C, "test");
  assert (eq(bcr3.eval_auto_bounds("central", BTagEntry::FLAV_C, 0.5, 100.), 200., 1e-3));
  assert (eq(bcr3.eval_auto_bounds("up",      BTagEntry::FLAV_C, 0.5, 100.), 210., 1e-3));
  assert (eq(bcr3.eval_auto_bounds("down",    BTagEntry::FLAV_C, 0.5, 100.), 190., 1e-3));
  assert (eq(bcr3.eval_auto_bounds("central", BTagEntry::FLAV_C, 0.5, 20.), 100., 1e-3));  // low
  assert (eq(bcr3.eval_auto_bounds("up",      BTagEntry::FLAV_C, 0.5, 20.), 110., 1e-3));  // low
  assert (eq(bcr3.eval_auto_bounds("down",    BTagEntry::FLAV_C, 0.5, 20.),  90., 1e-3));  // low
  assert (eq(bcr3.eval_auto_bounds("central", BTagEntry::FLAV_C, 0.5, 999.), 1000., 1e-3));  // high
  assert (eq(bcr3.eval_auto_bounds("up",      BTagEntry::FLAV_C, 0.5, 999.), 1100., 1e-3));  // high
  assert (eq(bcr3.eval_auto_bounds("down",    BTagEntry::FLAV_C, 0.5, 999.),  900., 1e-3));  // high

  return 0;
}

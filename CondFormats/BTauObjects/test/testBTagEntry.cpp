#include <assert.h>
#include <iostream>
#include <string>
#include <TF1.h>
#include <TH1F.h>

#include "../src/headers.h"

int main()
{
  using namespace std;

  // default constructor
  auto b1 = BTagEntry();

  // function constructor
  auto f1 = TF1("", "[0]*x");
  f1.SetParameter(0, 2);
  auto b2 = BTagEntry(
    &f1,
    BTagEntry::Parameters(BTagEntry::OP_TIGHT, "comb", "up", BTagEntry::FLAV_C)
  );
  assert (b2.formula == string("2*x"));

  // histo constructor linear formula
  auto h1 = TH1F("h1", "", 2, 0., 2.);
  h1.Fill(0.5, 1);
  h1.Fill(1.5, 2);
  auto b3 = BTagEntry(
    &h1,
    BTagEntry::Parameters(BTagEntry::OP_TIGHT, "comb", "up", BTagEntry::FLAV_C)
  );
  assert (b3.formula == string("x<0 ? 0. : x<1 ? 1 : x<2 ? 2 : 0"));

  // histo constructor bin tree formula
  auto h2 = TH1F("h2", "", 15, 0., 15.);
  for (int i=-2; i<17; ++i) {
    h2.Fill(i+.5, i+.5);
  }
  auto b3_1 = BTagEntry(
    &h2,
    BTagEntry::Parameters(BTagEntry::OP_TIGHT, "comb", "up", BTagEntry::FLAV_C)
  );
  assert (b3_1.formula == string("x<8 ? (x<4 ? (x<2 ? (x<1 ? (x<0 ? 0:0.5) : (1.5)) : (x<3 ? 2.5:3.5)) : (x<6 ? (x<5 ? 4.5:5.5) : (x<7 ? 6.5:7.5))) : (x<12 ? (x<10 ? (x<9 ? 8.5:9.5) : (x<11 ? 10.5:11.5)) : (x<14 ? (x<13 ? 12.5:13.5) : (x<15 ? 14.5:0)))"));

  // csv constructor
  string csv = "0, comb, up, 0, 1, 2, 3, 4, 5, 6, \"2*x\" \n";
  auto b4 = BTagEntry(csv);
  auto csv2 = b4.makeCSVLine();
  assert (b4.params.etaMin == 1);
  assert (b4.params.etaMax == 2);
  assert (b4.params.ptMin == 3);
  assert (b4.params.ptMax == 4);
  assert (b4.params.discrMin == 5);
  assert (b4.params.discrMax == 6);
  assert (csv == csv2);

  return 0;
}


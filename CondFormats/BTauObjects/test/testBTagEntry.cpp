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

  // histo constructor
  auto h1 = TH1F("name", "title", 2, 0., 2.);
  h1.Fill(0.5, 1);
  h1.Fill(1.5, 2);
  auto b3 = BTagEntry(
    &h1,
    BTagEntry::Parameters(BTagEntry::OP_TIGHT, "comb", "up", BTagEntry::FLAV_C)
  );
  assert (b3.formula == string("x<0 ? 0. : x<1 ? 1 : x<2 ? 2 : 0"));

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


#include <assert.h>
#include <iostream>
#include <string>
#include <TF1.h>
#include <TH1F.h>

#include "../src/headers.h"

int main()
{
  using namespace std;

  auto par1 = BTagEntry::Parameters(BTagEntry::OP_TIGHT, "CoMb", "cEnTrAl_");
  assert (par1.measurementType == std::string("comb"));
  assert (par1.sysType == string("central_"));

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
  auto h1 = TH1F("h1", "", 3, 0., 1.);  // lin.
  auto h2 = TH1F("h2", "", 100, 0., 1.);  // bin. tree
  auto sin = TF1("sin", "sin(x)");
  for (float f=0.01f; f<1.f; f+=.01f) {
    h1.Fill(f, sin.Eval(f)/30.);
    h2.Fill(f, sin.Eval(f));
  }
  auto f3_1 = TF1("", BTagEntry(&h1, par1).formula.c_str());
  auto f3_2 = TF1("", BTagEntry(&h2, par1).formula.c_str());
  for (float f=0.01f; f<1.f; f+=.01f) {
    assert (fabs(h1.GetBinContent(h1.FindBin(f)) - f3_1.Eval(f)) < 1e-5);
    assert (fabs(h2.GetBinContent(h2.FindBin(f)) - f3_2.Eval(f)) < 1e-5);
  }

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


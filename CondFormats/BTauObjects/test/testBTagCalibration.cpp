#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>

#include "../src/headers.h"

int main()
{
  using namespace std;

  string csv1(
    "0, comb, up, 0, 1, 2, 3, 4, 5, 6, \"2*x\" \n"
    "0, comb, central, 0, 1, 2, 3, 4, 5, 6, \"2*x\" \n"
    "0, comb, central, 0, 1, 2, 3, 4, 6, 7, \"2*x\" \n"
    " \n \t    \t"
    "0, ttbar, central, 0, 1, 2, 3, 4, 6, 7, \"2*x\" \n"
    "1, comb, central, 0, 1, 2, 3, 4, 6, 7, \"2*x\" \n"
    "0, comb, down, 0, 1, 2, 3, 4, 5, 6, \"2*x\" \n"
  );
  stringstream csv1Stream(csv1);
  BTagCalibration b1("csv");
  b1.readCSV(csv1Stream);

  // assert correct length of vectors
  auto e1 = b1.getEntries(
    BTagEntry::Parameters(BTagEntry::OP_LOOSE, "comb", "central")
  );
  assert (e1.size() == 2);
  auto e2 = b1.getEntries(
    BTagEntry::Parameters(BTagEntry::OP_LOOSE, "comb", "up")
  );
  assert (e2.size() == 1);
  auto e3 = b1.getEntries(
    BTagEntry::Parameters(BTagEntry::OP_MEDIUM, "comb", "central")
  );
  assert (e3.size() == 1);

  // check csv output (ordering arbitrary)
  string tggr = "testTagger";
  string csv2_1("0, comb, up, 0, 1, 2, 3, 4, 5, 6, \"2*x\" \n");
  string csv2_2("0, comb, down, 0, 1, 2, 3, 4, 5, 6, \"2*x\" \n");
  stringstream csv2Stream1;
  stringstream csv2Stream2;
  csv2Stream1 << tggr << ";" << BTagEntry::makeCSVHeader() << csv2_1 << csv2_2;
  csv2Stream2 << tggr << ";" << BTagEntry::makeCSVHeader() << csv2_2 << csv2_1;
  BTagCalibration b2(tggr);
  b2.readCSV(csv2Stream1);

  stringstream csv3Stream;
  b2.makeCSV(csv3Stream);
  assert (
    csv2Stream1.str() == csv3Stream.str() ||
    csv2Stream2.str() == csv3Stream.str()
  );

  return 0;
}


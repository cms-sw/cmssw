#include <iostream>
#include <map>
#include "TROOT.h"
#include "TColor.h"
#include <cassert>
#include <cstdlib>
#include "TString.h"

using namespace std;

static const map<TString, Style_t> stylemap{{"kSolid", kSolid},
                                            {"kDashed", kDashed},
                                            {"kDotted", kDotted},
                                            {"kDashDotted", kDashDotted},
                                            {"kFullSquare", kFullSquare},
                                            {"kFullCircle", kFullCircle},
                                            {"kFullTriangleDown", kFullTriangleDown}};

Style_t StyleParser(TString input) {
  // 1) remove all space
  input.ReplaceAll(" ", "");

  // 2) if number, then just return it
  if (input.IsDec())
    return input.Atoi();

  // 3) if the first char is not a k, then crash
  assert(input(0) == 'k');

  return stylemap.at(input);
}

#include <iostream>
#include <map>
#include "TROOT.h"
#include "TColor.h"

using namespace std;

static const map<TString, Color_t> colormap{{"kWhite", kWhite},
                                            {"kBlack", kBlack},
                                            {"kGray", kGray},
                                            {"kRed", kRed},
                                            {"kGreen", kGreen},
                                            {"kBlue", kBlue},
                                            {"kYellow", kYellow},
                                            {"kMagenta", kMagenta},
                                            {"kCyan", kCyan},
                                            {"kOrange", kOrange},
                                            {"kSpring", kSpring},
                                            {"kTeal", kTeal},
                                            {"kAzure", kAzure},
                                            {"kViolet", kViolet},
                                            {"kPink", kPink}};

Color_t parser(TString input) {
  // 1) remove all space
  input.ReplaceAll(" ", "");

  // 2) if number, then just return it
  if (input.IsDec())
    return input.Atoi();

  // 3) if the first char is not a k, then crash
  if (input(0) != 'k')
    exit(EXIT_FAILURE);

  // 4) in case of + or -, needs to split the string
  Ssiz_t Plus = input.First('+'), Minus = input.First('-');

  // 5) only one symbol is allowed
  if (Plus != TString::kNPOS && Minus != TString::kNPOS)
    exit(EXIT_FAILURE);

  // 6) treat the three cases: +, - or nothing
  if (Plus != TString::kNPOS) {
    TString Left = input(0, Plus), Right = input(Plus + 1);
    cout << Left << ' ' << Right << endl;
    return colormap.at(Left) + Right.Atoi();
  } else if (Minus != TString::kNPOS) {
    TString Left = input(0, Minus), Right = input(Minus + 1);
    return colormap.at(Left) - Right.Atoi();
  } else {
    return colormap.at(input);
  }
}

Color_t ColorParser(TString input) { return parser(input); }

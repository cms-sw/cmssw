#include "PhysicsTools/RooStatsCms/interface/ResonanceCalculators.hh"

#include "TStopwatch.h"

void runResCalc(ResonanceCalculatorAbs& rc, const char* label)
{
  // start the stopwatch
  TStopwatch t;
  t.Start();

  // make a name for the output root file
  char rootfilename[100];
  std::sprintf(rootfilename,"%s_%d.root",label, rc.getRandomSeed());

  // run the calculator
  std::vector<std::pair<double, double> > teststats;
  double bumpMass, bumpTestStat;
  double sig=rc.calculate(rootfilename, teststats, bumpMass, bumpTestStat); // go!

  // print some basic output
  std::cout << "\n\n";
  std::cout << "*********************************************************************" << std::endl;
  std::cout << "Most significant bump found at " << bumpMass << std::endl;
  if(rc.getNumPseudoExperiments()==0) {
    std::cout << "LEE un-corrected significance=" << sig << std::endl;
    std::cout << "LEE un-corrected p-value=" << ResonanceCalculatorAbs::zScoreToPValue(sig) << std::endl;
  } else {
    std::cout << "LEE un-corrected significance=" << bumpTestStat << std::endl;
    std::cout << "LEE un-corrected p-value=" << ResonanceCalculatorAbs::zScoreToPValue(bumpTestStat) << std::endl;

    std::cout << "LEE corrected significance=" << sig << std::endl;
    std::cout << "LEE corrected p-value=" << ResonanceCalculatorAbs::zScoreToPValue(sig) << std::endl;

    std::pair<double, double> pvalueRange=ResonanceCalculatorAbs::getPValueRange(teststats, bumpTestStat, 1.0-0.68);
    std::pair<double, double> zscoreRange=ResonanceCalculatorAbs::getZScoreRange(teststats, bumpTestStat, 1.0-0.68);
    std::cout << "LEE corrected significance 68% C.L. range=[" << zscoreRange.first << ", " << zscoreRange.second << "]" << std::endl;
    std::cout << "LEE corrected p-value 68% C.L. range=[" << pvalueRange.first << ", " << pvalueRange.second << "]" << std::endl;

    pvalueRange=ResonanceCalculatorAbs::getPValueRange(teststats, bumpTestStat, 1.0-0.95);
    zscoreRange=ResonanceCalculatorAbs::getZScoreRange(teststats, bumpTestStat, 1.0-0.95);
    std::cout << "LEE corrected significance 95% C.L. range=[" << zscoreRange.first << ", " << zscoreRange.second << "]" << std::endl;
    std::cout << "LEE corrected p-value 95% C.L. range=[" << pvalueRange.first << ", " << pvalueRange.second << "]" << std::endl;
  }
  std::cout << "*********************************************************************" << std::endl;

  // stop and print the stopwatch
  t.Stop();
  t.Print();

  return;
}


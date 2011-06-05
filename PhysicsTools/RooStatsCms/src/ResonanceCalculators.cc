#include "PhysicsTools/RooStatsCms/interface/ResonanceCalculators.hh"

#include "TStopwatch.h"

void runResCalc(ResonanceCalculatorAbs& rc, const char* label, bool goFast)
{
  // start the stopwatch
  TStopwatch t;
  t.Start();

  // make a name for the output root file
  char rootfilename[100];
  std::sprintf(rootfilename,"%s_%d.root",label, rc.getRandomSeed());

  // run the calculator
  std::vector<std::pair<double, double> > teststatsfloat, teststatsfixed;
  double bumpMass, bumpTestStat;
  rc.calculate(rootfilename, teststatsfloat, teststatsfixed, bumpMass, bumpTestStat, goFast); // go!

  // print some basic output
  std::cout << "\n\n";
  std::cout << "***************************************************************************************************" << std::endl;
  std::cout << "Most significant bump found at " << bumpMass << std::endl;
  std::cout << "LEE un-corrected significance from likelihood ratio  = " << bumpTestStat << std::endl;
  std::cout << "LEE un-corrected p-value from likelihood ratio       = " << ResonanceCalculatorAbs::zScoreToPValue(bumpTestStat) << std::endl;
  std::cout << std::endl;
  if(rc.getNumPseudoExperiments()>0) {

    double pvalueFloat=ResonanceCalculatorAbs::getPValue(teststatsfloat, bumpTestStat);
    double pvalueFixed=ResonanceCalculatorAbs::getPValue(teststatsfixed, bumpTestStat);
    double zscoreFloat=ResonanceCalculatorAbs::getZScore(teststatsfloat, bumpTestStat);
    double zscoreFixed=ResonanceCalculatorAbs::getZScore(teststatsfixed, bumpTestStat);
    
    std::pair<double, double> pvalueFloat68Range=ResonanceCalculatorAbs::getPValueRange(teststatsfloat, bumpTestStat, 1.0-0.68);
    std::pair<double, double> zscoreFloat68Range=ResonanceCalculatorAbs::getZScoreRange(teststatsfloat, bumpTestStat, 1.0-0.68);
    std::pair<double, double> pvalueFloat95Range=ResonanceCalculatorAbs::getPValueRange(teststatsfloat, bumpTestStat, 1.0-0.95);
    std::pair<double, double> zscoreFloat95Range=ResonanceCalculatorAbs::getZScoreRange(teststatsfloat, bumpTestStat, 1.0-0.95);

    std::pair<double, double> pvalueFixed68Range=ResonanceCalculatorAbs::getPValueRange(teststatsfixed, bumpTestStat, 1.0-0.68);
    std::pair<double, double> zscoreFixed68Range=ResonanceCalculatorAbs::getZScoreRange(teststatsfixed, bumpTestStat, 1.0-0.68);
    std::pair<double, double> pvalueFixed95Range=ResonanceCalculatorAbs::getPValueRange(teststatsfixed, bumpTestStat, 1.0-0.95);
    std::pair<double, double> zscoreFixed95Range=ResonanceCalculatorAbs::getZScoreRange(teststatsfixed, bumpTestStat, 1.0-0.95);

    std::cout << "LEE uncorrected significance from pseudo-experiments = " << zscoreFixed << "\n"
	      << "                                                     = [" << zscoreFixed68Range.first << ", "
	      << zscoreFixed68Range.second << "] @ 68% C.L.\n"
	      << "                                                     = [" << zscoreFixed95Range.first << ", "
	      << zscoreFixed95Range.second << "] @ 95% C.L." << std::endl;
    std::cout << "LEE uncorrected p-value from pseudo-experiments      = " << pvalueFixed << "\n"
	      << "                                                     = [" << pvalueFixed68Range.first << ", "
	      << pvalueFixed68Range.second << "] @ 68% C.L.\n"
	      << "                                                     = [" << pvalueFixed95Range.first << ", "
	      << pvalueFixed95Range.second << "] @ 95% C.L." << std::endl;
    std::cout << std::endl;

    std::cout << "LEE corrected significance from pseudo-experiments   = " << zscoreFloat << "\n"
	      << "                                                     = [" << zscoreFloat68Range.first << ", "
	      << zscoreFloat68Range.second << "] @ 68% C.L.\n"
	      << "                                                     = [" << zscoreFloat95Range.first << ", "
	      << zscoreFloat95Range.second << "] @ 95% C.L." << std::endl;
    std::cout << "LEE corrected p-value from pseudo-experiments        = " << pvalueFloat << "\n"
	      << "                                                     = [" << pvalueFloat68Range.first << ", "
	      << pvalueFloat68Range.second << "] @ 68% C.L.\n"
	      << "                                                     = [" << pvalueFloat95Range.first << ", "
	      << pvalueFloat95Range.second << "] @ 95% C.L." << std::endl;

    std::cout << std::endl;

  }
  std::cout << "***************************************************************************************************" << std::endl;

  // stop and print the stopwatch
  t.Stop();
  t.Print();

  return;
}


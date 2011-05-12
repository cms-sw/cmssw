#include "Analysis/Statistics/interface/ResonanceCalculators.hh"
#include "TStopwatch.h"

int main(int argc, char* argv[])
{
  if(argc<7) {
    std::cout << "USAGE: rescalc filename nbins minbin maxbin minmass maxmass [numPEs] [randomseed]" << std::endl;
    std::cout << "   filename: space delimited file which contains the data points\n";
    std::cout << "   nbins: number of bins to used to contain the data\n";
    std::cout << "   minbin/maxbin: minimum and maximum bin values\n";
    std::cout << "   minmass/maxmass: minimum and maximum resonance mass values to search for\n";
    std::cout << "   numPEs: number of pseudo-experiments to throw (0 is the default)\n";
    std::cout << "   randomseed: the random seed used to generate teh pseudo-experiments\n";
    return 0;
  }
  const char* filename=argv[1];
  int nbins=atoi(argv[2]);
  double minbin=atof(argv[3]);
  double maxbin=atof(argv[4]);
  double minmass=atof(argv[5]);
  double maxmass=atof(argv[6]);
  int numPEs=argc>=8 ? atoi(argv[7]) : 0;
  int randomseed=argc>=9 ? atoi(argv[8]) : 1;

  SimpleResCalc::setPrintLevel(2);
  SimpleResCalc rc;
  rc.setBinnedData(filename, nbins, minbin, maxbin);
  rc.setNumBinsToDraw(nbins);
  rc.setMinMaxSignalMass(minmass, maxmass);
  rc.setNumPseudoExperiments(numPEs);
  rc.setRandomSeed(randomseed);
  rc.setFitStrategy(2);
  char rootfilename[100];
  std::sprintf(rootfilename,"output_%d.root",randomseed);

  TStopwatch t;
  t.Start();
  double sig=rc.calculate(rootfilename); // go!
  t.Stop();
  t.Print();
  if(numPEs==0) std::cout << "LEE un-corrected signifcance=" << sig << std::endl;
  else          std::cout << "LEE corrected significance=" << sig << std::endl;
  return 0;
}

#include "CalibTracker/SiStripLorentzAngle/plugins/EnsembleCalibrationLA.h"
#include "CalibTracker/SiStripLorentzAngle/interface/Book.h"
#include <TChain.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TFile.h>
#include <boost/foreach.hpp>
#include <fstream>

namespace sistrip {

EnsembleCalibrationLA::EnsembleCalibrationLA(const edm::ParameterSet& conf) :
  inputFiles( conf.getParameter<std::vector<std::string> >("InputFiles") ),
  inFileLocation( conf.getParameter<std::string>("InFileLocation")),
  Prefix( conf.getUntrackedParameter<std::string>("Prefix","")),
  maxEvents( conf.getUntrackedParameter<unsigned>("MaxEvents",0)),
  samples( conf.getParameter<unsigned>("Samples")),
  nbins( conf.getParameter<unsigned>("NBins")),
  lowBin( conf.getParameter<double>("LowBin")),
  highBin( conf.getParameter<double>("HighBin")),
  useWIDTH( conf.getUntrackedParameter<bool>("useWIDTH",true)),
  useRATIO( conf.getUntrackedParameter<bool>("useRATIO",true)),
  useSQRTVAR( conf.getUntrackedParameter<bool>("useSQRTVAR",true))
{std::cout << conf << std::endl;}

void EnsembleCalibrationLA::
endJob() 
{
  Book book("la_ensemble");
  TChain* chain = new TChain("la_ensemble"); 
  BOOST_FOREACH(std::string file, inputFiles) chain->Add((file+inFileLocation).c_str());

  int methods = 0;
  if(useWIDTH)   methods|= LA_Filler_Fitter::WIDTH;
  if(useRATIO)   methods|= LA_Filler_Fitter::RATIO;
  if(useSQRTVAR) methods|= LA_Filler_Fitter::SQRTVAR;

  LA_Filler_Fitter laff(methods,samples,nbins,lowBin,highBin,maxEvents);
  laff.fill(chain,book);
  laff.fit(book);
  laff.summarize_ensembles(book);

  std::pair<std::string, std::vector<LA_Filler_Fitter::EnsembleSummary> > ensemble;
  BOOST_FOREACH(ensemble, laff.ensemble_summary(book)) {
    fstream file((Prefix+ensemble.first+".dat").c_str(),std::ios::out);
    BOOST_FOREACH(LA_Filler_Fitter::EnsembleSummary summary, ensemble.second)
      file << summary << std::endl;

    std::pair<std::pair<float,float>,std::pair<float,float> > line = LA_Filler_Fitter::offset_slope(ensemble.second);
    file << std::endl << std::endl
	 << "# Best Fit Line: "	 << line.first.first <<"("<< line.first.second<<")   +   x* "
	                         << line.second.first<<"("<< line.second.second<<")"           << std::endl
	 << "# Mean Reported Uncertainty / Sigma: " << LA_Filler_Fitter::pull(ensemble.second) << std::endl;
    file.close();
 }
  
  { 
    TFile file((Prefix+"sampleFits.root").c_str(),"RECREATE");
    for(Book::const_iterator hist = book.begin(".*(profile|ratio|reconstruction)"); hist!=book.end(); ++hist)
      (*hist)->Write();
    file.Close();
  }
  
  {
    TFile file((Prefix+"ensembleFits.root").c_str(),"RECREATE");
    for(Book::const_iterator hist = book.begin(".*(measure|merr|ensembleReco)"); hist!=book.end(); ++hist)
      (*hist)->Write();
    file.Close();
  }
  
}

}

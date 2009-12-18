#include "CalibTracker/SiStripLorentzAngle/plugins/EnsembleCalibrationLA.h"
#include "CalibTracker/SiStripCommon/interface/Book.h"
#include <TChain.h>
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
  vMethods( conf.getParameter<std::vector<int> >("Methods"))
{}

void EnsembleCalibrationLA::
endJob() 
{
  Book book("la_ensemble");
  TChain*const chain = new TChain("la_ensemble"); 
  BOOST_FOREACH(std::string file, inputFiles) chain->Add((file+inFileLocation).c_str());

  int methods = 0;  BOOST_FOREACH(unsigned method, vMethods) methods|=method;

  LA_Filler_Fitter 
    laff(methods,samples,nbins,lowBin,highBin,maxEvents);
  laff.fill(chain,book);           
  laff.fit(book);                  
  laff.summarize_ensembles(book);  

  write_ensembles_text(book);
  write_ensembles_plots(book);
  write_samples_plots(book);
}

void EnsembleCalibrationLA::
write_ensembles_text(const Book& book) const {
  std::pair<std::string, std::vector<LA_Filler_Fitter::EnsembleSummary> > ensemble;
  BOOST_FOREACH(ensemble, LA_Filler_Fitter::ensemble_summary(book)) {
    fstream file((Prefix+ensemble.first+".dat").c_str(),std::ios::out);
    BOOST_FOREACH(LA_Filler_Fitter::EnsembleSummary summary, ensemble.second)
      file << summary << std::endl;
    
    const std::pair<std::pair<float,float>,std::pair<float,float> > line = LA_Filler_Fitter::offset_slope(ensemble.second);
    const float pull =  LA_Filler_Fitter::pull(ensemble.second);

    file << std::endl << std::endl
	 << "# Best Fit Line: "	 
	 << line.first.first <<"("<< line.first.second<<")   +   x* "
	 << line.second.first<<"("<< line.second.second<<")"           
	 << std::endl
	 << "# Pull (average sigma of (x_measure-x_truth)/e_measure): " << pull 
	 << std::endl
	 << "LA_Calibration( METHOD_XXXXX , xxx, " << line.second.first << ", " << line.first.first << ", " << pull << ")," << std::endl;
    file.close();
  } 
}

void EnsembleCalibrationLA::
write_ensembles_plots(const Book& book) const {
  TFile file((Prefix+"sampleFits.root").c_str(),"RECREATE");
  for(Book::const_iterator hist = book.begin(".*(profile|ratio|reconstruction|symm|symmchi2|_w\\d)"); hist!=book.end(); ++hist)
    hist->second->Write();
  file.Close();
}
 
void EnsembleCalibrationLA::
write_samples_plots(const Book& book) const {
  TFile file((Prefix+"ensembleFits.root").c_str(),"RECREATE");
  for(Book::const_iterator hist = book.begin(".*(measure|merr|ensembleReco|pull)"); hist!=book.end(); ++hist)
    hist->second->Write();
  file.Close();
}
  
}


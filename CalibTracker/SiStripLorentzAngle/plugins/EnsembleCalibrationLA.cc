#include "CalibTracker/SiStripLorentzAngle/plugins/EnsembleCalibrationLA.h"
#include "CalibTracker/SiStripCommon/interface/Book.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include <TChain.h>
#include <TFile.h>
#include <boost/lexical_cast.hpp>
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

void EnsembleCalibrationLA::endJob() 
{
  Book book("la_ensemble");
  TChain*const chain = new TChain("la_ensemble"); 
  for(auto const& file : inputFiles) chain->Add((file+inFileLocation).c_str());

  int methods = 0;
  for(unsigned int method : vMethods) methods|=method;

  LA_Filler_Fitter laff(methods,samples,nbins,lowBin,highBin,maxEvents,tTopo_);
  laff.fill(chain,book);           
  laff.fit(book);                  
  laff.summarize_ensembles(book);  

  write_ensembles_text(book);
  write_ensembles_plots(book);
  write_samples_plots(book);
  write_calibrations();
}

void EnsembleCalibrationLA::endRun(const edm::Run&, const edm::EventSetup& eSetup)
{
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  tTopo_ = tTopoHandle.product();
}

void EnsembleCalibrationLA::write_ensembles_text(const Book& book)
{
  for(auto const& ensemble : LA_Filler_Fitter::ensemble_summary(book)) {
    std::fstream file((Prefix+ensemble.first+".dat").c_str(),std::ios::out);
    for(auto const& summary : ensemble.second)
      file << summary << std::endl;

    const std::pair<std::pair<float,float>,std::pair<float,float> > line = LA_Filler_Fitter::offset_slope(ensemble.second);
    const float pull =  LA_Filler_Fitter::pull(ensemble.second);

    unsigned index = 15;
    std::string label;
    {
      std::cout << ensemble.first << std::endl;
      boost::regex format(".*(T[IO]B)_layer(\\d)([as])_(.*)");
      if(boost::regex_match(ensemble.first,format)) {
	const bool TIB = "TIB" == boost::regex_replace(ensemble.first, format, "\\1");
	const bool stereo = "s" == boost::regex_replace(ensemble.first, format, "\\3");
	const unsigned layer = boost::lexical_cast<unsigned>(boost::regex_replace(ensemble.first, format, "\\2"));
	label = boost::regex_replace(ensemble.first, format, "\\4");
	index = LA_Filler_Fitter::layer_index(TIB,stereo,layer);

	calibrations[label].slopes[index]=line.second.first;
	calibrations[label].offsets[index]=line.first.first;
	calibrations[label].pulls[index]=pull;
      }
    }

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

void EnsembleCalibrationLA::
write_calibrations() const {
  std::fstream file((Prefix+"calibrations.dat").c_str(),std::ios::out);
  for(auto const& cal : calibrations) {
    file << cal.first << std::endl
	 << "\t slopes(";    for(float i : cal.second.slopes) file << i<< ","; file << ")" << std::endl
	 << "\t offsets(";   for(float i : cal.second.offsets) file << i<< ","; file << ")" << std::endl
	 << "\t pulls(";     for(float i : cal.second.pulls) file << i<< ","; file << ")" << std::endl;
  }
  file.close();
}
  
}


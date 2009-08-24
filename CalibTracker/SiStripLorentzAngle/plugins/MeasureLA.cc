#include "CalibTracker/SiStripLorentzAngle/plugins/MeasureLA.h"
#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <TChain.h>

namespace sistrip {

MeasureLA::MeasureLA(const edm::ParameterSet& conf) :
  inputFiles( conf.getParameter<std::vector<std::string> >("inputFiles") ),
  outputHistograms( conf.getParameter<std::string>("OutputHistograms")),
  byLayer( conf.getParameter<bool>("byLayer")),
  byModule( conf.getParameter<bool>("byModule")),
  chi2ndof_cut( conf.getParameter<double>("Chi2NDOF_cut") ),
  nEntries_cut( conf.getParameter<unsigned>("NEntries_cut") ),
  fp_(conf.getParameter<edm::FileInPath>("SiStripDetInfo") )
{
  TChain* chain = new TChain("la_chain");
  BOOST_FOREACH(std::string file, inputFiles) chain->Add((file+"/calibTree/tree").c_str());
  
  int method = LA_Filler_Fitter::RATIO|LA_Filler_Fitter::WIDTH|LA_Filler_Fitter::SQRTVAR;
  LA_Filler_Fitter laff(method, true, byModule);
  laff.fill(chain, book);
  LA_Filler_Fitter::fit(book);
  
  setWhatProduced(this,&MeasureLA::produce);
}
  
boost::shared_ptr<SiStripLorentzAngle> MeasureLA::
produce(const SiStripLorentzAngleRcd& ) {
  boost::shared_ptr<SiStripLorentzAngle> lorentzAngle(new SiStripLorentzAngle());

  std::map<uint32_t,LA_Filler_Fitter::Result> 
    module_results = LA_Filler_Fitter::module_results(book, LA_Filler_Fitter::SQRTVAR);
  
  //python pset:: file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),         
  BOOST_FOREACH(const uint32_t& detid, SiStripDetInfoFileReader(fp_.fullPath()).getAllDetIds()) {
    float la = module_results[detid].measure / module_results[detid].field ;
    lorentzAngle->putLorentzAngle( detid, la );
  }

  return lorentzAngle;
}
      
}

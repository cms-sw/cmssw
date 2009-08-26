#include "CalibTracker/SiStripLorentzAngle/plugins/MeasureLA.h"
#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <TChain.h>
#include <TFile.h>

namespace sistrip {

MeasureLA::MeasureLA(const edm::ParameterSet& conf) :
  inputFiles( conf.getParameter<std::vector<std::string> >("InputFiles") ),
  inFileLocation( conf.getParameter<std::string>("InFileLocation")),
  fp_(conf.getParameter<edm::FileInPath>("SiStripDetInfo") ),
  maxEvents( conf.getUntrackedParameter<unsigned>("MaxEvents",0)),
  reports( conf.getParameter<std::vector<edm::ParameterSet> >("Reports")),
  measurementPreferences( conf.getParameter<std::vector<edm::ParameterSet> >("MeasurementPreferences")),
  calibrations( conf.getParameter<std::vector<edm::ParameterSet> >("Calibrations"))
{
  TChain* chain = new TChain("la_ensemble"); 
  BOOST_FOREACH(std::string file, inputFiles) chain->Add((file+inFileLocation).c_str());
  
  int32_t methods = 0;  bool byModule(false), byLayer(false);
  append_methods_and_granularity( methods, byModule, byLayer, reports);
  append_methods_and_granularity( methods, byModule, byLayer, measurementPreferences );

  LA_Filler_Fitter laff(methods, byLayer, byModule, maxEvents);
  laff.fill(chain, book);
  LA_Filler_Fitter::fit(book);
  process_reports();

  setWhatProduced(this,&MeasureLA::produce);
}
  
void MeasureLA::
append_methods_and_granularity(int32_t& methods, bool& byModule, bool& byLayer, const std::vector<edm::ParameterSet>& vpset) {
  BOOST_FOREACH(edm::ParameterSet p, vpset) {
    methods |= p.getParameter<int32_t>("Method"); 
    byModule = byModule || p.getParameter<bool>("ByModule");
    byLayer  = byLayer  || !p.getParameter<bool>("ByModule");
  }  
}

void MeasureLA::
process_reports() {
  BOOST_FOREACH(edm::ParameterSet p, reports) {
    bool byMod = p.getParameter<bool>("ByModule");
    std::string name = p.getParameter<std::string>("ReportName");
    LA_Filler_Fitter::Method method = (LA_Filler_Fitter::Method) p.getParameter<int32_t>("Method");

    typedef std::pair<uint32_t,LA_Filler_Fitter::Result> mr_t;
    typedef std::pair<std::string,LA_Filler_Fitter::Result> lr_t;
    fstream file((name+".dat").c_str(),std::ios::out);
    if(byMod) BOOST_FOREACH(mr_t result, LA_Filler_Fitter::module_results(book,method)) {calibrate(result); file<< result.first <<"\t"<< result.second <<std::endl;}
    else      BOOST_FOREACH(lr_t result,  LA_Filler_Fitter::layer_results(book,method)) {calibrate(result); file<< result.first <<"\t"<< result.second <<std::endl;}
    file.close();
    
    TFile tfile((name+".root").c_str(),"RECREATE");
    std::string key = ( byMod ? ".*_module.*" : ".*_layer.*") + LA_Filler_Fitter::method(method);
    for(Book::const_iterator hist = book.begin(key); hist!=book.end(); ++hist)
      (*hist)->Write();
    tfile.Close();
  }
}

void MeasureLA::
calibrate(std::pair<uint32_t,LA_Filler_Fitter::Result>& r) {
  
}
void MeasureLA::
calibrate(std::pair<std::string,LA_Filler_Fitter::Result>& r) {

}

boost::shared_ptr<SiStripLorentzAngle> MeasureLA::
produce(const SiStripLorentzAngleRcd& ) {
  boost::shared_ptr<SiStripLorentzAngle> lorentzAngle(new SiStripLorentzAngle());

  std::map<uint32_t,LA_Filler_Fitter::Result> 
    module_results = LA_Filler_Fitter::module_results(book, LA_Filler_Fitter::SQRTVAR);
  
  BOOST_FOREACH(const uint32_t& detid, SiStripDetInfoFileReader(fp_.fullPath()).getAllDetIds()) {
    float la = module_results[detid].measure / module_results[detid].field ;
    lorentzAngle->putLorentzAngle( detid, la );
  }

  return lorentzAngle;
}
      
}

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
  reports( conf.getParameter<edm::VParameterSet>("Reports")),
  measurementPreferences( conf.getParameter<edm::VParameterSet>("MeasurementPreferences")),
  calibrations(conf.getParameter<edm::VParameterSet>("Calibrations"))
{
  BOOST_FOREACH(edm::ParameterSet p, calibrations) {
    std::pair<uint32_t,LA_Filler_Fitter::Method> key(p.getParameter<uint32_t>("Pitch"),
						     (LA_Filler_Fitter::Method) p.getParameter<int32_t>("Method"));
    offset[key] = p.getParameter<double>("Offset");
    slope[key] = p.getParameter<double>("Slope");
    error_scaling[key] = p.getParameter<double>("ErrorScaling");
  }

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
append_methods_and_granularity(int32_t& methods, bool& byModule, bool& byLayer, const edm::VParameterSet& vpset) {
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
    if(byMod) BOOST_FOREACH(mr_t result, LA_Filler_Fitter::module_results(book,method)) {
      calibrate( calibration_key(result.first,method), 
		 result.second); 
      file<< result.first <<"\t"<< result.second <<std::endl;
    }
    else BOOST_FOREACH(lr_t result,  LA_Filler_Fitter::layer_results(book,method)) {
      calibrate(calibration_key(result.first,method), 
		result.second);
      file<< result.first <<"\t"<< result.second <<std::endl;
    }
    file.close();
    
    TFile tfile((name+".root").c_str(),"RECREATE");
    std::string key = ( byMod ? ".*_module.*" : ".*_layer.*") + LA_Filler_Fitter::method(method);
    for(Book::const_iterator hist = book.begin(key); hist!=book.end(); ++hist)
      (*hist)->Write();
    for(Book::const_iterator hist = book.begin(".*_symm"); hist!=book.end(); ++hist)
      (*hist)->Write();
    tfile.Close();
  }
}

void MeasureLA::
calibrate(std::pair<uint32_t,LA_Filler_Fitter::Method> key, LA_Filler_Fitter::Result& result) {
  result.calibratedMeasurement = ( result.measure - offset[key] ) / slope[key] ;
  result.calibratedError = result.measureErr / (slope[key] * error_scaling[key] );
}

std::pair<uint32_t,LA_Filler_Fitter::Method> MeasureLA::
calibration_key(std::string layer,LA_Filler_Fitter::Method method) {
  uint32_t pitch = 0;
  if(layer.find("TIB_layer1") != std::string::npos || layer.find("TIB_layer2") != std::string::npos) pitch=80; else
    if(layer.find("TIB_layer3") != std::string::npos || layer.find("TIB_layer4") != std::string::npos) pitch=120; else
      if(layer.find("TOB_layer5") != std::string::npos || layer.find("TOB_layer6") != std::string::npos) pitch=122; else
	if(layer.find("TOB")   /* Layers 1,2,3,4 */                                    != std::string::npos) pitch=183;
  return std::make_pair(pitch,method);
}

std::pair<uint32_t,LA_Filler_Fitter::Method> MeasureLA::
calibration_key(uint32_t detid,LA_Filler_Fitter::Method method) {
  uint32_t pitch =  
    SiStripDetId(detid).subDetector() == SiStripDetId::TIB 
    ? TIBDetId(detid).layer() < 3 ?  80 : 120
    : TOBDetId(detid).layer() > 4 ? 122 : 183; 
  return std::make_pair(pitch,method);
}

boost::shared_ptr<SiStripLorentzAngle> MeasureLA::
produce(const SiStripLorentzAngleRcd& ) {
  boost::shared_ptr<SiStripLorentzAngle> lorentzAngle(new SiStripLorentzAngle());
  /*
  std::map<uint32_t,LA_Filler_Fitter::Result> 
    module_results = LA_Filler_Fitter::module_results(book, LA_Filler_Fitter::SQRTVAR);
  
  BOOST_FOREACH(const uint32_t& detid, SiStripDetInfoFileReader(fp_.fullPath()).getAllDetIds()) {
    float la = module_results[detid].measure / module_results[detid].field ;
    lorentzAngle->putLorentzAngle( detid, la );
  }
  */
  return lorentzAngle;
}
      
}

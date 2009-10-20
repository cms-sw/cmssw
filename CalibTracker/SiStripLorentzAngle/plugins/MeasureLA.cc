#include "CalibTracker/SiStripLorentzAngle/plugins/MeasureLA.h"
#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <TChain.h>
#include <TFile.h>

namespace sistrip {

void MeasureLA::
store_methods_and_granularity( const edm::VParameterSet& vpset) {
  BOOST_FOREACH(edm::ParameterSet p, vpset) {
    methods |= p.getParameter<int32_t>("Method"); 
    byModule = byModule || p.getParameter<int32_t>("Granularity");
    byLayer  = byLayer  || !p.getParameter<int32_t>("Granularity");
  }
}


MeasureLA::MeasureLA(const edm::ParameterSet& conf) :
  inputFiles( conf.getParameter<std::vector<std::string> >("InputFiles") ),
  inFileLocation( conf.getParameter<std::string>("InFileLocation")),
  fp_(conf.getParameter<edm::FileInPath>("SiStripDetInfo") ),
  maxEvents( conf.getUntrackedParameter<unsigned>("MaxEvents",0)),
  reports( conf.getParameter<edm::VParameterSet>("Reports")),
  measurementPreferences( conf.getParameter<edm::VParameterSet>("MeasurementPreferences")),
  calibrations(conf.getParameter<edm::VParameterSet>("Calibrations")),
  methods(0), byModule(false), byLayer(false)
{
  store_methods_and_granularity( reports );
  store_methods_and_granularity( measurementPreferences );
  store_calibrations();

  TChain* chain = new TChain("la_data"); 
  BOOST_FOREACH(std::string file, inputFiles) chain->Add((file+inFileLocation).c_str());
  
  LA_Filler_Fitter laff(methods, byLayer, byModule, maxEvents);
  laff.fill(chain, book);
  laff.fit(book);
  summarize_module_muH_byLayer();
  process_reports();

  setWhatProduced(this,&MeasureLA::produce);
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

void MeasureLA::
summarize_module_muH_byLayer() {
  for(int m = LA_Filler_Fitter::FIRST_METHOD; m <= LA_Filler_Fitter::LAST_METHOD; m<<=1) {
    LA_Filler_Fitter::Method method = (LA_Filler_Fitter::Method)m;
    std::pair<uint32_t,LA_Filler_Fitter::Result> result;
    BOOST_FOREACH(result, LA_Filler_Fitter::module_results(book, method)) {
      
      calibrate( calibration_key(result.first,method), result.second); 
      std::string label = LA_Filler_Fitter::layerLabel(result.first) + granularity(MODULESUMMARY) + LA_Filler_Fitter::method(method);
      label = boost::regex_replace(label,boost::regex("layer"),"");

      
      double mu_H = fabs( result.second.calibratedMeasurement / result.second.field );
      double sigma_mu_H = result.second.calibratedError / result.second.field;
      double weight = pow(1./sigma_mu_H, 2);
      
      book.fill(mu_H, label, 150,-0.05,0.1, weight);
    }
    for(Book::const_iterator it = book.begin(".*"+granularity(MODULESUMMARY)+".*"); it!=book.end(); ++it) {
      if((*it)->GetEntries()) (*it)->Fit("gaus","LLQ");
    }
  }
}

void MeasureLA::
process_reports() {
  BOOST_FOREACH(edm::ParameterSet p, reports) {
    GRANULARITY gran = (GRANULARITY) p.getParameter<int32_t>("Granularity");
    std::string name = p.getParameter<std::string>("ReportName");
    LA_Filler_Fitter::Method method = (LA_Filler_Fitter::Method) p.getParameter<int32_t>("Method");

    write_report_plots( name, method, gran);
    switch(gran) {
    case LAYER: write_report_text( name, method, LA_Filler_Fitter::layer_results(book, method) ); break;
    case MODULE: write_report_text( name, method, LA_Filler_Fitter::module_results(book, method)); break;
    case MODULESUMMARY: write_report_text_ms( name, method); break;
    }
  }
}

void MeasureLA::
write_report_plots(std::string name, LA_Filler_Fitter::Method method, GRANULARITY gran ) {
  TFile file((name+".root").c_str(),"RECREATE");
  std::string key = ".*" + granularity(gran) + ".*("+LA_Filler_Fitter::method(method)+"|"+LA_Filler_Fitter::method(method,0)+".*)";
  for(Book::const_iterator hist = book.begin(key); hist!=book.end(); ++hist) 
    (*hist)->Write();
  file.Close();
}

template <class T>
void MeasureLA::
write_report_text(std::string name, LA_Filler_Fitter::Method method, std::map<T,LA_Filler_Fitter::Result> results) {
  fstream file((name+".dat").c_str(),std::ios::out);
  std::pair<T,LA_Filler_Fitter::Result> result;
  BOOST_FOREACH(result, results) {
    calibrate( calibration_key(result.first,method), result.second); 
    file << result.first << "\t" << result.second << std::endl;
  }
  file.close();
}

void MeasureLA::
write_report_text_ms(std::string name, LA_Filler_Fitter::Method method) {
  fstream file((name+".dat").c_str(),std::ios::out);
  std::string key = ".*"+granularity(MODULESUMMARY)+LA_Filler_Fitter::method(method);
  for(Book::const_iterator it = book.begin(key); it!=book.end(); ++it) {
    TF1* f = (*it)->GetFunction("gaus");
    if(f) {
      file << it.name() << "\t"
	   << f->GetParameter(1) << "\t"
	   << f->GetParError(1) << "\t"
	   << f->GetParameter(2) << "\t"
	   << f->GetParError(2) << std::endl;
    }
  }
  file.close();
}
  
void MeasureLA::
store_calibrations() {
  BOOST_FOREACH(edm::ParameterSet p, calibrations) {
    std::pair<uint32_t,LA_Filler_Fitter::Method> 
      key( p.getParameter<uint32_t>("Pitch"), (LA_Filler_Fitter::Method) 
	   p.getParameter<int32_t>("Method"));
    offset[key] = p.getParameter<double>("Offset");
    slope[key] = p.getParameter<double>("Slope");
    error_scaling[key] = p.getParameter<double>("ErrorScaling");
  }
}

inline
void MeasureLA::
calibrate(std::pair<uint32_t,LA_Filler_Fitter::Method> key, LA_Filler_Fitter::Result& result) {
  result.calibratedMeasurement = ( result.measure - offset[key] ) / slope[key] ;
  result.calibratedError = result.measureErr * error_scaling[key] / slope[key] ;
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
      
}

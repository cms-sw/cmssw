#include "CalibTracker/SiStripLorentzAngle/plugins/MeasureLA.h"
#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <boost/lexical_cast.hpp>
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
  reports( conf.getParameter<edm::VParameterSet>("Reports")),
  measurementPreferences( conf.getParameter<edm::VParameterSet>("MeasurementPreferences")),
  calibrations(conf.getParameter<edm::VParameterSet>("Calibrations")),
  methods(0), byModule(false), byLayer(false), 
  localybin(conf.getUntrackedParameter<double>("LocalYBin",0.0)),
  stripsperbin(conf.getUntrackedParameter<unsigned>("StripsPerBin",0)),
  maxEvents( conf.getUntrackedParameter<unsigned>("MaxEvents",0))
{
  store_methods_and_granularity( reports );
  store_methods_and_granularity( measurementPreferences );
  store_calibrations();

  TChain*const chain = new TChain("la_data"); 
  BOOST_FOREACH(const std::string file, inputFiles) chain->Add((file+inFileLocation).c_str());
  
  LA_Filler_Fitter laff(methods, byLayer, byModule, localybin, stripsperbin, maxEvents);
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
    const LA_Filler_Fitter::Method method = (LA_Filler_Fitter::Method)m;
    std::pair<uint32_t,LA_Filler_Fitter::Result> result;
    BOOST_FOREACH(result, LA_Filler_Fitter::module_results(book, method)) {
      
      calibrate( calibration_key(result.first,method), result.second); 
      std::string label = LA_Filler_Fitter::layerLabel(result.first) + granularity(MODULESUMMARY) + LA_Filler_Fitter::method(method);
      label = boost::regex_replace(label,boost::regex("layer"),"");
      
      const double mu_H = -result.second.calMeasured.first / result.second.field;
      const double sigma_mu_H = result.second.calMeasured.second / result.second.field;
      const double weight = pow(1./sigma_mu_H, 2);
      
      book.fill(mu_H, label, 150,-0.05,0.1, weight);
    }
    for(Book::iterator it = book.begin(".*"+granularity(MODULESUMMARY)+".*"); it!=book.end(); ++it) {
      if(it->second->GetEntries()) it->second->Fit("gaus","LLQ");
    }
  }
}

void MeasureLA::
process_reports() const {
  BOOST_FOREACH(edm::ParameterSet p, reports) {
    const GRANULARITY gran = (GRANULARITY) p.getParameter<int32_t>("Granularity");
    const std::string name = p.getParameter<std::string>("ReportName");
    const LA_Filler_Fitter::Method method = (LA_Filler_Fitter::Method) p.getParameter<int32_t>("Method");

    write_report_plots( name, method, gran);
    switch(gran) {
    case LAYER: write_report_text( name, method, LA_Filler_Fitter::layer_results(book, method) ); break;
    case MODULE: write_report_text( name, method, LA_Filler_Fitter::module_results(book, method)); break;
    case MODULESUMMARY: write_report_text_ms( name, method); break;
    }
  }
  
  { TFile widthsFile("widths.root","RECREATE"); 
    for(Book::const_iterator it = book.begin(".*_width"); it!=book.end(); it++) if(it->second) it->second->Write();
    widthsFile.Close();}
}

void MeasureLA::
write_report_plots(std::string name, LA_Filler_Fitter::Method method, GRANULARITY gran ) const {
  TFile file((name+".root").c_str(),"RECREATE");
  const std::string key = ".*" + granularity(gran) + ".*("+LA_Filler_Fitter::method(method)+"|"+LA_Filler_Fitter::method(method,0)+".*)";
  for(Book::const_iterator hist = book.begin(key); hist!=book.end(); ++hist) 
    if(hist->second) hist->second->Write();
  file.Close();
}

template <class T>
void MeasureLA::
write_report_text(std::string name, const LA_Filler_Fitter::Method& _method, const std::map<T,LA_Filler_Fitter::Result>& _results) const {
  LA_Filler_Fitter::Method method = _method;
  std::map<T,LA_Filler_Fitter::Result>results = _results;
  fstream file((name+".dat").c_str(),std::ios::out);
  std::pair<T,LA_Filler_Fitter::Result> result;
  BOOST_FOREACH(result, results) {
    calibrate( calibration_key(result.first,method), result.second); 
    file << result.first << "\t" << result.second << std::endl;
  }
  file.close();
}

void MeasureLA::
write_report_text_ms(std::string name, LA_Filler_Fitter::Method method) const {
  fstream file((name+".dat").c_str(),std::ios::out);
  const std::string key = ".*"+granularity(MODULESUMMARY)+LA_Filler_Fitter::method(method);
  for(Book::const_iterator it = book.begin(key); it!=book.end(); ++it) {
    const TF1*const f = it->second->GetFunction("gaus");
    if(f) {
      file << it->first << "\t"
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
    LA_Filler_Fitter::Method method = (LA_Filler_Fitter::Method) p.getParameter<int32_t>("Method");
    std::vector<double> slopes(p.getParameter<std::vector<double> >("Slopes"));    assert(slopes.size()==14);
    std::vector<double> offsets(p.getParameter<std::vector<double> >("Offsets"));  assert(offsets.size()==14);
    std::vector<double> pulls(p.getParameter<std::vector<double> >("Pulls"));      assert(pulls.size()==14);
    
    for(unsigned i=0; i<14; i++) {
      const std::pair<unsigned,LA_Filler_Fitter::Method> key( i, method);
      offset[key] = offsets[i];
      slope[key] = slopes[i];
      error_scaling[key] = pulls[i];
    }
  }
}

inline
void MeasureLA::
calibrate(const std::pair<unsigned,LA_Filler_Fitter::Method> key, LA_Filler_Fitter::Result& result) const {
  result.calMeasured = std::make_pair<float,float>( ( result.measured.first - offset.find(key)->second ) / slope.find(key)->second ,
						    result.measured.second * error_scaling.find(key)->second / slope.find(key)->second );
}
  
std::pair<uint32_t,LA_Filler_Fitter::Method> MeasureLA::
calibration_key(const std::string layer, const LA_Filler_Fitter::Method method) {
  boost::regex format(".*(T[IO]B)_layer(\\d)([as]).*");
  const bool TIB = "TIB" == boost::regex_replace(layer, format, "\\1");
  const bool stereo = "s" == boost::regex_replace(layer, format, "\\3");
  const unsigned layerNum = boost::lexical_cast<unsigned>(boost::regex_replace(layer, format, "\\2"));
  return std::make_pair(LA_Filler_Fitter::layer_index(TIB,stereo,layerNum),method);
}

std::pair<uint32_t,LA_Filler_Fitter::Method> MeasureLA::
calibration_key(const uint32_t detid, const LA_Filler_Fitter::Method method) {
  const bool TIB = SiStripDetId(detid).subDetector() == SiStripDetId::TIB;
  const bool stereo = TIB ? TIBDetId(detid).stereo() : TOBDetId(detid).stereo();
  const unsigned layer = TIB ? TIBDetId(detid).layer() : TOBDetId(detid).layer();
  return std::make_pair(LA_Filler_Fitter::layer_index(TIB,stereo,layer),method);
}

}

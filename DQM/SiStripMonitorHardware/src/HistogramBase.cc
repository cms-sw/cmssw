#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "DQM/SiStripMonitorHardware/interface/HistogramBase.hh"



void HistogramBase::fillHistogram(MonitorElement* histogram, 
				  double value, 
				  double weight)
{
  if (histogram) histogram->Fill(value,weight);
}



void HistogramBase::fillTkHistoMap(TkHistoMap *aMap,
				   uint32_t & detid,
				   float value
				   ){
  if (aMap) aMap->fill(detid,value);
}


bool HistogramBase::isTkHistoMapEnabled(std::string aName){
  return histogramConfig_[aName].enabled;
}

void HistogramBase::getConfigForHistogram(const std::string& configName, 
					  const edm::ParameterSet& psetContainingConfigPSet, 
					  std::ostringstream* pDebugStream
					  )
{

  HistogramConfig config;
  config.enabled = false;
  config.nBins = 0;
  config.min = config.max = 0.;

  const std::string psetName = configName+std::string("HistogramConfig");

  bool isTimeHisto = configName.find("vsTime") != configName.npos;

  if (psetContainingConfigPSet.exists(psetName)) {
    const edm::ParameterSet& pset = psetContainingConfigPSet.getUntrackedParameter<edm::ParameterSet>(psetName);
    config.enabled = (pset.exists("Enabled") ? pset.getUntrackedParameter<bool>("Enabled") : true);
    if (config.enabled) {
      config.nBins = (pset.exists("NBins") ? pset.getUntrackedParameter<unsigned int>("NBins") : 600);
      config.min = (pset.exists("Min") ? pset.getUntrackedParameter<double>("Min") : 0);
      config.max = (pset.exists("Max") ? pset.getUntrackedParameter<double>("Max") : 3600);
      if (config.nBins) {
        if (pDebugStream) (*pDebugStream) << "[HistogramBase]\tHistogram: " << configName << "\tEnabled"
                                          << "\tNBins: " << config.nBins << "\tMin: " << config.min << "\tMax: " << config.max << std::endl;
      } else {
        if (pDebugStream) (*pDebugStream) << "[HistogramBase]\tHistogram: " << configName << "\tEnabled" << std::endl;
      }
    } else {
      config.enabled = false;
      config.nBins = 0;
      config.min = config.max = 0.;
      if (pDebugStream) (*pDebugStream) << "[HistogramBase]\tHistogram: " << configName << "\tDisabled" << std::endl;
    }
  }
  else {
    config.enabled = false;
    config.nBins = 0;
    config.min = config.max = 0.;
    if (pDebugStream) (*pDebugStream) << "[HistogramBase]\tHistogram: " << configName << "\tDisabled" << std::endl;
  }

  //for timehisto, if global parameter exists => overwrite individual settings
  if (psetContainingConfigPSet.exists("TimeHistogramConfig") && isTimeHisto)
    {
      const edm::ParameterSet& pset = psetContainingConfigPSet.getUntrackedParameter<edm::ParameterSet>("TimeHistogramConfig");
      config.nBins = (pset.exists("NBins") ? pset.getUntrackedParameter<unsigned int>("NBins") : 600);
      config.min = (pset.exists("Min") ? pset.getUntrackedParameter<double>("Min") : 0);
      config.max = (pset.exists("Max") ? pset.getUntrackedParameter<double>("Max") : 3600);
    } 
  
  histogramConfig_[configName] = config;
}

MonitorElement* HistogramBase::bookHistogram(const std::string& configName,
					     const std::string& name, 
					     const std::string& title,
					     const unsigned int nBins, 
					     const double min, 
					     const double max,
					     const std::string& xAxisTitle
					     )
{

  if (histogramConfig_[configName].enabled) {
    MonitorElement* histo = dqm_->book1D(name,title,nBins,min,max);
    histo->setAxisTitle(xAxisTitle,1);
    return histo;
  } else {
    return NULL;
  }

}

MonitorElement* HistogramBase::bookHistogram(const std::string& configName,
					     const std::string& name, 
					     const std::string& title, 
					     const std::string& xAxisTitle
					     )
{
  return bookHistogram(configName,name,title,histogramConfig_[configName].nBins,histogramConfig_[configName].min,histogramConfig_[configName].max,xAxisTitle);

}

MonitorElement* HistogramBase::book2DHistogram(const std::string& configName,
					       const std::string& name, 
					       const std::string& title,
					       const unsigned int nBins, 
					       const double min, 
					       const double max,
					       const unsigned int nBinsY, 
					       const double minY, 
					       const double maxY,
					       const std::string& xAxisTitle,
					       const std::string& yAxisTitle
					       )
{
  if (histogramConfig_[configName].enabled) {
    MonitorElement* histo = dqm_->book2D(name,title,nBins,min,max,nBinsY,minY,maxY);
    histo->setAxisTitle(xAxisTitle,1);
    histo->setAxisTitle(yAxisTitle,2);
    return histo;
  } else {
    return NULL;
  }
}

MonitorElement* HistogramBase::bookProfile(const std::string& configName,
					   const std::string& name,
					   const std::string& title,
					   const double minY, 
					   const double maxY
					   )
{
 
  if (histogramConfig_[configName].enabled) {
    MonitorElement* histo = dqm_->bookProfile(name,
					      title,
					      histogramConfig_[configName].nBins,
					      histogramConfig_[configName].min,
					      histogramConfig_[configName].max,
					      minY,
					      maxY
					      );

    histo->setAxisTitle("",1);
    //automatically set the axis range: will accomodate new values keeping the same number of bins.
    histo->getTProfile()->SetBit(TH1::kCanRebin);
    return histo;
  } else {
    return NULL;
  }
}
 

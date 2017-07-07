#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "DQM/SiStripMonitorHardware/interface/HistogramBase.hh"



void HistogramBase::fillHistogram(HistogramConfig & histogram, 
				  double value, 
				  double weight)
{
  if (histogram.monitorEle) histogram.monitorEle->Fill(value,weight);
}


void HistogramBase::fillHistogram(MonitorElement* histogram,
				  double value, 
				  double weight)
{
  if (histogram) histogram->Fill(value,weight);
}


void HistogramBase::fillTkHistoMap(TkHistoMap* aMap,
				   uint32_t & detid,
				   float value
				   ){
  if (aMap) aMap->fill(detid,value);
}


void HistogramBase::getConfigForHistogram(HistogramConfig & aConfig,
					  const std::string& configName, 
					  const edm::ParameterSet& psetContainingConfigPSet, 
					  std::ostringstream* pDebugStream
					  )
{

  aConfig.monitorEle = nullptr;
  aConfig.enabled = false;
  aConfig.nBins = 0;
  aConfig.min = aConfig.max = 0.;

  const std::string psetName = configName+std::string("HistogramConfig");

  bool isTimeHisto = configName.find("vsTime") != configName.npos;

  if (psetContainingConfigPSet.exists(psetName)) {
    const edm::ParameterSet& pset = psetContainingConfigPSet.getUntrackedParameter<edm::ParameterSet>(psetName);
    aConfig.enabled = (pset.exists("Enabled") ? pset.getUntrackedParameter<bool>("Enabled") : true);
    if (aConfig.enabled) {
      aConfig.nBins = (pset.exists("NBins") ? pset.getUntrackedParameter<unsigned int>("NBins") : 600);
      aConfig.min = (pset.exists("Min") ? pset.getUntrackedParameter<double>("Min") : 0);
      aConfig.max = (pset.exists("Max") ? pset.getUntrackedParameter<double>("Max") : 3600);
      if (aConfig.nBins) {
        if (pDebugStream) (*pDebugStream) << "[HistogramBase]\tHistogram: " << configName << "\tEnabled"
                                          << "\tNBins: " << aConfig.nBins << "\tMin: " << aConfig.min << "\tMax: " << aConfig.max << std::endl;
      } else {
        if (pDebugStream) (*pDebugStream) << "[HistogramBase]\tHistogram: " << configName << "\tEnabled" << std::endl;
      }
    } else {
      aConfig.enabled = false;
      aConfig.nBins = 0;
      aConfig.min = aConfig.max = 0.;
      if (pDebugStream) (*pDebugStream) << "[HistogramBase]\tHistogram: " << configName << "\tDisabled" << std::endl;
    }
  }
  else {
    aConfig.enabled = false;
    aConfig.nBins = 0;
    aConfig.min = aConfig.max = 0.;
    if (pDebugStream) (*pDebugStream) << "[HistogramBase]\tHistogram: " << configName << "\tDisabled" << std::endl;
  }

  //for timehisto, if global parameter exists => overwrite individual settings
  if (psetContainingConfigPSet.exists("TimeHistogramConfig") && isTimeHisto)
    {
      const edm::ParameterSet& pset = psetContainingConfigPSet.getUntrackedParameter<edm::ParameterSet>("TimeHistogramConfig");
      aConfig.nBins = (pset.exists("NBins") ? pset.getUntrackedParameter<unsigned int>("NBins") : 600);
      aConfig.min = (pset.exists("Min") ? pset.getUntrackedParameter<double>("Min") : 0);
      aConfig.max = (pset.exists("Max") ? pset.getUntrackedParameter<double>("Max") : 3600);
    } 
  
}

void HistogramBase::bookHistogram(DQMStore::IBooker & ibooker , HistogramConfig & aConfig,
				  const std::string& name, 
				  const std::string& title,
				  const unsigned int nBins, 
				  const double min, 
				  const double max,
				  const std::string& xAxisTitle
				  )
{

  if (aConfig.enabled) {
    aConfig.monitorEle = ibooker.book1D(name,title,nBins,min,max);
    aConfig.monitorEle->setAxisTitle(xAxisTitle,1);
  } else {
    aConfig.monitorEle = nullptr;
  }

}

void HistogramBase::bookHistogram(DQMStore::IBooker & ibooker , HistogramConfig & aConfig,
				  MonitorElement* & aHist,
				  const std::string& name, 
				  const std::string& title,
				  const unsigned int nBins, 
				  const double min, 
				  const double max,
				  const std::string& xAxisTitle
				  )
{

  if (aConfig.enabled) {
    aHist = ibooker.book1D(name,title,nBins,min,max);
    aHist->setAxisTitle(xAxisTitle,1);
  } else {
    aHist = nullptr;
  }

}

void HistogramBase::bookHistogram(DQMStore::IBooker & ibooker , HistogramConfig & aConfig,
				  const std::string& name, 
				  const std::string& title, 
				  const std::string& xAxisTitle
				  )
{
  return bookHistogram(ibooker , aConfig,name,title,aConfig.nBins,aConfig.min,aConfig.max,xAxisTitle);
  
}

void HistogramBase::book2DHistogram(DQMStore::IBooker & ibooker , HistogramConfig & aConfig,
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
  if (aConfig.enabled) {
    aConfig.monitorEle = ibooker.book2D(name,title,nBins,min,max,nBinsY,minY,maxY);
    aConfig.monitorEle->setAxisTitle(xAxisTitle,1);
    aConfig.monitorEle->setAxisTitle(yAxisTitle,2);
  } else {
    aConfig.monitorEle=nullptr;
  }
}


void HistogramBase::book2DHistogram(DQMStore::IBooker & ibooker , HistogramConfig & aConfig,
				    MonitorElement* & aHist,
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
  if (aConfig.enabled) {
    aHist = ibooker.book2D(name,title,nBins,min,max,nBinsY,minY,maxY);
    aHist->setAxisTitle(xAxisTitle,1);
    aHist->setAxisTitle(yAxisTitle,2);
  } else {
    aHist=nullptr;
  }
}


void HistogramBase::bookProfile(DQMStore::IBooker & ibooker , HistogramConfig & aConfig,
				const std::string& name,
				const std::string& title,
				const unsigned int nBins, 
				const double min, 
				const double max,
				const double minY, 
				const double maxY,
				const std::string& xAxisTitle,
				const std::string& yAxisTitle
				)
{
  
  if (aConfig.enabled) {
    aConfig.monitorEle = ibooker.bookProfile(name,
					   title,
					   nBins,
					   min,
					   max,
					   minY,
					   maxY
					   );
    
    aConfig.monitorEle->setAxisTitle(xAxisTitle,1);
    aConfig.monitorEle->setAxisTitle(yAxisTitle,2);
  } else {
    aConfig.monitorEle = nullptr;
  }
}

void HistogramBase::bookProfile(DQMStore::IBooker & ibooker , HistogramConfig & aConfig,
				const std::string& name,
				const std::string& title,
				const double minY, 
				const double maxY,
				const std::string& xAxisTitle,
				const std::string& yAxisTitle
				)
{
  
  bookProfile(ibooker , aConfig,
	      name,
	      title,
	      aConfig.nBins,
	      aConfig.min,
	      aConfig.max,
	      minY,
	      maxY,
	      xAxisTitle,
	      yAxisTitle);

  //automatically set the axis range: will accomodate new values keeping the same number of bins.
  if (aConfig.monitorEle) aConfig.monitorEle->getTProfile()->SetCanExtend(TH1::kAllAxes);
}
 

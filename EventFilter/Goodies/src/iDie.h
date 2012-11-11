#ifndef EVENTFILTER_GOODIES_IDIE_H
#define EVENTFILTER_GOODIES_IDIE_H

//XDAQ
#include "xdaq/Application.h"

#include "xdata/String.h"
#include "xdata/Double.h"
#include "xdata/Float.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Boolean.h"
#include "xdata/ActionListener.h"

#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/Method.h"

#include "xgi/Utils.h"
#include "xgi/Input.h"
#include "xgi/Output.h"
#include "xgi/Method.h"

#include "toolbox/net/URN.h"
#include "toolbox/fsm/exception/Exception.h"
#include "toolbox/task/TimerListener.h"

//C++2011
#include <atomic>

//C++
#include <list>
#include <vector>
#include <deque>

//C
#include <sys/time.h>
#include <math.h>

//ROOT
#include "TFile.h"
#include "TTree.h"

//framework
#include "FWCore/Framework/interface/EventProcessor.h"
#include "DQMServices/Core/src/DQMService.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

//CMSSW EventFilter
#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/TriggerReportDef.h"

#define MODNAMES 25

namespace evf {

  int modlistSortFunction( const void *a, const void *b);

  namespace internal{
   struct fu{
      time_t tstamp;
      unsigned int ccount;
      std::vector<pid_t> cpids;
      std::vector<std::string> signals;
      std::vector<std::string> stacktraces;
    };
   struct rate{
     int nproc;
     int nsub;
     int nrep;
     int npath;
     int nendpath;
     int ptimesRun[evf::max_paths];
     int ptimesPassedPs[evf::max_paths];
     int ptimesPassedL1[evf::max_paths];
     int ptimesPassed[evf::max_paths];
     int ptimesFailed[evf::max_paths];
     int ptimesExcept[evf::max_paths];
     int etimesRun[evf::max_endpaths];
     int etimesPassedPs[evf::max_endpaths];
     int etimesPassedL1[evf::max_endpaths];
     int etimesPassed[evf::max_endpaths];
     int etimesFailed[evf::max_endpaths];
     int etimesExcept[evf::max_endpaths];
   };

  }
  typedef std::map<std::string,internal::fu> fmap;
  typedef fmap::iterator ifmap;
  
  class iDie :
    public xdaq::Application,
    public xdata::ActionListener,
    public toolbox::task::TimerListener
  {
  public:
    //
    // xdaq instantiator macro
    //
    XDAQ_INSTANTIATOR();
  
    //
    // construction/destruction
    //
    iDie(xdaq::ApplicationStub *s);
    virtual ~iDie();
    //UI
    void defaultWeb(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void summaryTable(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void detailsTable(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void dumpTable(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void updater(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void iChoke(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void iChokeMiniInterface(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void spotlight(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    //AI
    void postEntry(xgi::Input*in,xgi::Output*out)
      throw (xgi::exception::Exception);
    void postEntryiChoke(xgi::Input*in,xgi::Output*out)
      throw (xgi::exception::Exception);
    
    // *fake* fsm soap command callback
    xoap::MessageReference fsmCallback(xoap::MessageReference msg)
      throw (xoap::exception::Exception);

    // xdata:ActionListener interface
    void actionPerformed(xdata::Event& e);

    //toolbox::Task::TimerListener interface
    void timeExpired(toolbox::task::TimerEvent& e);

  private:

    struct sorted_indices{
      sorted_indices(const std::vector<int> &arr) : arr_(arr)
      {
	ind_.resize(arr_.size(),0);
	unsigned int i = 0;
	while(i<ind_.size()) {ind_[i] = i; i++;}
	std::sort(ind_.rbegin(),ind_.rend(),*this);
      }
      int operator[](size_t ind) const {return arr_[ind_[ind]];}
      
      bool operator()(const size_t a, const size_t b) const
      {
	return arr_[a]<arr_[b];
      }
      int ii(size_t ind){return ind_[ind];}
      std::vector<int> ind_;
      const std::vector<int> &arr_;
    };
    //
    // private member functions
    //
    class lsStat;
    class commonLsStat;

    void reset();
    void parseModuleLegenda(std::string);
    void parseModuleHisto(const char *, unsigned int);
    void parsePathLegenda(std::string);
    void parseDatasetLegenda(std::string);
    void parsePathHisto(const unsigned char *, unsigned int);
    void initFramework();
    void deleteFramework();
    void initMonitorElements();
    void initMonitorElementsStreams();
    void initMonitorElementsDatasets();
    void fillDQMStatHist(unsigned int nbsIdx, unsigned int lsid);
    void fillDQMModFractionHist(unsigned int nbsIdx, unsigned int lsid, unsigned int nonIdle,
		                 std::vector<std::pair<unsigned int, unsigned int>> offenders);
 
    void updateRollingHistos(unsigned int nbsIdx, unsigned int lsid, lsStat * lst, commonLsStat * clst, bool roll);
    void updateStreamHistos(unsigned int forls, commonLsStat *clst, commonLsStat *prevclst);
    void updateDatasetHistos(unsigned int forls, commonLsStat *clst, commonLsStat *prevclst);
    void doFlush();
    void perLumiFileSaver(unsigned int lsid);
    void perTimeFileSaver();
    void initDQMEventInfo();
    void setRunStartTimeStamp();

    //
    // member data
    //

    // message logger
    Logger                          log_;
    std::string                     dqmState_;		
    // monitored parameters
    xdata::String                   url_;
    xdata::String                   class_;
    xdata::UnsignedInteger32        instance_;
    xdata::String                   hostname_;

    xdata::UnsignedInteger32        runNumber_;
    unsigned int                    lastRunNumberSet_;
    bool                            runActive_;

    xdata::UnsignedInteger32        flashRunNumber_;

    //CPU load flashlist
    std::list<std::string>          monNames_;
    xdata::InfoSpace                *cpuInfoSpace_;
    xdata::UnsignedInteger32        flashLoadLs_;
    std::atomic<unsigned int>       cpuLoadLastLs_;
    std::atomic<unsigned int>       cpuLoadSentLs_;

    float                           cpuLoad_[4000];
    float                           cpuLoadPS_[4000];
    float                           cpuLoadTime7_[4000];
    float                           cpuLoadTime12_[4000];
    float                           cpuLoadTime16_[4000];
    float                           cpuLoadRate_[4000];

    xdata::Float                    flashLoad_;
    xdata::Float                    flashLoadPS_;
    xdata::Float                    flashLoadTime7_;
    xdata::Float                    flashLoadTime12_;
    xdata::Float                    flashLoadTime16_;
    xdata::Float                    flashLoadRate_;


    //CPU peak load flashlist
    xdata::InfoSpace                *cpuInfoSpaceMax_;
    xdata::UnsignedInteger32        flashLoadMaxLs_;
    std::atomic<unsigned int>       loadMaxLs_;
 
    float                           loadMax_;
    float                           loadMaxPS_;
    float                           loadMaxTime7_;
    float                           loadMaxTime12_;
    float                           loadMaxTime16_;
    float                           loadMaxRate_;

    xdata::Float                    flashLoadMax_;
    xdata::Float                    flashLoadMaxPS_;
    xdata::Float                    flashLoadMaxTime7_;
    xdata::Float                    flashLoadMaxTime12_;
    xdata::Float                    flashLoadMaxTime16_;
    xdata::Float                    flashLoadMaxRate_;

    //EventInfo
    MonitorElement * runId_;
    MonitorElement * lumisecId_;
    MonitorElement * eventId_;
    MonitorElement * eventTimeStamp_;
    MonitorElement * runStartTimeStamp_;

    MonitorElement * processTimeStampMe_;
    MonitorElement * processLatencyMe_;
    MonitorElement * processEventsMe_;
    MonitorElement * processEventRateMe_;
    MonitorElement * nUpdatesMe_;
    MonitorElement * processIdMe_;
    MonitorElement * processStartTimeStampMe_;
    MonitorElement * hostNameMe_;
    MonitorElement * processNameMe_;
    MonitorElement * workingDirMe_;
    MonitorElement * cmsswVerMe_;

    float runTS_;
    float latencyTS_;

    xdata::String                   dqmCollectorHost_;
    xdata::String                   dqmCollectorPort_;
    fmap                            fus_;
    
    unsigned int                    totalCores_;
    unsigned int                    nstates_;   
    std::vector<int>                cpuentries_;
    std::vector<std::vector<int> >  cpustat_;
    std::vector<std::string>        mapmod_;
    unsigned int                    last_ls_;
    std::vector<TriggerReportStatic>trp_;
    std::vector<int>                trpentries_;
    std::vector<std::string>        mappath_;
    //root stuff
    TFile                          *f_;
    TTree                          *t_;
    TBranch                        *b_;
    TBranch                        *b1_;
    TBranch                        *b2_;
    TBranch                        *b3_;
    TBranch                        *b4_;
    int                            *datap_;
    TriggerReportStatic            *trppriv_;
    internal::rate                  r_;

    //message statistics 
    int                             nModuleLegendaMessageReceived_;
    int                             nPathLegendaMessageReceived_;
    int                             nModuleLegendaMessageWithDataReceived_;
    int                             nPathLegendaMessageWithDataReceived_;
    int                             nModuleHistoMessageReceived_;
    int                             nPathHistoMessageReceived_;
    timeval                         runStartDetectedTimeStamp_;
    timeval                         lastModuleLegendaMessageTimeStamp_;
    timeval                         lastPathLegendaMessageTimeStamp_;

    int                             nDatasetLegendaMessageReceived_;
    int                             nDatasetLegendaMessageWithDataReceived_;
    timeval                         lastDatasetLegendaMessageTimeStamp_;

    //DQM histogram statistics
    std::vector<unsigned int> epInstances;
    std::vector<unsigned int> epMax;
    std::vector<float> HTscaling;
    std::vector<unsigned int> nbMachines;
    std::vector<float> machineWeight;
    std::vector<float> machineWeightInst;

    std::vector<std::string > endPathNames_;
    std::vector<std::string > datasetNames_;

    class commonLsStat {
      
      public:
      unsigned int ls_;
      std::vector<float> rateVec_;
      std::vector<float> busyVec_;
      std::vector<float> busyCPUVec_;
      std::vector<float> busyVecTheor_;
      std::vector<float> busyCPUVecTheor_;
      std::vector<unsigned int> nbMachines;
      std::vector<unsigned int> endPathCounts_; 
      std::vector<unsigned int> datasetCounts_; 
      commonLsStat(unsigned int lsid,unsigned int classes) {
        for (size_t i=0;i<classes;i++) {
	  rateVec_.push_back(0.);
	  busyVec_.push_back(0.);
	  busyCPUVec_.push_back(0.);
	  busyVecTheor_.push_back(0.);
	  busyCPUVecTheor_.push_back(0.);
	  nbMachines.push_back(0);
	}
	ls_=lsid;
      }

      void setBusyForClass(unsigned int classIdx,float rate,float busy,float busyTheor, float busyCPU, float busyCPUTheor, unsigned int nMachineReports) {
	rateVec_[classIdx]=rate;
	busyVec_[classIdx]=busy;
	busyCPUVec_[classIdx]=busyCPU;
	busyVecTheor_[classIdx]=busyTheor;
	busyCPUVecTheor_[classIdx]=busyCPUTheor;
	nbMachines[classIdx]=nMachineReports;
      }

      float getTotalRate() {
	float totRate=0;
	for (size_t i=0;i<rateVec_.size();i++) totRate+=rateVec_[i];
	return totRate;
      } 

      float getBusyTotalFrac(bool procstat,std::vector<float> & machineWeightInst) {
	double sum=0;
	double sumMachines=0;
	for (size_t i=0;i<busyVec_.size();i++) {
	  if (!procstat)
	    sum+=machineWeightInst[i]*nbMachines.at(i)*busyVec_[i];
	  else
	    sum+=machineWeightInst[i]*nbMachines.at(i)*busyCPUVec_[i];
	  sumMachines+=machineWeightInst[i]*nbMachines.at(i);
	}
	if (sumMachines>0)
	  return float(sum/sumMachines);
	else return 0.;
      }

      float getBusyTotalFracTheor(bool procstat,std::vector<float> & machineWeight) {
	float sum=0;
	float sumMachines=0;
	for (size_t i=0;i<busyVecTheor_.size() && i<nbMachines.size();i++) {
	  if (!procstat)
	    sum+=machineWeight[i]*nbMachines[i]*busyVecTheor_[i];
	  else
	    sum+=machineWeight[i]*nbMachines[i]*busyCPUVecTheor_[i];
	  sumMachines+=machineWeight[i]*nbMachines[i];
	}
	if (sumMachines>0)
	  return sum/sumMachines;
	else return 0.;
      }

      unsigned int getNReports() {
        unsigned int sum=0;
	for (size_t i=0;i<nbMachines.size();i++) sum+=nbMachines[i];
	return sum;
      }

      std::string printInfo() {
	std::ostringstream info;
	for (size_t i=0;i<rateVec_.size();i++) {
	  info << i << "/r:" << rateVec_[i] <<"/b:"<<busyVec_[i]<<"/n:"<<nbMachines[i]<<"; ";
	}
	return info.str();
      }
    };

    class lsStat {
      public:
      unsigned int ls_;
      bool updated_;
      unsigned int nbSubs_;
      unsigned int nSampledNonIdle_;
      unsigned int nSampledNonIdle2_;
      unsigned int nSampledIdle_;
      unsigned int nSampledIdle2_;
      unsigned int nProc_;
      unsigned int nProc2_;
      unsigned int nCPUBusy_;
      unsigned int nReports_;
      unsigned int nMaxReports_;
      double rateAvg;
      double rateErr;
      double evtTimeAvg;
      double evtTimeErr;
      double fracWaitingAvg;
      double fracCPUBusy_;
      unsigned int nmodulenames_;
      unsigned int sumDeltaTms_;
      float avgDeltaT_;
      float avgDeltaT2_;
      std::pair<unsigned int,unsigned int> *moduleSamplingSums;

      lsStat(unsigned int ls, unsigned int nbSubs,unsigned int maxreps,unsigned int nmodulenames):
	ls_(ls),updated_(true),nbSubs_(nbSubs),
	nSampledNonIdle_(0),nSampledNonIdle2_(0),nSampledIdle_(0),nSampledIdle2_(0),
	nProc_(0),nProc2_(0),nCPUBusy_(0),nReports_(0),nMaxReports_(maxreps),nmodulenames_(nmodulenames),
	sumDeltaTms_(0),avgDeltaT_(23),avgDeltaT2_(0)
      {
        moduleSamplingSums = new std::pair<unsigned int,unsigned int>[nmodulenames_];
	for (unsigned int i=0;i<nmodulenames_;i++) {
	  moduleSamplingSums[i].first=i;
	  moduleSamplingSums[i].second=0;
	}
      }

      ~lsStat() {
         delete moduleSamplingSums;
      }

      void update(unsigned int nSampledNonIdle,unsigned int nSampledIdle, 
	          unsigned int nProc,unsigned int ncpubusy, unsigned int deltaTms)
      {
	nReports_++;
	nSampledNonIdle_+=nSampledNonIdle;
	nSampledNonIdle2_+=pow(nSampledNonIdle,2);
	nSampledIdle_+=nSampledIdle;
	nSampledIdle2_+=pow(nSampledIdle,2);
	nProc_+=nProc;
	nProc2_+=pow(nProc,2);
	nCPUBusy_+=ncpubusy;
	sumDeltaTms_+=deltaTms;
	updated_=true;
      }

      std::pair<unsigned int,unsigned int> * getModuleSamplingPtr() {
        return moduleSamplingSums;
      }

      void deleteModuleSamplingPtr() {
        delete moduleSamplingSums;
	moduleSamplingSums=nullptr;
        nmodulenames_=0;
      }

      void calcStat()
      {
	if (!updated_) return;
	if (nReports_) {
	  float tinv = 0.001/nReports_;
	  fracCPUBusy_=nCPUBusy_*tinv;
	  avgDeltaT_ = avgDeltaT2_ = sumDeltaTms_*tinv;
	  if (avgDeltaT_==0.) {
	    avgDeltaT_=23.;//default value
	    avgDeltaT2_=0;
	  }
	  rateAvg=nProc_ / avgDeltaT_;
	  rateErr=sqrt(fabs(nProc2_ - pow(nProc_,2)))/avgDeltaT_;
	}
	else {
	  fracCPUBusy_=0.;
	  rateAvg=0.;
	  rateErr=0.;
	  avgDeltaT_=23.;
	}

	evtTimeAvg=0.;evtTimeErr=0.;fracWaitingAvg=1.;
	unsigned int sampled = nSampledNonIdle_+nSampledIdle_;
	if (rateAvg!=0. && sampled) {
	    float nAllInv = 1./sampled;
	    fracWaitingAvg= nSampledIdle_*nAllInv;
	    double nSampledIdleErr2=fabs(nSampledIdle2_ - pow(nSampledIdle_,2));
	    double nSampledNonIdleErr2=fabs(nSampledNonIdle2_ - pow(nSampledNonIdle_,2));
	    double fracWaitingAvgErr= sqrt(
			            (pow(nSampledIdle_,2)*nSampledNonIdleErr2
				     + pow(nSampledNonIdle_,2)*nSampledIdleErr2))*pow(nAllInv,2);
	    float rateAvgInv=1./rateAvg;
	    evtTimeAvg=nbSubs_ * nReports_ * (1.-fracWaitingAvg)*rateAvgInv;
	    evtTimeErr = nbSubs_ * nReports_ * sqrt(pow(fracWaitingAvg*rateErr*pow(rateAvgInv,2),2) + pow(fracWaitingAvgErr*rateAvgInv,2));
	}
	updated_=false;
      }

      float getRate() {
	if (updated_) calcStat();
	return rateAvg;
      }

      float getRateErr() {
	if (updated_) calcStat();
	return rateErr;
      }

      float getRatePerMachine() {
	if (updated_) calcStat();
	if (nReports_)
	return rateAvg/(1.*nReports_);
	return 0.;
      }

      float getRateErrPerMachine() {
	if (updated_) calcStat();
	if (nReports_)
	return rateErr/(1.*nReports_);
	return 0.;
      }

      float getEvtTime() {
	if (updated_) calcStat();
	return evtTimeAvg;
      }

      float getEvtTimeErr() {
	if (updated_) calcStat();
	return evtTimeErr;
      }

      unsigned int getNSampledNonIdle() {
	if (updated_) calcStat();
        return nSampledNonIdle_;
      }

      float getFracBusy() {
	if (updated_) calcStat();
	return 1.-fracWaitingAvg;
      }

      float getFracCPUBusy() {
	if (updated_) calcStat();
	return fracCPUBusy_;
      }

      unsigned int getReports() {
        return nReports_;
      }

      float getDt() {
	if (updated_) calcStat();
        return avgDeltaT2_;
      }

      std::vector<std::pair<unsigned int, unsigned int>> getOffendersVector() {
        std::vector<std::pair<unsigned int, unsigned int>> ret;
	if (updated_) calcStat();
	if (moduleSamplingSums) {
	  //make a copy for sorting
	  std::pair<unsigned int,unsigned int> *moduleSumsCopy = new std::pair<unsigned int,unsigned int>[nmodulenames_];
	  memcpy(moduleSumsCopy,moduleSamplingSums,nmodulenames_*sizeof(std::pair<unsigned int,unsigned int>));

	  std::qsort((void *)moduleSumsCopy, nmodulenames_,
	             sizeof(std::pair<unsigned int,unsigned int>), modlistSortFunction);

	  unsigned int count=0;
	  unsigned int saveidx=0;
	  while (saveidx < MODNAMES && count<nmodulenames_)
	  {
            if (moduleSumsCopy[count].first==2) {count++;continue;}
            ret.push_back(moduleSumsCopy[count]);
	    saveidx++;
	    count++;
	  }
	  delete moduleSumsCopy;
	}
        return ret;
      }

      float getOffenderFracAt(unsigned int x) {
        if (x<nmodulenames_) {
	  if (updated_) calcStat();
	  float total = nSampledNonIdle_+nSampledIdle_;
	  if (total>0.) {
	    for (size_t i=0;i<nmodulenames_;i++) {
	      if (moduleSamplingSums[i].first==x)
	      return moduleSamplingSums[i].second/total;
	    }
	  }
	}
	return 0.;
      }
    };


    //DQM
    boost::shared_ptr<std::vector<edm::ParameterSet> > pServiceSets_;
    edm::ServiceToken               serviceToken_;
    edm::EventProcessor             *evtProcessor_;
    bool                            meInitialized_;
    bool                            meInitializedStreams_;
    bool                            meInitializedDatasets_;
    DQMService                      *dqmService_;
    DQMStore                        *dqmStore_;
    std::string                     configString_;
    xdata::Boolean                  dqmEnabled_;
    xdata::Boolean                  debugMode_;

    std::map<unsigned int,int> nbSubsList;
    std::map<int,unsigned int> nbSubsListInv;
    unsigned int nbSubsClasses;
    std::vector<MonitorElement*> meVecRate_;
    std::vector<MonitorElement*> meVecTime_;
    std::vector<MonitorElement*> meVecOffenders_;
    MonitorElement * rateSummary_;
    MonitorElement * reportPeriodSummary_;
    MonitorElement * timingSummary_;
    MonitorElement * busySummary_;
    MonitorElement * busySummary2_;
    MonitorElement * busySummaryUncorr1_;
    MonitorElement * busySummaryUncorr2_;
    MonitorElement * fuReportsSummary_;
    MonitorElement * daqBusySummary_;
    MonitorElement * daqBusySummary2_;
    MonitorElement * daqTotalRateSummary_;
    MonitorElement * busyModules_;
    unsigned int summaryLastLs_;
    std::vector<std::map<unsigned int, unsigned int> > occupancyNameMap;
    //1 queue per number of subProcesses (and one common)
    std::deque<commonLsStat*> commonLsHistory;
    std::deque<lsStat*> * lsHistory;

    //endpath statistics
    std::vector<MonitorElement *> endPathRates_;

    //dataset statistics
    std::vector<MonitorElement *> datasetRates_;

    std::vector<unsigned int> currentLs_;

    xdata::UnsignedInteger32 saveLsInterval_;
    unsigned int ilumiprev_;
    std::list<std::string> pastSavedFiles_;
    xdata::String dqmSaveDir_;
    xdata::Boolean dqmFilesWritable_;
    xdata::String topLevelFolder_;
    unsigned int savedForLs_;
    std::string fileBaseName_;
    bool writeDirectoryPresent_;

    timeval * reportingStart_;
    unsigned int lastSavedForTime_;

    unsigned int dsMismatch;
  }; // class iDie

  int modlistSortFunction( const void *a, const void *b)
  {
    std::pair<unsigned int,unsigned int> intOne = *((std::pair<unsigned int,unsigned int>*)a);
    std::pair<unsigned int,unsigned int> intTwo = *((std::pair<unsigned int,unsigned int>*)b);
    if (intOne.second > intTwo.second)
      return -1;
    if (intOne.second == intTwo.second)
      return 0;
    return 1;
  }

  float fround(float val, float mod) {
    return val - fmod(val,mod);
  }

} // namespace evf


#endif

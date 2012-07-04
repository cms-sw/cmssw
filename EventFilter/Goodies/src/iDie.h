#ifndef EVENTFILTER_GOODIES_IDIE_H
#define EVENTFILTER_GOODIES_IDIE_H

#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/TriggerReportDef.h"

#include "xdata/String.h"
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

#include "xdaq/Application.h"

#include "toolbox/net/URN.h"
#include "toolbox/fsm/exception/Exception.h"


#include <vector>
#include <deque>

#include <sys/time.h>

#include "TFile.h"
#include "TTree.h"

#include "FWCore/Framework/interface/EventProcessor.h"
#include "DQMServices/Core/src/DQMService.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#define MODLZSIZE 25
#define MODLZSIZELUMI 20
#define MOD_OCC_THRESHOLD 5

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
  
  class iDie : public xdaq::Application,
    public xdata::ActionListener
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
    void parsePathHisto(const unsigned char *, unsigned int);
    void initFramework();
    void deleteFramework();
    void initMonitorElements();
    void fillDQMStatHist(unsigned int nbsIdx, unsigned int lsid);
    void fillDQMModFractionHist(unsigned int nbsIdx, unsigned int lsid, unsigned int nonIdle,
		                 std::vector<std::pair<unsigned int, unsigned int>> offenders);
 
    void updateRollingHistos(unsigned int nbsIdx, unsigned int lsid, lsStat & lst, commonLsStat & clst, bool roll);
    void doFlush();
    void perLumiFileSaver(unsigned int lsid);
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

    //DQM histogram statistics
    std::vector<unsigned int> epInstances;
    std::vector<unsigned int> epMax;
    std::vector<float> HTscaling;
    std::vector<unsigned int> nbMachines;
    std::vector<float> machineWeight;
    std::vector<float> machineWeightInst;

    class commonLsStat {
      
      public:
      unsigned int ls_;
      std::vector<unsigned int> rateVec_;
      std::vector<float> busyVec_;
      std::vector<float> busyCPUVec_;
      std::vector<float> busyVecTheor_;
      std::vector<float> busyCPUVecTheor_;
      std::vector<unsigned int> nbMachines;
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
      void setBusyForClass(unsigned int classIdx,unsigned int rate,float busy,float busyTheor, float busyCPU, float busyCPUTheor, unsigned int nMachineReports) {
	rateVec_[classIdx]=rate;
	busyVec_[classIdx]=busy;
	busyCPUVec_[classIdx]=busyCPU;
	busyVecTheor_[classIdx]=busyTheor;
	busyCPUVecTheor_[classIdx]=busyCPUTheor;
	nbMachines[classIdx]=nMachineReports;
      }

      unsigned int getTotalRate() {
	unsigned int totRate=0;
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
      std::pair<unsigned int,unsigned int> *moduleSamplingSums;

      lsStat(unsigned int ls, unsigned int nbSubs,unsigned int maxreps,unsigned int nmodulenames):
	ls_(ls),updated_(false),nbSubs_(nbSubs),
	nSampledNonIdle_(0),nSampledNonIdle2_(0),nSampledIdle_(0),nSampledIdle2_(0),
	nProc_(0),nProc2_(0),nCPUBusy_(0),nReports_(0),nMaxReports_(maxreps),nmodulenames_(nmodulenames)
      {
        moduleSamplingSums = new std::pair<unsigned int,unsigned int>[nmodulenames_];
	for (unsigned int i=0;i<nmodulenames_;i++) {
	  moduleSamplingSums[i].first=i;
	  moduleSamplingSums[i].second=0;
	}
      }

      void update(unsigned int nSampledNonIdle,unsigned int nSampledIdle, unsigned int nProc,unsigned int ncpubusy) {
	nReports_++;
	nSampledNonIdle_+=nSampledNonIdle;
	nSampledNonIdle2_+=pow(nSampledNonIdle,2);
	nSampledIdle_+=nSampledIdle;
	nSampledIdle2_+=pow(nSampledIdle,2);
	nProc_+=nProc;
	nProc2_+=pow(nProc,2);
	nCPUBusy_+=ncpubusy;
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
	rateAvg=nProc_ / 23.;
	rateErr=sqrt(fabs(nProc2_ - pow(nProc_,2)))/23.;
	if (rateAvg==0.) {rateErr=0.;evtTimeAvg=0.;evtTimeErr=0.;fracWaitingAvg=0;}
	else {
	  if (nSampledNonIdle_+nSampledIdle_!=0) {
	    float nAllInv = 1./(nSampledNonIdle_+nSampledIdle_);
	    fracWaitingAvg= nSampledIdle_*nAllInv;
	    double nSampledIdleErr2=fabs(nSampledIdle2_ - pow(nSampledIdle_,2));
	    double nSampledNonIdleErr2=fabs(nSampledNonIdle2_ - pow(nSampledNonIdle_,2));
	    double fracWaitingAvgErr= sqrt(
			            (pow(nSampledIdle_,2)*nSampledNonIdleErr2
				     + pow(nSampledNonIdle_,2)*nSampledIdleErr2))*pow(nAllInv,2);
	    if (rateAvg) {
	      float rateAvgInv=1./rateAvg;
	      evtTimeAvg=nbSubs_ * nReports_ * (1.-fracWaitingAvg)*rateAvgInv;
	      evtTimeErr = nbSubs_ * nReports_ * (fracWaitingAvg*rateErr*pow(rateAvgInv,2) + fracWaitingAvgErr*rateAvgInv);
	    }
	    else {
              evtTimeAvg=0;
	      evtTimeErr=0;
	    }
	  }
	}
	if (nReports_) fracCPUBusy_=nCPUBusy_/(nReports_*1000.);
	else fracCPUBusy_=0.;
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

      std::vector<std::pair<unsigned int, unsigned int>> getOffendersVector() {
        std::vector<std::pair<unsigned int, unsigned int>> ret;
	if (updated_) calcStat();
	if (moduleSamplingSums) {
          std::qsort((void *)moduleSamplingSums, nmodulenames_,
	             sizeof(std::pair<unsigned int,unsigned int>), modlistSortFunction);
	  unsigned int count=0;
	  unsigned int saveidx=0;
	  while (saveidx < MODLZSIZE && count<nmodulenames_ && saveidx<MODLZSIZE)
	  {
            if (moduleSamplingSums[count].first==2) {count++;continue;}
            ret.push_back(moduleSamplingSums[count]);
	    saveidx++;
	    count++;
	  }
	}
        return ret;
      }
    };


    //DQM
    boost::shared_ptr<std::vector<edm::ParameterSet> > pServiceSets_;
    edm::ServiceToken               serviceToken_;
    edm::EventProcessor             *evtProcessor_;
    bool                            meInitialized_;
    DQMService                      *dqmService_;
    DQMStore                        *dqmStore_;
    std::string                     configString_;
    bool                            dqmDisabled_;

    std::map<unsigned int,int> nbSubsList;
    std::map<int,unsigned int> nbSubsListInv;
    unsigned int nbSubsClasses;
    std::vector<MonitorElement*> meVecRate_;
    std::vector<MonitorElement*> meVecTime_;
    std::vector<MonitorElement*> meVecOffenders_;
    MonitorElement * rateSummary_;
    MonitorElement * timingSummary_;
    MonitorElement * busySummary_;
    MonitorElement * busySummary2_;
    MonitorElement * fuReportsSummary_;
    MonitorElement * daqBusySummary_;
    unsigned int summaryLastLs_;
    std::vector<std::map<unsigned int, unsigned int> > occupancyNameMap;
    //1 queue per number of subProcesses (and one common)
    std::deque<commonLsStat> commonLsHistory;
    std::deque<lsStat> *lsHistory;

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


} // namespace evf


#endif

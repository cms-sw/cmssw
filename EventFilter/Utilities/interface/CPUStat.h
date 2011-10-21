#ifndef EVF_CPUSTAT
#define EVF_CPUSTAT

#include <vector>
#include <string>
#include <iostream>
#include "EventFilter/Utilities/interface/AsciiRollingChart.h"

namespace evf{

  class CurlPoster;

  class CPUStat{
  public:
    CPUStat(unsigned int nstates, unsigned int nproc, unsigned int instance, std::string iDieUrl);
    ~CPUStat();
    void addEntry(int sta)
    {
      mstat_[sta]++;
      entries_++;
    }
    void setNproc(int nproc){
      mstat_[nstates_+2]=nproc;
    }
    void reset()
    {
      for(int i = 0; i < nstates_; i++)
	mstat_[i]=0;	
      entries_ = 0;
    }
    void printStat(){
      std::cout << "dump of microstates avg.:" << entries_ << " samples" << std::endl;
      for(int i = 0; i < nstates_; i++)
	if(mstat_[i]!=0) std::cout << i << " " << float(mstat_[i])/float(entries_)
				   << std::endl;
    }
    std::string &getChart(){return chart_.draw();}
    void sendStat(unsigned int);
    void sendLegenda(const std::vector<std::string> &);
  private:
    std::string iDieUrl_;
    CurlPoster *poster_;
    int nstates_;
    int nproc_;
    int instance_;
    int entries_;
    int *mstat_;
    AsciiRollingChart chart_;
  };
}
#endif

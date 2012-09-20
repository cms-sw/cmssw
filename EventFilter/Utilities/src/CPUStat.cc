#include "EventFilter/Utilities/interface/CPUStat.h"
#include "EventFilter/Utilities/interface/CurlPoster.h"

namespace evf{

CPUStat::CPUStat(unsigned int nstates,
		 unsigned int nproc,
		 unsigned int instance,
		 std::string iDieUrl) : iDieUrl_(iDieUrl)
				      , nstates_(nstates)
				      , nproc_(nproc)
				      , instance_(instance)
				      , entries_(0)
				      , mstat_(new int[nstates_+3])
				      , chart_("busy fraction",50)
{
  poster_ = new CurlPoster(iDieUrl_);
  for(int i = 0; i < nstates_; i++)
    mstat_[i]=0;	
  mstat_[nstates_]=nproc_;
  mstat_[nstates_+1]=instance_;
  mstat_[nstates_+2]=0;
}
CPUStat::~CPUStat()
{
  delete poster_;
  delete mstat_;
}

  void CPUStat::sendStat(unsigned int lsid)
{
  chart_.flip(lsid,float(entries_-mstat_[2])/float(entries_));
  poster_->postBinary((unsigned char *)mstat_,(nstates_+4)*sizeof(int),lsid,"/postChoke");
}

void CPUStat::sendLegenda(const std::vector<std::string> &mapmod)
{
  std::string message;
  unsigned int i = 0;
  while(i<mapmod.size()){
    message+=mapmod[i];
    if(++i!=mapmod.size()) message+=",";
  }
  poster_->postString(message.c_str(),message.length(),0,CurlPoster::leg,"/postChoke");
}

}

#include "EventFilter/Utilities/interface/CPUStat.h"
#include "EventFilter/Utilities/interface/CurlPoster.h"

namespace evf{

CPUStat::CPUStat(unsigned int nstates, std::string iDieUrl) : iDieUrl_(iDieUrl)
							    , nstates_(nstates)
							    , entries_(0)
							    , mstat_(new int[nstates_])
{
  poster_ = new CurlPoster(iDieUrl_);
  for(int i = 0; i < nstates_; i++)
    mstat_[i]=0;	
}
CPUStat::~CPUStat()
{
  delete poster_;
  delete mstat_;
}

void CPUStat::sendStat(unsigned int lsid)
{
  poster_->postBinary((unsigned char *)mstat_,(nstates_+1)*sizeof(int),lsid,"/postChoke");
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

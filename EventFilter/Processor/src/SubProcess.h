#ifndef EVENTFILTER_PROCESSOR_SUB_PROCESS_H
#define EVENTFILTER_PROCESSOR_SUB_PROCESS_H

#include "EventFilter/Utilities/interface/MsgBuf.h"
#include <string>

#include <iostream>
#include <boost/shared_ptr.hpp>

// subprocess states: -1000 never started -1: crashed 0: exited successfully 1: running
// @@EM ToDo: replace magic numbers with enum

namespace evf{
  
  class SubProcess{
  public:
    SubProcess()
      : ind_(100000)
      , pid_(-1)
      , alive_(-1000)
      , restart_countdown_(0)
      , save_nbp_(0)
      , save_nba_(0)
      , save_ndqm_(0)
      , save_scalers_(0)
      , reported_inconsistent_(false)
      {}
    SubProcess(int ind, pid_t pid)
      : ind_(ind)
      , pid_(pid)
      , alive_(-1)
      , mqm_(new MasterQueue(monitor_queue_offset_+ind))
      , mqs_(new MasterQueue(ind))
      , restart_countdown_(0)
      , save_nbp_(0)
      , save_nba_(0)
      , save_ndqm_(0)
      , save_scalers_(0)
      , reported_inconsistent_(false)
      {
	mqm_->drain();
	mqs_->drain();
      }
    SubProcess(const SubProcess &b)
      : ind_(b.ind_)
      , pid_(b.pid_)
      , alive_(b.alive_)
      , mqm_(b.mqm_)
      , mqs_(b.mqs_)
      , restart_countdown_(b.restart_countdown_)
      , reported_inconsistent_(b.reported_inconsistent_)
      {
      }
    SubProcess &operator=(const SubProcess &b)
      {
	ind_=b.ind_;
	pid_=b.pid_;
	alive_=b.alive_;
	mqm_=b.mqm_;
	mqs_=b.mqs_;
	save_nbp_ = b.save_nbp_;
	save_nba_ = b.save_nba_;
	save_ndqm_ = b.save_ndqm_;
        save_scalers_ = b.save_scalers_;
	restart_countdown_=b.restart_countdown_;
	reported_inconsistent_=b.reported_inconsistent_;
	return *this;
      }
    virtual ~SubProcess()
      {
      }
    void disconnect()
      {
	mqm_->drain();
	mqs_->drain();
	mqs_->disconnect();
	mqm_->disconnect();
	save_nbp_ = 0;
	save_nba_ = 0;
	save_ndqm_ = 0;
	save_scalers_ = 0;
      }
    void setStatus(int st){
      alive_ = st;
      if(alive_ != 1) //i.e. process is no longer alive
	{
	  //save counters after last update
	  save_nbp_= prg_.nbp;
	  save_nba_= prg_.nba;
	  save_ndqm_ = prg_.dqm;
	  save_scalers_ = prg_.trp;
	}
    }
    int queueId(){return (mqm_.get()!=0 ? mqm_->id() : 0);}
    int queueStatus(){return (mqm_.get() !=0 ? mqm_->status() : 0);}
    int queueOccupancy(){return (mqm_.get() !=0 ? mqm_->occupancy() : -1);}
    int controlQueueOccupancy(){return (mqs_.get() !=0 ? mqs_->occupancy() : -1);}
    pid_t queuePidOfLastSend(){return (mqm_.get() !=0 ? mqm_->pidOfLastSend() : -1);}
    pid_t queuePidOfLastReceive(){return (mqm_.get() !=0 ? mqm_->pidOfLastReceive() : -1);}
    pid_t pid() const {return pid_;}
    int alive() const {return alive_;}
    struct prg &params(){return prg_;}
    void setParams(struct prg *p)
      {
	prg_.ls  = p->ls;
	prg_.ps  = p->ps;
	prg_.nbp = p->nbp + save_nbp_;
	prg_.nba = p->nba + save_nba_;
	prg_.Ms  = p->Ms;
	prg_.ms  = p->ms;
	prg_.dqm = p->dqm + save_ndqm_;
	prg_.trp = p->trp + save_scalers_;
      }
    int post(MsgBuf &ptr, bool isMonitor)
      {
	//	std::cout << "post called for sp " << ind_ << " type " << ptr->mtype 
	//	  << " queue ids " << mqm_->id() << " " << mqs_->id() << std::endl;
	if(isMonitor) return mqm_->post(ptr); else return mqs_->post(ptr);
      }
    unsigned long rcv(MsgBuf &ptr, bool isMonitor)
      {
	//	std::cout << "receive called for sp " << ind_ << " type " << ptr->mtype 
	//  << " queue ids " << mqm_->id() << " " << mqs_->id() << std::endl;
	if(isMonitor) return mqm_->rcv(ptr); else return mqs_->rcv(ptr);
      }
    unsigned long rcvNonBlocking(MsgBuf &ptr, bool isMonitor)
      {
	//	std::cout << "receivenb called for sp " << ind_ << " type " << ptr->mtype 
	//	  << " queue ids " << mqm_->id() << " " << mqs_->id() << std::endl;
	if(isMonitor) 
	  return mqm_->rcvNonBlocking(ptr); 
	else 
	  return mqs_->rcvNonBlocking(ptr);
      }
    int forkNew()
      {
	mqm_->drain();
	mqs_->drain();
	pid_t retval = -1;
	retval = fork();
	reported_inconsistent_ = false;
	if(retval>0)
	  {
	    pid_ = retval;
	    alive_=1;
	  }
	if(retval==0)
	  {
	    //	  freopen(filename,"w",stdout); // send all console output from children to /dev/null
	    freopen("/dev/null","w",stderr);
	  }
	return retval;
      }
    std::string const &reasonForFailed()const {return reasonForFailed_;}
    bool inInconsistentState() const {return reported_inconsistent_;}
    void setReasonForFailed(std::string r){reasonForFailed_ = r;}
    void setReportedInconsistent(){reported_inconsistent_ = true;}
    unsigned int &countdown(){return restart_countdown_;}
  private:
    int ind_;
    pid_t pid_;
    int alive_;
    boost::shared_ptr<MasterQueue> mqm_; //to be turned to real object not pointer later
    boost::shared_ptr<MasterQueue> mqs_;
    std::string reasonForFailed_;
    struct prg prg_;
    unsigned int restart_countdown_;
    static const unsigned int monitor_queue_offset_ = 200;
    int save_nbp_;
    int save_nba_;
    unsigned int save_ndqm_;
    unsigned int save_scalers_;
    bool reported_inconsistent_;
  };


}
#endif

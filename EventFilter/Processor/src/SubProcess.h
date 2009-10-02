#ifndef EVENTFILTER_PROCESSOR_SUB_PROCESS_H
#define EVENTFILTER_PROCESSOR_SUB_PROCESS_H

#include "MsgBuf.h"
#include <string>

#include <iostream>
#include <boost/shared_ptr.hpp>

namespace evf{

  class SubProcess{
  public:
    SubProcess()
      : ind_(100000)
      , pid_(-1)
      , alive_(-1)
      , restart_countdown_(0)
      {
      }
    SubProcess(int ind, pid_t pid)
      : ind_(ind)
      , pid_(pid)
      , alive_(-1)
      , mqm_(new MasterQueue(monitor_queue_offset_+ind))
      , mqs_(new MasterQueue(ind))
      , restart_countdown_(0)
      {
	std::cout << "subprocess " << ind << " create command queue" << std::endl;
	std::cout << "subprocess " << ind << " done" << std::endl;
      }
    SubProcess(const SubProcess &b)
      : ind_(b.ind_)
      , pid_(b.pid_)
      , alive_(b.alive_)
      , mqm_(b.mqm_)
      , mqs_(b.mqs_)
      , restart_countdown_(b.restart_countdown_)
      {
	std::cout << "subprocess " << ind_ << " in copyconstructor" << std::endl;
      }
    SubProcess &operator=(const SubProcess &b)
      {
	ind_=b.ind_;
	pid_=b.pid_;
	alive_=b.alive_;
	mqm_=b.mqm_;
	mqs_=b.mqs_;
	restart_countdown_=b.restart_countdown_;

	std::cout << "subprocess " << ind_ << " in assignmentoperator" << std::endl;
	return *this;
      }
    virtual ~SubProcess()
      {
	std::cout << "subprocess " << ind_ << " in destructor" << std::endl;
      }
    void disconnect()
      {
	mqs_->disconnect();
	mqm_->disconnect();
      }
    int queueId(){return (mqm_.get()!=0 ? mqm_->id() : 0);}
    int queueStatus(){return (mqm_.get() !=0 ? mqm_->status() : 0);}
    int queueOccupancy(){return (mqm_.get() !=0 ? mqm_->occupancy() : -1);}
    pid_t queuePidOfLastSend(){return (mqm_.get() !=0 ? mqm_->pidOfLastSend() : -1);}
    pid_t queuePidOfLastReceive(){return (mqm_.get() !=0 ? mqm_->pidOfLastReceive() : -1);}
    pid_t pid() const {return pid_;}
    int &alive() {return alive_;}
    struct prg &params(){return prg_;}
    void setParams(struct prg *p)
      {
	prg_.ls  = p->ls;
	prg_.ps  = p->ps;
	prg_.nbp = p->nbp;
	prg_.nba = p->nba;
	prg_.Ms  = p->Ms;
	prg_.ms  = p->ms;
      }
    int post(MsgBuf &ptr, bool isMonitor)
      {
	std::cout << "post called for sp " << ind_ 
		  << " queue ids " << mqm_->id() << " " << mqs_->id() << std::endl;
	if(isMonitor) return mqm_->post(ptr); else return mqs_->post(ptr);
      }
    unsigned long rcv(MsgBuf &ptr, bool isMonitor)
      {
	std::cout << "receive called for sp " << ind_ 
		  << " queue ids " << mqm_->id() << " " << mqs_->id() << std::endl;
	if(isMonitor) return mqm_->rcv(ptr); else return mqs_->rcv(ptr);
      }
    unsigned long rcvNonBlocking(MsgBuf &ptr, bool isMonitor)
      {
	std::cout << "receivenb called for sp " << ind_ 
		  << " queue ids " << mqm_->id() << " " << mqs_->id() << std::endl;
	if(isMonitor) 
	  return mqm_->rcvNonBlocking(ptr); 
	else 
	  return mqs_->rcvNonBlocking(ptr);
      }
    int forkNew()
      {
	pid_t retval = -1;
	retval = fork();
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
    void setReasonForFailed(std::string r){reasonForFailed_ = r;}
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
  };


}
#endif

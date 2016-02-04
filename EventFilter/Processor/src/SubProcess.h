#ifndef EVENTFILTER_PROCESSOR_SUB_PROCESS_H
#define EVENTFILTER_PROCESSOR_SUB_PROCESS_H

#include "EventFilter/Utilities/interface/MsgBuf.h"
#include "EventFilter/Utilities/interface/MasterQueue.h"
#include "EventFilter/Utilities/interface/SlaveQueue.h"
#include <string>
#include <vector>

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
      , restart_count_(0)
      , save_nbp_(0)
      , save_nba_(0)
      , save_ndqm_(0)
      , save_scalers_(0)
      , reported_inconsistent_(false)
      , nfound_invalid_(0)
      {}
    SubProcess(int ind, pid_t pid)
      : ind_(ind)
      , pid_(pid)
      , alive_(-1)
      , mqm_(new MasterQueue(monitor_queue_offset_+ind))
      , mqs_(new MasterQueue(ind))
      , restart_countdown_(0)
      , restart_count_(0)
      , save_nbp_(0)
      , save_nba_(0)
      , save_ndqm_(0)
      , save_scalers_(0)
      , reported_inconsistent_(false)
      , nfound_invalid_(0)
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
      , restart_count_(b.restart_count_)
      , reported_inconsistent_(b.reported_inconsistent_)
      , nfound_invalid_(b.nfound_invalid_)
      , postponed_trigger_updates_(b.postponed_trigger_updates_)
      {
      }
    SubProcess &operator=(const SubProcess &b);

    virtual ~SubProcess()
      {
      }
    void disconnect();

    void setStatus(int st);

    int queueId(){return (mqm_.get()!=0 ? mqm_->id() : 0);}
    int queueStatus(){return (mqm_.get() !=0 ? mqm_->status() : 0);}
    int queueOccupancy(){return (mqm_.get() !=0 ? mqm_->occupancy() : -1);}
    int controlQueueOccupancy(){return (mqs_.get() !=0 ? mqs_->occupancy() : -1);}
    pid_t queuePidOfLastSend(){return (mqm_.get() !=0 ? mqm_->pidOfLastSend() : -1);}
    pid_t queuePidOfLastReceive(){return (mqm_.get() !=0 ? mqm_->pidOfLastReceive() : -1);}
    pid_t pid() const {return pid_;}
    int alive() const {return alive_;}
    struct prg &params(){return prg_;}
    void setParams(struct prg *p);
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
    int postSlave(MsgBuf &ptr, bool isMonitor)
      {
	//	std::cout << "post called for sp " << ind_ << " type " << ptr->mtype 
	//	  << " queue ids " << mqm_->id() << " " << mqs_->id() << std::endl;
	if(isMonitor) return sqm_->post(ptr); else return sqs_->post(ptr);
      }
    unsigned long rcvSlave(MsgBuf &ptr, bool isMonitor)
      {
	//	std::cout << "receive called for sp " << ind_ << " type " << ptr->mtype 
	//  << " queue ids " << mqm_->id() << " " << mqs_->id() << std::endl;
	if(isMonitor) return sqm_->rcv(ptr); else return sqs_->rcv(ptr);
      }
    unsigned long rcvSlaveNonBlocking(MsgBuf &ptr, bool isMonitor)
      {
	//	std::cout << "receivenb called for sp " << ind_ << " type " << ptr->mtype 
	//	  << " queue ids " << mqm_->id() << " " << mqs_->id() << std::endl;
	if(isMonitor) 
	  return sqm_->rcvNonBlocking(ptr); 
	else 
	  return sqs_->rcvNonBlocking(ptr);
      }

    int forkNew();

    std::string const &reasonForFailed()const {return reasonForFailed_;}
    bool inInconsistentState() const {return reported_inconsistent_;}
    void setReasonForFailed(std::string r){reasonForFailed_ = r;}
    void setReportedInconsistent(){reported_inconsistent_ = true;}
    int &countdown(){return restart_countdown_;}    
    unsigned int &restartCount(){return restart_count_;}
    int get_save_nbp() const {return save_nbp_;}
    int get_save_nba() const {return save_nba_;}
    void found_invalid() {nfound_invalid_++;}
    unsigned int nfound_invalid() const { return nfound_invalid_;}
    void add_postponed_trigger_update(MsgBuf &);
    bool check_postponed_trigger_update(MsgBuf &, unsigned int);
    static const unsigned int monitor_queue_offset_ = 200;

  private:
    int ind_;
    pid_t pid_;
    int alive_;
    boost::shared_ptr<MasterQueue> mqm_; //to be turned to real object not pointer later
    boost::shared_ptr<MasterQueue> mqs_;
    SlaveQueue*                    sqm_;  // every subprocess will create its instance at fork 
    SlaveQueue*                    sqs_; 
    std::string reasonForFailed_;
    struct prg prg_;
    int restart_countdown_;
    unsigned int restart_count_;

    int save_nbp_;
    int save_nba_;
    unsigned int save_ndqm_;
    unsigned int save_scalers_;
    bool reported_inconsistent_;
    unsigned int nfound_invalid_;

    std::vector<MsgBuf> postponed_trigger_updates_;
  };


}
#endif

#include "SubProcess.h"
#include "FileDescriptorHandler.h"

namespace evf{

  SubProcess &SubProcess::operator=(const SubProcess &b)
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

  void SubProcess::disconnect()
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

  void SubProcess::setStatus(int st){
    alive_ = st;
    if(alive_ != 1) //i.e. process is no longer alive
      {
	//save counters after last update
	save_nbp_ = prg_.nbp;
	save_nba_ = prg_.nba;
	save_ndqm_  = prg_.dqm;
	save_scalers_ = prg_.trp;
      }
  }

  void SubProcess::setParams(struct prg *p)
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

  int SubProcess::forkNew()
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
	FileDescriptorHandler a; //handle socket file descriptors left open at fork
	sqm_ = new SlaveQueue(monitor_queue_offset_+ind_);
	sqs_ = new SlaveQueue(ind_);
      }
    return retval;
  }

}

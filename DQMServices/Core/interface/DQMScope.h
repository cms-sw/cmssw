#ifndef DQMSERVICES_CORE_DQM_SCOPE_H
#define DQMSERVICES_CORE_DQM_SCOPE_H

/** Gateway to accessing DQM core in threads other than the CMSSW thread.  */
class DQMScope {
public:
  DQMScope();
  ~DQMScope();
};

#endif  // DQMSERVICES_CORE_DQM_SCOPE_H

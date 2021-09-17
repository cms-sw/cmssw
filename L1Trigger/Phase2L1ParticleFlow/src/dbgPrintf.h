#ifndef L1Trigger_Phase2L1ParticleFlow_dbgPrintf_h
#define L1Trigger_Phase2L1ParticleFlow_dbgPrintf_h

template <typename... Args>
inline void dbgPrintf(const char *formatString, Args &&...args) {
#ifdef L1PF_DEBUG
  printf(formatString, std::forward<Args>(args)...);
#endif
}

#endif

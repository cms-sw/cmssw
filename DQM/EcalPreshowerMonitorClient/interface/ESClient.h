#ifndef ESClient_H
#define ESClient_H

#include <string>

class DQMStore;

class ESClient {

   public:

      virtual void analyze(void)      = 0;
      virtual void beginJob(DQMStore* dqmStore)     = 0;
      virtual void endJob(void)       = 0;
      virtual void beginRun(void)     = 0;
      virtual void endRun(void)       = 0;
      virtual void setup(void)	=0;
      virtual void cleanup(void)	=0;
      //  virtual int getEvtPerJob( void ) = 0;
      //  virtual int getEvtPerRun( void ) = 0;
      virtual void endLumiAnalyze(void)   =0;

      virtual ~ESClient(void) {}

};

#endif // ESClient_H


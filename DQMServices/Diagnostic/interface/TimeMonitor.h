#ifndef DIAGNOSTIC_TIME_MONITOR_H
# define DIAGNOSTIC_TIME_MONITOR_H

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include <map>
#include <time.h>
#include <fstream>
#include <SealBase/TimeInfo.h>



//<<<<<< INCLUDES                                                       >>>>>>
//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

struct TimeData 
{
    seal::TimeInfo::NanoSecs doneTime;
    seal::TimeInfo::NanoSecs startTime;
    std::ofstream file;
    int count;
    int iterCounter;
    int messSize;
};

class TimeMonitor
{
public:
    TimeMonitor ();
    ~TimeMonitor (void);
    seal::TimeInfo::NanoSecs	tempTime;	//time at which a source finshes sending monitoring information about a directory
    const int N_ITER;

    std::map <const char*, TimeData*> timeMap;
    std::map <const char*, bool> doneMap;
    
    void set (const char* sourcename, seal::TimeInfo::NanoSecs, int currSize = 0);
    void setDone (const char*);
    
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // DIAGNOSTIC_TIME_MONITOR_H

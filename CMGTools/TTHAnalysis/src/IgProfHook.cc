#include "CMGTools/TTHAnalysis/interface/IgProfHook.h"

// we try to stick to pure C within the signal handlers
#include <stdio.h>
#include <dlfcn.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

void (*igProfRequestDump_)(const char *);
int igProfDumpNumber_ = 0;

void igProfDumpNow(int) {
    char buff[50];
    igProfDumpNumber_++;
    sprintf(buff,"dump.%d.%d.out.gz", getpid(), igProfDumpNumber_);
    igProfRequestDump_(buff);
    fprintf(stderr, "Dumped to %s\n", buff); fflush(stderr);
}

bool setupIgProfDumpHook() {
    if (void *sym = dlsym(0, "igprof_dump_now")) {
        igProfRequestDump_ = __extension__ (void(*)(const char *)) sym;
        fprintf(stderr, "IgProf dump hook enabled. Do kill -USR2 %d to request a dump.\n", int(getpid())); fflush(stderr);
    } else {
        fprintf(stderr, "Not being profiled by IgProf. The command you should use to profile this is:\n"); 
        fprintf(stderr, "  igprof -mp -z -t python python -i $CMSSW_BASE/src/CMGTools/RootTools/python/fwlite/MultiLoop.py cfg.py -f \n\n");
        fflush(stderr);
        return false;
    }
    signal(SIGUSR2,igProfDumpNow);
    return true;
}

SetupIgProfDumpHook::SetupIgProfDumpHook() {}
SetupIgProfDumpHook::~SetupIgProfDumpHook() {}
void SetupIgProfDumpHook::start() { setupIgProfDumpHook(); }

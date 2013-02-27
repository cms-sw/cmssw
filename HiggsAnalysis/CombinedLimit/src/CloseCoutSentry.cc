#include "../interface/CloseCoutSentry.h"

#include <cstdio>
#include <unistd.h>

bool CloseCoutSentry::open_ = true;
int  CloseCoutSentry::fdOut_ = 0;
int  CloseCoutSentry::fdErr_ = 0;
FILE * CloseCoutSentry::trueStdOut_ = 0;

CloseCoutSentry::CloseCoutSentry(bool silent) :
    silent_(silent), stdOutIsMine_(false)
{
    if (silent_) {
        if (open_) {
            open_ = false;
            if (fdOut_ == 0 && fdErr_ == 0) {
                fdOut_ = dup(1);
                fdErr_ = dup(2);
            }
            freopen("/dev/null", "w", stdout);
            freopen("/dev/null", "w", stderr);
        } else {
            silent_ = false; 
        }
    }
}

CloseCoutSentry::~CloseCoutSentry() 
{
    clear();
}

void CloseCoutSentry::clear() 
{
    if (stdOutIsMine_) { fclose(trueStdOut_); trueStdOut_ = 0; }
    if (silent_) {
        reallyClear();
        silent_ = false;
    }
}

void CloseCoutSentry::reallyClear() 
{
    if (fdOut_ != fdErr_) {
        char buf[50];
        sprintf(buf, "/dev/fd/%d", fdOut_); freopen(buf, "w", stdout);
        sprintf(buf, "/dev/fd/%d", fdErr_); freopen(buf, "w", stderr);
        open_   = true;
        fdOut_ = fdErr_ = 0; 
    }
}

void CloseCoutSentry::breakFree() 
{
    reallyClear();
}

FILE *CloseCoutSentry::trueStdOut() 
{
    if (open_) return stdout;
    if (trueStdOut_) return trueStdOut_;
    stdOutIsMine_ = true;
    char buf[50];
    sprintf(buf, "/dev/fd/%d", fdOut_); trueStdOut_ = fopen(buf, "w");
    return trueStdOut_;
}

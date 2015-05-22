#ifndef CMGTools_TTHAnalysis_IgProfHook_h
#define CMGTools_TTHAnalysis_IgProfHook_h

bool setupIgProfDumpHook() ;

class SetupIgProfDumpHook {
    public:
        SetupIgProfDumpHook() ;
        ~SetupIgProfDumpHook() ;
        void start() ; 
};

#endif

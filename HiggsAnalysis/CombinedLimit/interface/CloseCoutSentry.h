#ifndef HiggsAnalysis_CombinedLimit_CloseCoutSentry_
#define HiggsAnalysis_CombinedLimit_CloseCoutSentry_
/** This class redirects cout and cerr to /dev/null when created,
    and restores them back when destroyed.                        */

class CloseCoutSentry {
    public:
        CloseCoutSentry(bool silent = true) ;
        ~CloseCoutSentry();
        void clear() ;
    private:
        bool silent_;
        int fdOut_, fdErr_;
        static bool open_;
};

#endif

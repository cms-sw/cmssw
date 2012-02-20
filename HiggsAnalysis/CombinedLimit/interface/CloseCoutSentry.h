#ifndef HiggsAnalysis_CombinedLimit_CloseCoutSentry_
#define HiggsAnalysis_CombinedLimit_CloseCoutSentry_
/** This class redirects cout and cerr to /dev/null when created,
    and restores them back when destroyed.                        */

class CloseCoutSentry {
    public:
        CloseCoutSentry(bool silent = true) ;
        ~CloseCoutSentry();
        // clear, if I was the one closing it
        void clear() ;
        // break through any sentry, even the ones above myself (for critical error messages, or debug)
        static void breakFree() ;
    private:
        bool silent_;
        static int fdOut_, fdErr_;
        static bool open_;
        // always clear, even if I was not the one closing it
        void static reallyClear() ;
};

#endif

#ifndef HiggsAnalysis_CombinedLimit_ProfilingTools_
#define HiggsAnalysis_CombinedLimit_ProfilingTools_
bool setupIgProfDumpHook() ;

//#include <boost/unordered_map.hpp>
class PerfCounter {
    public:
        PerfCounter() : value_(0) {}
        static PerfCounter & get(const char *name) ;

        void   add(double increment=1.0) { value_ += increment; }
        double get() const { return value_; }

        static void add(const char *name, double increment=1.0) { PerfCounter::get(name).add(increment); }
        static void enable() ;
        static void printAll() ;
    private:
        double value_;
};

#endif

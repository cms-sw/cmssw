#ifndef HiggsAnalysis_CombinedLimit_SequentialMinimizer_h
#define HiggsAnalysis_CombinedLimit_SequentialMinimizer_h

struct RooAbsReal;
struct RooRealVar;
#include <vector>

class OneDimMinimizer {
    public:
        OneDimMinimizer() : nll_(0), var_(0) {}
        OneDimMinimizer(RooAbsReal *nll, RooRealVar *variable) : nll_(nll), var_(variable) {}

        RooRealVar & var() { return *var_; }

        /// do N steps of golden bisection (or until tolerance in x, y is reached)
        /// return true if converged, false if finished steps
        bool minimize(int steps=1, double ytol=0, double xtol = 0);

        /// improve minimum, re-evaluating points (other parameters of the function might have changed)
        /// return value is:
        ///    0: minimum changed, and did not converge there yet
        ///    1: minimum changed, but did converge to it
        ///    2: minimum did not change significantly
        /// in case 2, then the final position is NOT changed at all.
        /// force = true will instead force the update even if it's trivial
        enum ImproveRet { Unchanged = 2, Done = 1, NotDone = 0 };
        ImproveRet improve(int steps=1, double ytol=0, double xtol = 0, bool force=true);
    private:
        RooAbsReal *nll_;
        RooRealVar *var_;
        double xi_[3], yi_[3];

        /// basic loop
        /// return false if ended steps, true if reached tolerance
        bool doloop(int steps, double ytol, double xtol) ;

        /// search for a triplet of points bracketing the maximum. return false in case of errors
        void seek() ;

        /// re-search for a triplet of points bracketing the maximum. return false in case of errors
        /// assume that value and error make sense
        void reseek() ;

        /// do the golden bisection
        void goldenBisection();

        /// do the parabola fit
        bool parabolaStep();

        /// do the parabola fit
        double parabolaFit();

        /// evaluate function at x
        double eval(double x) ;

        inline void assign(int to, int from) { xi_[to] = xi_[from]; yi_[to] = yi_[from]; }
};

class SequentialMinimizer {
    public:
        SequentialMinimizer(RooAbsReal *nll, RooRealVar *poi=0) ;
       
        bool minimize(double ytol=0, int bigsteps=1000, int smallsteps=5);
        bool improve(double ytol=0,  int bigsteps=1000, int smallsteps=5);
    private:
        enum State { Cleared, Ready, Active, Done };
        struct Worker : public OneDimMinimizer {
            Worker() : OneDimMinimizer(), xtol(0), state(Cleared) {}
            Worker(RooAbsReal *nll, RooRealVar *var) : OneDimMinimizer(nll,var), xtol(0), state(Cleared) {}
            double xtol;
            State state;     
        };
        RooAbsReal *nll_;
        bool                hasPoi_;
        Worker              poiWorker_;
        std::vector<Worker> nuisWorkers_;
};
#endif

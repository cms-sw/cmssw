#ifndef HiggsAnalysis_CombinedLimit_SequentialMinimizer_h
#define HiggsAnalysis_CombinedLimit_SequentialMinimizer_h

struct RooAbsReal;
struct RooRealVar;
#include <vector>
#include <memory>
#include <Math/Minimizer.h>

namespace cmsmath {

    /// Basic struct to call a function
    struct MinimizerContext {
        MinimizerContext(const ROOT::Math::IMultiGenFunction *function) : func(function), x(func->NDim()), nCalls(0) {}
        // convenience methods
        double eval() const { nCalls++; return (*func)(&x[0]); }
        double setAndEval(unsigned int i, double xi) const { x[i] = xi; return eval(); }
        double cleanEval(unsigned int i, double xi) const { double x0 = x[i]; x[i] = xi; double y = eval(); x[i] = x0; return y; }
        // data, fixed
        const ROOT::Math::IMultiGenFunction * func;
        // data, mutable
        mutable std::vector<double> x;
        mutable unsigned int nCalls;
    };

    class OneDimMinimizer {
        public:
            OneDimMinimizer() : f_(0), idx_(0) {}
            OneDimMinimizer(const MinimizerContext &ctx, unsigned int idx) :
                f_(&ctx), idx_(idx) {}
            OneDimMinimizer(const MinimizerContext &ctx, unsigned int idx, double xmin, double xmax, double xstep, const std::string &name) : 
                f_(&ctx), idx_(idx), name_(name), xmin_(xmin), xmax_(xmax), xstep_(xstep) {}

            const std::string &  name() const { return name_; }
            const char        * cname() const { return name_.c_str(); }

            bool isInit() const { return f_ != 0; }
            void init(const MinimizerContext &ctx, unsigned int idx, double xmin, double xmax, double xstep, const std::string &name) {
                f_ = &ctx; idx_ = idx; 
                xmin_ = xmin; xmax_ = xmax; xstep_ = xstep;
                name_ = name;
            }
            void initUnbound(const MinimizerContext &ctx, unsigned int idx, double xstep, const std::string &name) {
                init(ctx, idx, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), xstep, name);
            }
            // out of line to avoid including TString everywhere
            void initDefault(const MinimizerContext &ctx, unsigned int idx) ;

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
            // Function
            const MinimizerContext * f_;
            // Index of myself
            unsigned int idx_;
            // My name
            std::string name_;

            // Point (x and y)
            double xi_[3], yi_[3];

            // Bounds and step
            double xmin_, xmax_, xstep_;

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
            inline double &x() { return f_->x[idx_]; }
            inline double eval() { return f_->eval(); }
            inline double eval(double x) { return f_->cleanEval(idx_, x); }

            inline void assign(int to, int from) { xi_[to] = xi_[from]; yi_[to] = yi_[from]; }
    };

    class SequentialMinimizer : public ROOT::Math::Minimizer {
        public:
            SequentialMinimizer(const char *name=0) : ROOT::Math::Minimizer() {}

            /// reset for consecutive minimizations - implement if needed 
            virtual void Clear() ;

            /// set the function to minimize
            virtual void SetFunction(const ROOT::Math::IMultiGenFunction & func) ; 

            /// set free variable 
            virtual bool SetVariable(unsigned int ivar, const std::string & name, double val, double step) ; 

            /// set upper/lower limited variable (override if minimizer supports them )
            virtual bool SetLimitedVariable(unsigned int ivar, const std::string & name, double val, double  step, double  lower, double  upper) ; 

            /// set fixed variable (override if minimizer supports them )
            virtual bool SetFixedVariable(unsigned int ivar, const std::string & name, double val) ; 

            /// method to perform the minimization
            virtual  bool Minimize() ; 

            /// return minimum function value
            virtual double MinValue() const { return minValue_;  }

            /// return expected distance reached from the minimum
            virtual double Edm() const { return edm_; }

            /// return  pointer to X values at the minimum 
            virtual const double *  X() const { return & func_->x[0]; }

            /// return pointer to gradient values at the minimum 
            virtual const double *  MinGradient() const { return 0; }  

            /// number of function calls to reach the minimum 
            virtual unsigned int NCalls() const { return func_->nCalls; }    

            /// this is <= Function().NDim() which is the total 
            /// number of variables (free+ constrained ones) 
            virtual unsigned int NDim() const { return nDim_; }

            /// number of free variables (real dimension of the problem) 
            /// this is <= Function().NDim() which is the total 
            virtual unsigned int NFree() const { return nFree_;   }

            /// minimizer provides error and error matrix
            virtual bool ProvidesError() const { return false; } 

            /// return errors at the minimum 
            virtual const double * Errors() const { return 0; }

            virtual double CovMatrix(unsigned int i, unsigned int j) const { return 0; }

            // these have to be public for ROOT to handle
            enum State { Cleared, Ready, Active, Done, Fixed, Unknown };
            struct Worker : public OneDimMinimizer {
                Worker() : OneDimMinimizer(), state(Unknown) {}
                State state;     
            };
        protected:
            bool minimize(int smallsteps=5);
            bool improve(int smallsteps=5);

            std::auto_ptr<MinimizerContext> func_;
            unsigned int nDim_, nFree_;

            // status information
            double minValue_;            
            double edm_;

            // Workers
            std::vector<Worker> workers_;
            State state_;
                    
    };

} // namespace
#endif

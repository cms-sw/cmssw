#include "../interface/SequentialMinimizer.h"

#include <cmath>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <limits>
#include "TString.h"
#include "RooRealVar.h"
#include "RooAbsReal.h"
#include "RooLinkedListIter.h"
#include <Math/MinimizerOptions.h>
#include <Math/Factory.h>
#include <boost/foreach.hpp>
#include "../interface/ProfilingTools.h"
#define foreach BOOST_FOREACH

#define DEBUG_ODM_printf if (0) printf
//#define DEBUG_SM_printf   if (0) printf
//#define DEBUGV_SM_printf  if (0) printf
// #define DEBUG_ODM_printf printf
#define DEBUG_SM_printf  if (fDebug > 1) printf
#define DEBUGV_SM_printf if (fDebug > 2) printf  

namespace { 
    const double GOLD_R1 = 0.61803399 ;
    const double GOLD_R2 = 1-0.61803399 ;
    const double XTOL    = 10*std::sqrt(std::numeric_limits<double>::epsilon());
}
void cmsmath::OneDimMinimizer::initDefault(const MinimizerContext &ctx, unsigned int idx) {
    init(ctx, idx, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 1.0, Form("#%d", idx));
}


bool cmsmath::OneDimMinimizer::minimize(int steps, double ytol, double xtol) 
{
    // get initial bracket
    seek();
    
    bool done = doloop(steps,ytol,xtol);
    parabolaStep();
    x() = xi_[1];
    xstep_ = xi_[2] - xi_[0];
    return done;
}

cmsmath::OneDimMinimizer::ImproveRet cmsmath::OneDimMinimizer::improve(int steps, double ytol, double xtol, bool force) 
{
    double x0 = x();
    if (x0 < xi_[0] || x0 > xi_[2]) {
        // could happen if somebody outside this loop modifies some parameters
        DEBUG_ODM_printf("ODM: ALERT: variable %s outside bounds x = [%.4f, %.4f, %.4f], x0 = %.4f\n", name_.c_str(), xi_[0], xi_[1], xi_[2], x0);
        x0 = x() = xi_[1]; 
    } else {
        xi_[1] = x0;
    }
    double y0 = eval();
    yi_[1] = y0;
    yi_[0] = eval(xi_[0]);
    yi_[2] = eval(xi_[2]);
    if (xtol == 0) xtol = (fabs(xi_[1])+XTOL)*XTOL;

    DEBUG_ODM_printf("ODM: start of improve %s x = [%.4f, %.4f, %.4f], y = [%.4f, %.4f, %.4f]\n", name_.c_str(), xi_[0], xi_[1], xi_[2], yi_[0], yi_[1], yi_[2]);

    if (yi_[1] <= yi_[0] && yi_[1] <= yi_[2]) {
        if (ytol > 0 && (std::max(yi_[2],yi_[0]) - yi_[1]) < ytol) {
            DEBUG_ODM_printf("ODM: immediate ytol for %s: ymin %.8f, ymax %.8f, diff %.8f\n", name_.c_str(), yi_[1], std::max(yi_[2],yi_[0]), std::max(yi_[2],yi_[0]) - yi_[1]);
            if (!force || parabolaStep()) return Unchanged;
        }
        if (xtol > 0 && (xi_[2] - xi_[0]) < xtol) {
            DEBUG_ODM_printf("ODM: immediate xtol for %s: xmin %.8f, xmax %.8f, diff %.8f\n", name_.c_str(), xi_[0], xi_[2], xi_[2] - xi_[0]);
            if (!force || parabolaStep()) return Unchanged;
        }
    } else {
        reseek();
    }

    //post-condition: always a sorted interval
    assert(xi_[0] < xi_[2]);
    assert(xi_[0] <= xi_[1] && xi_[1] <= xi_[2]);
    // if midpoint is not not one of the extremes, it's not higher than that extreme
    assert(xi_[1] == xi_[0] || yi_[1] <= yi_[0]); 
    assert(xi_[1] == xi_[2] || yi_[1] <= yi_[2]);

    bool done = doloop(steps,ytol,xtol);
    parabolaStep(); 

    //post-condition: always a sorted interval
    assert(xi_[0] < xi_[2]);
    assert(xi_[0] <= xi_[1] && xi_[1] <= xi_[2]);
    // if midpoint is not not one of the extremes, it's not higher than that extreme
    assert(xi_[1] == xi_[0] || yi_[1] <= yi_[0]); 
    assert(xi_[1] == xi_[2] || yi_[1] <= yi_[2]);


    if (ytol > 0 && fabs(yi_[1] - y0) < ytol) {
        DEBUG_ODM_printf("ODM: final ytol for %s: ymin(old) %.8f, ymin(new) %.8f, diff %.8f\n", name_.c_str(), y0, yi_[1], y0 -yi_[1]);
        if (!force) x() = x0;
        return Unchanged;
    } 

    if (xtol > 0 && fabs(xi_[1] - x0) < xtol) {
        x() = (force ? xi_[1] : x0);
        return Unchanged;
    }
    DEBUG_ODM_printf("ODM: doloop for %s is %s\n", name_.c_str(), done ? "Done" : "NotDone");
    DEBUG_ODM_printf("ODM: end of improve %s x = [%.4f, %.4f, %.4f], y = [%.4f, %.4f, %.4f]\n", name_.c_str(), xi_[0], xi_[1], xi_[2], yi_[0], yi_[1], yi_[2]);
    x() = xi_[1];
    xstep_ = xi_[2] - xi_[0];
    return done ? Done : NotDone; 
}

bool cmsmath::OneDimMinimizer::doloop(int steps, double ytol, double xtol) 
{
    if (steps <= 0) steps = 100;
    for (int i = 0; i < steps; ++i) {
        if (xtol > 0 && (xi_[2] - xi_[0]) < xtol) {
            return true;
        }
        goldenBisection();
        if (ytol > 0 && (std::max(yi_[2],yi_[0]) - yi_[1]) < ytol) {
            DEBUG_ODM_printf("ODM: intermediate ytol for %s: ymin %.8f, ymax %.8f, diff %.8f\n", name_.c_str(), yi_[1], std::max(yi_[2],yi_[0]), std::max(yi_[2],yi_[0]) - yi_[1]);
            return true;
        }
        DEBUG_ODM_printf("ODM: step %d/%d done for %s: ymin %.8f, ymax %.8f, diff %.8f\n", i+1, steps, name_.c_str(), yi_[1], std::max(yi_[2],yi_[0]), std::max(yi_[2],yi_[0]) - yi_[1]);
        DEBUG_ODM_printf("ODM: %s x = [%.4f, %.4f, %.4f], y = [%.4f, %.4f, %.4f]\n", name_.c_str(), xi_[0], xi_[1], xi_[2], yi_[0], yi_[1], yi_[2]);
    }

    return false;
}

void cmsmath::OneDimMinimizer::seek() 
{
    if (std::isfinite(xmax_-xmin_)) {
        xstep_ = std::max(xstep_, 0.2*(xmax_-xmin_));
    } else {
        xstep_ = 1.0;
    }
    reseek();
}
void cmsmath::OneDimMinimizer::reseek() 
{
    double xtol2 = 2*(fabs(xi_[1])+XTOL)*XTOL;
    if (xstep_ < xtol2) xstep_ = xtol2;
    xi_[1] = x(); 
    yi_[1] = eval(xi_[1]);
    xi_[0] = std::max(xmin_, xi_[1]-xstep_);
    yi_[0] = (xi_[0] == xi_[1] ? yi_[1] : eval(xi_[0]));
    xi_[2] = std::min(xmax_, xi_[1]+xstep_);
    yi_[2] = (xi_[2] == xi_[1] ? yi_[1] : eval(xi_[2]));
    assert(xi_[0] < xi_[2]);
    assert(xi_[0] <= xi_[1] && xi_[1] <= xi_[2]);

    for (;;) {
        //DEBUG_ODM_printf("ODM: bracketing %s in x = [%.4f, %.4f, %.4f], y = [%.4f, %.4f, %.4f]\n", name_.c_str(), xi_[0], xi_[1], xi_[2], yi_[0], yi_[1], yi_[2]);
        if (yi_[0] < yi_[1]) {
            assign(2,1); // 2:=1
            assign(1,0); // 1:=0
            xi_[0] = std::max(xmin_, xi_[1]-xstep_);
            yi_[0] = (xi_[0] == xi_[1] ? yi_[1] : eval(xi_[0]));
        } else if (yi_[2]  < yi_[1]) {
            assign(0,1); // 0:=1
            assign(1,2); // 1:=2
            xi_[2] = std::min(xmax_, xi_[1]+xstep_);
            yi_[2] = (xi_[2] == xi_[1] ? yi_[1] : eval(xi_[2]));
        } else if (yi_[2] == yi_[1] && yi_[1] == yi_[0]) {
            // function is identical in three points --> constant?
            // try a scan
            const int nPoints = 20;
            double xi[nPoints], yi[nPoints];
            double dx = (xmax_-xmin_)/nPoints, x = xmin_ - 0.5*dx;
            bool isConst = true;
            int iFound = 0; 
            for (int i = 0; i < nPoints; ++i, x += dx) {
                xi[i] = x; yi[i] = eval(x);
                if (yi[i] != yi_[1]) isConst = false;
                if (yi[i] < yi[iFound]) { iFound = i; }
            }
            if (isConst) break;
            xi_[0] = (iFound == 0        ? xmin_ : xi[iFound-1]);
            xi_[2] = (iFound > nPoints-1 ? xmax_ : xi[iFound+1]);
            xi_[1] = iFound; yi_[1] = yi_[iFound];
            break;
        } else {
            xstep_ /= 2;
            break;
        }
        xstep_ *= 2;
    }
    //DEBUG_ODM_printf("ODM: bracketed minimum of %s in [%.4f, %.4f, %.4f]\n", name_.c_str(), xi_[0], xi_[1], xi_[2]);
    //post-condition: always a sorted interval
    assert(xi_[0] < xi_[2]);
    assert(xi_[0] <= xi_[1] && xi_[1] <= xi_[2]);
    // if midpoint is not not one of the extremes, it's not higher than that extreme
    assert(xi_[1] == xi_[0] || yi_[1] <= yi_[0]); 
    assert(xi_[1] == xi_[2] || yi_[1] <= yi_[2]);
}

void cmsmath::OneDimMinimizer::goldenBisection() 
{
    //pre-condition: always a sorted interval
    assert(xi_[0] < xi_[2]);
    assert(xi_[0] <= xi_[1] && xi_[1] <= xi_[2]);
    // if midpoint is not not one of the extremes, it's not higher than that extreme
    assert(xi_[1] == xi_[0] || yi_[1] <= yi_[0]); 
    assert(xi_[1] == xi_[2] || yi_[1] <= yi_[2]);
    if (xi_[0] == xi_[1] || xi_[1] == xi_[2]) {
        int isame = (xi_[0] == xi_[1] ? 0 : 2);
        /// pre-condition: the endpoint equal to x1 is not the highest
        assert(yi_[isame] <= yi_[2-isame]);
        xi_[1] = 0.5*(xi_[0]+xi_[2]);
        yi_[1] = eval(xi_[1]);
        if (yi_[1] < yi_[isame]) {
            // maximum is in the interval-
            // leave as is, next bisection steps will find it
        } else {
            // maximum remains on the boundary, leave both points there
            assign(2-isame, 1);
            assign(1, isame); 
        }
    } else {
        int inear = 2, ifar = 0;
        if (xi_[2]-xi_[1] > xi_[1] - xi_[0]) {
            inear = 0, ifar = 2;
        } else {
            inear = 2, ifar = 0;
        }
        double xc = xi_[1]*GOLD_R1 + xi_[ifar]*GOLD_R2;
        double yc = eval(xc);
        //DEBUG_ODM_printf("ODM: goldenBisection:\n\t\tfar = (%.2f,%.8f)\n\t\tnear = (%.2f,%.8f)\n\t\tcenter  = (%.2f,%.8f)\n\t\tnew  = (%.2f,%.8f)\n",
        //            xi_[ifar],  yi_[ifar], xi_[inear], yi_[inear], xi_[1], yi_[1], xc, yc);
        if (yc < yi_[1]) {   // then use 1, c, far
            assign(inear, 1);
            xi_[1] = xc; yi_[1] = yc;
        } else {  // then use c, 1, near
            xi_[ifar] = xc; yi_[ifar] = yc;
        }
    }
    //post-condition: always a sorted interval
    assert(xi_[0] < xi_[2]);
    assert(xi_[0] <= xi_[1] && xi_[1] <= xi_[2]);
    // if midpoint is not not one of the extremes, it's not higher than that extreme
    assert(xi_[1] == xi_[0] || yi_[1] <= yi_[0]); 
    assert(xi_[1] == xi_[2] || yi_[1] <= yi_[2]);

}

double cmsmath::OneDimMinimizer::parabolaFit() 
{
    if (xi_[0] == xi_[1] || xi_[1] == xi_[2]) { 
        return xi_[1]; 
    }
    double dx0 = xi_[1] - xi_[0], dx2 = xi_[1] - xi_[2];
    double dy0 = yi_[1] - yi_[0], dy2 = yi_[1] - yi_[2];
    double num = dx0*dx0*dy2 - dx2*dx2*dy0;
    double den = dx0*dy2     - dx2*dy0;
    if (den != 0) {
        double x = xi_[1] - 0.5*num/den;
        if (xi_[0] < x && x < xi_[2]) {
            return x;
        }
    } 
    return xi_[1];
}

bool cmsmath::OneDimMinimizer::parabolaStep() {
    double xc = parabolaFit();
    if (xc != xi_[1]) {
        double yc = eval(xc);
        if (yc < yi_[1]) {
            xi_[1] = xc; 
            yi_[1] = yc;
            //post-condition: always a sorted interval
            assert(xi_[0] < xi_[2]);
            assert(xi_[0] <= xi_[1] && xi_[1] <= xi_[2]);
            // if midpoint is not not one of the extremes, it's not higher than that extreme
            assert(xi_[1] == xi_[0] || yi_[1] <= yi_[0]); 
            assert(xi_[1] == xi_[2] || yi_[1] <= yi_[2]);
            return true;
        }
    }
    return false;
}

void cmsmath::OneDimMinimizer::moveTo(double x) {
    if (x == xmax_) {
        xi_[0] = xmax_ - (xi_[2]-xi_[0]); yi_[0] = eval(xi_[0]);
        xi_[1] = xmax_; yi_[1] = eval(xmax_);
        xi_[2] = xmax_; yi_[2] = yi_[1];
    } else if (x == xmin_) {
        xi_[2] = xmin_ + (xi_[2]-xi_[0]); yi_[2] = eval(xi_[0]);
        xi_[1] = xmin_; yi_[1] = eval(xmin_);
        xi_[0] = xmin_; yi_[0] = yi_[1];
    } else {
        double dx = xi_[2] - xi_[0];
        xi_[1] = x; yi_[0] = eval(x);
        xi_[0] = std::max(xmin_, x-dx); yi_[0] = eval(xi_[0]);
        xi_[2] = std::min(xmax_, x+dx); yi_[2] = eval(xi_[2]);
    }
}

void cmsmath::SequentialMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction & func) {
    DEBUG_SM_printf("SequentialMinimizer::SetFunction: nDim = %u\n", func.NDim());
    func_.reset(new MinimizerContext(&func));
    nFree_ = nDim_ = func_->x.size();
    // create dummy workers
    workers_.clear();
    workers_.resize(nDim_);
    // reset states
    Clear();
}

void cmsmath::SequentialMinimizer::Clear() {
    DEBUGV_SM_printf("SequentialMinimizer::Clear()\n");
    minValue_ = std::numeric_limits<double>::quiet_NaN();
    edm_      = std::numeric_limits<double>::infinity();
    state_ = Cleared;
    foreach(Worker &w, workers_) w.state = Cleared;
}

bool cmsmath::SequentialMinimizer::SetVariable(unsigned int ivar, const std::string & name, double val, double step) {
    DEBUGV_SM_printf("SequentialMinimizer::SetVariable(idx %u, name %s, val %g, step %g)\n", ivar, name.c_str(), val, step);
    assert(ivar < nDim_);
    func_->x[ivar] = val;
    workers_[ivar].initUnbound(*func_, ivar, step, name);
    workers_[ivar].state = Cleared;
    return true;
}

bool cmsmath::SequentialMinimizer::SetLimitedVariable(unsigned int ivar, const std::string & name, double val, double  step, double  lower, double  upper) {
    DEBUGV_SM_printf("SequentialMinimizer::SetLimitedVariable(idx %u, name %s, var %g, step %g, min %g, max %g)\n", ivar, name.c_str(), val, step, lower, upper);
    assert(ivar < nDim_);
    func_->x[ivar] = val;
    workers_[ivar].init(*func_, ivar, lower, upper, step, name);
    workers_[ivar].state = Cleared;
    return true;
}

bool cmsmath::SequentialMinimizer::SetFixedVariable(unsigned int ivar, const std::string & name, double val) {
    DEBUGV_SM_printf("SequentialMinimizer::SetFixedVariable(idx %u, name %s, var %g)\n", ivar, name.c_str(), val);
    assert(ivar < nDim_);
    func_->x[ivar] = val;
    workers_[ivar].initUnbound(*func_, ivar, 1.0, name);
    workers_[ivar].state = Fixed;
    return true;
}

bool cmsmath::SequentialMinimizer::Minimize() {
    return minimize();
}

bool cmsmath::SequentialMinimizer::minimize(int smallsteps) 
{
    for (unsigned int i = 0; i < nDim_; ++i) {
        Worker &w = workers_[i];
        if (!w.isInit() || w.state == Unknown) throw std::runtime_error(Form("SequentialMinimizer::worker[%u/%u] not initialized!\n", i, nDim_));
        if (w.state != Fixed) {
            w.minimize(1); 
            w.state = Ready; 
        }
    }
    state_ = Ready;
    return improve(smallsteps);
}

bool cmsmath::SequentialMinimizer::improve(int smallsteps)
{
    static int nFailWakeUpAttempts = runtimedef::get("SeqMinimizer_nFailWakeUpAttempts");

    // catch improve before minimize case
    if (state_ == Cleared) return minimize(smallsteps);

    // setup default tolerances and steps
    double ytol = Tolerance()/sqrt(workers_.size());
    int bigsteps = MaxIterations()*20;

    // list of done workers (latest-done on top)
    std::list<Worker*> doneWorkers;

    // start with active workers, for all except constants
    foreach(Worker &w, workers_) {
        if (w.state != Fixed) w.state = Active;
    }

    state_ = Active;
    for (int i = 0; i < bigsteps; ++i) {
        DEBUG_SM_printf("Start of loop. Strategy %d, State is %s\n",fStrategy,(state_ == Done ? "DONE" : "ACTIVE"));
        State newstate = Done;
        int oldActiveWorkers = 0, newActiveWorkers = 0;
        foreach(Worker &w, workers_) {
            OneDimMinimizer::ImproveRet iret = OneDimMinimizer::Unchanged;
            if (w.state == Done || w.state == Fixed) continue;
            iret = w.improve(smallsteps,ytol); 
            oldActiveWorkers++; 
            if (iret == OneDimMinimizer::Unchanged) {
                DEBUGV_SM_printf("\tMinimized %s:  Unchanged. NLL = %.8f\n", w.cname(), func_->eval());
                w.state = Done;
                w.nUnaffected = 0;
                doneWorkers.push_front(&w);
            } else {
                DEBUGV_SM_printf("\tMinimized %s:  Changed. NLL = %.8f\n", w.cname(), func_->eval());
                w.state = Active;
                newstate = Active;
                newActiveWorkers++;
            }
        }
        if (fStrategy >= 2 && newActiveWorkers <= 30) { // arbitrary cut-off
            DEBUG_SM_printf("Middle of loop. Strategy %d, active workers %d: firing full minimizer\n", fStrategy, newActiveWorkers);
            if (doFullMinim()) newstate = Done;
        }
        if (newstate == Done) {
            DEBUG_SM_printf("Middle of loop. Strategy %d, State is %s, active workers %d --> %d \n",fStrategy,(state_ == Done ? "DONE" : "ACTIVE"), oldActiveWorkers, newActiveWorkers);
            oldActiveWorkers = 0; newActiveWorkers = 0;
            double y0 = func_->eval();
            std::list<Worker*>::iterator it = doneWorkers.begin();
            // The topmost worker was added on the list just now, so by definition it's already done.
            // We save a reference to it, remove it from the done list, and if the loop doesn't end there we set it to active again 
            Worker* firstWorker = *it; 
            it = doneWorkers.erase(it);
            // Then we check all the others
            while( it != doneWorkers.end()) {
                Worker &w = **it;
                if (nFailWakeUpAttempts && w.nUnaffected >= nFailWakeUpAttempts) { ++it; continue; }
                OneDimMinimizer::ImproveRet iret = w.improve(smallsteps,ytol,0,/*force=*/true);
                oldActiveWorkers++;
                if (iret == OneDimMinimizer::Unchanged) {
                    DEBUGV_SM_printf("\tMinimized %s:  Unchanged. NLL = %.8f\n", w.cname(), func_->eval());
                    w.nUnaffected++;
                    ++it;
                } else {
                    DEBUGV_SM_printf("\tMinimized %s:  Changed. NLL = %.8f\n", w.cname(), func_->eval());
                    w.state = Active;
                    newstate = Active;
                    w.nUnaffected = 0;
                    it = doneWorkers.erase(it);
                    newActiveWorkers++;
                    if (fStrategy == 0) break;
                }
            }
            if (newstate == Active) { // wake up him too
                firstWorker->state = Active; 
                firstWorker->nUnaffected = 0; 
            }
            double y1 = func_->eval();
            edm_ = y0 - y1;
        }
        DEBUG_SM_printf("End of loop. Strategy %d, State is %s, active workers %d --> %d \n",fStrategy,(state_ == Done ? "DONE" : "ACTIVE"), oldActiveWorkers, newActiveWorkers);
        if (state_ == Done && newstate == Done) {
            DEBUG_SM_printf("Converged after %d big steps\n",i);
            minValue_ = func_->eval();
            fStatus   = 0;
            return true;
        }
        state_ = newstate;
        if (func_->nCalls > MaxFunctionCalls()) break;
    }
    DEBUG_SM_printf("Failed do converge after %d big steps\n",bigsteps);
    fStatus   = -1;
    minValue_ = func_->eval();
    return false;
}

namespace cmsmath {
    class SubspaceMultiGenFunction : public ROOT::Math::IMultiGenFunction {
        public:
           SubspaceMultiGenFunction(const ROOT::Math::IMultiGenFunction *f, int nDim, const int *idx, double *xi) :
                f_(f), nDim_(nDim), idx_(idx), x_(xi) {}
           virtual IBaseFunctionMultiDim * Clone() const { return new SubspaceMultiGenFunction(*this); }
           virtual unsigned int NDim() const { return nDim_; }
        private:
           virtual double DoEval(const double * x) const {
                for (int i = 0; i < nDim_; ++i) x_[idx_[i]] = x[i];
                return (*f_)(x_);
           }
           const ROOT::Math::IMultiGenFunction *f_; 
           const int  nDim_;
           const int *idx_;
           double    *x_;
    };
}

bool cmsmath::SequentialMinimizer::doFullMinim()
{
    if (fullMinimizer_.get() == 0) {
        fullMinimizer_.reset(ROOT::Math::Factory::CreateMinimizer("Minuit2", ""));
        fullMinimizer_->SetTolerance(Tolerance());
        fullMinimizer_->SetStrategy(Strategy()-2);
    }
    subspaceIndices_.clear();
    for (int i = 0, n = workers_.size(); i < n; ++i) {
        if (workers_[i].state == Active) subspaceIndices_.push_back(i);
    }
    fullMinimizer_->Clear();
    SubspaceMultiGenFunction subfunc(func_->func, subspaceIndices_.size(), &subspaceIndices_[0], &func_->x[0]);
    fullMinimizer_->SetFunction(subfunc);
    for (int i = 0, n = subspaceIndices_.size(); i < n; ++i) {
        int j = subspaceIndices_[i];
        Worker &w = workers_[j];
        fullMinimizer_->SetLimitedVariable(i, w.name(), func_->x[j], w.step(), w.min(), w.max());
    }
    bool ok = fullMinimizer_->Minimize();
    if (ok) {
        const double *ximin = fullMinimizer_->X();
        // move to the right place
        for (int i = 0, n = subspaceIndices_.size(); i < n; ++i) {
            func_->x[subspaceIndices_[i]] = ximin[i];
        }
        // update all workers
        for (int i = 0, n = subspaceIndices_.size(); i < n; ++i) {
            int j = subspaceIndices_[i];
            Worker &w = workers_[j];
            w.moveTo( ximin[i] );
        }

    }
    return ok;
}


#include <TPluginManager.h>
namespace {
    static int load_seqmin() {
        gPluginMgr->AddHandler("ROOT::Math::Minimizer", "SeqMinimizer", "cmsmath::SequentialMinimizer", "HiggsAnalysisCombinedLimit", "SequentialMinimizer(const char *)");
        return 1;
    }
    static int loaded_seqmin = load_seqmin();
}

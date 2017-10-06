#ifndef CalibCalorimetry_HcalAlgos_ConstantStepOdeSolver_h_
#define CalibCalorimetry_HcalAlgos_ConstantStepOdeSolver_h_

#include <vector>
#include <iostream>
#include "FWCore/Utilities/interface/Exception.h"

#include "CalibCalorimetry/HcalAlgos/interface/AbsODERHS.h"

//
// ODE solver with a constant time step. The derived classes are supposed
// to implement concrete ODE solving algorithms (Runge-Kutta, etc).
//
class ConstantStepOdeSolver
{
public:
    inline ConstantStepOdeSolver()
        : rhs_(nullptr), dt_(0.0), dim_(0), runLen_(0), lastIntegrated_(0) {}

    inline ConstantStepOdeSolver(const AbsODERHS& rhs) :
        rhs_(nullptr), dt_(0.0), dim_(0), runLen_(0), lastIntegrated_(0)
    {
        rhs_ = rhs.clone();
    }

    // The copy constructor and the assignment operator are explicitly provided
    ConstantStepOdeSolver(const ConstantStepOdeSolver& r);
    ConstantStepOdeSolver& operator=(const ConstantStepOdeSolver& r);

    inline virtual ~ConstantStepOdeSolver() {delete rhs_;}

    // Access the equation right hand side
    inline void setRHS(const AbsODERHS& rhs)
    {
        delete rhs_;
        rhs_ = rhs.clone();
    }
    inline const AbsODERHS* getRHS() const {return rhs_;}
    inline AbsODERHS* getRHS() {return rhs_;}

    // Inspectors (will be valid after at least one "run" call)
    inline unsigned lastDim() const {return dim_;}
    inline unsigned lastRunLength() const {return runLen_;}
    inline double lastDeltaT() const {return dt_;}
    inline double lastMaxT() const {return runLen_ ? dt_*(runLen_-1U) : 0.0;}

    inline double getTime(const unsigned idx) const
    {
        if (idx >= runLen_) throw cms::Exception(
            "In ConstantStepOdeSolver::getTime: index out of range");
        return idx*dt_;
    }

    inline double getCoordinate(const unsigned which, const unsigned idx) const
    {
        if (which >= dim_ || idx >= runLen_) throw cms::Exception(
            "In ConstantStepOdeSolver::getCoordinate: index out of range");
        return historyBuffer_[dim_*idx + which];
    }

    // Integrate the node with the given number and get
    // the value of the integral at the given history point
    double getIntegrated(unsigned which, unsigned idx) const;

    // Truncate some coordinate
    void truncateCoordinate(unsigned which, double minValue, double maxValue);

    // Linear interpolation methods will be used in case the "cubic"
    // argument is false, and cubic in case it is true
    double interpolateCoordinate(unsigned which, double t,
                                 bool cubic = false) const;

    // Interpolate the integrated node
    double interpolateIntegrated(unsigned which, double t,
                                 bool cubic = false) const;

    // Get the time of the peak position
    double getPeakTime(unsigned which) const;

    // Solve the ODE and remember the history
    void run(const double* initialConditions, unsigned lenConditions,
             double dt, unsigned nSteps);

    // Set the history from some external source. The size
    // of the data array should be at least dim*runLen.
    void setHistory(double dt, const double* data,
                    unsigned dim, unsigned runLen);

    // Write out the history
    void writeHistory(std::ostream& os, double dt, bool cubic = false) const;

    // Write out the integrated node
    void writeIntegrated(std::ostream& os, unsigned which,
                         double dt, bool cubic = false) const;

    // The following method must be overriden by derived classes
    virtual const char* methodName() const = 0;

protected:
    AbsODERHS* rhs_;

private:
    // The following method must be overriden by derived classes
    virtual void step(double t, double dt,
                      const double* x, unsigned lenX,
                      double* coordIncrement) const = 0;

    // The following integration corresponds to the cubic
    // interpolation of the coordinate
    void integrateCoordinate(const unsigned which);

    double dt_;
    unsigned dim_;
    unsigned runLen_;
    unsigned lastIntegrated_;

    std::vector<double> historyBuffer_;
    std::vector<double> chargeBuffer_;
};


// Dump the coordinate history as it was collected
inline std::ostream& operator<<(std::ostream& os,
                                const ConstantStepOdeSolver& s)
{
    s.writeHistory(os, s.lastDeltaT());
    return os;
}


// A few concrete ODE solvers
class EulerOdeSolver : public ConstantStepOdeSolver
{
public:
    inline EulerOdeSolver() : ConstantStepOdeSolver() {}

    inline explicit EulerOdeSolver(const AbsODERHS& rhs)
        : ConstantStepOdeSolver(rhs) {}

    inline const char* methodName() const override {return "Euler";}

private:
    void step(double t, double dt,
              const double* x, unsigned lenX,
              double* coordIncrement) const override;
};


class RK2 : public ConstantStepOdeSolver
{
public:
    inline RK2() : ConstantStepOdeSolver() {}

    inline explicit RK2(const AbsODERHS& rhs) : ConstantStepOdeSolver(rhs) {}

    inline const char* methodName() const override {return "2nd order Runge-Kutta";}

private:
    void step(double t, double dt,
              const double* x, unsigned lenX,
              double* coordIncrement) const override;

    mutable std::vector<double> buf_;
};


class RK4 : public ConstantStepOdeSolver
{
public:
    inline RK4() : ConstantStepOdeSolver() {}

    inline explicit RK4(const AbsODERHS& rhs) : ConstantStepOdeSolver(rhs) {}

    inline const char* methodName() const override {return "4th order Runge-Kutta";}

private:
    void step(double t, double dt,
              const double* x, unsigned lenX,
              double* coordIncrement) const override;

    mutable std::vector<double> buf_;
};

#endif // CalibCalorimetry_HcalAlgos_ConstantStepOdeSolver_h_

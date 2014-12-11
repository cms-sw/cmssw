#include <cassert>
#include "FWCore/Utilities/interface/Exception.h"

#include "CalibCalorimetry/HcalAlgos/interface/PadeTableODE.h"

PadeTableODE::PadeTableODE(const unsigned padeRow, const unsigned padeColumn)
    : row_(padeRow),
      col_(padeColumn)
{
    if (row_ > 2U) throw cms::Exception(
      "In PadeTableODE constructor: Pade table row number out of range");
    if (col_ > 3U) throw cms::Exception(
      "In PadeTableODE constructor: Pade table column number out of range");
}

void PadeTableODE::calculate(const double tau, const double currentIn,
                             const double dIdt, const double d2Id2t,
                             const double* x, const unsigned lenX,
                             const unsigned firstNode, double* derivative) const
{
    // Check input sanity
    if (lenX < firstNode + col_) throw cms::Exception(
        "In PadeTableODE::calculate: insufficient number of variables");
    if (tau <= 0.0) throw cms::Exception(
        "In PadeTableODE::calculate: delay time is not positive");
    if (col_) assert(x);
    assert(derivative);

    switch (col_)
    {
    case 0U:
        // Special case: no ODE to solve
        derivative[firstNode] = 0.0;
        switch (row_)
        {
        case 2U:
            derivative[firstNode] += 0.5*tau*tau*d2Id2t;
        case 1U:
            derivative[firstNode] -= tau*dIdt;
        case 0U:
            derivative[firstNode] += currentIn;
            break;

        default:
            assert(0);
        }
        break;

    case 1U:
        // First order ODE to solve
        switch (row_)
        {
        case 0U:
            derivative[firstNode] = (currentIn - x[firstNode])/tau;
            break;

        case 1U:
            derivative[firstNode] = 2.0*(currentIn - x[firstNode])/tau - dIdt;
            break;

        case 2U:
            derivative[firstNode] = 3.0*(currentIn - x[firstNode])/tau - 
                                    2.0*dIdt + 0.5*tau*d2Id2t;
            break;

        default:
            assert(0);
        }
        break;

    case 2U:
        // Second order ODE to solve
        derivative[firstNode] = x[firstNode+1];
        switch (row_)
        {
        case 0U:
            derivative[firstNode+1] = 
                2.0*(currentIn-x[firstNode]-tau*x[firstNode+1])/tau/tau;
            break;

        case 1U:
            derivative[firstNode+1] = (6.0*(currentIn - x[firstNode]) - 
                                       2.0*tau*dIdt -
                                       4.0*tau*x[firstNode+1])/tau/tau;
            break;

        case 2U:
            derivative[firstNode+1] = 
                12.0*(currentIn - x[firstNode])/tau/tau - 
                6.0*(x[firstNode+1] + dIdt)/tau + d2Id2t;
            break;

        default:
            assert(0);
        }
        break;

    case 3U:
        // Third order ODE to solve
        derivative[firstNode] = x[firstNode+1];
        derivative[firstNode+1] = x[firstNode+2];
        switch (row_)
        {
        case 0U:
            derivative[firstNode+2] = 
                6.0*(currentIn - x[firstNode] - tau*x[firstNode+1] -
                     0.5*tau*tau*x[firstNode+2])/tau/tau/tau;
            break;

        case 1U:
            derivative[firstNode+2] = 24.0/tau/tau/tau*(
                currentIn - x[firstNode] - 0.25*tau*dIdt -
                0.75*tau*x[firstNode+1] - 0.25*tau*tau*x[firstNode+2]);
            break;

        case 2U:
            derivative[firstNode+2] = 60.0/tau/tau/tau*(
                currentIn - x[firstNode] - 0.4*tau*dIdt +
                0.05*tau*tau*d2Id2t - 0.6*tau*x[firstNode+1] -
                0.15*tau*tau*x[firstNode+2]);
            break;

        default:
            assert(0);
        }
        break;

    default:
        //
        // In principle, it is possible to proceed a bit further, but
        // we will soon encounter difficulties. For example, row 0 and
        // column 4 is going to generate a 4th order differential
        // equation for which all roots of the characteristic equation
        // still have negative real parts. The most "inconvenient" pair
        // of roots there is (-0.270556 +- 2.50478 I) which leads
        // to oscillations with damping. The characteristic equation
        // of 5th and higher order ODEs are going to have roots with
        // positive real parts. Unless additional damping is
        // purposefully introduced into the system, numerical
        // solutions of such equations will just blow up.
        //
        assert(0);
    }
}

void PadeTableODE::setParameters(const double* /* pars */, const unsigned nPars)
{
    assert(nPars == 0U);
}

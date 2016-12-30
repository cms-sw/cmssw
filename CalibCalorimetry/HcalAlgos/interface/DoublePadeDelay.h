#ifndef CalibCalorimetry_HcalAlgos_DoublePadeDelay_h_
#define CalibCalorimetry_HcalAlgos_DoublePadeDelay_h_

#include <cassert>
#include <algorithm>

#include "CalibCalorimetry/HcalAlgos/interface/AbsElectronicODERHS.h"

//
// Two differential equations using the Pade delay scheme. The control
// equation and the output equation are coupled only via the timing
// parameter of the output equation (this timing is determined by the
// control output). In this particular model, there is no feedback from
// the output into the control.
//
template<class ODE1, class ODE2, class DelayTimeModel1, class DelayTimeModel2>
class DoublePadeDelay : public AbsElectronicODERHS
{
public:
    inline DoublePadeDelay(const unsigned padeRow1, const unsigned padeColumn1,
                           const unsigned padeRow2, const unsigned padeColumn2)
        : ode1_(padeRow1, padeColumn1), ode2_(padeRow2, padeColumn2)
    {
        validate();
    }

    inline DoublePadeDelay(const unsigned padeRow1, const unsigned padeColumn1,
                           const unsigned padeRow2, const unsigned padeColumn2,
                           const HcalInterpolatedPulse& pulse)
        : AbsElectronicODERHS(pulse),
          ode1_(padeRow1, padeColumn1),
          ode2_(padeRow2, padeColumn2)
    {
        validate();
    }

    inline virtual DoublePadeDelay* clone() const 
        {return new DoublePadeDelay(*this);}

    inline virtual void calc(const double t,
                             const double* x, const unsigned lenX,
                             double* derivative)
    {
        if (!allParametersSet()) throw cms::Exception(
            "In DoublePadeDelay::calc: timing and/or ODE parameters not set");

        // The input signal
        const double currentIn = inputPulse_(t);

        // The output signal
        const double currentOut = x[outputNode()];

        // Numbers of parameters used by the member objects
        const unsigned npTau1 = tau1_.nParameters();
        const unsigned npOde1 = ode1_.nParameters();
        const unsigned npTau2 = tau2_.nParameters();
        const unsigned npOde2 = ode2_.nParameters();

        // Parameters for this code.
        // Order of parameters in the overall parameter set is:
        // parameters for tau1, then for ode1, then tau2, then ode2,
        // then parameters of this code.
        const double* pstart = &params_[npTau1 + npOde1 + npTau2 + npOde2];
        const double* pars = pstart;
        const double ctlGainOut = *pars++;
        const double inGainOut  = *pars++;
        const double outGainOut = *pars++;
        assert(thisCodeNumPars == static_cast<unsigned>(pars - pstart));

        // Save a little bit of time by not calculating the input
        // signal derivatives in case they will not be needed
        const unsigned row = std::max(ode1_.getPadeRow(), ode2_.getPadeRow());
        const double dIdt = row ? inputPulse_.derivative(t) : 0.0;
        const double d2Id2t = row > 1U ? inputPulse_.secondDerivative(t) : 0.0;

        // Set the timing parameters of the control circuit
        unsigned firstPar = npTau1 + npOde1;
        const double tau2 = tau2_(currentIn, &params_[firstPar], npTau2);

        // Set the ODE parameters for the control circuit
        firstPar += npTau2;
        if (npOde2)
            ode2_.setParameters(&params_[firstPar], npOde2);

        // Run the control circuit
        const unsigned ctrlNode = controlNode();
        double control;
        if (ctrlNode < AbsElectronicODERHS::invalidNode)
        {
            // The control circuit solves an ODE
            control = x[ctrlNode];
            ode2_.calculate(tau2, currentIn, dIdt, d2Id2t,
                            x, lenX, ctrlNode, derivative);
        }
        else
        {
            // The control circuit does not solve an ODE.
            // Instead, it drives its output directly.
            ode2_.calculate(tau2, currentIn, dIdt, d2Id2t,
                            0, 0U, 0U, &control);
        }

        // Timing parameter for the output circuit (the preamp)
        const double vtau = ctlGainOut*control   +
                             inGainOut*currentIn + 
                            outGainOut*currentOut;
        const double tau = tau1_(vtau, &params_[0], npTau1);

        // ODE parameters for the output circuit
        if (npOde1)
            ode1_.setParameters(&params_[npTau1], npOde1);

        // Run the output circuit
        ode1_.calculate(tau, currentIn, dIdt, d2Id2t, x, lenX, 0U, derivative);
    }

    inline unsigned numberOfNodes() const
        {return ode1_.getPadeColumn() + ode2_.getPadeColumn();}

    inline unsigned nParameters() const
    {
        const unsigned npTau1 = tau1_.nParameters();
        const unsigned npOde1 = ode1_.nParameters();
        const unsigned npTau2 = tau2_.nParameters();
        const unsigned npOde2 = ode2_.nParameters();
        return npTau1 + npOde1 + npTau2 + npOde2 + thisCodeNumPars;
    }

    inline unsigned outputNode() const {return 0U;}

    // The second ODE is the one for control. It's output node
    // is the control node.
    inline unsigned controlNode() const
    {
        if (ode2_.getPadeColumn())
            // ode2 has a real output node
            return ode1_.getPadeColumn();
        else
            // ode2 does not have a real output node
            return AbsElectronicODERHS::invalidNode;
    }

private:
    static const unsigned thisCodeNumPars = 3U;

    inline void validate() const
    {
        // Basically, we need to avoid the situation in which
        // we need to solve the differential equation for the control
        // circuit but do not need to solve the differential equation
        // for the preamp. It this case we will not have a good way
        // to pass the preamp output to the simulator. The simplest
        // way to ensure correctness of the whole procedure is to require
        // that the preamp must always be modeled by an ODE. Indeed,
        // one will almost surely need to represent it by at least
        // a low-pass filter.
        if (!ode1_.getPadeColumn()) throw cms::Exception(
            "In DoublePadeDelay::validate: the output "
            "circuit must be modeled by an ODE");
    }

    ODE1 ode1_;
    ODE2 ode2_;
    DelayTimeModel1 tau1_;
    DelayTimeModel2 tau2_;
};

#endif // CalibCalorimetry_HcalAlgos_DoublePadeDelay_h_

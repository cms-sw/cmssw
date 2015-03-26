#include <cmath>
#include <cfloat>

#include "CalibCalorimetry/HcalAlgos/interface/QIE8Simulator.h"

QIE8Simulator::QIE8Simulator()
    : preampOutputCut_(-1.0e100), inputGain_(1.0),
      outputGain_(1.0), runCount_(0),
      chargeNode_(AbsElectronicODERHS::invalidNode),
      integrateToGetCharge_(false),
      useCubic_(false)
{
}

QIE8Simulator::QIE8Simulator(const AbsElectronicODERHS& model,
                             const unsigned chargeNode,
                             const bool interpolateCubic,
                             const double preampOutputCut,
                             const double inputGain,
                             const double outputGain)
    : solver_(model),
      preampOutputCut_(preampOutputCut),
      inputGain_(inputGain),
      outputGain_(outputGain),
      runCount_(0),
      chargeNode_(chargeNode),
      useCubic_(interpolateCubic)
{
    if (chargeNode >= AbsElectronicODERHS::invalidNode)
        throw cms::Exception(
            "In QIE8Simulator constructor: invalid charge collection node");
    integrateToGetCharge_ = chargeNode == model.outputNode();
    validateGain();
    zeroInitialConditions();
}

unsigned QIE8Simulator::run(const double dt, const double tstop,
                            const double tDigitize,
                            double* TS, const unsigned lenTS)
{
    if (chargeNode_ >= AbsElectronicODERHS::invalidNode)
        throw cms::Exception(
            "In QIE8Simulator::run: preamp model is not set");

    // Check arguments for validity
    if (dt <= 0.0) throw cms::Exception(
        "In QIE8Simulator::run: invalid time step");

    if (tstop < dt) throw cms::Exception(
        "In QIE8Simulator::run: invalid stopping time");

    if (lenTS && tDigitize < 0.0) throw cms::Exception(
        "In QIE8Simulator::run: can't digitize at t < 0");

    // Determine the number of time steps
    const double dsteps = tstop/dt;
    unsigned n = dsteps;
    if (dsteps != static_cast<double>(n))
        ++n;
    if (n >= maxlen)
        n = maxlen - 1;

    // Run the simulation
    AbsElectronicODERHS& rhs = modifiableRHS();
    const unsigned numNodes = rhs.numberOfNodes();
    if (numNodes)
    {
        assert(initialConditions_.size() == numNodes);
        solver_.run(&initialConditions_[0], numNodes, dt, n);
    }
    else
    {
        // Special situation: the simulation does not
        // need to solve any ODE. Instead, it will fill
        // out the output directly.
        if (!(integrateToGetCharge_ && chargeNode_ == 0U))
            throw cms::Exception("In QIE8Simulator::run: "
                                     "invalid mode of operation");
        const unsigned runLen = n + 1U;
        if (historyBuffer_.size() < runLen)
            historyBuffer_.resize(runLen);
        double* hbuf = &historyBuffer_[0];
        for (unsigned istep=0; istep<runLen; ++istep, ++hbuf)
            rhs.calc(istep*dt, 0, 0U, hbuf);
        solver_.setHistory(dt, &historyBuffer_[0], 1U, runLen);
    }

    // Truncate the preamp output if this will affect subsequent results
    if (integrateToGetCharge_ && preampOutputCut_ > -0.9e100)
        solver_.truncateCoordinate(chargeNode_, preampOutputCut_, DBL_MAX);

    // Digitize the accumulated charge
    unsigned filled = 0;
    if (lenTS)
    {
        assert(TS);
        const double lastTStop = solver_.lastMaxT();
        const double tsWidth = this->adcTSWidth();
        double oldCharge = getCharge(tDigitize);
        for (unsigned its=0; its<lenTS; ++its)
        {
            const double t0 = tDigitize + its*tsWidth;
            if (t0 < lastTStop)
            {
                double t1 = t0 + tsWidth;
                if (t1 > lastTStop)
                    t1 = lastTStop;
                else
                    ++filled;
                const double q = getCharge(t1);
                TS[its] = (q - oldCharge)*outputGain_;
                oldCharge = q;
            }
            else
                TS[its] = 0.0;
        }
    }
    ++runCount_;
    return filled;
}

double QIE8Simulator::preampOutput(const double t) const
{
    if (!runCount_) throw cms::Exception(
        "In QIE8Simulator::preampOutput: please run the simulation first");
    const unsigned preampNode = getRHS().outputNode();
    if (preampNode >= AbsElectronicODERHS::invalidNode)
        return 0.0;
    else
        return outputGain_*solver_.interpolateCoordinate(
            preampNode, t, useCubic_);
}

double QIE8Simulator::controlOutput(const double t) const
{
    if (!runCount_) throw cms::Exception(
        "In QIE8Simulator::controlOutput: please run the simulation first");
    const unsigned controlNode = getRHS().controlNode();
    if (controlNode >= AbsElectronicODERHS::invalidNode)
        return 0.0;
    else
        return solver_.interpolateCoordinate(controlNode, t, useCubic_);
}

double QIE8Simulator::preampPeakTime() const
{
    if (!runCount_) throw cms::Exception(
        "In QIE8Simulator::preampPeakTime: please run the simulation first");
    const unsigned preampNode = getRHS().outputNode();
    if (preampNode >= AbsElectronicODERHS::invalidNode) throw cms::Exception(
        "In QIE8Simulator::preampPeakTime: no preamp node in the circuit");
    return solver_.getPeakTime(preampNode);
}

void QIE8Simulator::setInitialConditions(
    const double* values, const unsigned sz)
{
    const unsigned nExpected = getRHS().numberOfNodes();
    if (sz != nExpected) throw cms::Exception(
        "In QIE8Simulator::setInitialConditions: unexpected number "
        "of initial conditions");
    assert(sz == initialConditions_.size());
    if (sz)
    {
        double* c = &initialConditions_[0];
        for (unsigned i=0; i<sz; ++i)
            *c++ = *values++;
    }
}

void QIE8Simulator::setRHS(const AbsElectronicODERHS& rhs,
                           const unsigned chargeNode,
                           const bool useCubicInterpolation)
{
    if (chargeNode >= AbsElectronicODERHS::invalidNode) 
        throw cms::Exception(
            "In QIE8Simulator::setRHS invalid charge collection node");
    solver_.setRHS(rhs);
    chargeNode_ = chargeNode;
    integrateToGetCharge_ = chargeNode == rhs.outputNode();
    useCubic_ = useCubicInterpolation;
    zeroInitialConditions();
}

void QIE8Simulator::zeroInitialConditions()
{
    const unsigned sz = getRHS().numberOfNodes();
    if (initialConditions_.size() != sz)
        initialConditions_.resize(sz);
    if (sz)
    {
        double* c = &initialConditions_[0];
        for (unsigned i=0; i<sz; ++i)
            *c++ = 0.0;
    }
}

void QIE8Simulator::validateGain() const
{
    if (inputGain_ <= 0.0) throw cms::Exception(
        "In QIE8Simulator::validateGain: invalid input gain");
}

double QIE8Simulator::lastStopTime() const
{
    if (!runCount_) throw cms::Exception(
        "In QIE8Simulator::lastStopTime: please run the simulation first");
    return solver_.lastMaxT();
}

double QIE8Simulator::totalIntegratedCharge(const double t) const
{
    if (!runCount_) throw cms::Exception(
        "In QIE8Simulator::totalIntegratedCharge: "
        "please run the simulation first");
    return outputGain_*getCharge(t);
}

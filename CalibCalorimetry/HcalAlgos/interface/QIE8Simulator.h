#ifndef CalibCalorimetry_HcalAlgos_QIE8Simulator_h_
#define CalibCalorimetry_HcalAlgos_QIE8Simulator_h_

#include "CalibCalorimetry/HcalAlgos/interface/ConstantStepOdeSolver.h"
#include "CalibCalorimetry/HcalAlgos/interface/AbsElectronicODERHS.h"

//
// This class is needed mostly in order to represent the charge
// to ADC conversion inside the QIE8 chip
//
class QIE8Simulator
{
public:
    static const unsigned maxlen = HcalInterpolatedPulse::maxlen;

    // In case the default constructor is used, "setRHS" method must be
    // called before running the simulation
    QIE8Simulator();

    // Constructor which includes a proper model
    QIE8Simulator(const AbsElectronicODERHS& model,
                  unsigned chargeNode,
                  bool interpolateCubic = false,
                  double preampOutputCut = -1.0e100,
                  double inputGain = 1.0,
                  double outputGain = 1.0);

    void setRHS(const AbsElectronicODERHS& rhs, unsigned chargeNode,
                bool interpolateCubic=false);

    inline const AbsElectronicODERHS& getRHS() const
    {
        const AbsODERHS* ptr = solver_.getRHS();
        if (!ptr) throw cms::Exception(
            "In QIE8Simulator::getRHS: RHS is not set");
        return *(static_cast<const AbsElectronicODERHS*>(ptr));
    }

    // Simple inspectors
    inline double getInputGain() const {return inputGain_;}
    inline double getOutputGain() const {return outputGain_;}
    inline unsigned long getRunCount() const {return runCount_;}
    inline double getPreampOutputCut() const {return preampOutputCut_;}

    // Examine preamp  model parameters
    inline unsigned nParameters() const
        {return getRHS().nParameters();}
    inline double getParameter(const unsigned which) const
        {return getRHS().getParameter(which);}

    // Set gains
    inline void setInputGain(const double g) {inputGain_ = g; validateGain();}
    inline void setOutputGain(const double g) {outputGain_ = g; validateGain();}

    // Set preamp initial conditions
    void setInitialConditions(const double* values, const unsigned len);
    void zeroInitialConditions();

    // Set preamp model parameters
    inline void setParameter(const unsigned which, const double p)
        {modifiableRHS().setParameter(which, p);}
    inline void setLeadingParameters(const double* values, const unsigned len)
        {modifiableRHS().setLeadingParameters(values, len);}

    // Set the minimum value for the preamp output
    inline void setPreampOutputCut(const double p)
        {preampOutputCut_ = p;}

    // Set the input pulse
    template<class Signal>
    inline void setInputSignal(const Signal& inputSignal)
    {
        modifiableRHS().setInputPulse(inputSignal);
        modifiableRHS().inputPulse() *= inputGain_;
    }

    // Get the input pulse
    inline const HcalInterpolatedPulse& getInputSignal() const
        {return getRHS().inputPulse();}

    // Set the input pulse data. This will not modify
    // signal begin and end times.
    template<class Real>
    inline void setInputShape(const Real* values, const unsigned len)
    {
        modifiableRHS().inputPulse().setShape(values, len);
        modifiableRHS().inputPulse() *= inputGain_;
    }

    // Scale the input pulse by some constant factor
    inline void scaleInputSignal(const double s)
        {modifiableRHS().inputPulse() *= s;}

    // Manipulate input pulse amplidude
    inline double getInputAmplitude() const
        {return getRHS().inputPulse().getPeakValue()/inputGain_;}

    inline void setInputAmplitude(const double a)
        {modifiableRHS().inputPulse().setPeakValue(a*inputGain_);}

    // Manipulate input pulse total charge
    inline double getInputIntegral() const
        {return getRHS().inputPulse().getIntegral()/inputGain_;}

    inline void setInputIntegral(const double d)
        {modifiableRHS().inputPulse().setIntegral(d*inputGain_);}

    // Manipulate input pulse timing
    inline double getInputStartTime() const
        {return getRHS().inputPulse().getStartTime();}

    inline void setInputStartTime(const double newStartTime)
        {modifiableRHS().inputPulse().setStartTime(newStartTime);}

    // Run the simulation. Parameters are as follows:
    //
    // dt        -- Simulation time step.
    //
    // tstop     -- At what time to stop the simulation. The actual
    //              stopping time will be the smaller of this parameter and
    //              dt*(maxlen - 1). The simulation always starts at t = 0.
    //
    // tDigitize -- When to start producing ADC counts. This argument
    //              must be non-negative.
    //              
    // TS, lenTS -- Array (and its length) where ADC counts will be
    //              placed on exit.
    //              
    // This method returns the number of "good" ADC time slices -- the
    // ones that completely covered by the simulation interval.
    //
    unsigned run(double dt, double tstop,
                 double tDigitize, double* TS, unsigned lenTS);

    // Inspect simulation results
    double lastStopTime() const;
    double totalIntegratedCharge(double t) const;
    double preampPeakTime() const;

    // The following methods with simply return 0.0 in case
    // there are no corresponding nodes in the circuit
    double preampOutput(double t) const;
    double controlOutput(double t) const;

    // Time slice width in nanoseconds
    static inline double adcTSWidth() {return 25.0;}

private:
    inline AbsElectronicODERHS& modifiableRHS()
    {
        AbsODERHS* ptr = solver_.getRHS();
        if (!ptr) throw cms::Exception(
            "In QIE8Simulator::modifiableRHS: no RHS");
        return *(static_cast<AbsElectronicODERHS*>(ptr));
    }

    inline double getCharge(const double t) const
    {
        double q;
        if (integrateToGetCharge_)
            q = solver_.interpolateIntegrated(chargeNode_, t, useCubic_);
        else
            q = solver_.interpolateCoordinate(chargeNode_, t, useCubic_);
        return q;
    }

    void validateGain() const;

    RK4 solver_;
    std::vector<double> initialConditions_;
    std::vector<double> historyBuffer_;
    double preampOutputCut_;
    double inputGain_;
    double outputGain_;
    unsigned long runCount_;
    unsigned chargeNode_;
    bool integrateToGetCharge_;
    bool useCubic_;
};

#endif // CalibCalorimetry_HcalAlgos_QIE8Simulator_h_

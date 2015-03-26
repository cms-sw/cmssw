#ifndef CalibCalorimetry_HcalAlgos_AbsODERHS_h_
#define CalibCalorimetry_HcalAlgos_AbsODERHS_h_

//
// Base class for the ODE right hand sides
//
class AbsODERHS
{
public:
    inline virtual ~AbsODERHS() {}

    virtual AbsODERHS* clone() const = 0;

    virtual void calc(double t, const double* x, unsigned lenX,
                      double* derivative) = 0;
};

#endif // CalibCalorimetry_HcalAlgos_AbsODERHS_h_

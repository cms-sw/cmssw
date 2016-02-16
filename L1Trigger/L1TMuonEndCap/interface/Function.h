//////////////////////////////////////////////////////////////////////////
//                            Function.h                               //
// =====================================================================//
//                                                                      //
//   We need functions to evaluate the success of the regression,       //
//   to fit a transformation of the trueValue instead of the trueValue, //
//   and to provide a preliminary fit. Define the interfaces here.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ADD_FUNCTION
#define ADD_FUNCTION

#include "TMath.h"
#include "L1Trigger/L1TMuonEndCap/interface/Event.h"
#include <cmath>

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

class PreliminaryFit
{
// Before building the regression, fit the events with a preliminary fit.
    public:
        // return true if the fit fails.
        virtual bool fit(Event* e) = 0;
        virtual const char* name() = 0;
        virtual int id() = 0;
};

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

class TransformFunction
{
// Sometimes it is useful to predict a transformation of the trueValue instead
// of the trueValue itself.
    public:
        // return true if the transform fails.
        virtual bool transform(Event* e) = 0;
        virtual bool invertTransformation(Event* e) = 0;
        virtual const char* name() = 0;
        virtual int id() = 0;
};

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

class MetricOfSuccess
{
// Judge how well the regression worked.
    public:
        virtual Double_t calculate(std::vector<Event*>& v) = 0;
};

#endif

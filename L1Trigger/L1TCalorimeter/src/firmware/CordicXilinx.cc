/**\class CordicXilinx CordicXilinx.cc L1Trigger/L1TCalorimeter/src/firmware/CordicXilinx.cc

 Description: Emulates parts of the Xilinx DSP IP CORDIC routine, as described in
              http://www.xilinx.com/support/documentation/ip_documentation/cordic/v6_0/pg105-cordic.pdf
              This class only implements the vector translation, returning magnitude and phase, given signed
              x and y inputs.  The inputs and outputs are not packed, so that normal signed integer
              arithmetic works as expected.  They can easily be packed into a fixed width by abs() and placing
              a sign bit at the appropriate offset.
              The applicable configuration parameters that are being emulated are the following:
                - Functional Selection: Translate
                - Phase Format: Radians
                - Round Mode: Truncate
                - Advanced Configuration Parameters: Iterations=0, Precision=0, Coarse Rotation, Compensation Scaling = Embedded Multiplier
              In addition, an arbitrary input and output width can be specified, HOWEVER, only in=24, out=19 has been rigorously tested
              against the Xilinx proprietary emulator.

 Tests: Full circle at various magnitudes, including maximum; a few billion random inputs
        Limited hardware comparisons have shown agreement as well.
        Test framework: https://github.com/nsmith-/cordic_test

 Original Author:  Nick Smith ( nick.smith@cern.ch )

*/

#include "L1Trigger/L1TCalorimeter/interface/CordicXilinx.h"

#include <vector>
#include <iostream>
#include <iomanip> 
#include <cassert>
#include <math.h>

CordicXilinx::CordicXilinx(int inputBits, int outputBits, bool debug) :
    inputBits_(inputBits),
    outputBits_(outputBits),
    debug_(debug)
{
    // Coarse rotation lowers necessary iterations by 2
    iterations_ = outputBits-2;
    // Internal precision is by default this value (when set to 0 in xilinx config)
    internalBits_ = outputBits+ceil(log((float) iterations_)/log(2.));

    double scaleFactor = 1.;
    for(int i=1; i<=iterations_; ++i)
    {
        int rotation = encodeAngle(atan(pow(2.,-i)));
        rotations_.push_back(rotation);
        scaleFactor *= pow(1+pow(2.,-2*i), -0.5);
    }
    scaleFactor_ = scaleFactor*pow(2., internalBits_-1)+0.5;
    
    // Precompute angles table for speed
    encodedAngles_[Pi] = encodeAngle(M_PI);
    encodedAngles_[HalfPi] = encodeAngle(M_PI/2);
    encodedAngles_[NHalfPi] = encodeAngle(-M_PI/2);

    if ( debug_ ) printf("Cordic setup: %d iterations, %d internal bits, scale factor = %d\n", iterations_, internalBits_, scaleFactor_);
}

int CordicXilinx::encodeAngle(const double angleFloat) const
{
    assert(abs(angleFloat)<=M_PI);
    // Xilinx seems to store rounded rotation table
    return angleFloat*pow(2., internalBits_-3)+0.5;
}

void CordicXilinx::operator() ( int32_t xInput , int32_t yInput , int32_t& aPhi , uint32_t& aMagnitude ) const
{
    // Assumption in algorithm is that arithmetic shifts are used for ints (as opposed to logical shifts)
    static_assert( ((int) -1)>>3 == (int) -1 , "Signed ints need to use arithmetic shifts for this algorithm to work properly!");

    // Input checks
    // Input is in 2QN format, and for xilinx
    // the max is +- 1.0000...
    assert(abs(xInput) <= (1<<(inputBits_-1)));
    assert(abs(yInput) <= (1<<(inputBits_-1)));
    
    // Rotation to get from current vector to origin
    // must invert to get aPhi
    int rotation(0);
    int x,y;

    // Debug tool
    auto printVals = [&x,&y,&rotation,this]
    {
        printf("x: % 8d y: % 8d phi: % 8d outphi: % 8d float phi = % f\n",
            x,
            y,
            rotation,
            (abs(rotation)>>(internalBits_-outputBits_)) * ((rotation>0) ? -1:1),
            rotation/pow(2., internalBits_-3)
        );
    };

    // Convert to internal precision
    if ( internalBits_ > inputBits_ )
    {
        x = xInput << (internalBits_-inputBits_);
        y = yInput << (internalBits_-inputBits_);
    }
    else
    {
        x = xInput >> (inputBits_-internalBits_);
        y = yInput >> (inputBits_-internalBits_);
    }
    if ( debug_ ) printVals();

    // Coarse rotate to [-pi/4,pi/4)
    if ( x-y >= 0 )
    {
        if ( x+y >= 0 )
        {
            // East (Correct) quadrant
        }
        else
        {
            // South, rotate by +pi/2
            int xtmp = -y;
            int ytmp = x;
            x = xtmp;
            y = ytmp;
            rotation += encodedAngles_[HalfPi];
        }
    }
    else
    {
        if ( x+y >= 0 )
        {
            // North, rotate by -pi/2
            int xtmp = y;
            int ytmp = -x;
            x = xtmp;
            y = ytmp;
            rotation += encodedAngles_[NHalfPi];
        }
        else
        {
            // West, rotate by pi
            x = -x;
            y = -y;
            rotation += encodedAngles_[Pi];
        }
    }
    if ( debug_ ) std::cout << "Coarse rotate" << std::endl;
    if ( debug_ ) printVals();

    if ( debug_ ) std::cout << "Starting iterations" << std::endl;
    for ( int i=1; i<=iterations_; ++i )
    {
        int sign = (y>=0) ? -1:1;
        int xtmp = x - sign*(y>>i);
        int ytmp = y + sign*(x>>i);
        x = xtmp;
        y = ytmp;
        rotation += sign*rotations_[i-1];
        if ( debug_ ) printVals();
    }
    
    // need a little extra room for the last multiplication
    aMagnitude = ((long) x * (long) scaleFactor_)>>(2*internalBits_-outputBits_-1);

    // Xilinx seems to just mod to [-pi,pi]
    if ( rotation > encodedAngles_[Pi] ) rotation -= 2*encodedAngles_[Pi]+1;
    aPhi = (-rotation)>>(internalBits_-outputBits_);
}

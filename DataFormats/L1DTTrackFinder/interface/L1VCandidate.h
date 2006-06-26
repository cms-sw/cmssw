//-------------------------------------------------
//
/**  \class L1VCandidate
 *
 *   Abstract Base Class for various types of
 *   muon, electron and jet candidates.
 *   Having a common object will allow good implementation of global
 *   trigger system which needs to sort these objects and 
 *   correlate them in detector eta-phi space
*/
//
//   $Date: 2006/06/01 00:00:00 $
//   $Revision: 1.1 $
//
//   Authors :
//   S. Dasu                  University of Wisconsin 
//   N. Neumeister            CERN EP
//
//--------------------------------------------------
#ifndef L1VCANDIDATE_H
#define L1VCANDIDATE_H

//---------------
// C++ Headers --
//---------------

#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1VCandidate {

  public:

    /// A string to identify the object
    virtual string name() const = 0;

    /// Charge of the candidate
    virtual int charge() const = 0;

    /// Pt of the candidate in trigger scale
    virtual unsigned pt() const = 0;

    /// Eta location of the candidate in trigger scale
    virtual unsigned eta() const = 0;

    /// Phi location of the candidate in trigger scale
    virtual unsigned phi() const = 0;

    /// Quality of the candidate - defined to be such that the larger
    /// its value it is of higher quality
    virtual unsigned quality() const = 0;

    /// Null candidate - i.e., not yet initialized if this returns true
    virtual bool empty() const = 0;

    /// Linearized values in a scale with least significant bit = lsbValue (GeV)
    /// and maximum number of bits = maxScale.  The scale should saturate
    /// to lsbValue * (2^maxScale - 1)
    virtual unsigned linearizedPt(float lsbValue, unsigned maxScale) const = 0;

    /// Scale converter helper function - e.g., can be used for setting 
    /// global trigger thresholds
    virtual unsigned triggerScale(float value) const = 0;

    /// Eta, Phi indices of "standard" region, i.e., 0.35x0.35 regions
    /// The trigger scale eta,phi for the candidate may be finer
    virtual unsigned etaRegionIndex() const = 0;
    virtual unsigned phiRegionIndex() const = 0;

    /// Access in floats is useful for histogramming ...
    virtual float ptValue() const = 0;
    virtual float etaValue() const = 0;
    virtual float phiValue() const = 0;

};

#endif

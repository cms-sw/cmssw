#ifndef IOMC_HLLHCEvtVtxGenerator_H
#define IOMC_HLLHCEvtVtxGenerator_H

/**
 * Generate event vertices given beams sizes, crossing angle
 * offset, and crab rotation. 
 * Attention: All values are assumed to be mm for spatial coordinates
 * and ns for time.
 * Attention: This class fix the the vertex time generation of HLLHCEvtVtxGenerator
 *
 * $Id: HLLHCEvtVtxGenerator_Fix.h,v 1.0 2015/03/15 10:34:38 Exp $
 */

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"

#include <string>

namespace CLHEP {
    class RandFlat;
}

namespace edm {
    class ConfigurationDescriptions;
}

class HLLHCEvtVtxGenerator : public BaseEvtVtxGenerator 
{
public:

    HLLHCEvtVtxGenerator(const edm::ParameterSet & p);

    virtual ~HLLHCEvtVtxGenerator();

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

    /// return a new event vertex
    virtual HepMC::FourVector* newVertex(CLHEP::HepRandomEngine*) ;

    virtual TMatrixD* GetInvLorentzBoost() {return 0;};
   
private:
    /** Copy constructor */
    HLLHCEvtVtxGenerator(const HLLHCEvtVtxGenerator &p);

    /** Copy assignment operator */
    HLLHCEvtVtxGenerator&  operator = (const HLLHCEvtVtxGenerator & rhs );
    
    //spatial and time offset for mean collision
    const double fMeanX, fMeanY, fMeanZ, fTimeOffset;

    //proton beam energy
    const double momeV;
    const double gamma;
    const double beta;
    const double betagamma;
    
    //crossing angle 
    const double phi;
    
    //crab cavity frequency
    const double wcc;

    // 800 MHz RF?
    const bool RF800;

    //beta crossing plane (m)
    const double betx;

    //beta separation plane (m)
    const double bets;

    //horizontal emittance 
    const double epsxn;

    //vertical emittance
    const double epssn;

    //bunch length
    const double sigs;

    //crabbing angle crossing
    const double alphax;

    //crabbing angle separation
    const double alphay;

    // ratio of crabbing angle to crossing angle
    const double oncc;

    //normalized crossing emittance
    const double epsx;

    //normlaized separation emittance
    const double epss;

    //size in x
    const double sigx;

    // crossing angle * crab frequency
    const double phiCR;
    
    //width for y plane
    double sigma(double z, double epsilon, double beta, double betagamma) const;

    //density with crabbing
    double integrandCC(double x, double z, double t) const;

    // 4D intensity
    double intensity(double x, double y,double z,double t) const;
};

#endif

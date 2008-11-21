   
// -*- C++ -*-

#include "HadronizerFtn.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepMC/GenEvent.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// #include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

namespace gen
{


extern "C" {

        void upinit_() { fortranCallback.upinit(); }
        void upevnt_() { fortranCallback.upevnt(); }

} // extern "C"


HadronizerFtn::HadronizerFtn(edm::ParameterSet const& ps) 
   : fCOMEnergy(ps.getParameter<double>("comEnergy")),
     fHepMCVerbosity(ps.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
     fMaxEventsToPrint(ps.getUntrackedParameter<int>("maxEventsToPrint", 0)),
     fGenEvent(0)
{ }


void HadronizerFtn::fillHeader()
{

   const lhef::HEPRUP* heprup = fPartonLevel->getHEPRUP();

   lhef::CommonBlocks::fillHEPRUP(heprup);   
   
   return;

}


void HadronizerFtn::fillEvent()
{

        const lhef::HEPEUP* hepeup = fPartonLevel->getHEPEUP();

        //if (iterations++) {
                if (hepeup_.nup = 0)
                return;
        //}

        lhef::CommonBlocks::fillHEPEUP(hepeup);

    return;
    
}


} // end namespace

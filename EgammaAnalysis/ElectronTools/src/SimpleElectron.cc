#ifndef SimpleElectron_STANDALONE
#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
SimpleElectron::SimpleElectron(const reco::GsfElectron &in, unsigned int runNumber, bool isMC) :
        run_(runNumber),
        eClass_(int(in.classification())), 
        r9_(in.full5x5_r9()),
        scEnergy_(in.superCluster()->rawEnergy() + in.isEB() ? 0 : in.superCluster()->preshowerEnergy()), 
        scEnergyError_(-999.),  // FIXME???
        trackMomentum_(in.trackMomentumAtVtx().R()), 
        trackMomentumError_(in.trackMomentumError()), 
        regEnergy_(in.correctedEcalEnergy()), 
        regEnergyError_(in.correctedEcalEnergyError()), 
        eta_(in.superCluster()->eta()), 
        isEB_(in.isEB()), 
        isMC_(isMC), 
        isEcalDriven_(in.ecalDriven()), 
        isTrackerDriven_(in.trackerDrivenSeed()), 
        newEnergy_(regEnergy_), 
        newEnergyError_(regEnergyError_),
        combinedMomentum_(in.p4(reco::GsfElectron::P4_COMBINATION).P()), 
        combinedMomentumError_(in.p4Error(reco::GsfElectron::P4_COMBINATION)),
        scale_(1.0), smearing_(0.0)
{
}


void SimpleElectron::writeTo(reco::GsfElectron & out) const 
{
    math::XYZTLorentzVector oldMomentum = out.p4();
    math::XYZTLorentzVector newMomentum = math::XYZTLorentzVector(oldMomentum.x()*getCombinedMomentum()/oldMomentum.t(),
                                                                  oldMomentum.y()*getCombinedMomentum()/oldMomentum.t(),
                                                                  oldMomentum.z()*getCombinedMomentum()/oldMomentum.t(),
                                                                  getCombinedMomentum());
    out.setCorrectedEcalEnergy(getNewEnergy());
    out.setCorrectedEcalEnergyError(getNewEnergyError());
    out.correctMomentum(newMomentum, getTrackerMomentumError(), getCombinedMomentumError());
}
#endif

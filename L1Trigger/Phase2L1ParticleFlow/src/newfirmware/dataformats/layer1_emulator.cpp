#include "layer1_emulator.h"
#include "emulator_io.h"


bool l1ct::HadCaloObjEmu::read(std::fstream & from) {
    src = nullptr; // not persistent
    return readObj<HadCaloObj>(from, *this);
}
bool l1ct::HadCaloObjEmu::write(std::fstream & to) const {
    return writeObj<HadCaloObj>(*this, to);
}

bool l1ct::EmCaloObjEmu::read(std::fstream & from) {
    src = nullptr; // not persistent
    return readObj<EmCaloObj>(from, *this);
}
bool l1ct::EmCaloObjEmu::write(std::fstream & to) const {
    return writeObj<EmCaloObj>(*this, to);
}

bool l1ct::TkObjEmu::read(std::fstream & from) {
    src = nullptr; // not persistent
    return 
        readObj<TkObj>(from, *this) && 
        readVar(from, hwChi2) &&
        readVar(from, hwStubs);
}
bool l1ct::TkObjEmu::write(std::fstream & to) const {
    return 
        writeObj<TkObj>(*this, to) &&
        writeVar(hwChi2, to) &&
        writeVar(hwStubs, to);
}

bool l1ct::MuObjEmu::read(std::fstream & from) {
    src = nullptr; // not persistent
    return readObj<MuObj>(from, *this);
}
bool l1ct::MuObjEmu::write(std::fstream & to) const {
    return writeObj<MuObj>(*this, to);
}

bool l1ct::PFChargedObjEmu::read(std::fstream & from) {
    srcTrack = nullptr; // not persistent
    srcCluster = nullptr; // not persistent
    srcMu = nullptr; // not persistent
    return readObj<PFChargedObj>(from, *this);
}
bool l1ct::PFChargedObjEmu::write(std::fstream & to) const {
    return writeObj<PFChargedObj>(*this, to);
}

bool l1ct::PFNeutralObjEmu::read(std::fstream & from) {
    srcCluster = nullptr; // not persistent
    return readObj<PFNeutralObj>(from, *this);
}
bool l1ct::PFNeutralObjEmu::write(std::fstream & to) const {
    return writeObj<PFNeutralObj>(*this, to);
}

bool l1ct::PuppiObjEmu::read(std::fstream & from) {
    srcTrack = nullptr; // not persistent
    srcCluster = nullptr; // not persistent
    srcMu = nullptr; // not persistent
    return readObj<PuppiObj>(from, *this);
}
bool l1ct::PuppiObjEmu::write(std::fstream & to) const {
    return writeObj<PuppiObj>(*this, to);
}

bool l1ct::PFRegionEmu::read(std::fstream & from) {
    return 
        readObj<PFRegion>(from, *this) &&
        readVar(from, etaCenter) &&
        readVar(from, etaMin) &&
        readVar(from, etaMax) &&
        readVar(from, phiCenter) &&
        readVar(from, phiHalfWidth) &&
        readVar(from, etaExtra) &&
        readVar(from, phiExtra);
}
bool l1ct::PFRegionEmu::write(std::fstream & to) const {
    return 
        writeObj<PFRegion>(*this, to) &&
        writeVar(etaCenter, to) &&
        writeVar(etaMin, to) &&
        writeVar(etaMax, to) &&
        writeVar(phiCenter, to) &&
        writeVar(phiHalfWidth, to) &&
        writeVar(etaExtra, to) &&
        writeVar(phiExtra, to);
}

bool l1ct::PVObjEmu::read(std::fstream & from) {
    return readAP(from, hwZ0);
}
bool l1ct::PVObjEmu::write(std::fstream & to) const {
    return writeAP(hwZ0, to);
}


bool l1ct::PFInputRegion::read(std::fstream & from) {
    return 
        region.read(from) &&
        readMany(from, hadcalo) &&
        readMany(from, emcalo) &&
        readMany(from, track) &&
        readMany(from, muon);
}
bool l1ct::PFInputRegion::write(std::fstream & to) const {
    return 
        region.write(to) &&
        writeMany(hadcalo, to) &&
        writeMany(emcalo, to) &&
        writeMany(track, to) &&
        writeMany(muon, to);
}

bool l1ct::OutputRegion::read(std::fstream & from) {
    return 
        readMany(from, pfcharged) &&
        readMany(from, pfneutral) &&
        readMany(from, pfphoton) &&
        readMany(from, pfmuon) &&
        readMany(from, puppi);
}
bool l1ct::OutputRegion::write(std::fstream & to) const {
    return 
        writeMany(pfcharged, to) &&
        writeMany(pfneutral, to) &&
        writeMany(pfphoton, to) &&
        writeMany(pfmuon, to) &&
        writeMany(puppi, to);
}

bool l1ct::Event::read(std::fstream & from) {
    return 
        readVar(from, run) &&
        readVar(from, lumi) &&
        readVar(from, event) &&
        readMany(from, pfinputs) &&
        readMany(from, pvs) &&
        readMany(from, out);
}
bool l1ct::Event::write(std::fstream & to) const {
    return 
        writeVar(run, to) &&
        writeVar(lumi, to) &&
        writeVar(event, to) &&
        writeMany(pfinputs, to) &&
        writeMany(pvs, to) &&
        writeMany(out, to);
}

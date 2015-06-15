#include "CMGTools/H2TauTau/interface/DiTauWithSVFitProducer.h"

#include "CMGTools/H2TauTau/interface/DiObjectUpdateFactory.h"
#include "CMGTools/H2TauTau/interface/DiTauObjectFactory.h"


namespace cmg {

typedef DiTauObjectFactory< pat::Tau, pat::Tau > DiTauPOProducer;
typedef DiTauObjectFactory< pat::Tau, pat::Electron > TauElePOProducer;
typedef DiTauObjectFactory< pat::Tau, pat::Muon > TauMuPOProducer;
typedef DiTauObjectFactory< pat::Muon, pat::Electron > MuElePOProducer;
typedef DiTauObjectFactory< pat::Muon, pat::Muon > DiMuPOProducer;

typedef DiObjectUpdateFactory< pat::Tau, pat::Muon > TauMuUpdateProducer;
typedef DiObjectUpdateFactory< pat::Tau, pat::Electron > TauEleUpdateProducer;
typedef DiObjectUpdateFactory< pat::Muon, pat::Electron  > MuEleUpdateProducer;
typedef DiObjectUpdateFactory< pat::Tau, pat::Tau> DiTauUpdateProducer;
typedef DiObjectUpdateFactory< pat::Muon, pat::Muon > DiMuUpdateProducer;

}

typedef DiTauWithSVFitProducer< pat::Tau, pat::Muon > TauMuWithSVFitProducer;
typedef DiTauWithSVFitProducer< pat::Tau, pat::Electron > TauEleWithSVFitProducer;
typedef DiTauWithSVFitProducer< pat::Muon, pat::Electron > MuEleWithSVFitProducer;
typedef DiTauWithSVFitProducer< pat::Tau, pat::Tau > TauTauWithSVFitProducer;
typedef DiTauWithSVFitProducer< pat::Muon, pat::Muon > DiMuWithSVFitProducer;

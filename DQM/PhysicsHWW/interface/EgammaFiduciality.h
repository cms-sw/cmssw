//
// This is an enum to describe the fiduciality 
// of Egamma objects (electrons and photons)
//

// Fiduciality in the calorimeter

#ifndef EgammaFiduciality_h
#define EgammaFiduciality_h

enum EgammaFiduciality {
	// used by photons and electrons
	ISEB,
	ISEBEEGAP,
	ISEE,
	ISEEGAP,
	// used only by electrons
	ISEBETAGAP,
	ISEBPHIGAP,
	ISEEDEEGAP,
	ISEERINGGAP,
	ISGAP
};

// seeding type used and corrections applied

enum EgammaElectronType {
	ISECALENERGYCORRECTED,	// if false, the electron "ecalEnergy" is just the supercluster energy 
	ISMOMENTUMCORRECTED,  	// has E-p combination been applied
	ISECALDRIVEN,
	ISTRACKERDRIVEN,
    ISCUTPRESELECTED,
    ISMVAPRESELECTED
};

#endif


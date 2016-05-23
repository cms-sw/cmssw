#ifndef SimpleElectron_H
#define SimpleElectron_H
 
#ifndef SimpleElectron_STANDALONE
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"        
#endif

class SimpleElectron
{
	public:
	    SimpleElectron(){}
	    SimpleElectron( unsigned int run, 
                        double eClass, 
                        double r9, 
                        double scEnergy, 
                        double scEnergyError, 
                        double trackMomentum, 
                        double trackMomentumError, 
                        double regEnergy, 
                        double regEnergyError, 
                        double combinedMomentum, 
                        double combinedMomentumError, 
                        double eta, 
                        bool isEB, 
                        bool isMC, 
                        bool isEcalDriven, 
                        bool isTrackerDriven
                      ) : 
                        run_(run),
                        eClass_(eClass), 
                        r9_(r9),
                        scEnergy_(scEnergy), 
                        scEnergyError_(scEnergyError), 
                        trackMomentum_(trackMomentum), 
                        trackMomentumError_(trackMomentumError), 
                        regEnergy_(regEnergy), 
                        regEnergyError_(regEnergyError), 
                        eta_(eta), 
                        isEB_(isEB), 
                        isMC_(isMC), 
                        isEcalDriven_(isEcalDriven), 
                        isTrackerDriven_(isTrackerDriven), 
                        newEnergy_(regEnergy_), 
                        newEnergyError_(regEnergyError_),
                        combinedMomentum_(combinedMomentum), 
                        combinedMomentumError_(combinedMomentumError),
                        scale_(1.0), smearing_(0.0)
        {}
	    ~SimpleElectron(){}	

#ifndef SimpleElectron_STANDALONE
        explicit SimpleElectron(const reco::GsfElectron &in, unsigned int runNumber, bool isMC) ;
        void writeTo(reco::GsfElectron & out) const ;
#endif

    	//accessors
    	double getNewEnergy() const {return newEnergy_;}
    	double getNewEnergyError() const {return newEnergyError_;}
    	double getCombinedMomentum() const {return combinedMomentum_;}
    	double getCombinedMomentumError() const {return combinedMomentumError_;}
    	double getScale() const {return scale_;}
    	double getSmearing() const {return smearing_;}
    	double getSCEnergy() const {return scEnergy_;}
    	double getSCEnergyError() const {return scEnergyError_;}
    	double getRegEnergy() const {return regEnergy_;}
    	double getRegEnergyError() const {return regEnergyError_;}
    	double getTrackerMomentum() const {return trackMomentum_;}
    	double getTrackerMomentumError() const {return trackMomentumError_;}
    	double getEta() const {return eta_;}
    	float getR9() const {return r9_;}
    	int getElClass() const {return eClass_;}
    	unsigned int getRunNumber() const {return run_;}
    	bool isEB() const {return isEB_;}
    	bool isMC() const {return isMC_;}
    	bool isEcalDriven() const {return isEcalDriven_;}
    	bool isTrackerDriven() const {return isTrackerDriven_;}
    
    	//setters
    	void setCombinedMomentum(double combinedMomentum){combinedMomentum_ = combinedMomentum;}
    	void setCombinedMomentumError(double combinedMomentumError){combinedMomentumError_ = combinedMomentumError;}
    	void setNewEnergy(double newEnergy){newEnergy_ = newEnergy;}
    	void setNewEnergyError(double newEnergyError){newEnergyError_ = newEnergyError;}

	private:
    	unsigned int run_; 
        double eClass_;
    	double r9_;
    	double scEnergy_; 
        double scEnergyError_; 
        double trackMomentum_; 
        double trackMomentumError_; 
        double regEnergy_; 
        double regEnergyError_;
    	double eta_;
    	bool isEB_;
    	bool isMC_;
    	bool isEcalDriven_;
    	bool isTrackerDriven_;
    	double newEnergy_; 
        double newEnergyError_; 
        double combinedMomentum_; 
        double combinedMomentumError_;
    	double scale_; 
        double smearing_;
};

#endif

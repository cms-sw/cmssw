#ifndef SINGLEPARTICLEWRAPPER_HH_
#define SINGLEPARTICLEWRAPPER_HH_

class SingleParticleWrapper
{
public:
	SingleParticleWrapper() {
		reset();
	}
	virtual ~SingleParticleWrapper() {};
	
	virtual void reset() {
		eEcal = 0;
		eHcal = 0;
		trueEnergy = -1;
		eta = 0;
		phi = 0;
		nEcalCluster = 0;
		nHcalCluster = 0;
		eSqEcal = 0;
		eSqHcal = 0;
		nPFCandidates = 0;
		ePFEcal = 0;
		ePFHcal = 0;
		eSqPFEcal = 0;
		eSqPFHcal = 0;
	};
	
	//True, Ecal and Hcal energy and angle depositions
	double eEcal;
	double eHcal;
	double ePFEcal;
	double ePFHcal;
	double trueEnergy;
	double eta;
	double phi;
	
	//Number of clusters in each of the Ecal and Hcal
	int nEcalCluster;
	int nHcalCluster;
	int nPFCandidates;
	
	//Sum of the squares of the Ecal and Hcal cluster energies
	double eSqEcal;
	double eSqHcal;
	double eSqPFEcal;
	double eSqPFHcal;
};

#endif /*SINGLEPARTICLEWRAPPER_HH_*/

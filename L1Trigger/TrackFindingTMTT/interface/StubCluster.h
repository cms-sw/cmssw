#ifndef __STUB_CLUSTER_H__
#define __STUB_CLUSTER_H__

#include <vector>
#include <set>


// Used to merge all clusters assigned to a track in a single layer into a single StubCluster
// whose coordinate is the mean of all the individual clusters.

namespace TMTT {

class Stub;
class TP;

class StubCluster{

    public :
	StubCluster( std::vector<const Stub *> stubs, double SectorPhi, int lres_dr = 1 );
	~StubCluster(){}

        void setLayerKF(unsigned int lay){layerKF_=lay;} // KF encoded tracker layer
        unsigned int layerKF() const {return layerKF_;}

	unsigned nStubs() const { return nstubs_; }
        std::vector<const Stub *> stubs() const { return stubs_; } // Returns original stubs that were usefd to make StubCluster.
	double r() const { return r_; }
	double phi() const { return phi_; }
	double z() const { return z_; }
	double eta() const { return eta_; }
	std::set<const TP *> assocTPs() const { return assocTPs_; } 
	unsigned layerId() const { return layerId_; }
        unsigned layerIdReduced() const { return layerIdReduced_; }
	unsigned endcapRing() const { return endcapRing_; }
	bool barrel() const { return barrel_; }
        bool tiltedBarrel() const { return tiltedBarrel_;}
        bool psModule() const { return psModule_;}
	//quadratic sum of the short side error of sensors and RMS of the sensor positions along the short side.
	double sigmaX() const { return sigmaX_; }
	//quadratic sum of the long side error of sensors and RMS of the sensor positions along the long side.
	double sigmaZ() const { return sigmaZ_; }
	//phi error from strip pitch
	double dphi_dl() const { return dphi_dl_; }
	//phi error from strip length in the endcap
	double dphi_dr() const { return dphi_dr_; }
	//total phi error
	double dphi() const { return dphi_; }
	//strip distance in cm from the center of the sensor.
	double deltal() const { return deltal_; }
	//strip distance from the center of the sensor.
	double deltai() const { return deltai_; }
        //alpha correction for non-radial strips in endcap 2S modules.
        double alpha() const { return alpha_;}
        // Angle between normal to module and beam-line along +ve z axis. (In range -PI/2 to PI/2).
        double moduleTilt() const { return moduleTilt_;} 

    private:
	std::vector<const Stub *> stubs_;
	unsigned nstubs_;
	double r_;
	double phi_;
	double z_;
	double eta_;
	std::set<const TP*> assocTPs_;
	unsigned layerId_;
	unsigned layerIdReduced_;
	unsigned endcapRing_;
        unsigned layerKF_;
	bool barrel_;
	bool tiltedBarrel_;
        bool psModule_;
	double sigmaX_;
	double sigmaZ_;
	double dphi_dl_;
	double dphi_dr_;
	double dphi_;
	double deltal_;
	double deltai_;
        double alpha_;
        double moduleTilt_;
};

}

#endif


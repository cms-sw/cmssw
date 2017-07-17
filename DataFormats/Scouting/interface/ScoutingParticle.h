#ifndef DataFormats_ScoutingParticle_h
#define DataFormats_ScoutingParticle_h

#include <vector>

//class for holding PF candidate information, for use in data scouting 
//IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class ScoutingParticle
{
    public:
        //constructor with values for all data fields
        ScoutingParticle(float pt, float eta, float phi, float m, 
		 int pdgId, int vertex):
            pt_(pt), eta_(eta), phi_(phi), m_(m), pdgId_(pdgId), vertex_(vertex) {}
        //default constructor
        ScoutingParticle():pt_(0), eta_(0), phi_(0), m_(0), pdgId_(0), vertex_(-1) {}

        //accessor functions
        float pt() const { return pt_; }
        float eta() const { return eta_; }
        float phi() const { return phi_; }
        float m() const { return m_; }
        int pdgId() const { return pdgId_; }
	int vertex() const { return vertex_; }

    private:
        float pt_;
        float eta_;
        float phi_;
        float m_;
        int pdgId_;
	int vertex_;
};

typedef std::vector<ScoutingParticle> ScoutingParticleCollection;

#endif

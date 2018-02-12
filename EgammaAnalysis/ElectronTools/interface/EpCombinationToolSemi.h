#ifndef EPCOMBINATIONTOOLSEMI_H
#define EPCOMBINATIONTOOLSEMI_H

#include <string>
#include <vector>
#include <utility>
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

class GBRForestD;

class EpCombinationToolSemi
{
    public:
        EpCombinationToolSemi();
        ~EpCombinationToolSemi();
        // forbid copy and assignment, since we have a custom deleter
        EpCombinationToolSemi(const EpCombinationToolSemi &other) = delete;
        EpCombinationToolSemi & operator=(const EpCombinationToolSemi &other) = delete;

        bool init(std::vector<const GBRForestD*> forest) ;
	std::pair<float, float> combine(reco::GsfElectron& electron) const;


    private:
	std::vector<const GBRForestD*> m_forest;
	float meanlimlow  ;
	float meanlimhigh ;
	float meanoffset  ;
	float meanscale   ;

	float sigmalimlow ;
	float sigmalimhigh;
	float sigmaoffset ;
	float sigmascale  ;

	float lowEnergyThr  ;
	float highEnergyThr ;
	float eOverPThr     ;
	float epDiffSigThr  ;
	float epSigThr      ;
	
	bool m_ownForest;

};


#endif

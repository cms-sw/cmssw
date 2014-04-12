#ifndef EPCOMBINATIONTOOL_H
#define EPCOMBINATIONTOOL_H

#include <string>
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"

class GBRForest;

class EpCombinationTool
{
    public:
        EpCombinationTool();
        ~EpCombinationTool();

        bool init(const std::string& regressionFile, const std::string& bdtName="");
//        float combine(float energy, float energyError,
//                float momentum, float momentumError, 
//                int electronClass,
//                bool isEcalDriven, bool isTrackerDriven, bool isEB);
	void combine(SimpleElectron & mySimpleElectron);


    private:
        GBRForest* m_forest;

};


#endif

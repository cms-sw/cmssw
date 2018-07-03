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
        // forbid copy and assignment, since we have a custom deleter
        EpCombinationTool(const EpCombinationTool &other) = delete;
        EpCombinationTool & operator=(const EpCombinationTool &other) = delete;

        bool init(const GBRForest *forest) ;
        bool init(const std::string& regressionFile, const std::string& bdtName);
	void combine(SimpleElectron & mySimpleElectron) const;


    private:
        const GBRForest* m_forest;
        bool  m_ownForest;

};


#endif

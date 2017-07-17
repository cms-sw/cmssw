#ifndef ElectronEnergyCalibrator_H
#define ElectronEnergyCalibrator_H

#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

namespace edm {
  class StreamID;
}

struct correctionValues
{
    double nRunMin;
    double nRunMax;
    double corrCat0;
    double corrCat1;
    double corrCat2;
    double corrCat3;
    double corrCat4;
    double corrCat5;
    double corrCat6;
    double corrCat7;
};

struct linearityCorrectionValues
{
    double ptMin;
    double ptMax;
    double corrCat0;
    double corrCat1;
    double corrCat2;
    double corrCat3;
    double corrCat4;
    double corrCat5;
};

class ElectronEnergyCalibrator
{
 public:
 ElectronEnergyCalibrator(const std::string pathData, 
			  const std::string pathLinData,
			  const std::string dataset, 
			  int correctionsType, 
			  bool applyLinearityCorrection, 
			  double lumiRatio, 
			  bool isMC, 
			  bool updateEnergyErrors, 
			  bool verbose, 
			  bool synchronization
			  ) : 
    pathData_(pathData), 
    pathLinData_(pathLinData), 
    dataset_(dataset), 
    correctionsType_(correctionsType), 
    applyLinearityCorrection_(applyLinearityCorrection),
    lumiRatio_(lumiRatio), 
    isMC_(isMC), 
    updateEnergyErrors_(updateEnergyErrors), 
    verbose_(verbose), 
    synchronization_(synchronization) {
      init();
    }

    void calibrate(SimpleElectron &electron, edm::StreamID const&);
    void correctLinearity(SimpleElectron &electron);
				  
 private:
    void init();
    void splitString( const std::string &fullstr, 
		      std::vector<std::string> &elements, 
		      const std::string &delimiter
		      );
    double stringToDouble(const std::string &str);
      
    double newEnergy_ ;
    double newEnergyError_ ;
    
    std::string pathData_;
    std::string pathLinData_;
    std::string dataset_;
    int correctionsType_;
    bool applyLinearityCorrection_;
    double lumiRatio_;
    bool isMC_;
    bool updateEnergyErrors_;
    bool verbose_;
    bool synchronization_;
    
    correctionValues corrValArray[100];
    correctionValues corrValMC;
    linearityCorrectionValues linCorrValArray[100];
    int nCorrValRaw, nLinCorrValRaw;
};

#endif


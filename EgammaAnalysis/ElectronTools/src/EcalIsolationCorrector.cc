#ifndef STANDALONE_ECALCORR
#include "EgammaAnalysis/ElectronTools/interface/EcalIsolationCorrector.h"
#else
#include "EcalIsolationCorrector.h"
#endif

EcalIsolationCorrector::EcalIsolationCorrector(bool forElectrons):isElectron_(forElectrons)
{}

EcalIsolationCorrector::RunRange EcalIsolationCorrector::checkRunRange(int runNumber) {
  
  EcalIsolationCorrector::RunRange runRange = RunAB;
  
  if (runNumber <= 203755 && runNumber > 197770)
    runRange = RunC;
  else if (runNumber > 203755)
    runRange = RunD;
  
  return runRange;
}

float EcalIsolationCorrector::correctForNoise(float iso, bool isBarrel, EcalIsolationCorrector::RunRange runRange, bool isData) {
  
  float result = iso;
  
  if (!isElectron_) {
    if (!isData) {
      if (runRange == RunAB) {
	if (!isBarrel)
	  result = (iso-0.2827)/1.0949;
	else
	  result = (iso-0.0931)/1.0738;
      } else if (runRange == RunC) {
	if (!isBarrel)
	  result = (iso-0.5690)/0.9217;
	else
	  result = (iso-0.1824)/0.9279;
      } else if (runRange == RunD) {
	if (!isBarrel) 
	  result = (iso-0.9997)/0.8781;
	else
	  result = (iso-0.0944)/0.8140;
      }
    } else {
      std::cout << "Warning: you should correct MC to data" << std::endl;
    }
  } else {
    if (!isData) {
      if (runRange == RunAB) {
	if (!isBarrel)
	  result = (iso+0.1174)/1.0012;
	else
	  result = (iso+0.2736)/0.9948;
      } else if (runRange == RunC) {
	if (!isBarrel)
	  result = (iso+0.2271)/0.9684;
	else
	  result = (iso+0.5962)/0.9568;
      } else if (runRange == RunD) {
	if (!isBarrel) 
	  result = (iso+0.2907)/1.0005;
	else
	  result = (iso+0.9098)/0.9395;
      }
    } else {
      std::cout << "Warning: you should correct MC to data" << std::endl;
    } 
  }
  
  return result;
}

#ifndef STANDALONE_ECALCORR
// GSF Electron Methods 
float EcalIsolationCorrector::correctForNoise(reco::GsfElectron e, int runNumber, bool isData) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;
  
  float iso = e.dr03EcalRecHitSumEt();
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);
  
  return correctForNoise(iso, e.isEB(), runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(reco::GsfElectron e, std::string runName, bool isData) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;  
  
  float iso = e.dr03EcalRecHitSumEt();
  EcalIsolationCorrector::RunRange runRange = RunAB;
  
  if (runName == "RunAB") 
    runRange = RunAB;
  else if (runName == "RunC")
    runRange = RunC;
  else if (runName == "RunD")
    runRange = RunD;
  else {
    std::cerr << "Error: Unknown run range " << runName << std::endl;
    abort();
  }
  
  return correctForNoise(iso, e.isEB(), runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(reco::GsfElectron e, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;
  
  float iso = e.dr03EcalRecHitSumEt();
  float combination = (intL_AB * correctForNoise(iso, e.isEB(), RunAB, isData) +
		       intL_C  * correctForNoise(iso, e.isEB(), RunC,  isData) +
		       intL_D  * correctForNoise(iso, e.isEB(), RunD,  isData))/(intL_AB + intL_C + intL_D);
  
  return combination;
}

// PAT Electron Methods 
float EcalIsolationCorrector::correctForNoise(pat::Electron e, int runNumber, bool isData) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;
  
  float iso = e.dr03EcalRecHitSumEt();
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);
  
  return correctForNoise(iso, e.isEB(), runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(pat::Electron e, std::string runName, bool isData) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;  
  
  float iso = e.dr03EcalRecHitSumEt();
  EcalIsolationCorrector::RunRange runRange = RunAB;
  
  if (runName == "RunAB") 
    runRange = RunAB;
  else if (runName == "RunC")
    runRange = RunC;
  else if (runName == "RunD")
    runRange = RunD;
  else {
    std::cerr << "Error: Unknown run range " << runName << std::endl;
    abort();
  }
  
  return correctForNoise(iso, e.isEB(), runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(pat::Electron e, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;
  
  float iso = e.dr03EcalRecHitSumEt();
  float combination = (intL_AB * correctForNoise(iso, e.isEB(), RunAB, isData) +
		       intL_C  * correctForNoise(iso, e.isEB(), RunC,  isData) +
		       intL_D  * correctForNoise(iso, e.isEB(), RunD,  isData))/(intL_AB + intL_C + intL_D);
  
  return combination;
}

// RECO Photon Method
float EcalIsolationCorrector::correctForNoise(reco::Photon p, int runNumber, bool isData) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);
  
  return correctForNoise(iso, p.isEB(), runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(reco::Photon p, std::string runName, bool isData) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  EcalIsolationCorrector::RunRange runRange = RunAB;
  
  if (runName == "RunAB") 
    runRange = RunAB;
  else if (runName == "RunC")
    runRange = RunC;
  else if (runName == "RunD")
    runRange = RunD;
  else {
    std::cerr << "Error: Unknown run range " << runName << std::endl;
    abort();
  }
  
  return correctForNoise(iso, p.isEB(), runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(reco::Photon p, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  float combination = (intL_AB * correctForNoise(iso, p.isEB(), RunAB, isData) +
		       intL_C  * correctForNoise(iso, p.isEB(), RunC,  isData) +
		       intL_D  * correctForNoise(iso, p.isEB(), RunD,  isData))/(intL_AB + intL_C + intL_D);
  
  return combination;
}

// PAT Photon Method
float EcalIsolationCorrector::correctForNoise(pat::Photon p, int runNumber, bool isData) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);
  
  return correctForNoise(iso, p.isEB(), runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(pat::Photon p, std::string runName, bool isData) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  EcalIsolationCorrector::RunRange runRange = RunAB;
  
  if (runName == "RunAB") 
    runRange = RunAB;
  else if (runName == "RunC")
    runRange = RunC;
  else if (runName == "RunD")
    runRange = RunD;
  else {
    std::cerr << "Error: Unknown run range " << runName << std::endl;
    abort();
  }
  
  return correctForNoise(iso, p.isEB(), runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(pat::Photon p, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  float combination = (intL_AB * correctForNoise(iso, p.isEB(), RunAB, isData) +
		       intL_C  * correctForNoise(iso, p.isEB(), RunC,  isData) +
		       intL_D  * correctForNoise(iso, p.isEB(), RunD,  isData))/(intL_AB + intL_C + intL_D);
  
  return combination;
}
#else
float EcalIsolationCorrector::correctForNoise(float iso, bool isBarrel, int runNumber, bool isData) {
  
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);
  
  return correctForNoise(iso, isBarrel, runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(float iso, bool isBarrel, std::string runName, bool isData) {
  
  EcalIsolationCorrector::RunRange runRange = RunAB;
  
  if (runName == "RunAB") 
    runRange = RunAB;
  else if (runName == "RunC")
    runRange = RunC;
  else if (runName == "RunD")
    runRange = RunD;
  else {
    std::cerr << "Error: Unknown run range " << runName << std::endl;
    abort();
  }
  
  return correctForNoise(iso, isBarrel, runRange, isData);
}

float EcalIsolationCorrector::correctForNoise(float iso , bool isBarrel, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  float combination = (intL_AB * correctForNoise(iso, isBarrel, RunAB, isData) +
		       intL_C  * correctForNoise(iso, isBarrel, RunC,  isData) +
		       intL_D  * correctForNoise(iso, isBarrel, RunD,  isData))/(intL_AB + intL_C + intL_D);
  
  return combination;
}
#endif

float EcalIsolationCorrector::correctForHLTDefinition(float iso, bool isBarrel, EcalIsolationCorrector::RunRange runRange) {
  
  float result = iso;
  
  if (!isElectron_) {
    if (runRange == RunAB) {
      if (!isBarrel)
	result = iso*0.8499-0.6510;
      else
	result = iso*0.8504-0.5658;
    } else if (runRange == RunC) {
      if (!isBarrel)
	result = iso*0.9346-0.9987;
      else
	result = iso*0.8529-0.6816;
    } else if (runRange == RunD) {
      if (!isBarrel) 
	result = iso*0.8318-0.9999;
      else
	result = iso*0.8853-0.8783;
    }
  } else {
    if (runRange == RunAB) {
      if (!isBarrel)
	result = iso*0.9849-0.6871;
      else
	result = iso*0.8542-0.3558;
    } else if (runRange == RunC) {
      if (!isBarrel)
	result = iso*0.9996-0.8485;
      else
	result = iso*0.9994-0.5085;
    } else if (runRange == RunD) {
      if (!isBarrel) 
	result = iso*0.9467-0.9998;
      else
	result = iso*0.8574-0.4862;
    }
  }
  
  return result;
}

#ifndef STANDALONE_ECALCORR
// GSF Electron Methods
float EcalIsolationCorrector::correctForHLTDefinition(reco::GsfElectron e, int runNumber, bool isData) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;
  
  float iso = e.dr03EcalRecHitSumEt();
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);
  
  if (!isData)
    iso = correctForNoise(iso, e.isEB(), runRange, false);
  
  return correctForHLTDefinition(iso, e.isEB(), runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(reco::GsfElectron e, std::string runName, bool isData) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;
  
  float iso = e.dr03EcalRecHitSumEt();
  EcalIsolationCorrector::RunRange runRange = RunAB;
  if (runName == "RunAB") 
    runRange = RunAB;
  else if (runName == "RunC")
    runRange = RunC;
  else if (runName == "RunD")
    runRange = RunD;
  else {
    std::cerr << "Error: Unknown run range " << runName << std::endl;
    abort();
  }
  
  if (!isData)
    iso = correctForNoise(iso, e.isEB(), runRange, false);
  
  return correctForHLTDefinition(iso, e.isEB(), runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(reco::GsfElectron e, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;
  
  float iso = e.dr03EcalRecHitSumEt();
  if (!isData)
    iso = correctForNoise(e, isData, intL_AB, intL_C, intL_D);
  
  float combination = (intL_AB * correctForHLTDefinition(iso, e.isEB(), RunAB) +
		       intL_C  * correctForHLTDefinition(iso, e.isEB(), RunC ) +
		       intL_D  * correctForHLTDefinition(iso, e.isEB(), RunD ))/(intL_AB + intL_C + intL_D);
  
  return combination;
}
 
// PAT Electron Methods
float EcalIsolationCorrector::correctForHLTDefinition(pat::Electron e, int runNumber, bool isData) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;
  
  float iso = e.dr03EcalRecHitSumEt();
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);
  
  if (!isData)
    iso = correctForNoise(iso, e.isEB(), runRange, false);
  
  return correctForHLTDefinition(iso, e.isEB(), runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(pat::Electron e, std::string runName, bool isData) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;
  
  float iso = e.dr03EcalRecHitSumEt();
  EcalIsolationCorrector::RunRange runRange = RunAB;
  if (runName == "RunAB") 
    runRange = RunAB;
  else if (runName == "RunC")
    runRange = RunC;
  else if (runName == "RunD")
    runRange = RunD;
  else {
    std::cerr << "Error: Unknown run range " << runName << std::endl;
    abort();
  }
  
  if (!isData)
    iso = correctForNoise(iso, e.isEB(), runRange, false);
  
  return correctForHLTDefinition(iso, e.isEB(), runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(pat::Electron e, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  if (!isElectron_)
    std::cerr << "Warning: this corrector is setup for photons and you are passing an electron !" << std::endl;
  
  float iso = e.dr03EcalRecHitSumEt();
  if (!isData)
    iso = correctForNoise(e, isData, intL_AB, intL_C, intL_D);
  
  float combination = (intL_AB * correctForHLTDefinition(iso, e.isEB(), RunAB) +
		       intL_C  * correctForHLTDefinition(iso, e.isEB(), RunC ) +
		       intL_D  * correctForHLTDefinition(iso, e.isEB(), RunD ))/(intL_AB + intL_C + intL_D);
  
  return combination;
}

// RECO Photon Methods
float EcalIsolationCorrector::correctForHLTDefinition(reco::Photon p, int runNumber, bool isData) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);
  
  if (!isData)
    iso = correctForNoise(iso, p.isEB(), runRange, false);
  
  return correctForHLTDefinition(iso, p.isEB(), runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(reco::Photon p, std::string runName, bool isData) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  
  EcalIsolationCorrector::RunRange runRange = RunAB;
  if (runName == "RunAB") 
    runRange = RunAB;
  else if (runName == "RunC")
    runRange = RunC;
  else if (runName == "RunD")
    runRange = RunD;
  else {
    std::cerr << "Error: Unknown run range " << runName << std::endl;
    abort();
  }
  
  if (!isData)
    iso = correctForNoise(iso, p.isEB(), runRange, false);
  
  return correctForHLTDefinition(iso, p.isEB(), runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(reco::Photon p, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  
  if (!isData)
    iso = correctForNoise(p, isData, intL_AB, intL_C, intL_D);
  
  float combination = (intL_AB * correctForHLTDefinition(iso, p.isEB(), RunAB) +
		       intL_C  * correctForHLTDefinition(iso, p.isEB(), RunC ) +
		       intL_D  * correctForHLTDefinition(iso, p.isEB(), RunD ))/(intL_AB + intL_C + intL_D);
  
  return combination;
}

// PAT Photon Methods
float EcalIsolationCorrector::correctForHLTDefinition(pat::Photon p, int runNumber, bool isData) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);
  
  if (!isData)
    iso = correctForNoise(iso, p.isEB(), runRange, false);
  
  return correctForHLTDefinition(iso, p.isEB(), runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(pat::Photon p, std::string runName, bool isData) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  
  EcalIsolationCorrector::RunRange runRange = RunAB;
  if (runName == "RunAB") 
    runRange = RunAB;
  else if (runName == "RunC")
    runRange = RunC;
  else if (runName == "RunD")
    runRange = RunD;
  else {
    std::cerr << "Error: Unknown run range " << runName << std::endl;
    abort();
  }
  
  if (!isData)
    iso = correctForNoise(iso, p.isEB(), runRange, false);
  
  return correctForHLTDefinition(iso, p.isEB(), runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(pat::Photon p, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  if (isElectron_)
    std::cerr << "Warning: this corrector is setup for electrons and you are passing a photon !" << std::endl;
  
  float iso = p.ecalRecHitSumEtConeDR03();
  
  if (!isData)
    iso = correctForNoise(p, isData, intL_AB, intL_C, intL_D);
  
  float combination = (intL_AB * correctForHLTDefinition(iso, p.isEB(), RunAB) +
		       intL_C  * correctForHLTDefinition(iso, p.isEB(), RunC ) +
		       intL_D  * correctForHLTDefinition(iso, p.isEB(), RunD ))/(intL_AB + intL_C + intL_D);
  
  return combination;
}
#else
float EcalIsolationCorrector::correctForHLTDefinition(float iso, bool isBarrel, int runNumber, bool isData) {
  
  EcalIsolationCorrector::RunRange runRange = checkRunRange(runNumber);
  
  if (!isData)
    iso = correctForNoise(iso, isBarrel, runRange, false);

  return correctForHLTDefinition(iso, isBarrel, runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(float iso, bool isBarrel, std::string runName, bool isData) {
  
  EcalIsolationCorrector::RunRange runRange = RunAB;
  if (runName == "RunAB") 
    runRange = RunAB;
  else if (runName == "RunC")
    runRange = RunC;
  else if (runName == "RunD")
    runRange = RunD;
  else {
    std::cerr << "Error: Unknown run range " << runName << std::endl;
    abort();
  }

  if (!isData)
    iso = correctForNoise(iso, isBarrel, runRange, false);

  return correctForHLTDefinition(iso, isBarrel, runRange);
}

float EcalIsolationCorrector::correctForHLTDefinition(float iso, bool isBarrel, bool isData, float intL_AB, float intL_C, float intL_D) {
  
  if (!isData)
    iso = correctForNoise(iso, isBarrel, false, intL_AB, intL_C, intL_D);
  
  float combination = (intL_AB * correctForHLTDefinition(iso, isBarrel, RunAB) +
		       intL_C  * correctForHLTDefinition(iso, isBarrel, RunC ) +
		       intL_D  * correctForHLTDefinition(iso, isBarrel, RunD ))/(intL_AB + intL_C + intL_D);

  return combination;
}
#endif



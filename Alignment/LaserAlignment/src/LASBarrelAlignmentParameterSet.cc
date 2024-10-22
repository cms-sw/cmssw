
#include "Alignment/LaserAlignment/interface/LASBarrelAlignmentParameterSet.h"

///
///
///
LASBarrelAlignmentParameterSet::LASBarrelAlignmentParameterSet() { Init(); }

///
/// whatever initialization is needed
///
void LASBarrelAlignmentParameterSet::Init(void) {
  // could use a single vector<vector<vector<pair<> > > >
  // but better split it in 6 parts
  for (int i = 0; i < 2; ++i) {  // twice; once for each endface
    tecPlusParameters.push_back(std::vector<std::pair<double, double> >(3));
    tecMinusParameters.push_back(std::vector<std::pair<double, double> >(3));
    tibPlusParameters.push_back(std::vector<std::pair<double, double> >(3));
    tibMinusParameters.push_back(std::vector<std::pair<double, double> >(3));
    tobPlusParameters.push_back(std::vector<std::pair<double, double> >(3));
    tobMinusParameters.push_back(std::vector<std::pair<double, double> >(3));
  }

  // the beam parameters (8 beams * 2 pars) are stored in one single container
  for (int i = 0; i < 8; ++i) {
    beamParameters.push_back(std::vector<std::pair<double, double> >(2));
  }
}

///
/// function for accessing a single parameter (pair<>);
/// indices are:
///  * aSubdetector = 0 (TEC+), 1 (TEC-), 2 (TIB+), 3 (TIB-), 4 (TOB+), 5 (TOB-)
///  * aDisk = 0 (lower z), 1 (upper z)
///  * aParameter: 0 (rotation angle), 1 (x displacement), 2 (y displacement)
///
std::pair<double, double>& LASBarrelAlignmentParameterSet::GetParameter(int aSubdetector, int aDisk, int aParameter) {
  if (aSubdetector < 0 || aSubdetector > 5) {
    throw cms::Exception("Laser Alignment")
        << " [LASBarrelAlignmentParameterSet::GetParameter] ERROR ** Illegal subdetector index: " << aSubdetector << "."
        << std::endl;
  }

  if (aDisk < 0 || aDisk > 1) {
    throw cms::Exception("Laser Alignment")
        << " [LASBarrelAlignmentParameterSet::GetParameter] ERROR ** Illegal endface index: " << aDisk << "."
        << std::endl;
  }

  if (aParameter < 0 || aParameter > 2) {
    throw cms::Exception("Laser Alignment")
        << " [LASBarrelAlignmentParameterSet::GetParameter] ERROR ** Illegal parameter index: " << aParameter << "."
        << std::endl;
  }

  // would use a switch here, but this creates problems..
  if (aSubdetector == 0)
    return tecPlusParameters.at(aDisk).at(aParameter);
  else if (aSubdetector == 1)
    return tecMinusParameters.at(aDisk).at(aParameter);
  else if (aSubdetector == 2)
    return tibPlusParameters.at(aDisk).at(aParameter);
  else if (aSubdetector == 3)
    return tibMinusParameters.at(aDisk).at(aParameter);
  else if (aSubdetector == 4)
    return tobPlusParameters.at(aDisk).at(aParameter);
  else
    return tobMinusParameters.at(aDisk).at(aParameter);
}

///
/// return a single beam parameter (pair<> for value, error).
/// we have eight beams with two parameters each (phi1, phi2)
///
std::pair<double, double>& LASBarrelAlignmentParameterSet::GetBeamParameter(int aBeam, int aParameter) {
  if (aBeam < 0 || aBeam > 7) {
    throw cms::Exception("Laser Alignment")
        << " [LASBarrelAlignmentParameterSet::GetBeamParameter] ERROR ** Illegal beam index: " << aBeam << "."
        << std::endl;
  }

  if (aParameter < 0 || aParameter > 1) {
    throw cms::Exception("Laser Alignment")
        << " [LASBarrelAlignmentParameterSet::GetBeamParameter] ERROR ** Illegal beam parameter index: " << aParameter
        << "." << std::endl;
  }

  return beamParameters.at(aBeam).at(aParameter);
}

///
/// pretty-print all parameter and error values
///
void LASBarrelAlignmentParameterSet::Print(void) {
  std::cout << std::endl << " [LASBarrelAlignmentParameterSet::Print] -- Parameter list [rad/mm]: " << std::endl;

  const std::string subdetNames[6] = {" TEC+  ", " TEC-  ", " TIB+  ", " TIB-  ", " TOB+  ", " TOB-  "};

  std::cout << " Detector parameters: " << std::endl;
  std::cout << " --------------------" << std::endl;
  std::cout << " Values:     PHI1         X1          Y1         PHI2         X2          Y2   " << std::endl;
  for (int subdet = 0; subdet < 6; ++subdet) {
    std::cout << subdetNames[subdet];
    for (int par = 0; par < 3; ++par)
      std::cout << std::right << std::setw(12) << std::setprecision(6) << std::fixed
                << GetParameter(subdet, 0, par).first;
    for (int par = 0; par < 3; ++par)
      std::cout << std::right << std::setw(12) << std::setprecision(6) << std::fixed
                << GetParameter(subdet, 1, par).first;
    std::cout << std::endl;
  }

  std::cout << " Errors:     PHI1         X1          Y1         PHI2         X2          Y2   " << std::endl;
  for (int subdet = 0; subdet < 6; ++subdet) {
    std::cout << subdetNames[subdet];
    for (int par = 0; par < 3; ++par)
      std::cout << std::right << std::setw(12) << std::setprecision(6) << std::fixed
                << GetParameter(subdet, 0, par).second;
    for (int par = 0; par < 3; ++par)
      std::cout << std::right << std::setw(12) << std::setprecision(6) << std::fixed
                << GetParameter(subdet, 1, par).second;
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << " Beam parameters: " << std::endl;
  std::cout << " ----------------" << std::endl;
  std::cout << " Values:     PHI1        PHI2" << std::endl;
  for (int beam = 0; beam < 8; ++beam) {
    std::cout << " beam " << beam;
    for (int par = 0; par < 2; ++par)
      std::cout << std::right << std::setw(12) << std::setprecision(6) << std::fixed
                << GetBeamParameter(beam, par).first;
    std::cout << std::endl;
  }

  std::cout << " Errors:     PHI1        PHI2" << std::endl;
  for (int beam = 0; beam < 8; ++beam) {
    std::cout << " beam " << beam;
    for (int par = 0; par < 2; ++par)
      std::cout << std::right << std::setw(12) << std::setprecision(6) << std::fixed
                << GetBeamParameter(beam, par).second;
    std::cout << std::endl;
  }

  std::cout << " [LASBarrelAlignmentParameterSet::Print] -- End parameter list." << std::endl;
}

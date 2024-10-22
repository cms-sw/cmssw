
#include "Alignment/LaserAlignment/interface/LASEndcapAlignmentParameterSet.h"

///
///
///
LASEndcapAlignmentParameterSet::LASEndcapAlignmentParameterSet() { Init(); }

///
/// whatever initialization is needed
///
void LASEndcapAlignmentParameterSet::Init(void) {
  // could use a single vector<vector<vector<pair<> > > >
  // but better split it in 2 parts

  for (unsigned int disk = 0; disk < 9; ++disk) {  // nine times; once for each disk
    tecPlusDiskParameters.push_back(std::vector<std::pair<double, double> >(3));
    tecMinusDiskParameters.push_back(std::vector<std::pair<double, double> >(3));

    // compiler won't let me init the pairs within push_back call..
    for (unsigned int par = 0; par < 3; ++par) {
      tecPlusDiskParameters.at(disk).at(par) = std::pair<double, double>(0., 0.);
      tecMinusDiskParameters.at(disk).at(par) = std::pair<double, double>(0., 0.);
    }
  }

  // once for each parameter
  for (unsigned int par = 0; par < 6; ++par) {
    tecPlusGlobalParameters.push_back(std::pair<double, double>(0., 0.));
    tecMinusGlobalParameters.push_back(std::pair<double, double>(0., 0.));
  }

  // beam parameters
  tecPlusBeamParameters.resize(8);
  tecMinusBeamParameters.resize(8);

  // now once for each beam: the 2 rings
  for (unsigned int beam = 0; beam < 8; ++beam) {
    tecPlusBeamParameters.at(beam).resize(2);
    tecMinusBeamParameters.at(beam).resize(2);

    // now once for each ring: the 2 parameters
    for (unsigned int ring = 0; ring < 2; ++ring) {
      tecPlusBeamParameters.at(beam).at(ring).resize(2);
      tecMinusBeamParameters.at(beam).at(ring).resize(2);

      // now once for each parameter: the pairs (value/error)
      for (unsigned int par = 0; par < 2; ++par) {
        tecPlusBeamParameters.at(beam).at(ring).at(par) = std::pair<double, double>(0., 0.);
        tecMinusBeamParameters.at(beam).at(ring).at(par) = std::pair<double, double>(0., 0.);
      }
    }
  }
}

///
/// function for accessing a single disk parameter (pair<>);
/// indices are:
///  * aSubdetector = 0 (TEC+), 1 (TEC-)
///  * aDisk = 0..8 (from inner to outer)
///  * aParameter: 0 (rotation angle), 1 (x displacement), 2 (y displacement)
///
std::pair<double, double>& LASEndcapAlignmentParameterSet::GetDiskParameter(int aSubdetector,
                                                                            int aDisk,
                                                                            int aParameter) {
  if (aSubdetector < 0 || aSubdetector > 1) {
    throw cms::Exception("Laser Alignment")
        << " [LASEndcapAlignmentParameterSet::GetDiskParameter] ERROR ** Illegal subdetector index: " << aSubdetector
        << "." << std::endl;
  }

  if (aDisk < 0 || aDisk > 8) {
    throw cms::Exception("Laser Alignment")
        << " [LASEndcapAlignmentParameterSet::GetDiskParameter] ERROR ** Illegal disk index: " << aDisk << "."
        << std::endl;
  }

  if (aParameter < 0 || aParameter > 2) {
    throw cms::Exception("Laser Alignment")
        << " [LASEndcapAlignmentParameterSet::GetDiskParameter] ERROR ** Illegal parameter index: " << aParameter << "."
        << std::endl;
  }

  if (aSubdetector == 0)
    return tecPlusDiskParameters.at(aDisk).at(aParameter);
  return tecMinusDiskParameters.at(aDisk).at(aParameter);
}

///
/// function for accessing a single global parameter (pair<>);
/// indices are:
///  * aSubdetector = 0 (TEC+), 1 (TEC-)
///  * aParameter: 0 (global rotation), 1 (global torsion),
///                2 (global x shift),  3 (global x shear),
///                4 (global y shift),  5 (global y shear)
///
std::pair<double, double>& LASEndcapAlignmentParameterSet::GetGlobalParameter(int aSubdetector, int aParameter) {
  if (aSubdetector < 0 || aSubdetector > 1) {
    throw cms::Exception("Laser Alignment")
        << " [LASEndcapAlignmentParameterSet::GetGlobalParameter] ERROR ** Illegal subdetector index: " << aSubdetector
        << "." << std::endl;
  }

  if (aParameter < 0 || aParameter > 5) {
    throw cms::Exception("Laser Alignment")
        << " [LASEndcapAlignmentParameterSet::GetGlobalParameter] ERROR ** Illegal parameter index: " << aParameter
        << "." << std::endl;
  }

  if (aSubdetector == 0)
    return tecPlusGlobalParameters.at(aParameter);
  return tecMinusGlobalParameters.at(aParameter);
}

///
/// function for accessing a single beam parameter (pair<>);
/// indices are:
///  * aSubdetector = 0 (TEC+), 1 (TEC-)
///  * aBeam = 0..7
///  * aParameter: 0 (deltaPhi on disk0), 1 (deltaPhi on disk8),
///
std::pair<double, double>& LASEndcapAlignmentParameterSet::GetBeamParameter(int aSubdetector,
                                                                            int aRing,
                                                                            int aBeam,
                                                                            int aParameter) {
  if (aSubdetector < 0 || aSubdetector > 1) {
    throw cms::Exception("Laser Alignment")
        << " [LASEndcapAlignmentParameterSet::GetBeamParameter] ERROR ** Illegal subdetector index: " << aSubdetector
        << "." << std::endl;
  }

  if (aRing < 0 || aRing > 1) {
    throw cms::Exception("Laser Alignment")
        << " [LASEndcapAlignmentParameterSet::GetBeamParameter] ERROR ** Illegal ring index: " << aRing << "."
        << std::endl;
  }

  if (aBeam < 0 || aBeam > 7) {
    throw cms::Exception("Laser Alignment")
        << " [LASEndcapAlignmentParameterSet::GetBeamParameter] ERROR ** Illegal beam index: " << aBeam << "."
        << std::endl;
  }

  if (aParameter < 0 || aParameter > 5) {
    throw cms::Exception("Laser Alignment")
        << " [LASEndcapAlignmentParameterSet::GetBeamParameter] ERROR ** Illegal parameter index: " << aParameter << "."
        << std::endl;
  }

  if (aSubdetector == 0)
    return tecPlusBeamParameters.at(aBeam).at(aRing).at(aParameter);
  return tecMinusBeamParameters.at(aBeam).at(aRing).at(aParameter);
}

///
/// pretty-printout of all parameter and error values
///
void LASEndcapAlignmentParameterSet::Print(void) {
  std::cout << " [LASEndcapAlignmentParameterSet::Print] -- Listing parameters:" << std::endl;
  std::cout << std::endl;
  std::cout << "  Disk parameters:" << std::endl;
  std::cout << " ----------------" << std::endl;
  for (int det = 0; det < 2; ++det) {
    std::cout
        << "  " << (det == 0 ? "TEC+" : "TEC-")
        << ":          dPHI \xb1  \bE                 dX \xb1  \bE                 dY \xb1  \bE          (rad/mm): "
        << std::endl;
    for (int disk = 0; disk < 9; ++disk) {
      std::cout << "  disk " << disk << ": ";
      for (int par = 0; par < 3; ++par)
        std::cout << std::right << std::setw(11) << std::fixed << std::setprecision(6)
                  << GetDiskParameter(det, disk, par).first << " \xb1 " << std::left << std::setw(9) << std::fixed
                  << std::setprecision(6) << GetDiskParameter(det, disk, par).second;
      std::cout << std::endl;
    }
  }

  for (int det = 0; det < 2; ++det) {
    std::cout << "  " << (det == 0 ? "TEC+" : "TEC-")
              << " global parameters in format: dPhi0\xb1 \be  dPhiT\xb1 \be  dX0\xb1 \be  dXT\xb1 \be  dY0\xb1 \be  "
                 "dYT\xb1 \be (rad/mm): "
              << std::endl;
    for (int par = 0; par < 6; ++par)
      std::cout << std::setw(11) << std::setprecision(6) << std::right << GetGlobalParameter(det, par).first << " \xb1 "
                << std::setw(9) << std::setprecision(6) << std::left << GetGlobalParameter(det, par).second;
    std::cout << std::endl;
  }

  for (int det = 0; det < 2; ++det) {
    std::cout << "  " << (det == 0 ? "TEC+" : "TEC-")
              << " beam parameters in format: dPhi1\xb1 \be dPhi2\xb1 \be (rad): " << std::endl;
    for (int ring = 0; ring < 2; ++ring) {
      std::cout << "   ring " << (ring == 0 ? "4" : "6") << ": " << std::endl;
      for (int beam = 0; beam < 8; ++beam) {
        std::cout << "     beam " << beam << ": ";
        for (int par = 0; par < 2; ++par)
          std::cout << std::setw(11) << std::setprecision(6) << std::right
                    << GetBeamParameter(det, ring, beam, par).first << " \xb1 " << std::setw(9) << std::setprecision(6)
                    << std::left << GetBeamParameter(det, ring, beam, par).second;
        std::cout << std::endl;
      }
    }
  }

  std::cout << " [LASEndcapAlignmentParameterSet::Print] -- End of list." << std::endl << std::endl;
}

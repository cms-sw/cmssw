#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#include <fstream>

void SiPixelCPEGenericErrorParm::fillCPEGenericErrorParm(double version, std::string file) {
  //--- Open the file
  std::ifstream in(file.c_str(), std::ios::in);

  //--- Currently do not need to store part of detector, but is in input file
  int part;
  set_version(version);

  DbEntry Entry;
  in >> part >> Entry.bias >> Entry.pix_height >> Entry.ave_Qclus >> Entry.sigma >> Entry.rms;

  while (!in.eof()) {
    errors_.push_back(Entry);

    in >> part >> Entry.bias >> Entry.pix_height >> Entry.ave_Qclus >> Entry.sigma >> Entry.rms;
  }
  //--- Finished parsing the file, we're done.
  in.close();

  //--- Specify the current binning sizes to use
  DbEntryBinSize ErrorsBinSize;
  //--- Part = 1 By
  ErrorsBinSize.partBin_size = 0;
  ErrorsBinSize.sizeBin_size = 40;
  ErrorsBinSize.alphaBin_size = 10;
  ErrorsBinSize.betaBin_size = 1;
  errorsBinSize_.push_back(ErrorsBinSize);
  //--- Part = 2 Bx
  ErrorsBinSize.partBin_size = 240;
  ErrorsBinSize.alphaBin_size = 1;
  ErrorsBinSize.betaBin_size = 10;
  errorsBinSize_.push_back(ErrorsBinSize);
  //--- Part = 3 Fy
  ErrorsBinSize.partBin_size = 360;
  ErrorsBinSize.alphaBin_size = 10;
  ErrorsBinSize.betaBin_size = 1;
  errorsBinSize_.push_back(ErrorsBinSize);
  //--- Part = 4 Fx
  ErrorsBinSize.partBin_size = 380;
  ErrorsBinSize.alphaBin_size = 1;
  ErrorsBinSize.betaBin_size = 10;
  errorsBinSize_.push_back(ErrorsBinSize);
}

std::ostream& operator<<(std::ostream& s, const SiPixelCPEGenericErrorParm& genericErrors) {
  for (unsigned int count = 0; count < genericErrors.errors_.size(); ++count) {
    s.precision(6);

    s << genericErrors.errors_[count].bias << " " << genericErrors.errors_[count].pix_height << " "
      << genericErrors.errors_[count].ave_Qclus << " " << std::fixed << genericErrors.errors_[count].sigma << " "
      << genericErrors.errors_[count].rms << std::endl;

    s.unsetf(std::ios_base::fixed);
  }
  return s;
}

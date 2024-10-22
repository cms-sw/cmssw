#ifndef L1Trigger_TrackFindingTracklet_interface_Util_h
#define L1Trigger_TrackFindingTracklet_interface_Util_h

#include <sstream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <string>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace trklet {

  //Converts string in binary to hex (used in writing out memory content)
  inline std::string hexFormat(const std::string& binary) {
    std::stringstream ss;

    unsigned int radix = 1, value = 0;
    for (int i = binary.length() - 1; i >= 0; i--) {
      if (binary.at(i) != '0' && binary.at(i) != '1')
        continue;
      value += (binary.at(i) - '0') * radix;
      if (radix == 8) {
        ss << std::hex << value;
        radix = 1;
        value = 0;
      } else
        radix <<= 1;
    }
    if (radix != 1)
      ss << std::hex << value;

    std::string str = ss.str() + "x0";
    std::reverse(str.begin(), str.end());
    return str;
  }

  inline double bendstrip(double r, double rinv, double stripPitch, double sensorSpacing) {
    double delta = r * sensorSpacing * 0.5 * rinv;
    double bend = delta / stripPitch;
    return bend;
  }

  inline double convertFEBend(
      double FEbend, double sensorSep, double sensorSpacing, double CF, bool barrel, double r = 0) {
    double bend = sensorSpacing * CF * FEbend / sensorSep;
    return bend;
  }

  inline double tan_theta(double r, double z, double z0, bool z0_max) {
    //Calculates tan(theta) = z_displaced/r
    //measure tan theta at different points to account for displaced tracks
    double tan;
    if (z0_max)
      tan = (z - z0) / r;
    else
      tan = (z + z0) / r;

    return tan;
  }

  inline double rinv(double phi1, double phi2, double r1, double r2) {
    assert(r1 < r2);  //Can not form tracklet should not call function with r2<=r1

    double dphi = phi2 - phi1;
    double dr = r2 - r1;

    return 2.0 * sin(dphi) / dr / sqrt(1.0 + 2 * r1 * r2 * (1.0 - cos(dphi)) / (dr * dr));
  }

  inline std::string convertHexToBin(const std::string& stubwordhex) {
    std::string stubwordbin = "";

    for (char word : stubwordhex) {
      std::string hexword = "";
      if (word == '0')
        hexword = "0000";
      else if (word == '1')
        hexword = "0001";
      else if (word == '2')
        hexword = "0010";
      else if (word == '3')
        hexword = "0011";
      else if (word == '4')
        hexword = "0100";
      else if (word == '5')
        hexword = "0101";
      else if (word == '6')
        hexword = "0110";
      else if (word == '7')
        hexword = "0111";
      else if (word == '8')
        hexword = "1000";
      else if (word == '9')
        hexword = "1001";
      else if (word == 'A')
        hexword = "1010";
      else if (word == 'B')
        hexword = "1011";
      else if (word == 'C')
        hexword = "1100";
      else if (word == 'D')
        hexword = "1101";
      else if (word == 'E')
        hexword = "1110";
      else if (word == 'F')
        hexword = "1111";
      else {
        throw cms::Exception("Inconsistency")
            << __FILE__ << " " << __LINE__ << " hex string format invalid: " << stubwordhex;
      }
      stubwordbin += hexword;
    }
    return stubwordbin;
  }

  inline int ilog2(double factor) {
    double power = log(factor) / log(2);
    int ipower = round(power);
    assert(std::abs(power - ipower) < 0.1);
    return ipower;
  }

  /******************************************************************************
 * Checks to see if a directory exists. Note: This method only checks the
 * existence of the full path AND if path leaf is a dir.
 *
 * @return   1 if dir exists AND is a dir,
 *           0 if dir does not exist OR exists but not a dir,
 *          -1 if an error occurred (errno is also set)
 *****************************************************************************/
  inline int dirExists(const std::string& path) {
    struct stat info;

    int statRC = stat(path.c_str(), &info);
    if (statRC != 0) {
      if (errno == ENOENT) {
        return 0;
      }  // something along the path does not exist
      if (errno == ENOTDIR) {
        return 0;
      }  // something in path prefix is not a dir
      return -1;
    }

    return (info.st_mode & S_IFDIR) ? 1 : 0;
  }

  //Open file - create directory if not existent.
  inline std::ofstream openfile(const std::string& dir, const std::string& fname, const char* file, int line) {
    if (dirExists(dir) != 1) {
      edm::LogVerbatim("Tracklet") << "Creating directory : " << dir;
      int fail = system((std::string("mkdir -p ") + dir).c_str());
      if (fail) {
        throw cms::Exception("BadDir") << file << " " << line << " could not create directory " << dir;
      }
    }

    std::ofstream out(dir + "/" + fname);

    if (out.fail()) {
      throw cms::Exception("BadFile") << file << " " << line << " could not create file " << fname << " in " << dir;
    }

    return out;
  }

  //Open file - create directory if not existent.
  //If first==true open file in create mode, if first==false open in append mode
  inline void openfile(
      std::ofstream& out, bool first, const std::string& dir, const std::string& fname, const char* file, int line) {
    if (dirExists(dir) != 1) {
      edm::LogVerbatim("Tracklet") << "Creating directory : " << dir;
      int fail = system((std::string("mkdir -p ") + dir).c_str());
      if (fail) {
        throw cms::Exception("BadDir") << file << " " << line << " could not create directory " << dir;
      }
    }

    if (first) {
      out.open(fname);
    } else {
      out.open(fname, std::ofstream::app);
    }

    if (out.fail()) {
      throw cms::Exception("BadFile") << file << " " << line << " could not create file " << fname << " in " << dir;
    }
  }

};  // namespace trklet
#endif

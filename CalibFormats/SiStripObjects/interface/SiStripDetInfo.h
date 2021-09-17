#ifndef CalibFormats_SiStripObjects_SiStripDetInfo_h
#define CalibFormats_SiStripObjects_SiStripDetInfo_h
// -*- C++ -*-
//
// Package:     CalibFormats/SiStripObjects
// Class  :     SiStripDetInfo
//
/**\class SiStripDetInfo SiStripDetInfo.h "SiStripDetInfo.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 28 May 2021 20:02:00 GMT
//

// system include files

// user include files
#include <map>
#include <vector>
#include <cstdint>
// forward declarations

class SiStripDetInfo {
public:
  struct DetInfo {
    DetInfo(){};
    DetInfo(unsigned short _nApvs, double _stripLength, float _thickness)
        : nApvs(_nApvs), stripLength(_stripLength), thickness(_thickness){};

    unsigned short nApvs;
    double stripLength;
    float thickness;
  };

  SiStripDetInfo(std::map<uint32_t, DetInfo> iDetData, std::vector<uint32_t> iIDs) noexcept
      : detData_{std::move(iDetData)}, detIds_{std::move(iIDs)} {}

  SiStripDetInfo() = default;
  ~SiStripDetInfo() = default;

  SiStripDetInfo(const SiStripDetInfo&) = default;
  SiStripDetInfo& operator=(const SiStripDetInfo&) = default;
  SiStripDetInfo(SiStripDetInfo&&) = default;
  SiStripDetInfo& operator=(SiStripDetInfo&&) = default;

  // ---------- const member functions ---------------------
  const std::vector<uint32_t>& getAllDetIds() const noexcept { return detIds_; }

  const std::pair<unsigned short, double> getNumberOfApvsAndStripLength(uint32_t detId) const;

  const float& getThickness(uint32_t detId) const;

  const std::map<uint32_t, DetInfo>& getAllData() const noexcept { return detData_; }

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

private:
  // ---------- member data --------------------------------
  std::map<uint32_t, DetInfo> detData_;
  std::vector<uint32_t> detIds_;
};

#endif

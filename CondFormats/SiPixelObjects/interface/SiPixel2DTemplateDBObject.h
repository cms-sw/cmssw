#ifndef CondFormats_SiPixelObjects_SiPixel2DTemplateDBObject_h
#define CondFormats_SiPixelObjects_SiPixel2DTemplateDBObject_h 1

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <map>
#include <cstdint>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// ******************************************************************************************
//! \class SiPixel2DTemplateDBObject
//!
// ******************************************************************************************

class SiPixel2DTemplateDBObject {
public:
  SiPixel2DTemplateDBObject()
      : index_(0), maxIndex_(0), numOfTempl_(1), version_(-99.9), isInvalid_(false), sVector_(0) {
    sVector_.reserve(1000000);
  }
  virtual ~SiPixel2DTemplateDBObject() {}

  //- Allows the dbobject to be read out like cout
  friend std::ostream& operator<<(std::ostream& s, const SiPixel2DTemplateDBObject& dbobject);

  //- Fills integer from dbobject
  SiPixel2DTemplateDBObject& operator>>(int& i) {
    isInvalid_ = false;
    if (index_ <= maxIndex_) {
      i = (int)(*this).sVector_[index_];
      index_++;
    } else
      (*this).setInvalid();
    return *this;
  }
  //- Fills float from dbobject
  SiPixel2DTemplateDBObject& operator>>(float& f) {
    isInvalid_ = false;
    if (index_ <= maxIndex_) {
      f = (*this).sVector_[index_];
      index_++;
    } else
      (*this).setInvalid();
    return *this;
  }

  //- Functions to monitor integrity of dbobject
  void setVersion(float version) { version_ = version; }
  void setInvalid() { isInvalid_ = true; }
  bool fail() { return isInvalid_; }

  //- Setter functions
  void push_back(float entry) { sVector_.push_back(entry); }
  void setIndex(int index) { index_ = index; }
  void setMaxIndex(int maxIndex) { maxIndex_ = maxIndex; }
  void setNumOfTempl(int numOfTempl) { numOfTempl_ = numOfTempl; }

  //- Accessor functions
  int index() const { return index_; }
  int maxIndex() const { return maxIndex_; }
  int numOfTempl() const { return numOfTempl_; }
  float version() const { return version_; }
  std::vector<float> const& sVector() const { return sVector_; }

  //- Able to set the index for template header
  void incrementIndex(int i) { index_ += i; }

  //- Allows storage of header (type = char[80]) in dbobject
  union char2float {
    char c[4];
    float f;
  };

  //- To be used to select template calibration based on detid
  void putTemplateIDs(std::map<unsigned int, short>& t_ID) { templ_ID = t_ID; }
  const std::map<unsigned int, short>& getTemplateIDs() const { return templ_ID; }

  bool putTemplateID(const uint32_t& detid, short& value) {
    std::map<unsigned int, short>::const_iterator id = templ_ID.find(detid);
    if (id != templ_ID.end()) {
      edm::LogError("SiPixel2DTemplateDBObject")
          << "2Dtemplate ID for DetID " << detid << " is already stored. Skipping this put" << std::endl;
      return false;
    } else
      templ_ID[detid] = value;
    return true;
  }

  short getTemplateID(const uint32_t& detid) const {
    std::map<unsigned int, short>::const_iterator id = templ_ID.find(detid);
    if (id != templ_ID.end())
      return id->second;
    else
      edm::LogError("SiPixel2DTemplateDBObject")
          << "2Dtemplate ID for DetID " << detid << " is not stored" << std::endl;
    return 0;
  }

  class Reader {
  public:
    Reader(SiPixel2DTemplateDBObject const& object) : object_(object), index_(0), isInvalid_(false) {}

    bool fail() { return isInvalid_; }

    int index() const { return index_; }
    //- Able to set the index for template header
    void incrementIndex(int i) { index_ += i; }

    //- Fills integer from dbobject
    Reader& operator>>(int& i) {
      isInvalid_ = false;
      if (index_ <= object_.maxIndex_) {
        i = (int)object_.sVector_[index_];
        index_++;
      } else
        isInvalid_ = true;
      return *this;
    }
    //- Fills float from dbobject
    Reader& operator>>(float& f) {
      isInvalid_ = false;
      if (index_ <= object_.maxIndex_) {
        f = object_.sVector_[index_];
        index_++;
      } else
        isInvalid_ = true;
      return *this;
    }

  private:
    SiPixel2DTemplateDBObject const& object_;
    int index_;
    bool isInvalid_;
  };
  friend class Reader;

private:
  int index_;
  int maxIndex_;
  int numOfTempl_;
  float version_;
  bool isInvalid_;
  std::vector<float> sVector_;
  std::map<unsigned int, short> templ_ID;

  COND_SERIALIZABLE;
};  //end SiPixel2DTemplateDBObject
#endif

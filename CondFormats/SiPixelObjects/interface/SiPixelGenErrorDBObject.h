#ifndef CondFormats_SiPixelObjects_SiPixelGenErrorDBObject_h
#define CondFormats_SiPixelObjects_SiPixelGenErrorDBObject_h 1

#include <vector>
#include <map>
#include <stdint.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// ******************************************************************************************
//! \class SiPixelGenErrorDBObject
//!
// ******************************************************************************************

class SiPixelGenErrorDBObject {
public:
	SiPixelGenErrorDBObject():index_(0),maxIndex_(0),numOfTempl_(1),version_(-99.9),isInvalid_(false),sVector_(0) {
		sVector_.reserve(1000000);
	}
	virtual ~SiPixelGenErrorDBObject(){}
	
	//- Allows the dbobject to be read out like cout
	friend std::ostream& operator<<(std::ostream& s, const SiPixelGenErrorDBObject& dbobject);

	//- Fills integer from dbobject
	SiPixelGenErrorDBObject& operator>>( int& i)
		{
			isInvalid_ = false;
			if(index_<=maxIndex_) {
				i = (int) (*this).sVector_[index_];
				index_++;
			}
			else
				(*this).setInvalid();
			return *this;
		}
	//- Fills float from dbobject
	SiPixelGenErrorDBObject& operator>>( float& f)
		{
			isInvalid_ = false;
			if(index_<=maxIndex_) {
				f = (*this).sVector_[index_];
				index_++;
			}
			else
				(*this).setInvalid();
			return *this;
		}

	//- Functions to monitor integrity of dbobject
	void setVersion(float version) {version_ = version;}
	void setInvalid() {isInvalid_ = true;}
	bool fail() {return isInvalid_;}

	//- Setter functions
	void push_back(float entry) {sVector_.push_back(entry);}
	void setIndex(int index) {index_ = index;}
	void setMaxIndex(int maxIndex) {maxIndex_ = maxIndex;}
	void setNumOfTempl(int numOfTempl) {numOfTempl_ = numOfTempl;}
	
	//- Accessor functions
	int index() const {return index_;}
	int maxIndex() const {return maxIndex_;}
	int numOfTempl() const {return numOfTempl_;}
	float version() const {return version_;}
	std::vector<float> sVector() const {return sVector_;}

	//- Able to set the index for GenError header 
	void incrementIndex(int i) {index_+=i;}

	//- Allows storage of header (type = char[80]) in dbobject
	union char2float
	{
		char  c[4];
		float f;
	};

	//- To be used to select GenError calibration based on detid
	void putGenErrorIDs(std::map<unsigned int,short>& t_ID) {templ_ID = t_ID;}
	const std::map<unsigned int,short>& getGenErrorIDs () const {return templ_ID;}

	bool putGenErrorID(const uint32_t& detid, short& value)
		{
			std::map<unsigned int,short>::const_iterator id=templ_ID.find(detid);
			if(id!=templ_ID.end()){
				edm::LogError("SiPixelGenErrorDBObject") << "GenError ID for DetID " << detid
																								 << " is already stored. Skipping this put" << std::endl;
				return false;
			}
			else templ_ID[detid] = value;
			return true;
		}

	short getGenErrorID(const uint32_t& detid) const
		{
			std::map<unsigned int,short>::const_iterator id=templ_ID.find(detid);
			if(id!=templ_ID.end()) return id->second;
			else edm::LogError("SiPixelGenErrorDBObject") << "GenError ID for DetID " << detid
																										<< " is not stored" << std::endl;
			return 0;
		}
	
private:
	int index_;
	int maxIndex_;
	int numOfTempl_;
	float version_;
	bool isInvalid_;
	std::vector<float> sVector_;
	std::map<unsigned int,short> templ_ID;
};//end SiPixelGenErrorDBObject
#endif

#ifndef CondFormats_SiPixelObjects_SiPixelTemplateDBObject_h
#define CondFormats_SiPixelObjects_SiPixelTemplateDBObject_h 1

#include <vector>
#include <map>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// ******************************************************************************************
//! \class SiPixelTemplateDBObject
//!
// ******************************************************************************************

class SiPixelTemplateDBObject {
public:
	SiPixelTemplateDBObject():index_(0),maxIndex_(0),numOfTempl_(1),isInvalid_(false),sVector_(0) {
		sVector_.reserve(1000000);
	}
	virtual ~SiPixelTemplateDBObject(){}
	
	typedef std::vector<std::string> vstring;
	
	void fillDB(const vstring& atitles);
	
	friend std::ostream& operator<<(std::ostream& s, const SiPixelTemplateDBObject& dbobject);

	//- Fills interger from dbobject
	SiPixelTemplateDBObject& operator>>( int& i)
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
	SiPixelTemplateDBObject& operator>>( float& f)
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
	void setInvalid() {isInvalid_ = true;}
	bool fail() {return isInvalid_;}
	
	//- Accessor functions
	int index() {return index_;}
	int numOfTempl() {return numOfTempl_;}
	std::vector<float> sVector() {return sVector_;}

	//- Able to set the index for template header 
	void incrementIndex(int i) {index_+=i;}

	//- Allows storage of header (type = char[80]) in dbobject
	union char2float
	{
		char  c[4];
		float f;
	};

	//- To be used to select template calibration based on detid
	void putTemplateIDs(std::map<unsigned int,short>& t_ID) {templ_ID = t_ID;}
	const std::map<unsigned int,short>& getTemplateIDs () const {return templ_ID;}

	bool putTemplateID(const uint32_t& detid, short& value)
		{
			std::map<unsigned int,short>::const_iterator id=templ_ID.find(detid);
			if(id!=templ_ID.end()){
				edm::LogError("SiPixelTemplateDBObject") << "Template ID for DetID " << detid << " is already stored. Skipping this put" << std::endl;
				return false;
			}
			else templ_ID[detid] = value;
			return true;
		}

	short getTemplateID(const uint32_t& detid) const
		{
			std::map<unsigned int,short>::const_iterator id=templ_ID.find(detid);
			if(id!=templ_ID.end()) return id->second;
			else edm::LogError("SiPixelTemplateDBObject") << "Template ID for DetID " << detid << " is not stored" << std::endl;
			return 0;
		}
	
				

	
private:
  int index_;
	int maxIndex_;
	int numOfTempl_;
	bool isInvalid_;
	std::vector<float> sVector_;
	std::map<unsigned int,short> templ_ID;
		};//end SiPixelTemplateDBObject
#endif

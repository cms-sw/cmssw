/*
 * JsonMonitorable.h
 *
 *  Created on: Oct 29, 2012
 *      Author: aspataru
 */

#ifndef JSON_MONITORABLE_H
#define JSON_MONITORABLE_H

#include <string>
#include <sstream>
#include <vector>
#include <memory>
//#include "EventFilter/Utilities/interface/Utils.h"


namespace jsoncollector {

enum MonType  { TYPEINT, TYPEDOUBLE, TYPESTRING, TYPEHISTOINT, TYPEHISTODOUBLE, TYPEUNDEFINED};
enum OperationType  { OPSUM, OPAVG, OPSAME, OPHISTO, OPCAT, OPUNKNOWN};

//static configuration class
class JsonMonConfig {

public:
	JsonMonConfig(MonType monType, OperationType operationType, std::string const& name, bool NAifZero, unsigned int nBins=0): 
		monType_(monType), operationType_(operationType), name_(name), NAifZero_(NAifZero),nBins_(nBins),
		sourceInfoPtr_(nullptr),definitionPtr_(nullptr)
	{
		//initialize global histogram buffer (only usable for TYPEHISTOINT and OPHISTO)
		if (nBins) binBuffer_.reset(new unsigned int[nBins]);
	}

	MonType getMonType() {return monType_;}
	void setMonType(MonType monType) {monType_ = monType;}

	OperationType getOperationType() {return operationType_;}
	void setOperationType(OperationType operationType) {operationType_ = operationType;}

	std::string const& getName() {return name_;}
	void setName(std::string const& name) {name_ = name;}

	bool getNAifZero() {return NAifZero_;}
	void setNAifZero(bool NAifZero) {NAifZero_ = NAifZero;}

	//histograms
	unsigned int getNBins() {return nBins_;}

	void setNBins(bool nBins) {
		nBins_=nBins;
		binBuffer_.reset(new unsigned int[nBins]);
	}

	std::auto_ptr<unsigned int> getBinBuffer() {return binBuffer_;}

	std::string *getSourceInfoPtr() {return sourceInfoPtr_;}
	void setSourceInfoPtr(std::string *sPtr) {sourceInfoPtr_=sPtr;}

	std::string *definitionPtr() {return definitionPtr_;}
	void setDefinitionPtr(std::string *dPtr) {definitionPtr_=dPtr;}

private:
	MonType monType_;
	OperationType operationType_;
	std::string name_;
	bool NAifZero_;
	unsigned int nBins_;
	std::auto_ptr<unsigned int> binBuffer_;//maybe separate this out to be cleaner
	std::string *sourceInfoPtr_;
	std::string *definitionPtr_;
};


class JsonMonitorable {

public:

	JsonMonitorable() : updates_(0), notSame_(false) {}

	virtual ~JsonMonitorable() {}

	virtual std::string toString() const = 0;

	virtual void resetValue() = 0;

	unsigned int getUpdates() {return updates_;}
	
	bool getNotSame() {return notSame_;}

protected:
	unsigned int updates_;
	bool notSame_;
};


class IntJ: public JsonMonitorable {

public:
	IntJ() : JsonMonitorable(), theVar_(0) {}

//	IntJ(int initVal) : theVar_(initVal), updates_(0), notSame_(0) {}

//	IntJ(const IntJ& other) {
//		theVar_ = other.value();
//		updates_ = other.getUpdates();
//		notSame_ = other.getNotSame();
//	}
	virtual ~IntJ() {}

	virtual std::string toString() const {
		std::stringstream ss;
		ss << theVar_;
		return ss.str();
	}
	virtual void resetValue() {
		theVar_=0;
		updates_=0;
		notSame_=0;
	}
	void operator=(int sth) {
		theVar_ = sth;
		updates_=1;
		notSame_=0;
	}
	int & value() {
		return theVar_;
	}
	void add(int added) {
		theVar_+=added;
		updates_++;
	}

private:
	int theVar_;
};


class DoubleJ: public JsonMonitorable {

public:
	DoubleJ() : JsonMonitorable(), theVar_(0) {}

//	DoubleJ(double initVar) {
//		theVar_ = initVar;
//		updates_ = 0;
//		notSame_ = 0;
//	}
//	DoubleJ(const DoubleJ& other) {
//		theVar_ = other.value();
//		updates_ = other.getUpdates();
//		notSame_ = other.getNotSame();
//	}
	virtual ~DoubleJ() {}

	virtual std::string toString() const {
		std::stringstream ss;
		ss << theVar_;
		return ss.str();
	}
	virtual void resetValue() {
		theVar_=0;
		updates_=0;
		notSame_=0;
	}
	void operator=(double sth) {
		theVar_ = sth;
		updates_=1;
		notSame_=0;
	}
	double & value() {
		return theVar_;
	}
	void add(double added) {
		theVar_+=added;
		updates_++;
	}

private:
	double theVar_;
};


class StringJ: public JsonMonitorable {

public:
	StringJ() :  JsonMonitorable() {}

//	StringJ(std::string initVal) {
//		theVar_ = initVal;
//		updates_ = 0;
//		notSame_ = 0;
//	}
//	StringJ(const StringJ& other) {
//		theVar_ = other.value();
//		updates_ = other.getUpdates();
//		notSame_ = other.getNotSame();
//	}
	virtual ~StringJ() {}

	virtual std::string toString() const {
		return theVar_;
	}
	virtual void resetValue() {
		theVar_=std::string();
		updates_ = 0;
		notSame_=0;
	}
	void operator=(std::string sth) {
		theVar_ = sth;
		updates_=1;
		notSame_=0;
	}
	std::string & value() {
		return theVar_;
	}
	void concatenate(std::string const& added) {
		theVar_+=added;
		updates_++;
	}

private:
	std::string theVar_;
};

//histograms filled at time intervals (later converted to full histograms or concatenated)
template<class T> class HistoJ: public JsonMonitorable {

public:
	HistoJ( int expectedUpdates = 1 , unsigned int maxUpdates = 0 ){
		expectedSize_=expectedUpdates;
		updates_ = 0;
		maxUpdates_ = maxUpdates;
		if (maxUpdates_ && maxUpdates_<expectedSize_) expectedSize_=maxUpdates_;
		histo_.reserve(expectedSize_);
	}
//	HistoJ(const HistoJ& other) {
//		histo_  = other.value();
//		expectedSize_ = other.getExpectedSize();
//		updates_ = other.getUpdates();
//		maxUpdates_ = other.getMaxUpdates();
//	}
	virtual ~HistoJ() {}

	std::string toCSV() const {
		std::stringstream ss;
		for (unsigned int i=0;i<updates_;i++) {
			ss << histo_[i];
			if (i!=histo_.size()-1) ss<<",";
		}
		return ss.str();
	}
	//this is only left for debugging
	virtual std::string toString() const {
		std::stringstream ss;
		ss << "[";
		if (histo_.size())
			for (unsigned int i = 0; i < histo_.size(); i++) {
				ss << histo_[i];
				if (i<histo_.size()-1) ss << ",";
			}
		ss << "]";
		return ss.str();
	}
	virtual void resetValue() {
		histo_.clear();
		histo_.reserve(expectedSize_);
		updates_=0;
	}
	void operator=(std::vector<T> & sth) {
		histo_ = sth;
	}

	std::vector<T> & value() {
		return histo_;
	}

	unsigned int getExpectedSize() {
		return expectedSize_;
	}

	unsigned int getMaxUpdates() {
		return maxUpdates_;
	}

	void setMaxUpdates(unsigned int maxUpdates) {
		maxUpdates_=maxUpdates;
		if (!maxUpdates_) return;
		if (expectedSize_>maxUpdates_) expectedSize_=maxUpdates_;
		//truncate what is over the limit
		if (maxUpdates_ && histo_.size()>maxUpdates_) {
			histo_().resize(maxUpdates_);
		}
		else histo_.reserve(expectedSize_);
	}

	unsigned int getSize() {
		return histo_.size();
	}

	void update(T val) {
		if (maxUpdates_ && updates_>=maxUpdates_) return;
		histo_.push_back(val);
		updates_++;
	}

private:
	std::vector<T> histo_;
	unsigned int expectedSize_;
	unsigned int maxUpdates_;
};


}

#endif

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

using std::string;
using std::stringstream;

namespace jsoncollector {

class JsonMonitorable {

public:
	virtual ~JsonMonitorable() {
	}
	;
	virtual string toString() const = 0;
	string& getName() {
		return varName_;
	}
	void setName(string name) {
		varName_ = name;
	}

private:
	string varName_;

};

class IntJ: public JsonMonitorable {

public:
	IntJ() {
		theVar_ = 0;
	}
	IntJ(int initVal) {
		theVar_ = initVal;
	}
	IntJ(const IntJ& other) {
		theVar_ = other.theVar_;
	}
	virtual ~IntJ() {
	}

	virtual string toString() const {
		stringstream ss;
		ss << theVar_;
		return ss.str();
	}

	void operator=(int sth) {
		theVar_ = sth;
	}

	int& value() {
		return theVar_;
	}

private:
	int theVar_;
};

class StringJ: public JsonMonitorable {

public:
	StringJ() {
		theVar_ = "";
	}
	StringJ(string initVal) {
		theVar_ = initVal;
	}
	StringJ(const StringJ& other) {
		theVar_ = other.theVar_;
	}
	virtual ~StringJ() {
	}

	virtual string toString() const {
		return theVar_;
	}

	void operator=(string sth) {
		theVar_ = sth;
	}

	string& value() {
		return theVar_;
	}

private:
	string theVar_;
};

class DoubleJ: public JsonMonitorable {

public:
	DoubleJ() {
		theVar_ = 0.0;
	}
	DoubleJ(double initVar) {
		theVar_ = initVar;
	}
	DoubleJ(const DoubleJ& other) {
		theVar_ = other.theVar_;
	}
	virtual ~DoubleJ() {
	}

	virtual string toString() const {
		stringstream ss;
		ss << theVar_;
		return ss.str();
	}

	void operator=(double sth) {
		theVar_ = sth;
	}

	double& value() {
		return theVar_;
	}

private:
	double theVar_;
};
}

#endif

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
#include <iostream>
#include "json.h"

namespace jsoncollector {

  enum MonType { TYPEINT, TYPEUINT, TYPEDOUBLE, TYPESTRING, TYPEUNDEFINED };
  enum OperationType { OPSUM, OPAVG, OPSAME, OPHISTO, OPCAT, OPUNKNOWN };

  class JsonMonitorable {
  public:
    JsonMonitorable() : updates_(0), notSame_(false) {}

    virtual ~JsonMonitorable() {}

    virtual std::string toString() const = 0;

    virtual void resetValue() = 0;

    unsigned int getUpdates() { return updates_; }

    bool getNotSame() const { return notSame_; }

    virtual void setName(std::string name) { name_ = name; }

    virtual std::string const& getName() const { return name_; }

  protected:
    std::string name_;
    unsigned int updates_;
    bool notSame_;
  };

  class JsonMonPtr {
  public:
    JsonMonPtr() : ptr_(nullptr) {}
    JsonMonPtr(JsonMonitorable* ptr) : ptr_(ptr) {}
    void operator=(JsonMonitorable* ptr) { ptr_ = ptr; }
    ~JsonMonPtr() {
      if (ptr_)
        delete ptr_;
      ptr_ = nullptr;
    }
    JsonMonitorable* operator->() { return ptr_; }
    JsonMonitorable* get() { return ptr_; }
    //JsonMonPtr& operator=(JsonMonPtr& ) = delete;
    //JsonMonPtr& operator=(JsonMonPtr&& other){ptr_=other.ptr_;return *this;}
  private:
    JsonMonitorable* ptr_;
  };

  class IntJ : public JsonMonitorable {
  public:
    IntJ() : JsonMonitorable(), theVar_(0) {}
    IntJ(long val) : JsonMonitorable(), theVar_(val) {}

    ~IntJ() override {}

    std::string toString() const override {
      std::stringstream ss;
      ss << theVar_;
      return ss.str();
    }
    void resetValue() override {
      theVar_ = 0;
      updates_ = 0;
      notSame_ = false;
    }
    void operator=(long sth) {
      theVar_ = sth;
      updates_ = 1;
      notSame_ = false;
    }
    long& value() { return theVar_; }
    long value() const { return theVar_; }

    void update(long sth) {
      theVar_ = sth;
      if (updates_ && theVar_ != sth)
        notSame_ = true;
      updates_++;
    }

    void add(long sth) {
      theVar_ += sth;
      updates_++;
    }

  private:
    long theVar_;
  };

  class DoubleJ : public JsonMonitorable {
  public:
    DoubleJ() : JsonMonitorable(), theVar_(0) {}
    DoubleJ(double val) : JsonMonitorable(), theVar_(val) {}

    ~DoubleJ() override {}

    std::string toString() const override {
      std::stringstream ss;
      ss << theVar_;
      return ss.str();
    }
    void resetValue() override {
      theVar_ = 0;
      updates_ = 0;
      notSame_ = false;
    }
    void operator=(double sth) {
      theVar_ = sth;
      updates_ = 1;
      notSame_ = false;
    }
    double& value() { return theVar_; }
    double value() const { return theVar_; }
    void update(double sth) {
      theVar_ = sth;
      if (updates_ && theVar_ != sth)
        notSame_ = true;
      updates_++;
    }

  private:
    double theVar_;
  };

  class StringJ : public JsonMonitorable {
  public:
    StringJ() : JsonMonitorable() {}
    StringJ(StringJ const& sJ) : JsonMonitorable() { theVar_ = sJ.value(); }

    ~StringJ() override {}

    std::string toString() const override { return theVar_; }
    void resetValue() override {
      theVar_ = std::string();
      updates_ = 0;
      notSame_ = false;
    }
    void operator=(std::string sth) {
      theVar_ = sth;
      updates_ = 1;
      notSame_ = false;
    }
    std::string& value() { return theVar_; }
    std::string const& value() const { return theVar_; }
    void concatenate(std::string const& added) {
      if (!updates_)
        theVar_ = added;
      else
        theVar_ += "," + added;
      updates_++;
    }
    void update(std::string const& newStr) {
      theVar_ = newStr;
      updates_ = 1;
    }

  private:
    std::string theVar_;
  };

  //histograms filled at time intervals (later converted to full histograms)
  template <class T>
  class HistoJ : public JsonMonitorable {
  public:
    HistoJ(int expectedUpdates = 1, unsigned int maxUpdates = 0) {
      expectedSize_ = expectedUpdates;
      updates_ = 0;
      maxUpdates_ = maxUpdates;
      if (maxUpdates_ && maxUpdates_ < expectedSize_)
        expectedSize_ = maxUpdates_;
      histo_.reserve(expectedSize_);
    }
    ~HistoJ() override {}

    std::string toCSV() const {
      std::stringstream ss;
      for (unsigned int i = 0; i < updates_; i++) {
        ss << histo_[i];
        if (i != histo_.size() - 1)
          ss << ",";
      }
      return ss.str();
    }

    std::string toString() const override {
      std::stringstream ss;
      ss << "[";
      if (!histo_.empty())
        for (unsigned int i = 0; i < histo_.size(); i++) {
          ss << histo_[i];
          if (i < histo_.size() - 1)
            ss << ",";
        }
      ss << "]";
      return ss.str();
    }
    virtual Json::Value toJsonValue() const {  //TODO
      Json::Value jsonValue(Json::arrayValue);
      for (unsigned int i = 0; i < histo_.size(); i++) {
        jsonValue.append(histo_[i]);
      }
      return jsonValue;
    }
    void resetValue() override {
      histo_.clear();
      histo_.reserve(expectedSize_);
      updates_ = 0;
    }
    void operator=(std::vector<T> const& sth) { histo_ = sth; }

    std::vector<T>& value() { return histo_; }
    std::vector<T> const& value() const { return histo_; }

    unsigned int getExpectedSize() const { return expectedSize_; }

    unsigned int getMaxUpdates() const { return maxUpdates_; }

    void setMaxUpdates(unsigned int maxUpdates) {
      maxUpdates_ = maxUpdates;
      if (!maxUpdates_)
        return;
      if (expectedSize_ > maxUpdates_)
        expectedSize_ = maxUpdates_;
      //truncate what is over the limit
      if (maxUpdates_ && histo_.size() > maxUpdates_) {
        histo_.resize(maxUpdates_);
      } else
        histo_.reserve(expectedSize_);
    }

    unsigned int getSize() const { return histo_.size(); }

    void update(T val) {
      if (maxUpdates_ && updates_ >= maxUpdates_)
        return;
      histo_.push_back(val);
      updates_++;
    }

  private:
    std::vector<T> histo_;
    unsigned int expectedSize_;
    unsigned int maxUpdates_;
  };

}  // namespace jsoncollector

#endif

// -*-c++-*-
// $Id$
// $Log$

#ifndef L1TCONFIGPARSER_H
#define L1TCONFIGPARSER_H

#include <string>

#include "DQMServices/ClientConfig/interface/DQMParserBase.h"

class L1TClientConfigParser: public DQMParserBase
{
public:  
  virtual ~L1TClientConfigParser();
  L1TClientConfigParser(std::string configFile = std::string("config.xml"));
  bool parse(std::string NewConfigFile);

  // helper class
  class MeInfo {
  private:
    std::string name_, options_;
  public:
    MeInfo(std::string name, std::string options = std::string("")):
      name_(name), options_(options)
    {
    }
    const char *getName() const
    {
      return name_.c_str();
    }
    const char *getOptions() const
    {
      return options_.c_str();
    }
    virtual ~MeInfo() {};
  };
public:
  //typedef std::vector<L1TClientConfigParser::MeInfo> MeInfoList;
  class MeInfoList {
  private:
    std::vector<MeInfo> list_;
    std::string name_;
  public:
    void clear() {
      list_.clear();
    }
    void push_back(const L1TClientConfigParser::MeInfo & l ) {
      list_.push_back(l);
    }
    typedef 
    std::vector<L1TClientConfigParser::MeInfo>::const_iterator const_iterator;
    const_iterator begin() const {
      return list_.begin();
    }
    const_iterator end() const {
      return list_.end();
    }
  public:
    void setTitle(std::string title) {
      name_ = std::string(title);
    }
    const char* title() const {
      return name_.c_str();
    }

    
  };
  typedef std::vector<L1TClientConfigParser::MeInfoList> MeInfoLists;
  
  MeInfoLists::const_iterator begin() {
    return summaryList_.begin();
  }
  MeInfoLists::const_iterator end() {
    return summaryList_.end();
  }

private:
  std::string fname_;
  MeInfoLists summaryList_;

};

std::ostream & operator << (std::ostream & out,
			    const L1TClientConfigParser::MeInfo &mei );



#endif //  L1TCONFIGPARSER_H

//#include <iostream>
#include "CondTools/Utilities/interface/CSVDataLineParser.h"
#include <boost/spirit/core.hpp>
#include <boost/spirit/utility/confix.hpp>
#include <boost/spirit/utility/escape_char.hpp>
using namespace boost::spirit;

struct generic_actor{
  std::vector<boost::any>& vec_;
  generic_actor(std::vector<boost::any>& vec ):vec_(vec){}
  template <typename ActionIteratorT>
  void operator ()(ActionIteratorT const& first, 
		   ActionIteratorT const& last) const{
    vec_.push_back(std::string(first, last-first));
  }
  void operator ()(int val) const{
    vec_.push_back(val);
  }
  void operator ()(double val) const{
    vec_.push_back(val);
  }
};

bool CSVDataLineParser::parse( const std::string& inputLine){
  boost::spirit::rule<> item_parser, list_parser,strparser,numparser;
  strparser=confix_p('\"',*c_escape_ch_p, '\"')[generic_actor(m_result)];
  numparser=(strict_real_p)[generic_actor(m_result)] | int_p[generic_actor(m_result)];
  list_parser=(numparser|strparser) >>*(ch_p(',')>>(numparser|strparser));
  parse_info<> result=boost::spirit::parse(inputLine.c_str(),list_parser);
  if(result.full){
    return true;
  }
  return false;
}

std::vector<boost::any> CSVDataLineParser::result() const{
  return m_result;
}

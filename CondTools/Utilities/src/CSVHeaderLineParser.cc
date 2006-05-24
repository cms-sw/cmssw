#include "CondTools/Utilities/interface/CSVHeaderLineParser.h"
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <boost/spirit/utility/lists.hpp>
using namespace boost::spirit;

bool CSVHeaderLineParser::parse( const std::string& inputLine){
  boost::spirit::rule<> list_parser;
  list_parser=list_p((*print_p)[push_back_a(m_result)],',');
  parse_info<> result=boost::spirit::parse(inputLine.c_str(),list_parser);
  if(result.full){
    return true;
  }
  return false;
}

std::vector<std::string> CSVHeaderLineParser::result() const{
  return m_result;
}

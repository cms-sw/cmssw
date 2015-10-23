#include "CondCore/Utilities/interface/JsonPrinter.h"
//
#include <sstream>

namespace cond {

  namespace utilities {
    
    JsonPrinter::JsonPrinter():m_values(){}
    
    JsonPrinter::JsonPrinter( const std::string& xName, const std::string& yName ):
      m_xName(xName),
      m_yName(yName),
      m_values(){
    }

    void JsonPrinter::append( const std::string& xValue, const std::string& yValue, const std::string& yError ){
      m_values.push_back( std::make_tuple( xValue, yValue, yError ) );
    }
    
    void JsonPrinter::append( const std::string& xValue, const std::string& yValue ){
      m_values.push_back( std::make_tuple( xValue, yValue, "0" ) );
    }

    std::string JsonPrinter::print(){
      std::stringstream ss;
      ss<<" { [ ";
      bool first = true;
      for( auto iv : m_values ){
	if(!first) ss << ",";
        ss <<" { ";
	ss <<" \""<<m_xName<<"\": "<<std::get<0>(iv)<<",";
	ss <<" \""<<m_yName<<"\": "<<std::get<1>(iv)<<",";
	ss <<" \""<<m_yName<<"_Error\": "<<std::get<2>(iv);
        ss <<" } ";
	first = false;
      }
      ss<<"] }";
      return ss.str();
    }

  }

}


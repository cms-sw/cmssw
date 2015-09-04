#ifndef CondCore_Utilities_JsonPrinter_h
#define CondCore_Utilities_JsonPrinter_h

#include <string>
#include <tuple>
#include <vector>

namespace cond {

  namespace utilities {
    
    class JsonPrinter {
    public:
      JsonPrinter();
      JsonPrinter( const std::string& xName, const std::string& yName );

      virtual ~JsonPrinter(){}

      void append( const std::string& xValue, const std::string& yValue, const std::string& yError );
      void append( const std::string& xValue, const std::string& yValue );

      std::string print();

    private:
      std::string m_xName = "X";
      std::string m_yName = "Y";
      std::vector<std::tuple<std::string,std::string,std::string> > m_values;
    };
  }

}

#endif


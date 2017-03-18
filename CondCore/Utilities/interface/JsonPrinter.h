#ifndef CondCore_Utilities_JsonPrinter_h
#define CondCore_Utilities_JsonPrinter_h

#include <iostream>

#include <string>
#include <tuple>
#include <vector>
#include <boost/python/list.hpp>
namespace cond {

  namespace utilities {

    class Plot {
    public:
      Plot(){}
      Plot( const std::string& title, const std::string& type ):
	m_title(title),m_type(type){
      }
      // return the type-name of the objects we handle, so the PayloadInspector can find corresponding tags
      virtual std::string objectType(){ return ""; }

      // return a title string to be used in the PayloadInspector
      std::string title(){
	return m_title;
      }

      std::string type(){
	return m_type;
      }

      //
      std::string info(){
	return m_info;
      }

      std::string data(){
	return m_data;
      }

      bool process( const std::string&, const boost::python::list& iovs ){
	// some action... +
	return processIovs( iovs );
      }

      virtual bool processIovs( const boost::python::list& iovs ){ 
	std::cout <<"default..."<<std::endl;
	return true; 
      }
    
      std::string m_title = "";

      std::string m_type = "";

      std::string m_info = "";

      std::string m_data = "";
    };

    class Plot1D : public Plot {
    public:
      Plot1D( const std::string& title ) : Plot( title, "Plot1D" ){
      }
    };
    
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


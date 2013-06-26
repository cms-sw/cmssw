/**
   This parser is really fragile!!
 */


#ifndef trivialParser_h
#define trivialParser_h

#include <string>
#include <map>
#include <fstream>

class trivialParser
{
  public :

    //!ctor
    explicit trivialParser (std::string configFile) ;
    //! return the value for that parameter
    double getVal (std::string name) ;

  private :
  
    //! container for the output
    std::map <std::string,double> m_config ;

  private :

    //! parse the cfg file
    void parse (std::string configFile) ;
    //! print the read params
    void print (std::string prefix = "") ;
    //! returns the next not commented line
    std::string getNextLine (std::ifstream & input) ;
    //! get rid of spaces
    void eraseSpaces (std::string & word) ;

} ;

#endif

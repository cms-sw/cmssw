#include "CondCore/ORA/interface/PoolToken.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Guid.h"
//
#include <cstring>
#include <cstdio>
// externals
#include "Reflex/Reflex.h"

namespace cond {
  
  static const char* fmt_tech = "[TECH=%08X]";
  static const char* fmt_oid  = "[OID=%08X-%08X]";
    
  std::pair<std::string,int> parseToken( const std::string& source ){
    if( source.empty() ) ora::throwException("Provided token is empty.","PoolToken::parseToken");
    std::string tmp = source;
    std::pair<std::string,int> oid;
    oid.first = "";
    oid.second = -1;
    for(char* p1 = (char*)tmp.c_str(); p1; p1 = ::strchr(++p1,'[')) {
      char* p2 = ::strchr(p1, '=');
      char* p3 = ::strchr(p1, ']');
      if ( p2 && p3 )   {
        char* val = p2+1;
        if ( ::strncmp("[DB=", p1, 4) == 0 )  {
          *p3 = 0;
        } else if ( ::strncmp("[CNT=", p1, 5) == 0 )  {
          *p3 = 0;
          oid.first = val;
        } else if ( ::strncmp(fmt_oid, p1, 5) == 0 )  {
          int nn;
          ::sscanf(p1, fmt_oid, &nn, &oid.second);
        } else    {
          *p3 = *p2 = 0;
        }
        *p3 = ']';
        *p2 = '=';
      }
    }
    return oid;
  }

  std::string writeToken( const std::string& containerName,
                          int oid0,
                          int oid1,
                          const std::string& className ){
    std::string str = writeTokenContainerFragment( containerName, className );
    char text[128];
    ::sprintf(text, fmt_oid, oid0, oid1);
    str += text;
    return str;
  }

  std::string writeTokenContainerFragment( const std::string& containerName, 
                                           const std::string& className ){
    
    char buff[20];
    std::string clguid("");
    //  first lookup the class guid in the dictionary
    Reflex::Type containerType = Reflex::Type::ByName( className );
    if( containerType ){
      Reflex::PropertyList props = containerType.Properties();
      if( props.HasProperty("ClassID")){
        clguid = props.PropertyAsString("ClassID");
      }
    }
    // if not found, generate one...
    if( clguid.empty() ){
      genMD5(className,buff);
      Guid* gd = reinterpret_cast<Guid*>(buff);
      clguid = gd->toString();
    }
    int tech = 0xB01;
    char text[128];
    std::string str = "[DB="+Guid::null()+"][CNT=" + containerName + "][CLID="+clguid+"]";
    ::sprintf(text, fmt_tech, tech);
    str += text;
    return str;
  }


}



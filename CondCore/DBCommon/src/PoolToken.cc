#include "CondCore/DBCommon/interface/PoolToken.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "Guid.h"
//
#include <cstring>
#include <cstdio>

namespace cond {
  
  static const char* fmt_tech = "[TECH=%08X]";
  static const char* fmt_oid  = "[OID=%08X-%08X]";
  //static const char* fmt_Guid = "%08lX-%04hX-%04hX-%02hhX%02hhX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX";
  static const char* guid_null = "00000000-0000-0000-0000-000000000000";
  
  void genMD5(const std::string& s, void* code);
  
  std::pair<std::string,int> parseToken( const std::string& source ){
    if( source.empty() ) throw cond::Exception("PoolToken::parseToken: Provided token is empty.");
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
    char buff[20];
    genMD5(className,buff);
    std::string clguid = ((Guid*)buff)->toString();
    int tech = 0xB01;
    char text[128];
    std::string str = "[DB="+std::string(guid_null)+"][CNT=" + containerName + "][CLID="+clguid+"]";
    ::sprintf(text, fmt_tech, tech);
    str += text;
    ::sprintf(text, fmt_oid, oid0, oid1);
    str += text;
    return str;
  }

}



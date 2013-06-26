#ifndef _CommonTools_Utils_StringToEnumValue_h_
#define _CommonTools_Utils_StringToEnumValue_h_


#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include <string>
#include <sstream>
#include <vector>


/**

   Convert a string into the enum value it corresponds to. Example:
   <code>
   int value = StringToEnumValue<EcalRecHit::Flags>("kGood");
   </code>

   \author Stefano Argiro
   \version $Id: StringToEnumValue.h,v 1.6 2013/01/25 23:27:41 wmtan Exp $
   \date 04 Mar 2011
*/

template <class MyType> 
int StringToEnumValue(const std::string & enumMemberName){
  edm::TypeWithDict dataType(typeid(MyType), kIsEnum);
  return dataType.stringToEnumValue(enumMemberName);
}


/**

   Convert a vector<string> into a vector<int> with the enum values 
   it corresponds to. Example:

   <code>
   std::vector<std::string> names;
   names.push_back("kWeird");
   names.push_back("kGood");
   
   std::vector<int> ints =  StringToEnumValue<EcalRecHit::Flags>(names);
   

   std::copy(ints.begin(), ints.end(), 
             std::ostream_iterator<int>(std::cout, "-"));
   </code>


*/


template <class MyType> 
std::vector<int> StringToEnumValue(const std::vector<std::string> & enumNames){
  
  using std::vector;
  using std::string;

  vector<int> ret;
  vector<string>::const_iterator str=enumNames.begin();
  for (;str!=enumNames.end();++str){
    ret.push_back( StringToEnumValue<MyType>(*str));
  }
  return ret;

} // 

#endif

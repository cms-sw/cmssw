#ifndef _CommonTools_Utils_StringToEnumValue_h_
#define _CommonTools_Utils_StringToEnumValue_h_


#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
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
   \version $Id: StringToEnumValue.h,v 1.3 2012/08/03 18:08:09 wmtan Exp $
   \date 04 Mar 2011
*/

template <class MyType> 
int StringToEnumValue(const std::string & enumName){
  
  edm::TypeWithDict dataType(typeid(MyType));
  edm::MemberWithDict member = dataType.memberByName(enumName);

  if (!member) {
    std::ostringstream err;
    err<<"StringToEnumValue Failure trying to convert " << enumName << " to int value";
    throw cms::Exception("ConversionError",err.str());
  }

  if (member.typeOf().typeInfo() != typeid(int)) {
    
    std::ostringstream err;
    err << "Type "<<  member.typeOf().name() << " is not Enum";
    throw cms::Exception("ConversionError",err.str());
  }
  return member.get().objectCast<int>();

} // 


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

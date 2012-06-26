#ifndef _CommonTools_Utils_StringToEnumValue_h_
#define _CommonTools_Utils_StringToEnumValue_h_


#include "FWCore/Utilities/interface/Exception.h"
#include <Reflex/Reflex.h>
#include <string>
#include <sstream>
#include <vector>


/**

   Convert a string into the enum value it corresponds to. Example:
   <code>
   int value = StringToEnumValue<EcalRecHit::Flags>("kGood");
   </code>

   \author Stefano Argiro
   \version $Id: StringToEnumValue.h,v 1.1 2011/03/07 14:49:48 gpetrucc Exp $
   \date 04 Mar 2011
*/

template <class MyType> 
int StringToEnumValue(const std::string & enumName){
  
  Reflex::Type rflxType = Reflex::Type::ByTypeInfo(typeid(MyType));
  Reflex::Member member = rflxType.MemberByName(enumName);

  if (!member) {
    std::ostringstream err;
    err<<"StringToEnumValue Failure trying to convert " << enumName << " to int value";
    throw cms::Exception("ConversionError",err.str());
  }

  if (member.TypeOf().TypeInfo() != typeid(int)) {
    
    std::ostringstream err;
    err << "Type "<<  member.TypeOf().Name() << " is not Enum";
    throw cms::Exception("ConversionError",err.str());
  }
  return Reflex::Object_Cast<int>(member.Get());

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

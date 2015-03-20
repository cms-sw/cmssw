#ifndef _CommonTools_Utils_StringToEnumValue_h_
#define _CommonTools_Utils_StringToEnumValue_h_


#include "FWCore/Utilities/interface/Exception.h"
#include "TEnum.h"
#include "TEnumConstant.h"
#include <cassert>
#include <string>
#include <sstream>
#include <vector>


/**

   Convert a string into the enum value it corresponds to. Example:
   <code>
   int value = StringToEnumValue<EcalRecHit::Flags>("kGood");
   </code>

   \author Stefano Argiro
   \version $Id: StringToEnumValue.h,v 1.5 2012/08/28 22:28:38 wmtan Exp $
   \date 04 Mar 2011
*/

template <typename MyEnum>
int StringToEnumValue(std::string const& enumConstName){
   TEnum* en = TEnum::GetEnum(typeid(MyEnum));
   if (en != nullptr){
      if (TEnumConstant const* enc = en->GetConstant(enumConstName.c_str())){
         return enc->GetValue();
      }
   }
   assert(0);
   return -1;
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

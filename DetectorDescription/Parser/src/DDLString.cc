#include "DetectorDescription/Parser/src/DDLString.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDString.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include <map>
#include <utility>

class DDCompactView;
class DDLElementRegistry;

DDLString::DDLString( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}
 
void
DDLString::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{}

void
DDLString::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  if (parent() == "ConstantsSection" || parent() == "DDDefinition")
  {
    // I do not like "newing" things without my control.  But this is
    // the only way I was able to get this to work.

    std::string * ts = new std::string((getAttributeSet().find("value"))->second);
    DDName ddn = getDDName(nmspace);
    DDString( ddn, ts );
    clear();
  }
}


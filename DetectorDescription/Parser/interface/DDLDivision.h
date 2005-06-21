#ifndef DDL_Division_H
#define DDL_Division_H

#include "DetectorDescription/DDParser/interface/DDXMLElement.h"
#include "DetectorDescription/DDParser/interface/DDDividedGeometryObject.h"
#include "DetectorDescription/DDCore/interface/DDDivision.h"

#include <string>
#include <map>

/// DDLDivision processes Division elements.
/** @class DDLDivision
 * @author Michael Case
 *
 *  DDLDivision.h  -  description
 *  -------------------
 *  begin: Friday, April 23, 2004
 *  email: case@ucdhep.ucdavis.edu
 *
 *
 */

class DDLDivision : public DDXMLElement
{
 public:

  /// Constructor
  DDLDivision();

  /// Destructor
  ~DDLDivision();

  void preProcessElement (const std::string& name, const std::string& nmspace); 

  void processElement (const std::string& name, const std::string& nmspace); 

 private:

  DDDividedGeometryObject* makeDivider(const DDDivision & div);
};

#endif


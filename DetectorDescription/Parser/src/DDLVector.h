#ifndef DDL_Vector_H
#define DDL_Vector_H

#include <map>
#include <string>
#include <vector>

#include "DDXMLElement.h"
#include "DetectorDescription/Core/interface/DDReadMapType.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDVector.h"

class DDCompactView;
class DDLElementRegistry;

///  DDLVector handles Rotation and ReflectionRotation elements.
/** @class DDLVector
 * @author Michael Case
 *
 *  DDLVector.h  -  description
 *  -------------------
 *  begin: Fri Nov 21 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 *
 *  This is the Vector container
 *
 */
class DDLVector final : public DDXMLElement
{

 public:

  DDLVector( DDLElementRegistry* myreg );

  void preProcessElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override;
  void processElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override;
  void clearall();

  ReadMapType<std::vector<double> >  & getMapOfVectors();
  ReadMapType<std::vector<std::string> >  & getMapOfStrVectors();

 private:
  std::vector<double> pVector;
  std::vector<std::string> pStrVector;
  ReadMapType< std::vector<double> > pVecMap;
  ReadMapType< std::vector<std::string> > pStrVecMap;
  std::string pNameSpace;
  void errorOut(const char* str) const;
  void do_makeDouble(char const* str, char const* end);
  void do_makeString(char const* str, char const* end);
  bool parse_numbers(char const* str);
  bool parse_strings(char const* str);
};

#endif

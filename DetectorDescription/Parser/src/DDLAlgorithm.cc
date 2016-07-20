#include "DetectorDescription/Parser/src/DDLAlgorithm.h"

#include <stddef.h>
#include <map>
#include <utility>

#include "DetectorDescription/Algorithm/interface/DDAlgorithmHandler.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/ExprAlgo/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLMap.h"
#include "DetectorDescription/Parser/src/DDLVector.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLAlgorithm::DDLAlgorithm( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

void
DDLAlgorithm::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  myRegistry_->getElement( "Vector" )->clear();
}

void
DDLAlgorithm::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DDXMLElement* myNumeric        = myRegistry_->getElement( "Numeric" );
  DDXMLElement* myString         = myRegistry_->getElement( "String" );
  DDXMLElement* myVector         = myRegistry_->getElement( "Vector" );
  DDXMLElement* myMap            = myRegistry_->getElement( "Map" );
  DDXMLElement* myrParent        = myRegistry_->getElement( "rParent" );

  DDName algoName( getDDName( nmspace ));  
  DDLogicalPart lp( DDName( myrParent->getDDName( nmspace )));
  DDXMLAttribute atts;

  // handle all Numeric elements in the Algorithm.
  DDNumericArguments nArgs;
  size_t i = 0;
  for( ; i < myNumeric->size(); ++i )
  {
    atts = myNumeric->getAttributeSet( i );
    nArgs[atts.find( "name" )->second] = myRegistry_->evaluator().eval( nmspace, atts.find( "value" )->second );
  }

  DDStringArguments sArgs;
  for( i = 0; i < myString->size(); ++i )
  {
    atts = myString->getAttributeSet( i );
    sArgs[atts.find( "name" )->second] = atts.find( "value" )->second;
  }

  DDAlgorithmHandler handler;
  atts = getAttributeSet();
  DDLVector* tv = dynamic_cast<DDLVector*>( myVector );
  DDLMap* tm = dynamic_cast<DDLMap*>( myMap );
  handler.initialize( algoName, lp, nArgs, tv->getMapOfVectors(), tm->getMapOfMaps(), sArgs, tv->getMapOfStrVectors());
  handler.execute( cpv );

  // clear used/referred to elements.
  myString->clear();
  myNumeric->clear();
  myVector->clear();
  myMap->clear();
  myrParent->clear();
  clear();
}


#include "TableNameRationalizer.h"
#include "MappingElement.h"
#include "MappingRules.h"
//
#include <sstream>
// externals
#include "RelationalAccess/ISchema.h"

ora::TableNameRationalizer::TableNameRationalizer( coral::ISchema& schema ):
  m_schema( schema ),
  m_maximumSizeForTableName( ora::TableNameRationalizer::defaultMaximumSizeForTableName ),
  m_validTableNames(),
  m_suggestedToValidName()
{}

ora::TableNameRationalizer::TableNameRationalizer( coral::ISchema& schema,
                                                   unsigned int maximumSizeForTableName ):
  m_schema( schema ),
  m_maximumSizeForTableName( maximumSizeForTableName ),
  m_validTableNames(),
  m_suggestedToValidName()
{}

ora::TableNameRationalizer::~TableNameRationalizer()
{}

std::string
ora::TableNameRationalizer::validNameForNewTable( const std::string& suggestion )
{
  // Check if this is an already validated name
  std::map< std::string, std::string >::const_iterator iValidName = m_suggestedToValidName.find( suggestion );
  if ( iValidName != m_suggestedToValidName.end() ) {
    return iValidName->second;
  }

  // Check name as is first
  if (  suggestion.size() <= m_maximumSizeForTableName &&
        ! m_schema.existsTable( suggestion ) &&
        ! m_schema.existsView( suggestion ) ) {
    m_suggestedToValidName.insert( std::make_pair( suggestion, suggestion ) );
    m_validTableNames.insert( suggestion );
    return suggestion;
  }

  unsigned int i=0;
  std::string candidateName( suggestion );
  while( m_schema.existsTable( candidateName ) || m_schema.existsView( candidateName ) || (m_validTableNames.find( candidateName ) != m_validTableNames.end())){
    candidateName = ora::MappingRules::newNameForSchemaObject( suggestion, i, ora::MappingRules::MaxTableNameLength );
    i++;
  }
  m_suggestedToValidName.insert( std::make_pair( suggestion, candidateName ) );
  m_validTableNames.insert( candidateName );
  return candidateName;
}

void
ora::TableNameRationalizer::rationalizeMappingElement( ora::MappingElement& element )
{
  element.alterTableName( this->validNameForNewTable( element.tableName() ) );
  for ( ora::MappingElement::iterator iElement = element.begin();
        iElement != element.end(); ++iElement ) {
    this->rationalizeMappingElement( iElement->second );
  }
}

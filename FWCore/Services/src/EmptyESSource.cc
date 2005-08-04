#include <set>
#include <algorithm>
#include <sstream>

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Services/src/EmptyESSource.h"

namespace edm {

EmptyESSource::EmptyESSource(const edm::ParameterSet & pset) :
   recordName_(pset.getParameter<std::string>("recordname"))
{
   std::vector<unsigned int> temp( pset.getParameter< std::vector<unsigned int> >("firstvalid") );
   for( std::vector<unsigned int>::iterator itValue = temp.begin();
        itValue != temp.end();
        ++itValue ) {
      setOfIOV_.insert( IOVSyncValue( *itValue) );
   }
   //std::copy( temp.begin(), temp.end(), inserter(setOfIOV_ , setOfIOV_.end()));
   
   
   eventsetup::EventSetupRecordKey recordKey = eventsetup::EventSetupRecordKey::TypeTag::findType( recordName_ );
   if ( recordKey == edm::eventsetup::EventSetupRecordKey() ) {
      std::ostringstream errorMessage;
      errorMessage<<" The Record type named \""<<recordName_<<"\" could not be found. Please check the spelling. \n"
         <<"If the spelling is fine, try to move the declaration of the TestingIntevalFinder to the end of the configuration file.";
      throw std::runtime_error( errorMessage.str().c_str() );
   }
   findingRecordWithKey( recordKey );
}
  
void 
EmptyESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue& iTime, 
                               edm::ValidityInterval& oInterval ) {
   oInterval = edm::ValidityInterval::invalidInterval();
   //if no intervals given, fail immediately
   if ( setOfIOV_.size() == 0 ) {
      return;
   }
   
   std::pair< std::set<edm::IOVSyncValue>::iterator, 
      std::set<edm::IOVSyncValue>::iterator > itFound = setOfIOV_.equal_range( iTime );
   
   if ( itFound.first == setOfIOV_.end() ) {
      return;
   }
   
   edm::IOVSyncValue endOfInterval = edm::IOVSyncValue::endOfTime();
   
   if( itFound.second != setOfIOV_.end() ) { 
      endOfInterval = edm::IOVSyncValue( (itFound.second->collisionID()) -1 );
   }
   oInterval = edm::ValidityInterval( *(itFound.first), endOfInterval);
}

}

// -*- C++ -*-
//
// Package:     L1TObjects
// Class  :     L1TriggerKeyList
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Fri Feb 29 21:00:24 CET 2008
// $Id: L1TriggerKeyList.cc,v 1.3 2009/04/06 01:58:49 wsun Exp $
//

// system include files

// user include files
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TriggerKeyList::L1TriggerKeyList()
{
}

// L1TriggerKeyList::L1TriggerKeyList(const L1TriggerKeyList& rhs)
// {
//    // do actual copying here;
// }

L1TriggerKeyList::~L1TriggerKeyList()
{
}

//
// assignment operators
//
// const L1TriggerKeyList& L1TriggerKeyList::operator=(const L1TriggerKeyList& rhs)
// {
//   //An exception safe implementation is
//   L1TriggerKeyList temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

bool
L1TriggerKeyList::addKey( const std::string& tscKey,
			  const std::string& payloadToken,
			  bool overwriteKey )
{
  std::pair< KeyToToken::iterator, bool > result =
    m_tscKeyToToken.insert( std::make_pair( tscKey, payloadToken ) ) ;

  if( !result.second && overwriteKey )
    {
      // Erase previous entry
      m_tscKeyToToken.erase( result.first ) ;

      // Try again
      result = m_tscKeyToToken.insert( std::make_pair( tscKey,
						       payloadToken ) ) ;
    }

  return result.second ;
}

bool
L1TriggerKeyList::addKey( const std::string& recordType,
			  const std::string& key,
			  const std::string& payloadToken,
			  bool overwriteKey )
{
  RecordToKeyToToken::iterator it = m_recordKeyToken.find( recordType ) ;

  if( it == m_recordKeyToken.end() )
    {
      it = m_recordKeyToken.insert( std::make_pair( recordType,
						    KeyToToken() ) ).first ;
    } 

  std::pair< KeyToToken::iterator, bool > result =
    it->second.insert( std::make_pair( key, payloadToken ) ) ;

  if( !result.second && overwriteKey )
    {
      // Erase previous entry
      it->second.erase( result.first ) ;

      // Try again
      result = it->second.insert( std::make_pair( key, payloadToken ) ) ;
    }

  return result.second ;
}

//
// const member functions
//

std::string
L1TriggerKeyList::token( const std::string& tscKey ) const
{
  KeyToToken::const_iterator it = m_tscKeyToToken.find( tscKey ) ;

  if( it == m_tscKeyToToken.end() )
    {
      return std::string() ;
    }
  else
    {
      return it->second;
    }
}

std::string
L1TriggerKeyList::token( const std::string& recordName,
			 const std::string& dataType,
			 const std::string& key ) const
{
  std::string recordType = recordName + "@" + dataType ;
  return token( recordType, key ) ;
}

std::string
L1TriggerKeyList::token( const std::string& recordType,
			 const std::string& key ) const
{
  RecordToKeyToToken::const_iterator it = m_recordKeyToken.find( recordType ) ;

  if( it == m_recordKeyToken.end() )
    {
      return std::string() ;
    } 
  else
    {
      KeyToToken::const_iterator it2 = it->second.find( key ) ;

      if( it2 == it->second.end() )
	{
	  return std::string() ;
	}
      else
	{
	  return it2->second ;
	}
    }
}

// std::string
// L1TriggerKeyList::objectKey( const std::string& recordName,
// 			     const std::string& dataType,
// 			     const std::string& payloadToken ) const
// {
//   return objectKey( recordName + "@" + dataType,
// 		    payloadToken ) ;
// }

// std::string
// L1TriggerKeyList::objectKey( const std::string& recordType,// "record@type"
// 			     const std::string& payloadToken ) const
// {
//   RecordToKeyToToken::const_iterator keyTokenMap =
//     m_recordKeyToken.find( recordType ) ;

//   if( keyTokenMap != m_recordKeyToken.end() )
//     {
//       // Find object key with matching payload token.
//       KeyToToken::const_iterator iKey = keyTokenMap.second.begin();
//       KeyToToken::const_iterator eKey = keyTokenMap.second.end() ;
//       for( ; iKey != eKey ; ++iKey )
// 	{
// 	  if( iKey->second == payloadToken )
// 	    {
// 	      return iKey->first ;
// 	    }
// 	}
//     }

//   return std::string() ;
// }

std::string
L1TriggerKeyList::objectKey( const std::string& recordName,
			     const std::string& payloadToken ) const
{
  RecordToKeyToToken::const_iterator iRecordType = m_recordKeyToken.begin() ;
  for( ; iRecordType != m_recordKeyToken.end() ; ++iRecordType )
    {
      // Extract record name from recordType
      std::string recordInMap( iRecordType->first, 0,
			       iRecordType->first.find_first_of("@") ) ;
      if( recordInMap == recordName )
	{
	  // Find object key with matching payload token.
	  KeyToToken::const_iterator iKey = iRecordType->second.begin();
	  KeyToToken::const_iterator eKey = iRecordType->second.end() ;
	  for( ; iKey != eKey ; ++iKey )
	    {
	      if( iKey->second == payloadToken )
		{
		  return iKey->first ;
		}
	    }
	}
    }

  return std::string() ;
}

std::string
L1TriggerKeyList::tscKey( const std::string& triggerKeyPayloadToken ) const
{
  // Find object key with matching payload token.
  KeyToToken::const_iterator iKey = m_tscKeyToToken.begin();
  KeyToToken::const_iterator eKey = m_tscKeyToToken.end() ;
  for( ; iKey != eKey ; ++iKey )
    {
      if( iKey->second == triggerKeyPayloadToken )
	{
	  return iKey->first ;
	}
    }

  return std::string() ;
}

//
// static member functions
//

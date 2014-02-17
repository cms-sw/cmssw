#ifndef PhysicsTools_PatAlgos_PATTriggerMatchSelector_h
#define PhysicsTools_PatAlgos_PATTriggerMatchSelector_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      PATTriggerMatchSelector
//
/**
  \class    pat::PATTriggerMatchSelector PATTriggerMatchSelector.h "PhysicsTools/PatAlgos/plugins/PATTriggerMatchSelector.h"
  \brief

   .

  \author   Volker Adler
  \version  $Id: PATTriggerMatchSelector.h,v 1.6 2010/12/11 22:12:59 vadler Exp $
*/
//
// $Id: PATTriggerMatchSelector.h,v 1.6 2010/12/11 22:12:59 vadler Exp $
//


#include <string>
#include <vector>
#include <map>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace pat {

  template< typename T1, typename T2 >
  class PATTriggerMatchSelector : public StringCutObjectSelector< T2 > {

    public:

      PATTriggerMatchSelector( const edm::ParameterSet & iConfig ) :
        StringCutObjectSelector< T2 >( iConfig.getParameter< std::string >( "matchedCuts" ) )
      {}

      bool operator()( const T1 & patObj, const T2 & trigObj ) const {
        return StringCutObjectSelector< T2 >::operator()( trigObj );
      }

  };

}


#endif

// -*- C++ -*-
//
// Package:    L1TriggerConfig
// Class:      DTTFTSCObjectKeysOnlineProd
// 
/**\class DTTFTSCObjectKeysOnlineProd DTTFTSCObjectKeysOnlineProd.h L1TriggerConfig/DTTFConfigProducers/src/DTTFTSCObjectKeysOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Thu Oct  2 21:43:50 CEST 2008
// $Id: DTTFTSCObjectKeysOnlineProd.cc,v 1.1 2008/10/13 03:24:50 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class DTTFTSCObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      DTTFTSCObjectKeysOnlineProd(const edm::ParameterSet&);
      ~DTTFTSCObjectKeysOnlineProd();

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;
   private:
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DTTFTSCObjectKeysOnlineProd::DTTFTSCObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBase( iConfig )
{}


DTTFTSCObjectKeysOnlineProd::~DTTFTSCObjectKeysOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DTTFTSCObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
  std::string dttfKey = pL1TriggerKey->subsystemKey( L1TriggerKey::kDTTF ) ;

  // SELECT LUT_KEY FROM CMS_DT_TF.DTTF_CONF WHERE DTTF_CONF.ID = dttfKey
  l1t::OMDSReader::QueryResults lutKeyResults =
    m_omdsReader.basicQuery( "LUT_KEY",
			     "CMS_DT_TF",
			     "DTTF_CONF",
			     "DTTF_CONF.ID",
			     m_omdsReader.singleAttribute( dttfKey  ) );


  if( lutKeyResults.queryFailed() ||
      lutKeyResults.numberRows() != 1 ) // check query successful
    {
      edm::LogError( "L1-O2O" ) << "Problem with DTTF key." ;
      return ;
    }

  std::string lutKey ;
  lutKeyResults.fillVariable( lutKey ) ;

  pL1TriggerKey->add( "L1MuDTEtaPatternLutRcd",
		      "L1MuDTEtaPatternLut",
		      lutKey ) ;

  pL1TriggerKey->add( "L1MuDTExtLutRcd",
		      "L1MuDTExtLut",
		      lutKey ) ;

  pL1TriggerKey->add( "L1MuDTPhiLutRcd",
		      "L1MuDTPhiLut",
		      lutKey ) ;

  pL1TriggerKey->add( "L1MuDTPtaLutRcd",
		      "L1MuDTPtaLut",
		      lutKey ) ;

  pL1TriggerKey->add( "L1MuDTQualPatternLutRcd",
		      "L1MuDTQualPatternLut",
		      lutKey ) ;

  pL1TriggerKey->add( "L1MuDTTFParametersRcd",
		      "L1MuDTTFParameters",
		      dttfKey ) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DTTFTSCObjectKeysOnlineProd);

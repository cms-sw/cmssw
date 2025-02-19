//-------------------------------------------------
//
//   \class L1MuTriggerPtScaleOnlineProducer
//
//   Description:  A class to produce the L1 mu emulator scales record in the event setup
//
//   $Date: 2008/11/24 18:59:59 $
//   $Revision: 1.1 $
//
//   Author :
//   W. Sun (copied from L1MuTriggerScalesProducer)
//
//--------------------------------------------------
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerPtScaleOnlineProducer.h"

// #define DEBUG_PT_SCALE
#ifdef DEBUG_PT_SCALE
#include <iostream>
#endif

#include <sstream>

using namespace std;

L1MuTriggerPtScaleOnlineProducer::L1MuTriggerPtScaleOnlineProducer(const edm::ParameterSet& ps)
  : L1ConfigOnlineProdBase<L1MuTriggerPtScaleRcd, L1MuTriggerPtScale>(ps),
    m_signedPacking(ps.getParameter<bool>("signedPackingPt")),
    m_nbitsPacking(ps.getParameter<int>("nbitPackingPt")),
    m_nBins(ps.getParameter<int>("nbinsPt"))
{
}

L1MuTriggerPtScaleOnlineProducer::~L1MuTriggerPtScaleOnlineProducer() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
boost::shared_ptr<L1MuTriggerPtScale> 
L1MuTriggerPtScaleOnlineProducer::newObject(const std::string& objectKey )
{
   using namespace edm::es;

   // find Pt key from main scales key
   l1t::OMDSReader::QueryResults keysRecord = 
         m_omdsReader.basicQuery(
          // SELECTed columns
          "SC_MUON_PT_FK",
	  // schema name
	  "CMS_GT",
	  // table name
          "L1T_SCALES",
	  // WHERE lhs
	  "L1T_SCALES.ID",
	  // WHERE rhs
	  m_omdsReader.singleAttribute( objectKey  ) );

   if( keysRecord.numberRows() != 1 ) // check if query was successful
   {
       throw cond::Exception("Problem finding L1MuTriggerScales associated "
                             "with scales key " + objectKey);
   }


   /*
SQL> describe cms_gt.l1t_scale_muon_pt;
 Name                                      Null?    Type
 ----------------------------------------- -------- ----------------------------
 ID                                        NOT NULL VARCHAR2(300)
 PT_GEV_BIN_LOW_0                                   NUMBER
 [...]
 PT_GEV_BIN_LOW_32                                  NUMBER
   */

   ScaleRecordHelper h("PT_GEV_BIN_LOW", m_nBins );

   vector<string> columns;
   h.pushColumnNames(columns);

   l1t::OMDSReader::QueryResults resultRecord = 
       m_omdsReader.basicQuery(
           // SELECTed columns
           columns,
           // schema name
           "CMS_GT",
           // table name
           "L1T_SCALE_MUON_PT",
           // WHERE lhs
           "L1T_SCALE_MUON_PT.ID",
           // WHERE rhs
           keysRecord);

   if(resultRecord.numberRows() != 1) { 
       throw cond::Exception("Couldn't find Pt scale record for scales key `" 
                             + objectKey + "'") ;
   }

   vector<double> scales;
   h.extractScales(resultRecord, scales);
   
   boost::shared_ptr<L1MuTriggerPtScale> result( new L1MuTriggerPtScale(m_nbitsPacking, m_signedPacking, m_nBins, scales) );
   
#ifdef DEBUG_PT_SCALE
   cout << "PT scale:" << endl << result->getPtScale()->print() << endl;
#endif


   return result ;
}

//-------------------------------------------------
//
//   \class L1MuTriggerScalesOnlineProducer
//
//   Description:  A class to produce the L1 mu emulator scales record in the event setup
//                 from the OMDS database.
//
//   $Date: 2012/09/21 13:41:19 $
//   $Revision: 1.3 $
//
//   Author :
//   Thomas Themel
//
//--------------------------------------------------
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerScalesOnlineProducer.h"
#include "L1TriggerConfig/L1ScalesProducers/interface/ScaleRecordHelper.h"
#include <cmath>

using namespace std;

L1MuTriggerScalesOnlineProducer::L1MuTriggerScalesOnlineProducer(const edm::ParameterSet& ps)
  : L1ConfigOnlineProdBase<L1MuTriggerScalesRcd, L1MuTriggerScales>(ps),
    m_scales( 
	      ps.getParameter<int>("nbitPackingDTEta"),
	      ps.getParameter<bool>("signedPackingDTEta"),
	      ps.getParameter<int>("nbinsDTEta"),
	      ps.getParameter<double>("minDTEta"),
	      ps.getParameter<double>("maxDTEta"),
	      ps.getParameter<int>("offsetDTEta"),

	      ps.getParameter<int>("nbitPackingCSCEta"),
	      ps.getParameter<int>("nbinsCSCEta"),
	      ps.getParameter<double>("minCSCEta"),
	      ps.getParameter<double>("maxCSCEta"),

	      ps.getParameter<std::vector<double> >("scaleRPCEta"),
	      ps.getParameter<int>("nbitPackingBrlRPCEta"),
	      ps.getParameter<bool>("signedPackingBrlRPCEta"),
	      ps.getParameter<int>("nbinsBrlRPCEta"),
	      ps.getParameter<int>("offsetBrlRPCEta"),
	      ps.getParameter<int>("nbitPackingFwdRPCEta"),
	      ps.getParameter<bool>("signedPackingFwdRPCEta"),
	      ps.getParameter<int>("nbinsFwdRPCEta"),
	      ps.getParameter<int>("offsetFwdRPCEta"),
	      // Fields that should now be generated from OMDS:
	      // TODO: Adjust m_scales's definition to be a bit
	      //       more accessible for the partial initialization.
	      //ps.getParameter<int>("nbitPackingGMTEta"),
	      0,
	      //ps.getParameter<int>("nbinsGMTEta"),
	      0,
	      //ps.getParameter<std::vector<double> >("scaleGMTEta"),
	      std::vector<double>(1),
	      //ps.getParameter<int>("nbitPackingPhi"),
	      0,
	      //ps.getParameter<bool>("signedPackingPhi"),
	      false,
	      //ps.getParameter<int>("nbinsPhi"),
	      0,
	      //ps.getParameter<double>("minPhi"),
	      0,
	      //ps.getParameter<double>("maxPhi") 
	      0	      
	      ),
    /* Metadata that's not yet in the database. */
    m_nbitPackingPhi(ps.getParameter<int>("nbitPackingPhi")),
    m_nbitPackingEta(ps.getParameter<int>("nbitPackingGMTEta")),
    m_nbinsEta(ps.getParameter<int>("nbinsGMTEta")),
    m_signedPackingPhi(ps.getParameter<bool>("signedPackingPhi"))
{
}

L1MuTriggerScalesOnlineProducer::~L1MuTriggerScalesOnlineProducer() {}


//
// member functions
//

class PhiScaleHelper { 
  public:
  
  static L1MuBinnedScale* makeBinnedScale(l1t::OMDSReader::QueryResults& record, int nBits, bool signedPacking) {
    short nbins=0;
    record.fillVariable(BinsColumn, nbins);
    float lowMark=0.;
    record.fillVariable(LowMarkColumn, lowMark);
    float step=0.;
    record.fillVariable(StepColumn, step);

    return new L1MuBinnedScale(nBits, signedPacking, 
			       nbins, deg2rad(lowMark), 
			       deg2rad(lowMark + nbins*step));

  }

  static void pushColumnNames(vector<string>& columns) { 
    columns.push_back(BinsColumn);
    columns.push_back(LowMarkColumn);
    columns.push_back(StepColumn);
  }

  static double deg2rad(double deg) { return deg*M_PI/180.0; }
  static double rad2deg(double rad) { return rad/M_PI*180.0; }

  static const string BinsColumn;
  static const string LowMarkColumn;
  static const string StepColumn;
};

const string PhiScaleHelper::BinsColumn = "PHI_BINS";
const string PhiScaleHelper::LowMarkColumn = "PHI_DEG_BIN_LOW_0";
const string PhiScaleHelper::StepColumn = "PHI_DEG_BIN_STEP";

// ------------ method called to produce the data  ------------
boost::shared_ptr<L1MuTriggerScales> L1MuTriggerScalesOnlineProducer::newObject(const std::string& objectKey ) 
{
   using namespace edm::es;   

   // The key we get from the O2O subsystem is the CMS_GMT.L1T_SCALES key,
   // but the eta/phi scales have their own subtables, so let's find 
   // out.
   vector<string> foreignKeys;

   const std::string etaKeyColumn("SC_MUON_ETA_FK");
   const std::string phiKeyColumn("SC_MUON_PHI_FK");

   foreignKeys.push_back(etaKeyColumn);
   foreignKeys.push_back(phiKeyColumn);

   l1t::OMDSReader::QueryResults keysRecord = 
         m_omdsReader.basicQuery(
          // SELECTed columns
          foreignKeys,
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
                             "with scales key `" + objectKey + "'");
   }


   std::string etaKeyValue;
   std::string phiKeyValue;
   keysRecord.fillVariable(etaKeyColumn, etaKeyValue);
   keysRecord.fillVariable(phiKeyColumn, phiKeyValue);

   vector<string> columns;

   // get the eta scales from the database
   ScaleRecordHelper etaHelper("ETA_BIN_LOW", m_nbinsEta);
   etaHelper.pushColumnNames(columns);

   l1t::OMDSReader::QueryResults etaRecord = 
         m_omdsReader.basicQuery(
          // SELECTed columns
          columns,
	  // schema name
	  "CMS_GT",
	  // table name
          "L1T_SCALE_MUON_ETA",
	  // WHERE lhs
	  "L1T_SCALE_MUON_ETA.ID",
	  // WHERE rhs
	  m_omdsReader.singleAttribute( etaKeyValue  ) );

   vector<double> etaScales;
   etaHelper.extractScales(etaRecord, etaScales);
   
   auto_ptr<L1MuSymmetricBinnedScale> ptrEtaScale(new L1MuSymmetricBinnedScale(m_nbitPackingEta, m_nbinsEta, etaScales));
   m_scales.setGMTEtaScale(*ptrEtaScale);

   columns.clear();   

   // get the phi scales from the database
   PhiScaleHelper phiHelper;

   l1t::OMDSReader::QueryResults phiRecord = 
         m_omdsReader.basicQuery(
          // SELECTed columns
          columns,
	  // schema name
	  "CMS_GT",
	  // table name
          "L1T_SCALE_MUON_PHI",
	  // WHERE lhs
	  "L1T_SCALE_MUON_PHI.ID",
	  // WHERE rhs
	  m_omdsReader.singleAttribute( phiKeyValue  ) );

   auto_ptr<L1MuBinnedScale> ptrPhiScale(phiHelper.makeBinnedScale(phiRecord, m_nbitPackingPhi, m_signedPackingPhi));

   m_scales.setPhiScale(*ptrPhiScale);

   boost::shared_ptr<L1MuTriggerScales> l1muscale =
     boost::shared_ptr<L1MuTriggerScales>( new L1MuTriggerScales( m_scales ) );

   return l1muscale ;
}

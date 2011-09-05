#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h" 


// OMDS test stuff
#include "RelationalAccess/ISession.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/IRelationalDomain.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IConnection.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralKernel/Context.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "FWCore/Framework/interface/Run.h"
// end OMDS test stuff

#include <vector>
using std::vector;
#include <iostream>



using std::cout;
using std::endl;
const int L1RCTProducer::crateFED[18][5]=
      {{613, 614, 603, 702, 718},
    {611, 612, 602, 700, 718},
    {627, 610, 601,716,   722},
    {625, 626, 609, 714, 722},
    {623, 624, 608, 712, 722},
    {621, 622, 607, 710, 720},
    {619, 620, 606, 708, 720},
    {617, 618, 605, 706, 720},
    {615, 616, 604, 704, 718},
    {631, 632, 648, 703, 719},
    {629, 630, 647, 701, 719},
    {645, 628, 646, 717, 723},
    {643, 644, 654, 715, 723},
    {641, 642, 653, 713, 723},
    {639, 640, 652, 711, 721},
    {637, 638, 651, 709, 721},
    {635, 636, 650, 707, 721},
    {633, 634, 649, 705, 719}};



L1RCTProducer::L1RCTProducer(const edm::ParameterSet& conf) : 
  rctLookupTables(new L1RCTLookupTables),
  rct(new L1RCT(rctLookupTables)),
  useEcal(conf.getParameter<bool>("useEcal")),
  useHcal(conf.getParameter<bool>("useHcal")),
  ecalDigis(conf.getParameter<std::vector<edm::InputTag> >("ecalDigis")),
  hcalDigis(conf.getParameter<std::vector<edm::InputTag> >("hcalDigis")),
  bunchCrossings(conf.getParameter<std::vector<int> >("BunchCrossings")),
  getFedsFromOmds(conf.getParameter<bool>("getFedsFromOmds")),
  queryDelayInLS(conf.getParameter<unsigned int>("queryDelayInLS")),
  fedUpdatedMask(0)
{
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();

 

}

L1RCTProducer::~L1RCTProducer()
{
  if(rct != 0) delete rct;
  if(rctLookupTables != 0) delete rctLookupTables;
  if(fedUpdatedMask != 0) delete fedUpdatedMask;
}


void L1RCTProducer::beginRun(edm::Run& run, const edm::EventSetup& eventSetup)
{
  // Testing out this OMDS stuff online
  // hodge-podged together from http://cmslxr.fnal.gov/lxr/source/CondTools/RunInfo/src/RunInfoRead.cc#191

  // NEED VALUES FOR: (keep these things hard-coded?)
  //  std::string m_connectionString = "oracle://CMS_OMDS_LB/CMS_TRG_R"; // try this one, they appear to be of this form
  //  std::string m_connectionString = "oracle://cms_orcoff_prod/CMS_COND_RUNINFO"; 
  std::string m_connectionString = "oracle://cms_orcoff_prod/CMS_RUNINFO"; // from Salvatore Di Guida
  //  std::string m_user = "cms_trg_r";
  //  std::string m_user = "CMS_COND_RUNINFO";
  std::string m_user = "CMS_COND_GENERAL_R";
  //  std::string m_pass = "X3lmdvu4"; // FIXMEEEEEEEE!
  std::string m_pass = ""; // FIXMEEEEEEEE!
  std::string m_authpath = "/afs/cern.ch/cms/DB/conddb";
  //  std::string m_tableToRead = "cms_runinfo.runsession_parameter"; // "to be cms_runinfo.runsession_parameter" ??  make it so?
  //  std::string m_tableToRead = "runsession_parameter"; 
  std::string m_tableToRead = "RUNSESSION_PARAMETER"; 

  RunInfo temp_sum;
  
//   coral::ISession* session = this->connect( m_connectionString,
// 					    m_user, m_pass );
//   cond::DbSession* session = this->connect( m_connectionString,
// 					    m_authpath );

  //make connection object
  cond::DbConnection         m_connection;

  //  std::cout << "Check 1: configuring connection" << std::endl;
  
  //set in configuration object authentication path
  m_connection.configuration().setAuthenticationPath(m_authpath);
  m_connection.configure();

  //  std::cout << "Check 2: creating session" << std::endl;

  //create session object from connection
  cond::DbSession session = m_connection.createSession();
 
  //  std::cout << "Check 3: opening session" << std::flush;

  session.open(m_connectionString,true);
     
  //  std::cout << "successful" << std::endl;

  try{

    //    std::cout << "Check 4: starting session transaction" << std::endl;
    //    session->transaction().start();
    session.transaction().start(true); // (true=readOnly)

    //    std::cout << "Check 5: creating schema" << std::endl;
    //    coral::ISchema& schema = session->schema("CMS_RUNINFO");
    coral::ISchema& schema = session.schema("CMS_RUNINFO");

    //condition 
    coral::AttributeList conditionData;
    conditionData.extend<int>( "n_run" );
    int r_number = run.run(); // ADDED, from FWCore/Common/interface/RunBase.h
    conditionData[0].data<int>() = r_number;

    std::string m_columnToRead_val = "VALUE";

    std::string m_tableToRead_fed = "RUNSESSION_STRING";
    coral::IQuery* queryV = schema.newQuery();  
    queryV->addToTableList(m_tableToRead);
    queryV->addToTableList(m_tableToRead_fed);
    queryV->addToOutputList(m_tableToRead_fed + "." + m_columnToRead_val, m_columnToRead_val);
    //queryV->addToOutputList(m_tableToRead + "." + m_columnToRead, m_columnToRead);
    //condition
    std::string condition = m_tableToRead + ".RUNNUMBER=:n_run AND " + m_tableToRead + ".NAME='CMS.LVL0:FED_ENABLE_MASK' AND RUNSESSION_PARAMETER.ID = RUNSESSION_STRING.RUNSESSION_PARAMETER_ID";
    //std::string condition = m_tableToRead + ".runnumber=:n_run AND " + m_tableToRead + ".name='CMS.LVL0:FED_ENABLE_MASK'";
    queryV->setCondition(condition, conditionData);
    coral::ICursor& cursorV = queryV->execute();
    std::string fed;
    if ( cursorV.next() ) {
      //cursorV.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = cursorV.currentRow();
      fed = row[m_columnToRead_val].data<std::string>();
    }
    else {
      fed="null";
    }
    //std::cout << "string fed emask == " << fed << std::endl;
    delete queryV;
    
    std::replace(fed.begin(), fed.end(), '%', ' ');
    std::stringstream stream(fed);
    for(;;) 
      {
	std::string word; 
	if ( !(stream >> word) ){break;}
	std::replace(word.begin(), word.end(), '&', ' ');
	std::stringstream ss(word);
	int fedNumber; 
	int val;
	ss >> fedNumber >> val;
	//std::cout << "fed:: " << fed << "--> val:: " << val << std::endl; 
	//val bit 0 represents the status of the SLINK, but 5 and 7 means the SLINK/TTS is ON but NA or BROKEN (see mail of alex....)
	if( (val & 0001) == 1 && (val != 5) && (val != 7) ) 
	  temp_sum.m_fed_in.push_back(fedNumber);
      } 
    std::cout << "feds in run:--> ";
    std::copy(temp_sum.m_fed_in.begin(), temp_sum.m_fed_in.end(), std::ostream_iterator<int>(std::cout, ", "));
    std::cout << std::endl;
    /*
      for (size_t i =0; i<temp_sum.m_fed_in.size() ; ++i)
      {
      std::cout << "fed in run:--> " << temp_sum.m_fed_in[i] << std::endl; 
      } 
    */
    
  }
  catch (const std::exception& e) { 
    std::cout << "Exception: " << e.what() << std::endl;
  }
  //  delete session;
  
  
  // End testing OMDS stuff
  
}

// // HACK right now for OMDS stuff, move into separate class

// // coral::ISession*
// // L1RCTProducer::connect( const std::string& connectionString,
// // 			const std::string& user,
// // 			const std::string& pass ) {
// //coral::ISession*
// cond::DbSession*
// L1RCTProducer::connect( const std::string& connectionString,
// 			const std::string& authPath ) {

//   //  coral::IConnection*         m_connection(0);

//   //   coral::Context& ctx = coral::Context::instance();
//   //   coral::IHandle<coral::IRelationalDomain> iHandle=ctx.query<coral::IRelationalDomain>("CORAL/RelationalPlugins/oracle");
//   //   if ( ! iHandle.isValid() ) {
//   //     throw std::runtime_error( "Could not load the OracleAccess plugin" );
//   //   }
//   //   std::pair<std::string, std::string> connectionAndSchema = iHandle->decodeUserConnectionString( connectionString );
//   //   if ( ! m_connection ) {
//   //     m_connection = iHandle->newConnection( connectionAndSchema.first );
//   //   }
//   //   if ( ! m_connection->isConnected() ) {
//   //     m_connection->connect();
//   //   }
//   //   coral::ISession* session = m_connection->newSession( connectionAndSchema.second );
//   //   if ( session ) {
//   //     session->startUserSession( user, pass );
//   //   }
//   //   return session;
  
  
//   //make connection object
//   cond::DbConnection*         m_connection(0);
//   //cond::DbConnection         m_connection;
  
//   //set in configuration object authentication path
//   m_connection->configuration().setAuthenticationPath(authPath);
//   m_connection->configure();

//   //create session object from connection
//   //  cond::DbSession* dbSes = &(m_connection->createSession());
//   cond::DbSession dbSes = m_connection->createSession();
 
//   //try to make connection
// //   if (dbSes)
// //     {
// //    dbSes->open(connectionString,true);
// //     }
//   dbSes.open(connectionString,true);
     
// //   //start a transaction (true=readOnly)
// //   dbSes->transaction().start(true);

//   return dbSes;

// }

// END HACK


void L1RCTProducer::beginLuminosityBlock(edm::LuminosityBlock& lumiSeg,const edm::EventSetup& context)
{
  updateConfiguration(context);
} 



void L1RCTProducer::updateConfiguration(const edm::EventSetup& eventSetup)
{
  // Refresh configuration information every event
  // Hopefully, this does not take too much time
  // There should be a call back function in future to
  // handle changes in configuration
  // parameters to configure RCT (thresholds, etc)
  edm::ESHandle<L1RCTParameters> rctParameters;
  eventSetup.get<L1RCTParametersRcd>().get(rctParameters);
  const L1RCTParameters* r = rctParameters.product();

  // list of RCT channels to mask
  edm::ESHandle<L1RCTChannelMask> channelMask;
  eventSetup.get<L1RCTChannelMaskRcd>().get(channelMask);
  const L1RCTChannelMask* cEs = channelMask.product();


  // list of Noisy RCT channels to mask
  edm::ESHandle<L1RCTNoisyChannelMask> hotChannelMask;
  eventSetup.get<L1RCTNoisyChannelMaskRcd>().get(hotChannelMask);
  const L1RCTNoisyChannelMask* cEsNoise = hotChannelMask.product();
  rctLookupTables->setNoisyChannelMask(cEsNoise);


  
  //Update the channel mask according to the FED VECTOR
  //This is the beginning of run. We delete the old
  //create the new and set it in the LUTs

  if(fedUpdatedMask!=0) delete fedUpdatedMask;

  fedUpdatedMask = new L1RCTChannelMask();
  // copy a constant object
  for (int i = 0; i < 18; i++)
    {
      for (int j = 0; j < 2; j++)
	{
	  for (int k = 0; k < 28; k++)
	    {
	      fedUpdatedMask->ecalMask[i][j][k] = cEs->ecalMask[i][j][k];
	      fedUpdatedMask->hcalMask[i][j][k] = cEs->hcalMask[i][j][k] ;
	    }
	  for (int k = 0; k < 4; k++)
	    {
	      fedUpdatedMask->hfMask[i][j][k] = cEs->hfMask[i][j][k];
	    }
	}
    }


  // adding fed mask into channel mask
  
  // get FED vector from RUNINFO
  edm::ESHandle<RunInfo> sum;
  eventSetup.get<RunInfoRcd>().get(sum);
  const RunInfo* summary=sum.product();

  // get FED vector from OMDS in case of online running


  std::vector<int> caloFeds;  // pare down the feds to the intresting ones
  // is this unneccesary?
  // Mike B : This will decrease the find speed so better do it
  const std::vector<int> Feds = summary->m_fed_in;
  for(std::vector<int>::const_iterator cf = Feds.begin(); cf != Feds.end(); ++cf){
    int fedNum = *cf;
    if(fedNum > 600 && fedNum <724) 
      caloFeds.push_back(fedNum);
  }

  for(int  cr = 0; cr < 18; ++cr){

    for(crateSection cs = c_min; cs <= c_max; cs = crateSection(cs +1)) {
      bool fedFound = false;


      //Try to find the FED
      std::vector<int>::iterator fv = std::find(caloFeds.begin(),caloFeds.end(),crateFED[cr][cs]);
      if(fv!=caloFeds.end())
	fedFound = true;

      if(!fedFound) {
	int eta_min=0;
	int eta_max=0;
	bool phi_even[2] = {false};//, phi_odd = false;
	bool ecal=false;
	
	switch (cs) {
	case ebEvenFed :
	  eta_min = minBarrel;
	  eta_max = maxBarrel;
	  phi_even[0] = true;
	  ecal = true;	
	  break;
	  
	case ebOddFed:
	  eta_min = minBarrel;
	  eta_max = maxBarrel;
	  phi_even[1] = true;
	  ecal = true;	
	  break;
	  
	case eeFed:
	  eta_min = minEndcap;
	  eta_max = maxEndcap;
	  phi_even[0] = true;
	  phi_even[1] = true;
	  ecal = true;	
	  break;	
	  
	case hbheFed:
	  eta_min = minBarrel;
	  eta_max = maxEndcap;
	  phi_even[0] = true;
	  phi_even[1] = true;
	  ecal = false;
	  break;

	case hfFed:	
	  eta_min = minHF;
	  eta_max = maxHF;

	  phi_even[0] = true;
	  phi_even[1] = true;
	  ecal = false;
	  break;
	default:
	  break;

	}
	for(int ieta = eta_min; ieta <= eta_max; ++ieta){
	  if(ieta<=28) // barrel and endcap
	    for(int even = 0; even<=1 ; even++){	 
	      if(phi_even[even]){
		if(ecal)
		  fedUpdatedMask->ecalMask[cr][even][ieta-1] = true;
		else
		  fedUpdatedMask->hcalMask[cr][even][ieta-1] = true;
	      }
	    }
	  else
	    for(int even = 0; even<=1 ; even++)
	      if(phi_even[even])
		  fedUpdatedMask->hfMask[cr][even][ieta-29] = true;
	
	}
      }
    }
  }


  //SCALES

  // energy scale to convert eGamma output
  edm::ESHandle<L1CaloEtScale> emScale;
  eventSetup.get<L1EmEtScaleRcd>().get(emScale);
  const L1CaloEtScale* s = emScale.product();


 // get energy scale to convert input from ECAL
  edm::ESHandle<L1CaloEcalScale> ecalScale;
  eventSetup.get<L1CaloEcalScaleRcd>().get(ecalScale);
  const L1CaloEcalScale* e = ecalScale.product();
  
  // get energy scale to convert input from HCAL
  edm::ESHandle<L1CaloHcalScale> hcalScale;
  eventSetup.get<L1CaloHcalScaleRcd>().get(hcalScale);
  const L1CaloHcalScale* h = hcalScale.product();

  // set scales
  rctLookupTables->setEcalScale(e);
  rctLookupTables->setHcalScale(h);


  rctLookupTables->setRCTParameters(r);
  rctLookupTables->setChannelMask(fedUpdatedMask);
  rctLookupTables->setL1CaloEtScale(s);
}


void L1RCTProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{


  std::auto_ptr<L1CaloEmCollection> rctEmCands (new L1CaloEmCollection);
  std::auto_ptr<L1CaloRegionCollection> rctRegions (new L1CaloRegionCollection);


  if(!(ecalDigis.size()==hcalDigis.size()&&hcalDigis.size()==bunchCrossings.size()))
      throw cms::Exception("BadInput")
	<< "From what I see the number of your your ECAL input digi collections.\n"
        <<"is different from the size of your HCAL digi input collections\n"
	<<"or the size of your BX factor collection" 
        <<"They must be the same to correspond to the same Bxs\n"
	<< "It does not matter if one of them is empty\n"; 




  // loop through and process each bx
    for (unsigned short sample = 0; sample < bunchCrossings.size(); sample++)
      {
	edm::Handle<EcalTrigPrimDigiCollection> ecal;
	edm::Handle<HcalTrigPrimDigiCollection> hcal;

	EcalTrigPrimDigiCollection ecalIn;
	HcalTrigPrimDigiCollection hcalIn;


	if(useHcal&&event.getByLabel(hcalDigis[sample], hcal))
	  hcalIn = *hcal;

	if(useEcal&&event.getByLabel(ecalDigis[sample],ecal))
	  ecalIn = *ecal;

	rct->digiInput(ecalIn,hcalIn);
	rct->processEvent();

      // Stuff to create
	for (int j = 0; j<18; j++)
	  {
	    L1CaloEmCollection isolatedEGObjects = rct->getIsolatedEGObjects(j);
	    L1CaloEmCollection nonisolatedEGObjects = rct->getNonisolatedEGObjects(j);
	    for (int i = 0; i<4; i++) 
	      {
		isolatedEGObjects.at(i).setBx(bunchCrossings[sample]);
		nonisolatedEGObjects.at(i).setBx(bunchCrossings[sample]);
		rctEmCands->push_back(isolatedEGObjects.at(i));
		rctEmCands->push_back(nonisolatedEGObjects.at(i));
	      }
	  }
      
      
	for (int i = 0; i < 18; i++)
	  {
	    std::vector<L1CaloRegion> regions = rct->getRegions(i);
	    for (int j = 0; j < 22; j++)
	      {
		regions.at(j).setBx(bunchCrossings[sample]);
		rctRegions->push_back(regions.at(j));
	      }
	  }

      }

  
  //putting stuff back into event
  event.put(rctEmCands);
  event.put(rctRegions);
  
}

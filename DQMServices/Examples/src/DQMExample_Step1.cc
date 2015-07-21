#include "DQMServices/Examples/interface/DQMExample_Step1.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


// Geometry
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"


// CORAL
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/MessageStream.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IPrimaryKey.h"
#include "RelationalAccess/IForeignKey.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
#include <vector>
#include <ctime>

//
// -------------------------------------- Constructor --------------------------------------------
//
DQMExample_Step1::DQMExample_Step1(const edm::ParameterSet& ps): m_connectionService(), m_session(), m_connectionString( "" )
{
  edm::LogInfo("DQMExample_Step1") <<  "Constructor  DQMExample_Step1::DQMExample_Step1 " << std::endl;
  
  // Get parameters from configuration file
  theElectronCollection_   = consumes<reco::GsfElectronCollection>(ps.getParameter<edm::InputTag>("electronCollection"));
  theCaloJetCollection_    = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  thePfMETCollection_      = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("pfMETCollection"));
  theConversionCollection_ = consumes<reco::ConversionCollection>(ps.getParameter<edm::InputTag>("conversionsCollection"));
  thePVCollection_         = consumes<reco::VertexCollection>(ps.getParameter<edm::InputTag>("PVCollection"));
  theBSCollection_         = consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamSpotCollection"));
  triggerEvent_            = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("TriggerEvent"));
  triggerResults_          = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  triggerFilter_           = ps.getParameter<edm::InputTag>("TriggerFilter");
  triggerPath_             = ps.getParameter<std::string>("TriggerPath");


  // cuts:
  ptThrL1_ = ps.getUntrackedParameter<double>("PtThrL1");
  ptThrL2_ = ps.getUntrackedParameter<double>("PtThrL2");
  ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet");
  ptThrMet_ = ps.getUntrackedParameter<double>("PtThrMet");
 
  //DQMStore
  //dbe_ = edm::Service<DQMStore>().operator->();
  
  //Database connection configuration parameters
  edm::ParameterSet connectionParameters = ps.getParameter<edm::ParameterSet>("DBParameters");
  std::string authPath = connectionParameters.getUntrackedParameter<std::string>("authPath", "");
  int messageLevel = connectionParameters.getUntrackedParameter<int>("messageLevel",0);
  coral::MsgLevel level = coral::Error;
  switch (messageLevel) {
  case 0 :
    level = coral::Error;
    break;    
  case 1 :
      level = coral::Warning;
    break;
  case 2 :
    level = coral::Info;
    break;
  case 3 :
    level = coral::Debug;
    break;
  default:
    level = coral::Error;
  }
  bool enableConnectionSharing = connectionParameters.getUntrackedParameter<bool>("enableConnectionSharing",true);
  int connectionTimeOut = connectionParameters.getUntrackedParameter<int>("connectionTimeOut",600);
  bool enableReadOnlySessionOnUpdateConnection = connectionParameters.getUntrackedParameter<bool>("enableReadOnlySessionOnUpdateConnection",true);
  int connectionRetrialPeriod = connectionParameters.getUntrackedParameter<int>("connectionRetrialPeriod",30);
  int connectionRetrialTimeOut = connectionParameters.getUntrackedParameter<int>("connectionRetrialTimeOut",180);
  bool enablePoolAutomaticCleanUp = connectionParameters.getUntrackedParameter<bool>("enablePoolAutomaticCleanUp",false);
  //connection string
  m_connectionString = ps.getParameter<std::string>("connect");
  //now configure the DB connection
  coral::IConnectionServiceConfiguration& coralConfig = m_connectionService.configuration();
  //TODO: set up the authentication mechanism
  
  // message streaming
  coral::MessageStream::setMsgVerbosity( level );
  //connection sharing
  if(enableConnectionSharing) coralConfig.enableConnectionSharing();
  else coralConfig.disableConnectionSharing();
  //connection timeout
  coralConfig.setConnectionTimeOut(connectionTimeOut);
  //read-only session on update connection
  if(enableReadOnlySessionOnUpdateConnection) coralConfig.enableReadOnlySessionOnUpdateConnections();
  else coralConfig.disableReadOnlySessionOnUpdateConnections();
  //connection retrial period
  coralConfig.setConnectionRetrialPeriod( connectionRetrialPeriod );
  //connection retrial timeout
  coralConfig.setConnectionRetrialTimeOut( connectionRetrialTimeOut );
  //pool automatic cleanup
  if(enablePoolAutomaticCleanUp) coralConfig.enablePoolAutomaticCleanUp();
  else coralConfig.disablePoolAutomaticCleanUp();
}

//
// -- Destructor
//
DQMExample_Step1::~DQMExample_Step1()
{
  edm::LogInfo("DQMExample_Step1") <<  "Destructor DQMExample_Step1::~DQMExample_Step1 " << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void DQMExample_Step1::dqmBeginRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::beginRun" << std::endl;
//open the CORAL session at beginRun:
  //connect to DB only if you have events to process!
  m_session.reset( m_connectionService.connect( m_connectionString, coral::Update ) );
  //do not run in production!
  //create the relevant tables
  coral::ISchema& schema = m_session->nominalSchema();
  m_session->transaction().start( false );
  bool dqmTablesExist = schema.existsTable( "HISTOGRAM" );
  if( ! dqmTablesExist )
  {
    int columnSize = 200;

    // Create the first table
    coral::TableDescription table1;
    table1.setName( "HISTOGRAM" );
    table1.insertColumn( "NAME", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );
    table1.insertColumn( "PATH", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );
    table1.insertColumn( "TIMESTAMP", coral::AttributeSpecification::typeNameForType<unsigned int>());
    table1.insertColumn( "TITLE", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );

    table1.setNotNullConstraint( "NAME" );
    table1.setNotNullConstraint( "PATH" );
    table1.setNotNullConstraint( "TIMESTAMP" );
    table1.setNotNullConstraint( "TITLE" );

    std::vector<std::string> columnsForPrimaryKey1;
    columnsForPrimaryKey1.push_back( "NAME" );
    columnsForPrimaryKey1.push_back( "PATH" );
    table1.setPrimaryKey( columnsForPrimaryKey1 );

    schema.createTable( table1 );
    std::cout << "Table1 created" << std::endl;

    // Create the second table
    coral::TableDescription table2;
    table2.setName( "HISTOGRAM_PROPS" );
    table2.insertColumn( "NAME", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );
    table2.insertColumn( "PATH", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );
    table2.insertColumn( "RUN_NUMBER", coral::AttributeSpecification::typeNameForType<unsigned int>() );
    table2.insertColumn( "X_BINS", coral::AttributeSpecification::typeNameForType<int>() );
    table2.insertColumn( "X_LOW", coral::AttributeSpecification::typeNameForType<double>() );
    table2.insertColumn( "X_UP", coral::AttributeSpecification::typeNameForType<double>() );
    table2.insertColumn( "Y_BINS", coral::AttributeSpecification::typeNameForType<int>() );
    table2.insertColumn( "Y_LOW", coral::AttributeSpecification::typeNameForType<double>() );
    table2.insertColumn( "Y_UP", coral::AttributeSpecification::typeNameForType<double>() );
    table2.insertColumn( "Z_BINS", coral::AttributeSpecification::typeNameForType<int>() );
    table2.insertColumn( "Z_LOW", coral::AttributeSpecification::typeNameForType<double>() );
    table2.insertColumn( "Z_UP", coral::AttributeSpecification::typeNameForType<double>() );

    table2.setNotNullConstraint( "NAME" );
    table2.setNotNullConstraint( "PATH" );
    table2.setNotNullConstraint( "RUN_NUMBER" );
    table2.setNotNullConstraint( "X_BINS" );
    table2.setNotNullConstraint( "X_LOW" );
    table2.setNotNullConstraint( "X_UP" );
    table2.setNotNullConstraint( "Y_BINS" );
    table2.setNotNullConstraint( "Y_LOW" );
    table2.setNotNullConstraint( "Y_UP" );
    table2.setNotNullConstraint( "Z_BINS" );
    table2.setNotNullConstraint( "Z_LOW" );
    table2.setNotNullConstraint( "Z_UP" );

    std::vector<std::string> columnsForPrimaryKey2;
    columnsForPrimaryKey2.push_back( "NAME" );
    columnsForPrimaryKey2.push_back( "PATH" );
    columnsForPrimaryKey2.push_back( "RUN_NUMBER" );
    table2.setPrimaryKey( columnsForPrimaryKey2 );

    std::vector<std::string> columnsForForeignKey2;
    columnsForForeignKey2.push_back( "NAME" );
    columnsForForeignKey2.push_back( "PATH" );

    table2.createForeignKey( "table2_FK", columnsForForeignKey2, "HISTOGRAM", columnsForPrimaryKey1 );

    schema.createTable( table2 );
    std::cout << "Table2 created" << std::endl;

    // Create the third table
    coral::TableDescription table3;
    table3.setName( "HISTOGRAM_VALUES" );
    table3.insertColumn( "NAME", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );
    table3.insertColumn( "PATH", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );
    table3.insertColumn( "RUN_NUMBER", coral::AttributeSpecification::typeNameForType<unsigned int>() );
    table3.insertColumn( "LUMISECTION", coral::AttributeSpecification::typeNameForType<unsigned int>() );
    table3.insertColumn( "ENTRIES", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "X_MEAN", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "X_MEAN_ERROR", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "X_RMS", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "X_RMS_ERROR", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "X_UNDERFLOW", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "X_OVERFLOW", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Y_MEAN", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Y_MEAN_ERROR", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Y_RMS", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Y_RMS_ERROR", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Y_UNDERFLOW", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Y_OVERFLOW", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Z_MEAN", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Z_MEAN_ERROR", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Z_RMS", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Z_RMS_ERROR", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Z_UNDERFLOW", coral::AttributeSpecification::typeNameForType<double>() );
    table3.insertColumn( "Z_OVERFLOW", coral::AttributeSpecification::typeNameForType<double>() );

    table3.setNotNullConstraint( "NAME" );
    table3.setNotNullConstraint( "PATH" );
    table3.setNotNullConstraint( "RUN_NUMBER" );
    table3.setNotNullConstraint( "LUMISECTION" );
    table3.setNotNullConstraint( "ENTRIES" );
    table3.setNotNullConstraint( "X_MEAN" );
    table3.setNotNullConstraint( "X_MEAN_ERROR" );
    table3.setNotNullConstraint( "X_RMS" );
    table3.setNotNullConstraint( "X_RMS_ERROR" );
    table3.setNotNullConstraint( "X_UNDERFLOW" );
    table3.setNotNullConstraint( "X_OVERFLOW" );
    table3.setNotNullConstraint( "Y_MEAN" );
    table3.setNotNullConstraint( "Y_MEAN_ERROR" );
    table3.setNotNullConstraint( "Y_RMS" );
    table3.setNotNullConstraint( "Y_RMS_ERROR" );
    table3.setNotNullConstraint( "Y_UNDERFLOW" );
    table3.setNotNullConstraint( "Y_OVERFLOW" );
    table3.setNotNullConstraint( "Z_MEAN" );
    table3.setNotNullConstraint( "Z_MEAN_ERROR" );
    table3.setNotNullConstraint( "Z_RMS" );
    table3.setNotNullConstraint( "Z_RMS_ERROR" );
    table3.setNotNullConstraint( "Z_UNDERFLOW" );
    table3.setNotNullConstraint( "Z_OVERFLOW" );

    std::vector<std::string> columnsForPrimaryKey3;
    columnsForPrimaryKey3.push_back( "NAME" );
    columnsForPrimaryKey3.push_back( "PATH" );
    columnsForPrimaryKey3.push_back( "RUN_NUMBER" );
    columnsForPrimaryKey3.push_back( "LUMISECTION" );
    table3.setPrimaryKey( columnsForPrimaryKey3 );

    std::vector<std::string> columnsForForeignKey3;
    columnsForForeignKey3.push_back( "NAME" );
    columnsForForeignKey3.push_back( "PATH" );
    columnsForForeignKey3.push_back( "RUN_NUMBER" );

    table3.createForeignKey( "table2_FK", columnsForForeignKey3, "HISTOGRAM_PROPS", columnsForPrimaryKey2 );

    schema.createTable( table3 );
    std::cout << "Table3 created" << std::endl;

  }
  m_session->transaction().commit();

  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::endLuminosityBlock" << std::endl;

  m_session->transaction().start(false);
  coral::ITableDataEditor& editor = m_session->nominalSchema().tableHandle( "HISTOGRAM" ).dataEditor();
  coral::AttributeList insertData;
  insertData.extend< std::string >( "NAME" );
  insertData.extend< std::string >( "PATH" );
  insertData.extend< unsigned int >( "TIMESTAMP" );
  insertData.extend< std::string >( "TITLE" );

  insertData[ "NAME" ].data< std::string >() = h_vertex_number->getName();
  insertData[ "PATH" ].data< std::string >() = h_vertex_number->getPathname();
  insertData[ "TIMESTAMP" ].data< unsigned int >() = std::time(nullptr);
  insertData[ "TITLE" ].data< std::string >() = h_vertex_number->getFullname();
  editor.insertRow( insertData );
  m_session->transaction().commit();
}

//
// -------------------------------------- bookHistos --------------------------------------------
//
void DQMExample_Step1::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::bookHistograms" << std::endl;
  
  //book at beginRun
  bookHistos(ibooker_);
}
//
// -------------------------------------- beginLuminosityBlock --------------------------------------------
//
void DQMExample_Step1::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& context) 
{
  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::beginLuminosityBlock" << std::endl;
}


//
// -------------------------------------- Analyze --------------------------------------------
//
void DQMExample_Step1::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
{
  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::analyze" << std::endl;


  //-------------------------------
  //--- Vertex Info
  //-------------------------------
  edm::Handle<reco::VertexCollection> vertexHandle;
  e.getByToken(thePVCollection_, vertexHandle);
  if ( !vertexHandle.isValid() ) 
    {
      edm::LogError ("DQMClientExample") << "invalid collection: vertex" << "\n";
      return;
    }
  
  int vertex_number = vertexHandle->size();
  reco::VertexCollection::const_iterator v = vertexHandle->begin();

  math::XYZPoint PVPoint(-999, -999, -999);
  if(vertex_number != 0)
    PVPoint = math::XYZPoint(v->position().x(), v->position().y(), v->position().z());
  
  PVPoint_=PVPoint;

  //-------------------------------
  //--- MET
  //-------------------------------
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  e.getByToken(thePfMETCollection_, pfMETCollection);
  if ( !pfMETCollection.isValid() )    
    {
      edm::LogError ("DQMClientExample") << "invalid collection: MET" << "\n";
      return;
    }
  //-------------------------------
  //--- Electrons
  //-------------------------------
  edm::Handle<reco::GsfElectronCollection> electronCollection;
  e.getByToken(theElectronCollection_, electronCollection);
  if ( !electronCollection.isValid() )
    {
      edm::LogError ("DQMClientExample") << "invalid collection: electrons" << "\n";
      return;
    }

  float nEle=0;
  int posEle=0, negEle=0;
  const reco::GsfElectron* ele1 = NULL;
  const reco::GsfElectron* ele2 = NULL;
  for (reco::GsfElectronCollection::const_iterator recoElectron=electronCollection->begin(); recoElectron!=electronCollection->end(); ++recoElectron)
    {
      //decreasing pT
      if( MediumEle(e,eSetup,*recoElectron) )
	{
	  if(!ele1 && recoElectron->pt() > ptThrL1_)
	    ele1 = &(*recoElectron);
	  
	  else if(!ele2 && recoElectron->pt() > ptThrL2_)
	    ele2 = &(*recoElectron);

	}
      
      if(recoElectron->charge()==1)
	posEle++;
      else if(recoElectron->charge()==-1)
	negEle++;

    } // end of loop over electrons
  
  nEle = posEle+negEle;
  
  //-------------------------------
  //--- Jets
  //-------------------------------
  edm::Handle<reco::CaloJetCollection> caloJetCollection;
  e.getByToken (theCaloJetCollection_,caloJetCollection);
  if ( !caloJetCollection.isValid() ) 
    {
      edm::LogError ("DQMClientExample") << "invalid collection: jets" << "\n";
      return;
    }

  int   nJet = 0;
  const reco::CaloJet* jet1 = NULL;
  const reco::CaloJet* jet2 = NULL;
  
  for (reco::CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin(); i_calojet != caloJetCollection->end(); ++i_calojet) 
    {
      //remove jet-ele matching
      if(ele1)
	if (Distance(*i_calojet,*ele1) < 0.3) continue;
      
      if(ele2)
	if (Distance(*i_calojet,*ele2) < 0.3) continue;
      
      if (i_calojet->pt() < ptThrJet_) continue;

      nJet++;
      
      if (!jet1) 
	jet1 = &(*i_calojet);
      
      else if (!jet2)
	jet2 = &(*i_calojet);
    }
  
  // ---------------------------
  // ---- Analyze Trigger Event
  // ---------------------------

  //check what is in the menu
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_,hltresults);
  
  if(!hltresults.isValid())
    {
      edm::LogError ("DQMClientExample") << "invalid collection: TriggerResults" << "\n";
      return;
    }
  
  bool hasFired = false;
  const edm::TriggerNames& trigNames = e.triggerNames(*hltresults);
  unsigned int numTriggers = trigNames.size();
  
  for( unsigned int hltIndex=0; hltIndex<numTriggers; ++hltIndex )
    {
      if (trigNames.triggerName(hltIndex)==triggerPath_ &&  hltresults->wasrun(hltIndex) &&  hltresults->accept(hltIndex))
	hasFired = true;
    }
  


  //access the trigger event
  edm::Handle<trigger::TriggerEvent> triggerEvent;
  e.getByToken(triggerEvent_, triggerEvent);
  if( triggerEvent.failedToGet() )
    {
      edm::LogError ("DQMClientExample") << "invalid collection: TriggerEvent" << "\n";
      return;
    }


  reco::Particle* ele1_HLT = NULL;
  int nEle_HLT = 0;

  size_t filterIndex = triggerEvent->filterIndex( triggerFilter_ );
  trigger::TriggerObjectCollection triggerObjects = triggerEvent->getObjects();
  if( !(filterIndex >= triggerEvent->sizeFilters()) )
    {
      const trigger::Keys& keys = triggerEvent->filterKeys( filterIndex );
      std::vector<reco::Particle> triggeredEle;
      
      for( size_t j = 0; j < keys.size(); ++j ) 
	{
	  trigger::TriggerObject foundObject = triggerObjects[keys[j]];
	  if( abs( foundObject.particle().pdgId() ) != 11 )  continue; //make sure that it is an electron
	  
	  triggeredEle.push_back( foundObject.particle() );
	  ++nEle_HLT;
	}
      
      if( triggeredEle.size() >= 1 ) 
	ele1_HLT = &(triggeredEle.at(0));
    }

  //-------------------------------
  //--- Fill the histos
  //-------------------------------

  //vertex
  h_vertex_number -> Fill( vertex_number );

  //met
  h_pfMet -> Fill( pfMETCollection->begin()->et() );

  //multiplicities
  h_eMultiplicity->Fill(nEle);       
  h_jMultiplicity->Fill(nJet);
  h_eMultiplicity_HLT->Fill(nEle_HLT);

  //leading not matched
  if(ele1)
    {
      h_ePt_leading->Fill(ele1->pt());
      h_eEta_leading->Fill(ele1->eta());
      h_ePhi_leading->Fill(ele1->phi());
    }
  if(ele1_HLT)
    {
      h_ePt_leading_HLT->Fill(ele1_HLT->pt());
      h_eEta_leading_HLT->Fill(ele1_HLT->eta());
      h_ePhi_leading_HLT->Fill(ele1_HLT->phi());
    }
  //leading Jet
  if(jet1)
    {
      h_jPt_leading->Fill(jet1->pt());
      h_jEta_leading->Fill(jet1->eta());
      h_jPhi_leading->Fill(jet1->phi());
    }


  //fill only when the trigger candidate mathes with the reco one
  if( ele1 && ele1_HLT && deltaR(*ele1_HLT,*ele1) < 0.3 && hasFired==true )
    {
      h_ePt_leading_matched->Fill(ele1->pt());
      h_eEta_leading_matched->Fill(ele1->eta());
      h_ePhi_leading_matched->Fill(ele1->phi());
      
      h_ePt_leading_HLT_matched->Fill(ele1_HLT->pt());
      h_eEta_leading_HLT_matched->Fill(ele1_HLT->eta());
      h_ePhi_leading_HLT_matched->Fill(ele1_HLT->phi());

      h_ePt_diff->Fill(ele1->pt()-ele1_HLT->pt());
    }
}
//
// -------------------------------------- endLuminosityBlock --------------------------------------------
//
void DQMExample_Step1::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) 
{
  //bool histogramRecordExist = false;
  bool histogramPropsRecordExist = false;

  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::endLuminosityBlock" << std::endl;
  //get the data from the histograms and fill the DB table

  coral::ISchema& schema = m_session->nominalSchema();
 /* m_session->transaction().start(true);
  coral::IQuery* queryHistogram = schema.tableHandle( "HISTOGRAM" ).newQuery();
  queryHistogram->addToOutputList( "NAME" );
  queryHistogram->addToOutputList( "PATH" );
  std::string condition = "NAME = \"" + h_vertex_number->getName() + "\" AND PATH = \"" + h_vertex_number->getPathname() + "\"";
  coral::AttributeList conditionData;
  queryHistogram->setCondition( condition, conditionData );
  queryHistogram->setMemoryCacheSize( 5 );
  coral::ICursor& cursor1 = queryHistogram->execute();
  int numberOfRows = 0;
  while(cursor1.next())
  {
    cursor1.currentRow().toOutputStream( std::cout ) << std::endl;
    ++numberOfRows;
  }
  delete queryHistogram;
  if ( numberOfRows == 1 )
  {
    histogramRecordExist = true;
  }
  m_session->transaction().commit();

  m_session->transaction().start(false);
  std::cout << "After query" << std::endl;
  if(!histogramRecordExist)
  {
      coral::ITableDataEditor& editor = m_session->nominalSchema().tableHandle( "HISTOGRAM" ).dataEditor();
      coral::AttributeList insertData;
      insertData.extend< std::string >( "NAME" );
      insertData.extend< std::string >( "PATH" );
      insertData.extend< unsigned int >( "TIMESTAMP" );
      insertData.extend< std::string >( "TITLE" );

      insertData[ "NAME" ].data< std::string >() = h_vertex_number->getName();
      insertData[ "PATH" ].data< std::string >() = h_vertex_number->getPathname();
      insertData[ "TIMESTAMP" ].data< unsigned int >() = std::time(nullptr);
      insertData[ "TITLE" ].data< std::string >() = h_vertex_number->getFullname();
      editor.insertRow( insertData );
  }
  m_session->transaction().commit();
*/
  m_session->transaction().start(true);
  coral::IQuery* queryHistogramProps = schema.tableHandle( "HISTOGRAM_PROPS" ).newQuery();
  queryHistogramProps->addToOutputList( "NAME" );
  queryHistogramProps->addToOutputList( "PATH" );
  queryHistogramProps->addToOutputList( "RUN_NUMBER" );

  std::string condition = "NAME = \"" + h_vertex_number->getName() + "\" AND PATH = \"" + h_vertex_number->getPathname() + "\"" + " AND RUN_NUMBER = \"" + std::to_string(lumiSeg.run()) + "\"";
  coral::AttributeList conditionData2;
  queryHistogramProps->setCondition( condition, conditionData2 );
  queryHistogramProps->setMemoryCacheSize( 5 );
  coral::ICursor& cursor2 = queryHistogramProps->execute();
  int numberOfRows = 0;
  while(cursor2.next())
  {
    cursor2.currentRow().toOutputStream( std::cout ) << std::endl;
    ++numberOfRows;
  }
  delete queryHistogramProps;
  if ( numberOfRows == 1 )
  {
    histogramPropsRecordExist = true;
  }
  m_session->transaction().commit();

  m_session->transaction().start(false);
  if(!histogramPropsRecordExist)
  {
      coral::ITableDataEditor& editor = m_session->nominalSchema().tableHandle( "HISTOGRAM_PROPS" ).dataEditor();
      coral::AttributeList insertData;
      insertData.extend< std::string >( "NAME" );
      insertData.extend< std::string >( "PATH" );
      insertData.extend< unsigned int >( "RUN_NUMBER" );
      insertData.extend< int >( "X_BINS" );
      insertData.extend< double >( "X_LOW" );
      insertData.extend< double >( "X_UP" );
      insertData.extend< int >( "Y_BINS" );
      insertData.extend< double >( "Y_LOW" );
      insertData.extend< double >( "Y_UP" );
      insertData.extend< int >( "Z_BINS" );
      insertData.extend< double >( "Z_LOW" );
      insertData.extend< double >( "Z_UP" );

      insertData[ "NAME" ].data< std::string >() = h_vertex_number->getName();
      insertData[ "PATH" ].data< std::string >() = h_vertex_number->getPathname();
      insertData[ "RUN_NUMBER" ].data< unsigned int >() = lumiSeg.run();
      insertData[ "X_BINS" ].data< int >() = h_vertex_number->getNbinsX(); //or h_vertex_number->getTH1()->GetNbinsX() ?
      insertData[ "X_LOW" ].data< double >() = h_vertex_number->getTH1()->GetXaxis()->GetXmin();
      insertData[ "X_UP" ].data< double >() = h_vertex_number->getTH1()->GetXaxis()->GetXmax();
      insertData[ "Y_BINS" ].data< int >() = 0; //h_vertex_number->getNbinsY();
      insertData[ "Y_LOW" ].data< double >() = 0.; //h_vertex_number->getTH1()->GetYaxis()->GetXMin();
      insertData[ "Y_UP" ].data< double >() = 0.; //h_vertex_number->getTH1()->GetYaxis()->GetXMax();
      insertData[ "Z_BINS" ].data< int >() = 0; //h_vertex_number->getNbinsZ();
      insertData[ "Z_LOW" ].data< double >() = 0.; //h_vertex_number->getTH1()->GetZaxis()->GetXMin();
      insertData[ "Z_UP" ].data< double >() = 0.; //h_vertex_number->getTH1()->GetZaxis()->GetXMax();
      editor.insertRow( insertData );
  }
  m_session->transaction().commit();

  m_session->transaction().start(false);

  coral::ITableDataEditor& editor = m_session->nominalSchema().tableHandle( "HISTOGRAM_VALUES" ).dataEditor();
  coral::AttributeList insertData;
  insertData.extend< std::string >( "NAME" );
  insertData.extend< std::string >( "PATH" );
  insertData.extend< unsigned int >( "RUN_NUMBER" );
  insertData.extend< unsigned int >( "LUMISECTION" );
  insertData.extend< double >( "ENTRIES" );
  insertData.extend< double >( "X_MEAN" );
  insertData.extend< double >( "X_MEAN_ERROR" );
  insertData.extend< double >( "X_RMS" );
  insertData.extend< double >( "X_RMS_ERROR" );
  insertData.extend< double >( "X_UNDERFLOW");
  insertData.extend< double >( "X_OVERFLOW" );
  insertData.extend< double >( "Y_MEAN" );
  insertData.extend< double >( "Y_MEAN_ERROR" );
  insertData.extend< double >( "Y_RMS" );
  insertData.extend< double >( "Y_RMS_ERROR" );
  insertData.extend< double >( "Y_UNDERFLOW");
  insertData.extend< double >( "Y_OVERFLOW" );
  insertData.extend< double >( "Z_MEAN" );
  insertData.extend< double >( "Z_MEAN_ERROR" );
  insertData.extend< double >( "Z_RMS" );
  insertData.extend< double >( "Z_RMS_ERROR" );
  insertData.extend< double >( "Z_UNDERFLOW");
  insertData.extend< double >( "Z_OVERFLOW" );

  insertData[ "NAME" ].data< std::string >() = h_vertex_number->getName();
  insertData[ "PATH" ].data< std::string >() = h_vertex_number->getPathname();
  insertData[ "RUN_NUMBER" ].data< unsigned int >() = lumiSeg.run();
  insertData[ "LUMISECTION" ].data< unsigned int >() = lumiSeg.luminosityBlock();
  insertData[ "ENTRIES" ].data< double >() = h_vertex_number->getEntries(); //or h_vertex_number->getTH1()->GetEntries() ?
  insertData[ "X_MEAN" ].data< double >() = h_vertex_number->getTH1()->GetMean();
  insertData[ "X_MEAN_ERROR" ].data< double >() = h_vertex_number->getTH1()->GetMeanError();
  insertData[ "X_RMS" ].data< double >() = h_vertex_number->getTH1()->GetRMS();
  insertData[ "X_RMS_ERROR" ].data< double >() = h_vertex_number->getTH1()->GetRMSError();
  insertData[ "X_UNDERFLOW" ].data< double >() = h_vertex_number->getTH1()->GetBinContent( 0 );
  insertData[ "X_OVERFLOW" ].data< double >() = h_vertex_number->getTH1()->GetBinContent( h_vertex_number->getTH1()->GetNbinsX() + 1 );
  insertData[ "Y_MEAN" ].data< double >() = h_vertex_number->getTH1()->GetMean( 2 );
  insertData[ "Y_MEAN_ERROR" ].data< double >() = h_vertex_number->getTH1()->GetMeanError( 2 );
  insertData[ "Y_RMS" ].data< double >() = h_vertex_number->getTH1()->GetRMS( 2 );
  insertData[ "Y_RMS_ERROR" ].data< double >() = h_vertex_number->getTH1()->GetRMSError( 2 );
  insertData[ "Y_UNDERFLOW" ].data< double >() = 0.;
  insertData[ "Y_OVERFLOW" ].data< double >() = 0.;
  insertData[ "Z_MEAN" ].data< double >() = h_vertex_number->getTH1()->GetMean( 3 );
  insertData[ "Z_MEAN_ERROR" ].data< double >() = h_vertex_number->getTH1()->GetMeanError( 3 );
  insertData[ "Z_RMS" ].data< double >() = h_vertex_number->getTH1()->GetRMS( 3 );
  insertData[ "Z_RMS_ERROR" ].data< double >() = h_vertex_number->getTH1()->GetRMSError( 3 );
  insertData[ "Z_UNDERFLOW" ].data< double >() = 0.;
  insertData[ "Z_OVERFLOW" ].data< double >() = 0.;
  editor.insertRow( insertData );
  m_session->transaction().commit();
}


//
// -------------------------------------- endRun --------------------------------------------
//
void DQMExample_Step1::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::endRun" << std::endl;

  //no more data to process:
  //close DB session
  m_session.reset();
}


//
// -------------------------------------- book histograms --------------------------------------------
//
void DQMExample_Step1::bookHistos(DQMStore::IBooker & ibooker_)
{
  ibooker_.cd();
  ibooker_.setCurrentFolder("Physics/TopTest");

  h_vertex_number = ibooker_.book1D("Vertex_number", "Number of event vertices in collection", 40,-0.5,   39.5 );
  h_pfMet        = ibooker_.book1D("pfMet",        "Pf Missing E_{T}; GeV"          , 20,  0.0 , 100);
  h_eMultiplicity = ibooker_.book1D("NElectrons","# of electrons per event",10,0.,10.);


  h_ePt_leading_matched = ibooker_.book1D("ElePt_leading_matched","Pt of leading electron",50,0.,100.);
  h_eEta_leading_matched = ibooker_.book1D("EleEta_leading_matched","Eta of leading electron",50,-5.,5.);
  h_ePhi_leading_matched = ibooker_.book1D("ElePhi_leading_matched","Phi of leading electron",50,-3.5,3.5);

  /// by run
  h_ePt_leading = ibooker_.book1D("ElePt_leading","Pt of leading electron",50,0.,100.);
  h_eEta_leading = ibooker_.book1D("EleEta_leading","Eta of leading electron",50,-5.,5.);
  h_ePhi_leading = ibooker_.book1D("ElePhi_leading","Phi of leading electron",50,-3.5,3.5);
  ///

  h_jMultiplicity = ibooker_.book1D("NJets","# of electrons per event",10,0.,10.);
  h_jPt_leading = ibooker_.book1D("JetPt_leading","Pt of leading Jet",150,0.,300.);
  h_jEta_leading = ibooker_.book1D("JetEta_leading","Eta of leading Jet",50,-5.,5.);
  h_jPhi_leading = ibooker_.book1D("JetPhi_leading","Phi of leading Jet",50,-3.5,3.5);

  h_eMultiplicity_HLT = ibooker_.book1D("NElectrons_HLT","# of electrons per event @HLT",10,0.,10.);
  h_ePt_leading_HLT = ibooker_.book1D("ElePt_leading_HLT","Pt of leading electron @HLT",50,0.,100.);
  h_eEta_leading_HLT = ibooker_.book1D("EleEta_leading_HLT","Eta of leading electron @HLT",50,-5.,5.);
  h_ePhi_leading_HLT = ibooker_.book1D("ElePhi_leading_HLT","Phi of leading electron @HLT",50,-3.5,3.5);

  h_ePt_leading_HLT_matched = ibooker_.book1D("ElePt_leading_HLT_matched","Pt of leading electron @HLT",50,0.,100.);
  h_eEta_leading_HLT_matched = ibooker_.book1D("EleEta_leading_HLT_matched","Eta of leading electron @HLT",50,-5.,5.);
  h_ePhi_leading_HLT_matched = ibooker_.book1D("ElePhi_leading_HLT_matched","Phi of leading electron @HLT",50,-3.5,3.5);

  h_ePt_diff = ibooker_.book1D("ElePt_diff_matched","pT(RECO) - pT(HLT) for mathed candidates",100,-10,10.);

  ibooker_.cd();  

}


//
// -------------------------------------- functions --------------------------------------------
//
double DQMExample_Step1::Distance( const reco::Candidate & c1, const reco::Candidate & c2 ) {
        return  deltaR(c1,c2);
}

double DQMExample_Step1::DistancePhi( const reco::Candidate & c1, const reco::Candidate & c2 ) {
        return  deltaPhi(c1.p4().phi(),c2.p4().phi());
}

// This always returns only a positive deltaPhi
double DQMExample_Step1::calcDeltaPhi(double phi1, double phi2) {
  double deltaPhi = phi1 - phi2;
  if (deltaPhi < 0) deltaPhi = -deltaPhi;
  if (deltaPhi > 3.1415926) {
    deltaPhi = 2 * 3.1415926 - deltaPhi;
  }
  return deltaPhi;
}

//
// -------------------------------------- electronID --------------------------------------------
//
bool DQMExample_Step1::MediumEle (const edm::Event & iEvent, const edm::EventSetup & iESetup, const reco::GsfElectron & electron)
{
    
  //********* CONVERSION TOOLS
  edm::Handle<reco::ConversionCollection> conversions_h;
  iEvent.getByToken(theConversionCollection_, conversions_h);
  
  bool isMediumEle = false; 
  
  float pt = electron.pt();
  float eta = electron.eta();
    
  int isEB            = electron.isEB();
  float sigmaIetaIeta = electron.sigmaIetaIeta();
  float DetaIn        = electron.deltaEtaSuperClusterTrackAtVtx();
  float DphiIn        = electron.deltaPhiSuperClusterTrackAtVtx();
  float HOverE        = electron.hadronicOverEm();
  float ooemoop       = (1.0/electron.ecalEnergy() - electron.eSuperClusterOverP()/electron.ecalEnergy());
  
  int mishits             = electron.gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS);
  int nAmbiguousGsfTracks = electron.ambiguousGsfTracksSize();
  
  reco::GsfTrackRef eleTrack  = electron.gsfTrack() ;
  float dxy           = eleTrack->dxy(PVPoint_);  
  float dz            = eleTrack->dz (PVPoint_);
  
  edm::Handle<reco::BeamSpot> BSHandle;
  iEvent.getByToken(theBSCollection_, BSHandle);
  const reco::BeamSpot BS = *BSHandle;
  
  bool isConverted = ConversionTools::hasMatchedConversion(electron, conversions_h, BS.position());
  
  // default
  if(  (pt > 12.) && (fabs(eta) < 2.5) &&
       ( ( (isEB == 1) && (fabs(DetaIn)  < 0.004) ) || ( (isEB == 0) && (fabs(DetaIn)  < 0.007) ) ) &&
       ( ( (isEB == 1) && (fabs(DphiIn)  < 0.060) ) || ( (isEB == 0) && (fabs(DphiIn)  < 0.030) ) ) &&
       ( ( (isEB == 1) && (sigmaIetaIeta < 0.010) ) || ( (isEB == 0) && (sigmaIetaIeta < 0.030) ) ) &&
       ( ( (isEB == 1) && (HOverE        < 0.120) ) || ( (isEB == 0) && (HOverE        < 0.100) ) ) &&
       ( ( (isEB == 1) && (fabs(ooemoop) < 0.050) ) || ( (isEB == 0) && (fabs(ooemoop) < 0.050) ) ) &&
       ( ( (isEB == 1) && (fabs(dxy)     < 0.020) ) || ( (isEB == 0) && (fabs(dxy)     < 0.020) ) ) &&
       ( ( (isEB == 1) && (fabs(dz)      < 0.100) ) || ( (isEB == 0) && (fabs(dz)      < 0.100) ) ) &&
       ( ( (isEB == 1) && (!isConverted) ) || ( (isEB == 0) && (!isConverted) ) ) &&
       ( mishits == 0 ) &&
       ( nAmbiguousGsfTracks == 0 )      
       )
    isMediumEle=true;
  
  return isMediumEle;
}

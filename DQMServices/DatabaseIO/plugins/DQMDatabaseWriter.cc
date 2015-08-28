#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "DQMDatabaseWriter.h"

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


//
// -------------------------------------- Constructor --------------------------------------------
//
DQMDatabaseWriter::DQMDatabaseWriter(const edm::ParameterSet& ps) : m_connectionService(), m_session(), m_connectionString( "" )
{
  edm::LogInfo("DQMDatabaseWriter") <<  "Constructor  DQMDatabaseWriter::DQMDatabaseWriter " << std::endl;

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
DQMDatabaseWriter::~DQMDatabaseWriter()
{
  edm::LogInfo("DQMDatabaseWriter") <<  "Destructor DQMDatabaseWriter::~DQMDatabaseWriter " << std::endl;
}

//
// -------------------------------------- beginJob --------------------------------------------
//
void DQMDatabaseWriter::initDatabase()
{
  edm::LogInfo("DQMDatabaseWriter") <<  "DQMDatabaseWriter::initDatabase " << std::endl;

  m_session.reset( m_connectionService.connect( m_connectionString, coral::Update ) );
  //TODO: do not run in production!
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
    //columnsForForeignKey3.push_back( "RUN_NUMBER" );

    table3.createForeignKey( "table2_FK", columnsForForeignKey3, "HISTOGRAM", columnsForPrimaryKey1 );

    schema.createTable( table3 );
  }
  m_session->transaction().commit();

  edm::LogInfo("DQMExample_Step1") <<  "DQMExample_Step1::endLuminosityBlock" << std::endl;
  m_session->transaction().start(false);

}

//
// -------------------------------------- dqmDbLumiDrop --------------------------------------------
//
void DQMDatabaseWriter::dqmDbLumiDrop(std::vector <MonitorElement *> & histograms, int lumisection, int run)
{
  edm::LogInfo("DQMDatabaseWriter") <<  "DQMDatabaseWriter::dqmDbLumiDrop " << std::endl;

  bool histogramPropsRecordExist;
  bool histogramValuesRecordExist;

  coral::ISchema& schema = m_session->nominalSchema();

  for (MonitorElement * histogram : histograms)
  {
    {
      m_session->transaction().start( false );
      coral::IQuery* queryHistogramProps = schema.tableHandle( "HISTOGRAM" ).newQuery();
      queryHistogramProps->addToOutputList( "NAME" );
      queryHistogramProps->addToOutputList( "PATH" );

      std::string condition = "NAME = \"" + histogram->getName() + "\" AND PATH = \"" + histogram->getPathname() + "\"";
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
      if ( numberOfRows != 1 )
      {
        coral::ITableDataEditor& editor = m_session->nominalSchema().tableHandle( "HISTOGRAM" ).dataEditor();
        coral::AttributeList insertData;
        insertData.extend< std::string >( "NAME" );
        insertData.extend< std::string >( "PATH" );
        insertData.extend< unsigned int >( "TIMESTAMP" );
        insertData.extend< std::string >( "TITLE" );

        insertData[ "NAME" ].data< std::string >() = histogram->getName();
        insertData[ "PATH" ].data< std::string >() = histogram->getPathname();
        insertData[ "TIMESTAMP" ].data< unsigned int >() = std::time(nullptr);
        insertData[ "TITLE" ].data< std::string >() = histogram->getFullname();
        editor.insertRow( insertData );
      }
      m_session->transaction().commit();
      m_session->transaction().start(false);
    }

    histogramPropsRecordExist = false;
    {
      coral::IQuery* queryHistogramProps = schema.tableHandle( "HISTOGRAM_PROPS" ).newQuery();
      queryHistogramProps->addToOutputList( "NAME" );
      queryHistogramProps->addToOutputList( "PATH" );
      queryHistogramProps->addToOutputList( "RUN_NUMBER" );

      std::string condition = "NAME = \"" + histogram->getName() + "\" AND PATH = \"" + histogram->getPathname() + "\"" + " AND RUN_NUMBER = \"" + std::to_string(run) + "\"";
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
    }

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

        insertData[ "NAME" ].data< std::string >() = histogram->getName();
        insertData[ "PATH" ].data< std::string >() = histogram->getPathname();
        insertData[ "RUN_NUMBER" ].data< unsigned int >() = run;
        insertData[ "X_BINS" ].data< int >() = histogram->getNbinsX(); //or histogram->getTH1()->GetNbinsX() ?
        insertData[ "X_LOW" ].data< double >() = histogram->getTH1()->GetXaxis()->GetXmin();
        insertData[ "X_UP" ].data< double >() = histogram->getTH1()->GetXaxis()->GetXmax();
        insertData[ "Y_BINS" ].data< int >() = 0; //histogram->getNbinsY();
        insertData[ "Y_LOW" ].data< double >() = 0.; //histogram->getTH1()->GetYaxis()->GetXMin();
        insertData[ "Y_UP" ].data< double >() = 0.; //histogram->getTH1()->GetYaxis()->GetXMax();
        insertData[ "Z_BINS" ].data< int >() = 0; //histogram->getNbinsZ();
        insertData[ "Z_LOW" ].data< double >() = 0.; //histogram->getTH1()->GetZaxis()->GetXMin();
        insertData[ "Z_UP" ].data< double >() = 0.; //histogram->getTH1()->GetZaxis()->GetXMax();
        editor.insertRow( insertData );
    }
    m_session->transaction().commit();
    m_session->transaction().start(false);

    histogramValuesRecordExist = false;
    {
      coral::IQuery* queryHistogramValues = schema.tableHandle( "HISTOGRAM_VALUES" ).newQuery();
      queryHistogramValues->addToOutputList( "NAME" );
      queryHistogramValues->addToOutputList( "PATH" );
      queryHistogramValues->addToOutputList( "RUN_NUMBER" );
      queryHistogramValues->addToOutputList( "LUMISECTION" );


      std::string condition = "NAME = \"" + histogram->getName() + "\" AND PATH = \"" + histogram->getPathname() + "\"" + " AND RUN_NUMBER = \"" + std::to_string(run) + "\""  + " AND LUMISECTION = \"" + std::to_string(lumisection) + "\"";
      coral::AttributeList conditionData2;
      queryHistogramValues->setCondition( condition, conditionData2 );
      queryHistogramValues->setMemoryCacheSize( 5 );
      coral::ICursor& cursor = queryHistogramValues->execute();
      int numberOfRows = 0;
      while(cursor.next())
      {
        cursor.currentRow().toOutputStream( std::cout ) << std::endl;
        ++numberOfRows;
      }
      delete queryHistogramValues;
      if ( numberOfRows == 1 )
      {
        histogramValuesRecordExist = true;
      }
      m_session->transaction().commit();
      m_session->transaction().start(false);
    }

    if(!histogramValuesRecordExist)
    {
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

      insertData[ "NAME" ].data< std::string >() = histogram->getName();
      insertData[ "PATH" ].data< std::string >() = histogram->getPathname();
      insertData[ "RUN_NUMBER" ].data< unsigned int >() = run;
      insertData[ "LUMISECTION" ].data< unsigned int >() = lumisection;
      insertData[ "ENTRIES" ].data< double >() = histogram->getEntries(); //or histogram->getTH1()->GetEntries() ?
      insertData[ "X_MEAN" ].data< double >() = histogram->getTH1()->GetMean();
      insertData[ "X_MEAN_ERROR" ].data< double >() = histogram->getTH1()->GetMeanError();
      insertData[ "X_RMS" ].data< double >() = histogram->getTH1()->GetRMS();
      insertData[ "X_RMS_ERROR" ].data< double >() = histogram->getTH1()->GetRMSError();
      insertData[ "X_UNDERFLOW" ].data< double >() = histogram->getTH1()->GetBinContent( 0 );
      insertData[ "X_OVERFLOW" ].data< double >() = histogram->getTH1()->GetBinContent( histogram->getTH1()->GetNbinsX() + 1 );
      insertData[ "Y_MEAN" ].data< double >() = histogram->getTH1()->GetMean( 2 );
      insertData[ "Y_MEAN_ERROR" ].data< double >() = histogram->getTH1()->GetMeanError( 2 );
      insertData[ "Y_RMS" ].data< double >() = histogram->getTH1()->GetRMS( 2 );
      insertData[ "Y_RMS_ERROR" ].data< double >() = histogram->getTH1()->GetRMSError( 2 );
      insertData[ "Y_UNDERFLOW" ].data< double >() = 0.;
      insertData[ "Y_OVERFLOW" ].data< double >() = 0.;
      insertData[ "Z_MEAN" ].data< double >() = histogram->getTH1()->GetMean( 3 );
      insertData[ "Z_MEAN_ERROR" ].data< double >() = histogram->getTH1()->GetMeanError( 3 );
      insertData[ "Z_RMS" ].data< double >() = histogram->getTH1()->GetRMS( 3 );
      insertData[ "Z_RMS_ERROR" ].data< double >() = histogram->getTH1()->GetRMSError( 3 );
      insertData[ "Z_UNDERFLOW" ].data< double >() = 0.;
      insertData[ "Z_OVERFLOW" ].data< double >() = 0.;
      editor.insertRow( insertData );
      m_session->transaction().commit();
    }
  }

  processLumi(run);
}

//
// -------------------------------------- dqmDbRunInitialize --------------------------------------------
//
void DQMDatabaseWriter::dqmDbRunInitialize(std::vector < std::pair <MonitorElement *, HistogramValues> > & histograms)
{
  histogramsPerRun = histograms;
}

//
// -------------------------------------- processLumi --------------------------------------------
//
void DQMDatabaseWriter::processLumi(int run)
{
  coral::ISchema& schema = m_session->nominalSchema();

  for (std::vector<std::pair <MonitorElement *, HistogramValues> >::iterator it = histogramsPerRun.begin() ; it != histogramsPerRun.end(); ++it)
  {
    if((*it).second.test_entries != 0)
    {
      (*it).second.test_x_mean = ((*it).second.test_x_mean*(*it).second.test_entries + (*it).first->getTH1()->GetMean()*(*it).first->getEntries())/((*it).second.test_entries+(*it).first->getEntries());
      (*it).second.test_x_mean_error = ((*it).second.test_x_mean_error*(*it).second.test_entries + (*it).first->getTH1()->GetMeanError()*(*it).first->getEntries())/((*it).second.test_entries+(*it).first->getEntries());
      (*it).second.test_x_rms = sqrt(((*it).second.test_entries*pow((*it).second.test_x_rms,2.0) + (*it).first->getEntries()*pow((*it).first->getTH1()->GetRMS(),2.0))/((*it).second.test_entries+(*it).first->getEntries()));
      (*it).second.test_x_rms_error = sqrt(((*it).second.test_entries*pow((*it).second.test_x_rms_error,2.0) + (*it).first->getEntries()*pow((*it).first->getTH1()->GetRMSError(),2.0))/((*it).second.test_entries+(*it).first->getEntries()));
    }
    else
    {
      (*it).second.test_run = run;
      (*it).second.test_x_mean = (*it).first->getTH1()->GetMean();
      (*it).second.test_x_mean_error = (*it).first->getTH1()->GetMeanError();
      (*it).second.test_x_rms = (*it).first->getTH1()->GetRMS();
      (*it).second.test_x_rms_error = (*it).first->getTH1()->GetRMSError();
    }
    (*it).second.test_entries += (*it).first->getEntries();


  }
}

//
// -------------------------------------- dqmDbRunDrop --------------------------------------------
//
void DQMDatabaseWriter::dqmDbRunDrop()
{
  bool histogramValuesRecordExist;
  static coral::ISchema& schema = m_session->nominalSchema();

  for (std::vector<std::pair <MonitorElement *, HistogramValues> >::iterator it = histogramsPerRun.begin() ; it != histogramsPerRun.end(); ++it)
  {
    {
      m_session->transaction().start( false );
      coral::IQuery* queryHistogramProps = schema.tableHandle( "HISTOGRAM" ).newQuery();

      queryHistogramProps->addToOutputList( "NAME" );
      queryHistogramProps->addToOutputList( "PATH" );

      std::string condition = "NAME = \"" + (*it).first->getName() + "\" AND PATH = \"" + (*it).first->getPathname() + "\"";
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
      if ( numberOfRows != 1 )
      {
        coral::ITableDataEditor& editor = m_session->nominalSchema().tableHandle( "HISTOGRAM" ).dataEditor();
        coral::AttributeList insertData;
        insertData.extend< std::string >( "NAME" );
        insertData.extend< std::string >( "PATH" );
        insertData.extend< unsigned int >( "TIMESTAMP" );
        insertData.extend< std::string >( "TITLE" );

        insertData[ "NAME" ].data< std::string >() = (*it).first->getName();
        insertData[ "PATH" ].data< std::string >() = (*it).first->getPathname();
        insertData[ "TIMESTAMP" ].data< unsigned int >() = std::time(nullptr);
        insertData[ "TITLE" ].data< std::string >() = (*it).first->getFullname();
        editor.insertRow( insertData );
      }
      m_session->transaction().commit();
      m_session->transaction().start(false);
    }


    bool histogramPropsRecordExist = false;
    {
      coral::IQuery* queryHistogramProps = schema.tableHandle( "HISTOGRAM_PROPS" ).newQuery();
      queryHistogramProps->addToOutputList( "NAME" );
      queryHistogramProps->addToOutputList( "PATH" );
      queryHistogramProps->addToOutputList( "RUN_NUMBER" );

      std::string condition = "NAME = \"" + (*it).first->getName() + "\" AND PATH = \"" + (*it).first->getPathname() + "\"" + " AND RUN_NUMBER = \"" + std::to_string(it->second.test_run) + "\"";
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
    }
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

        insertData[ "NAME" ].data< std::string >() = (*it).first->getName();
        insertData[ "PATH" ].data< std::string >() = (*it).first->getPathname();
        insertData[ "RUN_NUMBER" ].data< unsigned int >() = it->second.test_run;
        insertData[ "X_BINS" ].data< int >() = (*it).first->getNbinsX(); //or histogram->getTH1()->GetNbinsX() ?
        insertData[ "X_LOW" ].data< double >() = (*it).first->getTH1()->GetXaxis()->GetXmin();
        insertData[ "X_UP" ].data< double >() = (*it).first->getTH1()->GetXaxis()->GetXmax();
        insertData[ "Y_BINS" ].data< int >() = 0; //histogram->getNbinsY();
        insertData[ "Y_LOW" ].data< double >() = 0.; //histogram->getTH1()->GetYaxis()->GetXMin();
        insertData[ "Y_UP" ].data< double >() = 0.; //histogram->getTH1()->GetYaxis()->GetXMax();
        insertData[ "Z_BINS" ].data< int >() = 0; //histogram->getNbinsZ();
        insertData[ "Z_LOW" ].data< double >() = 0.; //histogram->getTH1()->GetZaxis()->GetXMin();
        insertData[ "Z_UP" ].data< double >() = 0.; //histogram->getTH1()->GetZaxis()->GetXMax();
        editor.insertRow( insertData );
    }
    m_session->transaction().commit();


    histogramValuesRecordExist = false;
    {
      m_session->transaction().start(true);
      coral::IQuery* queryHistogramValues = schema.tableHandle( "HISTOGRAM_VALUES" ).newQuery();
      queryHistogramValues->addToOutputList( "NAME" );
      queryHistogramValues->addToOutputList( "PATH" );
      queryHistogramValues->addToOutputList( "RUN_NUMBER" );
      queryHistogramValues->addToOutputList( "LUMISECTION" );

      std::string condition = "NAME = \"" + (*it).first->getName() + "\" AND PATH = \"" + (*it).first->getPathname() + "\"" + " AND RUN_NUMBER = \"" + std::to_string((*it).second.test_run) + "\"" + " AND LUMISECTION = \"" + std::to_string(0) + "\"";;
      coral::AttributeList conditionData2;
      queryHistogramValues->setCondition( condition, conditionData2 );
      queryHistogramValues->setMemoryCacheSize( 5 );
      coral::ICursor& cursor = queryHistogramValues->execute();
      int numberOfRows = 0;
      while(cursor.next())
      {
        cursor.currentRow().toOutputStream( std::cout ) << std::endl;
        ++numberOfRows;
      }
      delete queryHistogramValues;
      if ( numberOfRows == 1 )
      {
        histogramValuesRecordExist = true;
      }
      m_session->transaction().commit();
    }
    m_session->transaction().start(false);
    if(!histogramValuesRecordExist)
    {
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

      insertData[ "NAME" ].data< std::string >() = (*it).first->getName();
      insertData[ "PATH" ].data< std::string >() = (*it).first->getPathname();
      insertData[ "RUN_NUMBER" ].data< unsigned int >() = (*it).second.test_run;
      insertData[ "LUMISECTION" ].data< unsigned int >() = 0;

      insertData[ "ENTRIES" ].data< double >() = (*it).second.test_entries; //or h_ePt_leading->getTH1()->GetEntries() ?
      (*it).second.test_entries = 0;

      insertData[ "X_MEAN" ].data< double >() = (*it).second.test_x_mean;            //h_ePt_leading->getTH1()->GetMean();
      (*it).second.test_x_mean = 0;

      insertData[ "X_MEAN_ERROR" ].data< double >() = (*it).second.test_x_mean_error;      //h_ePt_leading->getTH1()->GetMeanError();
      (*it).second.test_x_mean_error = 0;

      insertData[ "X_RMS" ].data< double >() = (*it).second.test_x_rms;             //h_ePt_leading->getTH1()->GetRMS();
      (*it).second.test_x_rms = 0;

      insertData[ "X_RMS_ERROR" ].data< double >() = (*it).second.test_x_rms_error;      //h_ePt_leading->getTH1()->GetRMSError();
      (*it).second.test_x_rms_error = 0;

      insertData[ "X_UNDERFLOW" ].data< double >() = (*it).first->getTH1()->GetBinContent( 0 );
      insertData[ "X_OVERFLOW" ].data< double >() = (*it).first->getTH1()->GetBinContent( (*it).first->getTH1()->GetNbinsX() + 1 );
      insertData[ "Y_MEAN" ].data< double >() = (*it).first->getTH1()->GetMean( 2 );
      insertData[ "Y_MEAN_ERROR" ].data< double >() = (*it).first->getTH1()->GetMeanError( 2 );
      insertData[ "Y_RMS" ].data< double >() = (*it).first->getTH1()->GetRMS( 2 );
      insertData[ "Y_RMS_ERROR" ].data< double >() = (*it).first->getTH1()->GetRMSError( 2 );
      insertData[ "Y_UNDERFLOW" ].data< double >() = 0.;
      insertData[ "Y_OVERFLOW" ].data< double >() = 0.;
      insertData[ "Z_MEAN" ].data< double >() = (*it).first->getTH1()->GetMean( 3 );
      insertData[ "Z_MEAN_ERROR" ].data< double >() = (*it).first->getTH1()->GetMeanError( 3 );
      insertData[ "Z_RMS" ].data< double >() = (*it).first->getTH1()->GetRMS( 3 );
      insertData[ "Z_RMS_ERROR" ].data< double >() = (*it).first->getTH1()->GetRMSError( 3 );
      insertData[ "Z_UNDERFLOW" ].data< double >() = 0.;
      insertData[ "Z_OVERFLOW" ].data< double >() = 0.;
      editor.insertRow( insertData );
    }
    m_session->transaction().commit();
  }
  m_session.reset();
}

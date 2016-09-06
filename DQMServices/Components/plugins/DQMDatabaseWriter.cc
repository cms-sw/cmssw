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
    table1.insertColumn( "PATH", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );
    table1.insertColumn( "TIMESTAMP", coral::AttributeSpecification::typeNameForType<unsigned int>());
    table1.insertColumn( "TITLE", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );

    table1.setNotNullConstraint( "PATH" );
    table1.setNotNullConstraint( "TIMESTAMP" );
    table1.setNotNullConstraint( "TITLE" );

    std::vector<std::string> columnsForPrimaryKey1;
    columnsForPrimaryKey1.push_back( "PATH" );
    table1.setPrimaryKey( columnsForPrimaryKey1 );

    schema.createTable( table1 );

    // Create the second table
    coral::TableDescription table2;
    table2.setName( "HISTOGRAM_PROPS" );
    table2.insertColumn( "PATH", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );
    table2.insertColumn( "RUN_NUMBER", coral::AttributeSpecification::typeNameForType<int>() );
    table2.insertColumn( "DIMENSIONS", coral::AttributeSpecification::typeNameForType<int>() );
    table2.insertColumn( "X_AXIS", coral::AttributeSpecification::typeNameForType<std::string>() );
    table2.insertColumn( "Y_AXIS", coral::AttributeSpecification::typeNameForType<std::string>() );
    table2.insertColumn( "Z_AXIS", coral::AttributeSpecification::typeNameForType<std::string>() );

    table2.setNotNullConstraint( "PATH" );
    table2.setNotNullConstraint( "RUN_NUMBER" );
    table2.setNotNullConstraint( "DIMENSIONS" );
    table2.setNotNullConstraint( "X_AXIS" );

    std::vector<std::string> columnsForPrimaryKey2;
    columnsForPrimaryKey2.push_back( "PATH" );
    columnsForPrimaryKey2.push_back( "RUN_NUMBER" );
    table2.setPrimaryKey( columnsForPrimaryKey2 );

    std::vector<std::string> columnsForForeignKey2;

    columnsForForeignKey2.push_back( "PATH" );

    table2.createForeignKey( "table2_FK", columnsForForeignKey2, "HISTOGRAM", columnsForPrimaryKey1 );

    schema.createTable( table2 );

    // Create the third table
    coral::TableDescription table3;
    table3.setName( "HISTOGRAM_VALUES" );
    table3.insertColumn( "PATH", coral::AttributeSpecification::typeNameForType<std::string>(), columnSize, false );
    table3.insertColumn( "RUN_NUMBER", coral::AttributeSpecification::typeNameForType<int>() );
    table3.insertColumn( "LUMISECTION", coral::AttributeSpecification::typeNameForType<int>() );
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
    columnsForPrimaryKey3.push_back( "PATH" );
    columnsForPrimaryKey3.push_back( "RUN_NUMBER" );
    columnsForPrimaryKey3.push_back( "LUMISECTION" );
    table3.setPrimaryKey( columnsForPrimaryKey3 );

    std::vector<std::string> columnsForForeignKey3;
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
// -------------------------------------- dqmDbDrop --------------------------------------------
//
void DQMDatabaseWriter::dqmDbDrop(const HistoStats &stats, int lumisection, int run)
{
  edm::LogInfo("DQMDatabaseWriter") <<  "DQMDatabaseWriter::dqmDbDrop " << std::endl;

  bool histogramPropsRecordExist;
  bool histogramValuesRecordExist;
  bool histogramRecordExist;
  coral::ISchema& schema = m_session->nominalSchema();
  for (auto histogram : stats)
  {
    histogramRecordExist = false;
    {
      m_session->transaction().start( false );
      std::unique_ptr<coral::IQuery> queryHistogramProps(schema.tableHandle( "HISTOGRAM" ).newQuery());
      queryHistogramProps->addToOutputList( "PATH" );

      std::string condition = "PATH = :path";
      coral::AttributeList conditionData2;
      conditionData2.extend< std::string >("path");
      conditionData2["path"].data< std::string >() = histogram.path; 
      queryHistogramProps->setCondition( condition, conditionData2 );
      queryHistogramProps->setMemoryCacheSize( 5 );
      coral::ICursor& cursor2 = queryHistogramProps->execute();
      while(cursor2.next())
      {
        cursor2.currentRow().toOutputStream( std::cout ) << std::endl;
        histogramRecordExist = true;
      }
      if ( !histogramRecordExist )
      {
        coral::ITableDataEditor& editor = m_session->nominalSchema().tableHandle( "HISTOGRAM" ).dataEditor();
        coral::AttributeList insertData;
        insertData.extend< std::string >( "PATH" );
        insertData.extend< unsigned int >( "TIMESTAMP" );
        insertData.extend< std::string >( "TITLE" );

        insertData[ "PATH" ].data< std::string >() = histogram.path;
        insertData[ "TIMESTAMP" ].data< unsigned int >() = std::time(nullptr);
        insertData[ "TITLE" ].data< std::string >() = histogram.path;
        editor.insertRow( insertData );
      }
      m_session->transaction().commit();
      m_session->transaction().start(false);
    }

    histogramPropsRecordExist = false;
    {
      if (histogramRecordExist){
        std::unique_ptr<coral::IQuery> queryHistogramProps(schema.tableHandle( "HISTOGRAM_PROPS" ).newQuery());
        queryHistogramProps->addToOutputList( "PATH" );
        queryHistogramProps->addToOutputList( "RUN_NUMBER" );
        queryHistogramProps->addToOutputList( "DIMENSIONS" );
        queryHistogramProps->addToOutputList( "X_AXIS" );
        queryHistogramProps->addToOutputList( "Y_AXIS" );
        queryHistogramProps->addToOutputList( "Z_AXIS" );

        std::string condition = "PATH = :path AND RUN_NUMBER <= :run";
        coral::AttributeList conditionData2;
        conditionData2.extend< std::string >("path");
        conditionData2["path"].data< std::string >() = histogram.path;
        conditionData2.extend< int >("run");
        conditionData2["run"].data< int >() = run; 
        queryHistogramProps->setCondition( condition, conditionData2 );
        queryHistogramProps->setMemoryCacheSize( 5 );
        coral::ICursor& cursor2 = queryHistogramProps->execute();
        long diff = LONG_MAX;
        coral::AttributeList row;
        while(cursor2.next())
        {
          cursor2.currentRow().toOutputStream( std::cout ) << std::endl;
          int runNumber = cursor2.currentRow()["RUN_NUMBER"].data< int >();
          if (run - runNumber < diff){
            diff = run - runNumber;
            row = cursor2.currentRow();
            if (diff == 0) break;
          }
        }
        if (diff != LONG_MAX){
          histogramPropsRecordExist = true;
          if (diff == 0){
            if (row["DIMENSIONS"].data< int >() != histogram.dimNumber) exceptionThrow("Dimensions", histogram.path, run);
            if (row["X_AXIS"].data< std::string >() != dimensionJson(histogram.dimX)) exceptionThrow("X_AXIS", histogram.path, run);
            if (row["Y_AXIS"].data< std::string >() != dimensionJson(histogram.dimY)) exceptionThrow("Y_AXIS", histogram.path, run); 
            if (row["Z_AXIS"].data< std::string >() != dimensionJson(histogram.dimZ)) exceptionThrow("Z_AXIS", histogram.path, run);
            edm::LogWarning( "DQMDatabaseWriter" ) << "Trying to insert the path: " << histogram.path << " with the same run number: " << run
                                                   << " and already existing the same values." << std::endl;
          }else {
            if (row["DIMENSIONS"].data< int >() != histogram.dimNumber ||
                row["X_AXIS"].data< std::string >() != dimensionJson(histogram.dimX) ||
                row["Y_AXIS"].data< std::string >() != dimensionJson(histogram.dimY) || 
                row["Z_AXIS"].data< std::string >() != dimensionJson(histogram.dimZ)) 
                histogramPropsRecordExist = false;
          }
        }
      }
      m_session->transaction().commit();
      m_session->transaction().start(false);
    }

    if(!histogramPropsRecordExist)
    {
        coral::ITableDataEditor& editor = m_session->nominalSchema().tableHandle( "HISTOGRAM_PROPS" ).dataEditor();
        coral::AttributeList insertData;
        insertData.extend< std::string >( "PATH" );
        insertData.extend< int >( "RUN_NUMBER" );
        insertData.extend< int >( "DIMENSIONS" );
        insertData.extend< std::string >( "X_AXIS" );
        insertData.extend< std::string >( "Y_AXIS" );
        insertData.extend< std::string >( "Z_AXIS" );

        insertData[ "PATH" ].data< std::string >() = histogram.path;
        insertData[ "RUN_NUMBER" ].data< int >() = run;
        insertData[ "DIMENSIONS" ].data< int >() = histogram.dimNumber; 
        insertData[ "X_AXIS" ].data< std::string >() = dimensionJson(histogram.dimX);
        insertData[ "Y_AXIS" ].data< std::string >() = dimensionJson(histogram.dimY);
        insertData[ "Z_AXIS" ].data< std::string >() = dimensionJson(histogram.dimZ);
        editor.insertRow( insertData );
    }
    m_session->transaction().commit();
    m_session->transaction().start(false);

    histogramValuesRecordExist = false;
    {
      std::unique_ptr<coral::IQuery> queryHistogramValues(schema.tableHandle( "HISTOGRAM_VALUES" ).newQuery());
      queryHistogramValues->addToOutputList( "PATH" );
      queryHistogramValues->addToOutputList( "RUN_NUMBER" );
      queryHistogramValues->addToOutputList( "LUMISECTION" );

      std::string condition = "PATH = :path AND RUN_NUMBER = :run AND LUMISECTION = :lumisection";
      coral::AttributeList conditionData2;
      conditionData2.extend< std::string >("path");
      conditionData2["path"].data< std::string >() = histogram.path;
      conditionData2.extend< int >("run");
      conditionData2["run"].data< int >() = run;
      conditionData2.extend< int >("lumisection");
      conditionData2["lumisection"].data< int >() = lumisection;
      queryHistogramValues->setCondition( condition, conditionData2 );
      queryHistogramValues->setMemoryCacheSize( 5 );
      coral::ICursor& cursor = queryHistogramValues->execute();
      while(cursor.next())
      {
        cursor.currentRow().toOutputStream( std::cout ) << std::endl;
        histogramValuesRecordExist = true;
      }
      m_session->transaction().commit();
      m_session->transaction().start(false);
    }

    if(!histogramValuesRecordExist)
    {
      coral::ITableDataEditor& editor = m_session->nominalSchema().tableHandle( "HISTOGRAM_VALUES" ).dataEditor();
      coral::AttributeList insertData;
      insertData.extend< std::string >( "PATH" );
      insertData.extend< int >( "RUN_NUMBER" );
      insertData.extend< int >( "LUMISECTION" );
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

      insertData[ "PATH" ].data< std::string >() = histogram.path;
      insertData[ "RUN_NUMBER" ].data< int >() = run;
      insertData[ "LUMISECTION" ].data< int >() = lumisection;
      insertData[ "ENTRIES" ].data< double >() = histogram.entries;
      insertData[ "X_MEAN" ].data< double >() = histogram.dimX.mean;
      insertData[ "X_MEAN_ERROR" ].data< double >() = histogram.dimX.meanError;
      insertData[ "X_RMS" ].data< double >() = histogram.dimX.rms;
      insertData[ "X_RMS_ERROR" ].data< double >() = histogram.dimX.rmsError;
      insertData[ "X_UNDERFLOW" ].data< double >() = histogram.dimX.underflow;
      insertData[ "X_OVERFLOW" ].data< double >() = histogram.dimX.overflow;
      insertData[ "Y_MEAN" ].data< double >() = histogram.dimY.mean;
      insertData[ "Y_MEAN_ERROR" ].data< double >() = histogram.dimY.meanError;
      insertData[ "Y_RMS" ].data< double >() = histogram.dimY.rms;
      insertData[ "Y_RMS_ERROR" ].data< double >() = histogram.dimY.rmsError;
      insertData[ "Y_UNDERFLOW" ].data< double >() = 0.;
      insertData[ "Y_OVERFLOW" ].data< double >() = 0.;
      insertData[ "Z_MEAN" ].data< double >() = histogram.dimZ.mean;
      insertData[ "Z_MEAN_ERROR" ].data< double >() = histogram.dimZ.meanError;
      insertData[ "Z_RMS" ].data< double >() = histogram.dimZ.rms;
      insertData[ "Z_RMS_ERROR" ].data< double >() = histogram.dimZ.rmsError;
      insertData[ "Z_UNDERFLOW" ].data< double >() = 0.;
      insertData[ "Z_OVERFLOW" ].data< double >() = 0.;
      editor.insertRow( insertData );
      m_session->transaction().commit();
    }
  }
}

  std::string DQMDatabaseWriter::toString(boost::property_tree::ptree doc){

    boost::regex exp("\"([0-9]+(\\.[0-9]+)?)\"");
    std::stringstream ss;
    boost::property_tree::json_parser::write_json(ss, doc);
    std::string rv = boost::regex_replace(ss.str(), exp, "$1");
    return rv;
  }

  std::string DQMDatabaseWriter::dimensionJson(Dimension &dim){
    using boost::property_tree::ptree;
    if (dim.nBin == 0 && dim.low == 0 && dim.up == 0) return "";
    ptree doc;
    doc.put("bins", dim.nBin);
    doc.put("low", dim.low);
    doc.put("up", dim.up);
    return toString(doc);
  }

  void DQMDatabaseWriter::exceptionThrow(std::string quantity, std::string path, int run){
    throw cms::Exception( "DQMDatabaseWriter" ) << "Trying to insert already existing histogram: "
                                                << path <<" with the same run number: " 
                                                << run << " but different "
                                                << quantity << " values.\n" << std::endl;
  }



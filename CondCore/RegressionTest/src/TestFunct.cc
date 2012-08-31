#include "CondCore/RegressionTest/interface/TestFunct.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVProxy.h"

#include "RelationalAccess/ITableSchemaEditor.h"
#include "CoralBase/Blob.h"

//typedef TestPayloadClass Payload;
typedef RegressionTestPayload Payload;

TestFunct::TestFunct() {}

std::pair<int,int> TestFunct::GetMetadata(std::string mappingName)
{
 	cond::DbScopedTransaction trans(s);
 	std::pair<int,int> ret(-1,-1);
 	try {
 		trans.start(true);
 		
 		coral::ITable& mytable=s.nominalSchema().tableHandle("TEST_METADATA");
 		std::auto_ptr< coral::IQuery > query(mytable.newQuery());
 		coral::AttributeList BindVariableList;
 		std::string condition="NAME =:NAME";
 		BindVariableList.extend("NAME",typeid(std::string));
 		BindVariableList["NAME"].data<std::string>()=mappingName;
 		query->setCondition( condition, BindVariableList );
 		query->addToOutputList( "SEED" );
 		query->addToOutputList( "RUN" );
 		coral::ICursor& cursor = query->execute();
 		while( cursor.next() ) {
 			const coral::AttributeList& row = cursor.currentRow();
 			ret.first = row[ "SEED" ].data<int>();
 			ret.second =row["RUN" ].data<int>();
 		}
 		trans.commit();
 	} catch ( const cond::Exception& exc )
 	{
 		std::cout << "ERROR: "<<exc.what()<<std::endl;
 		return std::pair<int,int>(-1,-1);
 	}
 	return ret;
}

bool TestFunct::Read (std::string mappingName)
{
	cond::DbScopedTransaction trans(s);
	cond::MetaData  metadata(s);
	int refSeed =0;
        bool ret = false;
	try {
		trans.start(true);
		
		coral::ITable& mytable=s.nominalSchema().tableHandle("TEST_METADATA");
		std::auto_ptr< coral::IQuery > query(mytable.newQuery());
		coral::AttributeList BindVariableList;
		std::string condition="NAME =: NAME";
		BindVariableList.extend("NAME",typeid(std::string));
		BindVariableList["NAME"].data<std::string>()=mappingName;
		query->setCondition( condition, BindVariableList );
		query->addToOutputList( "SEED" );
		coral::ICursor& cursor = query->execute();
		while( cursor.next() ) {
			const coral::AttributeList& row = cursor.currentRow();
			refSeed=row[ "SEED" ].data<int>();
		}
		std::string readToken = metadata.getToken(mappingName);
		pool::Ref<Payload> readRef0 = s.getTypedObject<Payload>( readToken );  //v3	
		std::cout << "Object with id="<<readToken<<" has been read"<<std::endl;
		Payload tp = *readRef0;
		Payload tp2(refSeed);
		if(tp == tp2){
		  ret = true;
		} else {
		  std::cout <<" read failed : seed "<<refSeed<<std::endl;
		}
		trans.commit();
	} catch ( const cond::Exception& exc )
	{
		std::cout << "ERROR: "<<exc.what()<<std::endl;
		return false;
	}
	return ret;
}

bool TestFunct::ReadWithIOV(std::string mappingName, 
 			    int seed,
 			    int validity)
{
 	cond::DbScopedTransaction trans(s);
 	cond::MetaData  metadata(s);
        bool ret = false;
 	try {
 		trans.start(true);

 		std::string iovToken = metadata.getToken(mappingName);
 		cond::IOVProxy iov(s,iovToken, false, false );
 		cond::IOVProxy::const_iterator iPayload = iov.find( validity );
 		if( iPayload == iov.end() ){
 		  std::cout << "ERROR: no payload found in IOV for run="<<validity<<std::endl;
 		  return 1;
 		}
 		pool::Ref<Payload> readRef0 = s.getTypedObject<Payload>( iPayload->token() ); //v4	
 		std::cout << "Object with id="<<iPayload->token()<<" has been read"<<std::endl;
 		Payload tp = *readRef0;
 		Payload tp2(seed);
 		if(tp == tp2){
		  ret = true;
		} else {
 		  std::cout <<" read failed : seed="<<seed<<std::endl;
		}
 		trans.commit();
 	} catch ( const cond::Exception& exc )
 	{
 		std::cout << "ERROR: "<<exc.what()<<std::endl;
 		return false;
 	}
 	return ret;
}

bool TestFunct::ReadAll()
{
	cond::DbScopedTransaction trans(s);
	cond::MetaData  metadata(s);
	std::vector<std::string> tokenList;
	bool ret = true;
	try {
		trans.start(true);
		metadata.listAllTags(tokenList);
		for(unsigned int i=0; i<tokenList.size(); i++)
		{
		  if(!Read(tokenList[i])) ret = false;
		}
		trans.commit();
	} 
	catch ( const cond::Exception& exc )
	{
		std::cout << "ERROR: "<<exc.what()<<std::endl;
		return false;
	}
	return ret;
}

/**
namespace TestFuncImpl {
  // workaround to make possible the streaming of std::bitset
  static const char* BITSET_COLNAME = "DARRAYPA_M_BITSET";
  pool::Ref<Payload> initWriting( cond::DbSession& s ){
    cond::DbScopedTransaction trans(s);
    pool::Ref<Payload> ret;
    trans.start();
    coral::ISchema& schema = s.nominalSchema();
    if(!schema.existsTable("POOL_RSS_DB")){
      ret = s.storeObject( new Payload(1), "cont1" );
      trans.commit();
      trans.start();
      coral::ITableSchemaEditor& mainPayloadTableSchema = schema.tableHandle("REGRESSIONTESTPAYLOA").schemaEditor();
      mainPayloadTableSchema.insertColumn(BITSET_COLNAME,coral::AttributeSpecification::typeNameForId( typeid(coral::Blob) ));
      coral::ITableDataEditor& mappingTable = schema.tableHandle("POOL_OR_MAPPING_ELEMENTS").dataEditor();
      coral::AttributeList inputData;
      inputData.extend<std::string>( "ELEMENT_TYPE");
      inputData.extend<std::string>( "COLUMN_NAME");
      inputData.extend<std::string>( "VARIABLE_NAME");
      inputData[ "ELEMENT_TYPE" ].data<std::string>() = "Blob";
      inputData[ "COLUMN_NAME" ].data<std::string>() = BITSET_COLNAME;
      inputData[ "VARIABLE_NAME" ].data<std::string>() = "ArrayPayload::m_bitset";
      std::string setClause = "ELEMENT_TYPE =:ELEMENT_TYPE, COLUMN_NAME=:COLUMN_NAME";
      std::string whereClause = "VARIABLE_NAME =:VARIABLE_NAME";
      mappingTable.updateRows( setClause,whereClause, inputData  );
    }
    trans.commit();
    return ret;
  }
}
**/

bool TestFunct::Write (std::string mappingName, int payloadID)
{
		cond::DbScopedTransaction trans(s);
	   cond::MetaData  metadata(s);
	   std::string tok0("");
	try 
	{
	  //pool::Ref<Payload> dummy = TestFuncImpl::initWriting( s );
	    trans.start();
		coral::ITable& mytable=s.nominalSchema().tableHandle("TEST_METADATA");
		coral::AttributeList rowBuffer;
		coral::ITableDataEditor& dataEditor = mytable.dataEditor();
		dataEditor.rowBuffer( rowBuffer );
		rowBuffer["NAME"].data<std::string>()=mappingName;
		rowBuffer["SEED"].data<int>()=payloadID;
		rowBuffer["RUN"].data<int>()=-1;
		dataEditor.insertRow( rowBuffer );	
	        pool::Ref<Payload> myRef0 = s.storeObject(new Payload(payloadID), "cont1"); //v3
		tok0 = myRef0.toString(); //v3
	    metadata.addMapping(mappingName, tok0);
	    std::cout << "Stored object with id = "<<tok0<<std::endl;
	    trans.commit();
	} catch ( const cond::Exception& exc )
	{
		std::cout << "ERROR: "<<exc.what()<<std::endl;
		return false;
	}
	return true;
}

bool TestFunct::WriteWithIOV(std::string mappingName, 
 			     int payloadID, 
 			     int runValidity,
			     bool updateTestMetadata ){
   cond::DbScopedTransaction trans(s);
   cond::MetaData  metadata(s);
   std::string tok0("");
   try {
     //pool::Ref<Payload> dummy = TestFuncImpl::initWriting( s );
     cond::IOVEditor iov(s);
     trans.start();
     if( updateTestMetadata ){
       coral::ITable& mytable=s.nominalSchema().tableHandle("TEST_METADATA");
       coral::AttributeList rowBuffer;
       coral::ITableDataEditor& dataEditor = mytable.dataEditor();
       dataEditor.rowBuffer( rowBuffer );
       rowBuffer["NAME"].data<std::string>()=mappingName;
       rowBuffer["SEED"].data<int>()=payloadID;
       rowBuffer["RUN"].data<int>()= runValidity;
       dataEditor.insertRow( rowBuffer );
     }		
     pool::Ref<Payload> myRef0 = s.storeObject(new Payload(payloadID), "cont1"); //v3
     std::string payloadTok = myRef0.toString(); 
     iov.create( cond::runnumber );
     iov.append( runValidity, payloadTok );
     metadata.addMapping(mappingName, iov.token());
     trans.commit();
   } catch ( const cond::Exception& exc )
     {
       std::cout << "ERROR: "<<exc.what()<<std::endl;
       return false;
     }
   return true;    
}

bool TestFunct::CreateMetaTable ()
{
	cond::DbScopedTransaction trans(s);
	cond::MetaDataSchemaUtility metaUt(s);
	try
	{
		trans.start();
		metaUt.create();
		coral::ISchema& schema=s.nominalSchema();
		coral::TableDescription description;
		description.setName("TEST_METADATA");
		description.insertColumn(  "NAME", coral::AttributeSpecification::typeNameForId( typeid(std::string)) );
		description.insertColumn( "SEED", coral::AttributeSpecification::typeNameForId( typeid(int)) );
		description.insertColumn( "RUN", coral::AttributeSpecification::typeNameForId( typeid(int)) );
		std::vector<std::string> cols;
		cols.push_back( "NAME" );
		description.setPrimaryKey(cols);
		description.setNotNullConstraint("SEED");
		description.setNotNullConstraint("RUN");
		coral::ITable& table=schema.createTable(description);
		table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select);
		std::cout<<"Table created"<<std::endl;
		trans.commit();
	}catch( const coral::TableAlreadyExistingException& er ){
		std::cout<<"table alreay existing, not creating a new one"<<std::endl;
		return false;
	}
	return true;
}
bool TestFunct::DropTables(std::string connStr)
{
	std::set<std::string> exclude;
	exclude.insert("VERSION_TABLE");
	exclude.insert("TEST_STATUS");
	exclude.insert("SEQUENCES");
	exclude.insert("TEST_RESULTS");
	exclude.insert("RUN_HEADER");
	exclude.insert("RUN_RESULT");
	exclude.insert("RUN_STEP_RESULT");
	try
	{
		ora::SchemaUtils::cleanUp(connStr, exclude); //v4
	}
	catch ( const std::exception& exc )
	{
		std::cout <<" ERROR: "<<exc.what()<<std::endl;
		return false;
    }
	return true;
}
bool TestFunct::DropItem(std::string mappingName)
{
	cond::DbScopedTransaction trans(s);
	cond::MetaData  metadata(s);
	try {
		trans.start(false);
		std::string token = metadata.getToken(mappingName);
		s.deleteObject(token);
		metadata.deleteEntryByTag(mappingName);
		trans.commit();
	} 
	catch ( const cond::Exception& exc )
	{
		std::cout << "ERROR: "<<exc.what()<<std::endl;
		return false;
	}
	
	return true;
}

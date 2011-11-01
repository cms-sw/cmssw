#include "CondCore/RegressionTest/interface/TestFunct.h"


TestFunct::TestFunct() {}
bool TestFunct::Read (std::string mappingName)
{
	cond::DbScopedTransaction trans(s);
	cond::MetaData  metadata(s);
	int refSeed =0;
	try {
		trans.start(true);
		
		coral::ITable& mytable=s.nominalSchema().tableHandle("TEST_SEED");
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
		pool::Ref<TestPayloadClass> readRef0 = s.getTypedObject<TestPayloadClass>( readToken );  //v3	
		std::cout << "Object with id="<<readToken<<" has been read"<<std::endl;
		TestPayloadClass tp = *readRef0;
		TestPayloadClass tp2(refSeed);
		if(tp != tp2)
			std::cout <<" read failed : token "<<refSeed<<std::endl;
		trans.commit();
	} catch ( const cond::Exception& exc )
	{
		std::cout << "ERROR: "<<exc.what()<<std::endl;
		return 1;
	}
	return 0;
}
bool TestFunct::ReadAll()
{
	cond::DbScopedTransaction trans(s);
	cond::MetaData  metadata(s);
	std::vector<std::string> tokenList;
	try {
		trans.start(true);
		metadata.listAllTags(tokenList);
		for(unsigned int i=0; i<tokenList.size(); i++)
		{
			Read(tokenList[i]);
		}
		trans.commit();
	} 
	catch ( const cond::Exception& exc )
	{
		std::cout << "ERROR: "<<exc.what()<<std::endl;
		return 1;
	}
	return 0;
}
bool TestFunct::Write (std::string mappingName, int payloadID)
{
		cond::DbScopedTransaction trans(s);
	   cond::MetaData  metadata(s);
	   std::string tok0("");
	try 
	{
	    trans.start();
		coral::ITable& mytable=s.nominalSchema().tableHandle("TEST_SEED");
		coral::AttributeList rowBuffer;
		coral::ITableDataEditor& dataEditor = mytable.dataEditor();
		dataEditor.rowBuffer( rowBuffer );
		rowBuffer["NAME"].data<std::string>()=mappingName;
		rowBuffer["SEED"].data<int>()=payloadID;
		dataEditor.insertRow( rowBuffer );	
	    pool::Ref<TestPayloadClass> myRef0 = s.storeObject(new TestPayloadClass(payloadID), "cont1"); //v3
		tok0 = myRef0.toString(); //v3
	    metadata.addMapping(mappingName, tok0);
	    std::cout << "Stored object with id = "<<tok0<<std::endl;
	    trans.commit();
	} catch ( const cond::Exception& exc )
	{
		std::cout << "ERROR: "<<exc.what()<<std::endl;
		return 1;
	}
	return 0;
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
		description.setName("TEST_SEED");
		description.insertColumn(  "NAME", coral::AttributeSpecification::typeNameForId( typeid(std::string)) );
		description.insertColumn( "SEED", coral::AttributeSpecification::typeNameForId( typeid(int)) );
		std::vector<std::string> cols;
		cols.push_back( "NAME" );
		description.setPrimaryKey(cols);
		description.setNotNullConstraint("SEED");
		coral::ITable& table=schema.createTable(description);
		table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select);
		std::cout<<"Table created"<<std::endl;
		trans.commit();
	}catch( const coral::TableAlreadyExistingException& er ){
		std::cout<<"table alreay existing, not creating a new one"<<std::endl;
		return 1;
	}
	return 0;
}
bool TestFunct::DropTables(std::string connStr)
{
	std::set<std::string> exclude;
	exclude.insert("VERSION_TABLE");
	exclude.insert("TEST_STATUS");
	exclude.insert("SEQUENCES");
	exclude.insert("TEST_RESULTS");
	try
	{
		ora::SchemaUtils::cleanUp(connStr, exclude); //v4
	}
	catch ( const std::exception& exc )
	{
		std::cout <<" ERROR: "<<exc.what()<<std::endl;
		return 1;
    }
	return 0;
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
		return 1;
	}
	
	return 0;
}

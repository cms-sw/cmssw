import os
import coral
if __name__ == "__main__":
    try:
        os.putenv( "CORAL_AUTH_PATH", "." )
        svc = coral.ConnectionService()
        session = svc.connect( 'sqlite_file:testcoral.db',
                           accessMode = coral.access_Update )
        TreeTableName = 'TAGTREE_TABLE'
        TreeTableColumns = {'nodeid':'unsigned long', 'nodelabel':'string', 'lft':'unsigned long', 'rgt':'unsigned long', 'parentid':'unsigned long', 'tagid':'unsigned long', 'globalsince':'unsigned long long', 'globaltill':'unsigned long long'}
        TreeTableValues = {'nodeid':1, 'nodelabel':'testtest', 'lft':5, 'rgt':6, 'parentid':4, 'tagid':2, 'globalsince':1235, 'globaltill':9457}
        TreeTableNotNullColumns = ['nodelabel','lft','rgt','parentid']
        TreeTableUniqueColumns = ['nodelabel']
        TreeTablePK = ('nodeid')
        transaction=session.transaction()
        transaction.start(False)
        schema = session.nominalSchema()
        schema.dropIfExistsTable( TreeTableName )
        description = coral.TableDescription();
        description.setName( TreeTableName )
        for columnName, columnType in TreeTableColumns.items():
            description.insertColumn(columnName, columnType)
        for columnName in TreeTableNotNullColumns :
            description.setNotNullConstraint(columnName,True)
        for columnName in TreeTableUniqueColumns :
            description.setUniqueConstraint(columnName)
        description.setPrimaryKey(  TreeTablePK )
        TreeTableHandle = schema.createTable( description )
        print 'created'
        TreeTableHandle.privilegeManager().grantToPublic( coral.privilege_Select )
        editor=TreeTableHandle.dataEditor()
        inputData=coral.AttributeList()
        for name,type in TreeTableColumns.items():
            inputData.extend(name,type)
            inputData[name].setData(TreeTableValues[name])
        editor.insertRow(inputData)
        transaction.commit()
        transaction.start(True)
        query=session.nominalSchema().tableHandle(TreeTableName).newQuery()
        condition = 'nodelabel =:nodelabel'
        conditionData = coral.AttributeList()
        conditionData.extend( 'nodelabel','string' )
        query.setCondition( condition, conditionData)
        conditionData['nodelabel'].setData('testtest')
        cursor = query.execute()
        while ( cursor.next() ):
          tagid=cursor.currentRow()['tagid'].data()
          print 'tagid',tagid
          nodeid=cursor.currentRow()['nodeid'].data()
          print 'nodeid',nodeid
          nodelabel=cursor.currentRow()['nodelabel'].data()
          print 'nodelabel',nodelabel
          lft=cursor.currentRow()['lft'].data()
          print 'lft',lft
          rgt=cursor.currentRow()['rgt'].data()
          print 'rgt',rgt
          parentid=cursor.currentRow()['parentid'].data()
          print 'parentid',parentid
          globalsince=cursor.currentRow()['globalsince'].data()
          print 'globalsince',globalsince
          globaltill=cursor.currentRow()['globaltill'].data()
          print 'globaltill',globaltill    
        transaction.commit()
        del session
    except Exception, e:
        print "Failed in unit test"
        print str(e)
        transaction.rollback()
        del session
    


'''
central place for all the table names
'''
def schemaV2Tables():
       return ['REVISIONS','LUMIDATA','LUMISUMMARYV2','LUMINORMS','TRGDATA','LSTRG', 'HLTDATA', 'LSHLT']
def schemaV1Tables():
       return [ 'LUMISUMMARY','LUMIDETAIL','TRG','HLT']
def commonTables():
       return [ 'CMSRUNSUMMARY','TRGHLTMAP','LUMIVALIDATION']
def revisionTableName():
       return 'REVISIONS'

def cmsrunsummaryTableName():
       return 'CMSRUNSUMMARY'

def lumidataTableName():
	return 'LUMIDATA'

def lumisummaryTableName():
	return 'LUMISUMMARY'

def lumisummaryv2TableName():
	return 'LUMISUMMARYV2'
def lumidetailTableName():
	return 'LUMIDETAIL'
def luminormTableName():
       return 'LUMINORMS'
def trgdataTableName():
       return 'TRGDATA'

def lstrgTableName():
       return 'LSTRG'
       
def trgTableName():
	return 'TRG'

def hltTableName():
	return 'HLT'

def hltdataTableName():
       return 'HLTDATA'

def lshltTableName():
       return 'LSHLT'

def trghltMapTableName():
        return 'TRGHLTMAP'

def lumiresultTableName():
	return 'INTLUMI'

def lumihltresultTableName():
	return 'INTLUMIHLT'

def lumivalidationTableName():
       return 'LUMIVALIDATION'

def idTableName( dataTableName ):
	return dataTableName+"_ID"

def idTableColumnDefinition():
	return ('NEXTID','unsigned long long')

def revmapTableName( dataTableName ):
       return dataTableName+'_REV'

def entryTableName( dataTableName ):
       return dataTableName+'_ENTRIES'

if __name__ == "__main__":
    pass


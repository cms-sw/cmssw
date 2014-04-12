
'''
central place for all the table names
'''
def schemaV2Tables():
       return ['REVISIONS','LUMIDATA','PIXELLUMIDATA','LUMISUMMARYV2','LUMINORMS','TRGDATA','LSTRG', 'HLTDATA', 'LSHLT','INTGLUMIV2','PIXELLUMISUMMARYV2']
def schemaV1Tables():
       return [ 'LUMISUMMARY','LUMIDETAIL','TRG','HLT']
def commonTables():
       return [ 'CMSRUNSUMMARY','TRGHLTMAP','LUMIVALIDATION','FILLSCHEME']
def revisionTableName():
       return 'REVISIONS'
def fillschemeTableName():
	return 'FILLSCHEME'

def cmsrunsummaryTableName():
       return 'CMSRUNSUMMARY'

def lumidataTableName():
	return 'LUMIDATA'

def pixellumidataTableName():
	return 'PIXELLUMIDATA'

def lumisummaryTableName():
	return 'LUMISUMMARY'

def lumisummaryv2TableName():
	return 'LUMISUMMARYV2'
 
def pixellumisummaryv2TableName():
       return 'PIXELLUMISUMMARYV2'

def lumidetailTableName():
	return 'LUMIDETAIL'
 
def luminormTableName():
       return 'LUMINORMS'

def luminormv2TableName():
       return 'LUMINORMSV2'

def luminormv2dataTableName():
       return 'LUMINORMSV2DATA'

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

def tagRunsTableName():
       return 'TAGRUNS'

def tagsTableName():
       return 'TAGS'

def pixeltagRunsTableName():
       return 'PIXELTAGRUNS'

def pixeltagsTableName():
       return 'PIXELTAGS'

def trghltMapTableName():
        return 'TRGHLTMAP'

def lumiresultTableName():
	return 'INTLUMI'

def lumihltresultTableName():
	return 'INTLUMIHLT'

def lumivalidationTableName():
       return 'LUMIVALIDATION'

def intglumiTableName():
       return 'INTGLUMI'

def intglumiv2TableName():
       return 'INTGLUMIV2'
       
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


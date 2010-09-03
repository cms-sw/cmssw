
'''
central place for all the table names
'''
def cmsrunsummaryTableName():
       return 'CMSRUNSUMMARY'

#def lumirunsummaryTableName():
#	return 'LUMIRUNSUMMARY'

def lumisummaryTableName():
	return 'LUMISUMMARY'

def lumidetailTableName():
	return 'LUMIDETAIL'

def trgTableName():
	return 'TRG'

def hltTableName():
	return 'HLT'

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

if __name__ == "__main__":
    pass


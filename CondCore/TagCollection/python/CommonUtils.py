
def inventoryTableName():
	return 'TAGINVENTORY_TABLE'
def inventoryIDTableName():
	return 'TAGINVENTORY_IDS'
def treeTableName(treename):
	return 'TAGTREE_TABLE_'+str.upper(treename)
def treeIDTableName(treename):
	return 'TAGTREE_'+str.upper(treename)+'_IDS'

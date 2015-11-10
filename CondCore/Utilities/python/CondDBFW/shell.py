"""

Contains classes for shell part of framework - basically a collection of classes that are designed to be invoked on the command line.

"""

import querying

# function to setup database connection, based on given database name
def connect(connection_data=None):
	if connection_data == None:
		connection_data = {"db_alias":"orapro", "schema" : "cms_conditions", "host":"oracle", "secrets":"/afs/cern.ch/cms/DB/conddb/.cms_cond/netrc"}
	connection = querying.connect(connection_data)
	return connection
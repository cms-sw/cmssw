"""

Contains classes for shell part of framework - basically a collection of classes that are designed to be invoked on the command line.

"""

import querying
import sys

connections = []

# function to setup database connection, based on given database name
def connect(connection_data=None, mode="r", map_blobs=False, secrets=None, pooling=True):
	if connection_data == None:
		connection_data = "frontier://FrontierProd/CMS_CONDITIONS"
	connection = querying.connect(connection_data, mode=mode, map_blobs=map_blobs, secrets=secrets, pooling=pooling)
	connections.append(connection)
	return connection

def close_connections(verbose=True):
	global connections
	for connection in connections:
		connection_string = "%s/%s" % (connection.connection_data["database_name"], connection.connection_data["schema"])
		connection.tear_down()
		if verbose:
			print("Connection to %s was closed." % connection_string)
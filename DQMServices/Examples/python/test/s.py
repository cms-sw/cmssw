import socket
import sys
import sqlite3
import getopt
from ctypes import c_ushort

RUN_AND_LUMI	= "1"
RUN				= "2"
LUMINOSITY		= "3"
HISTOGRAM_NAME	= "4"

def parseRow(row):
	row = str(row)
	row = row.replace("u\'", "\"")
	row = row.replace("\'", "\"")
	print row
	row_length = len(row)
	if(row_length>1):
		row = row[1:-1]
	return row+'\n\r'

def hex16(data):
	'''16bit int->hex converter'''
	return  '0x%004x' % (c_ushort(data).value)
#------------------------------------------------------------------------------      
def int16(data):
	'''16bit hex->int converter'''
	return c_ushort(int(data,16)).value

def main(argv):
	database = ''
	default_database = 'db1.db'
	isDatabaseSpecified = False
	user = ''
	password = ''
	
	try:
		opts, args = getopt.getopt(argv,"hu:d:p:",["port="])
	except getopt.GetoptError:
		print 'database_server.py -d <databasepath> -u <username> -p <password>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'database_server.py -d <databasepath> -u <username> -p <password>'
			sys.exit()
		elif opt in ("-d"):
			database = arg
			isDatabaseSpecified = True
		elif opt in ("-o"):
			user = arg
		elif opt in ("-p"):
			password = arg
		elif opt in ("--port"):
			if 9000 <= int(arg) <= 11000:
				port = int(arg)
			else:
				print "WARNING: Allowed ports are 9000-11000, using 10000 by default"

	if isDatabaseSpecified:
		try:
			conn = sqlite3.connect(database)
			print "INFO: Connected to the database"
		except sqlite3.Error as e:
			print "ERROR: An error occurred:", e.args[0]
	else:
		try:
			conn = sqlite3.connect(default_database)
			print "INFO: Connected to the default database"
		except sqlite3.Error as e:
			print "ERROR: An error occurred:", e.args[0]

	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try:
	  port
	except NameError:
	  port = 10000
	server_address = ('localhost', port)
	print >>sys.stderr, 'INFO: Starting up on %s port %s' % server_address
	sock.bind(server_address)
	sock.listen(1)
	
	name = ''
	query_type = ''
	while True:
		# Wait for a connection
		print "INFO: Waiting for the query"
		connection, client_address = sock.accept()
		try:
			data_header = connection.recv(1)
			if(data_header == HISTOGRAM_NAME):
				print "QUERY: HISTOGRAM_NAME"
				query_type = HISTOGRAM_NAME
				luminosity = connection.recv(16)
				run = connection.recv(16)			
				name_length = connection.recv(16)
				name = connection.recv(int(name_length))
			elif(data_header == RUN_AND_LUMI):
				print "QUERY: RUN_AND_LUMI"
				query_type = RUN_AND_LUMI
				luminosity = connection.recv(16)
				run = connection.recv(16)
			elif(data_header == RUN):
				print "QUERY: RUN"
				query_type = RUN
				run = connection.recv(16)
			elif(data_header == LUMINOSITY):
				print "QUERY: LUMINOSITY"
				query_type = LUMINOSITY
				luminosity = connection.recv(16)
			else:
				print "ERROR: Request not recognized:", data_header
		except socket.error as e:
			# Clean up the connection
			connection.close()
			print "ERROR: An error occurred:", e.args[0]
			sys.exit(2)
			
		c = conn.cursor()
		result = ''
		if(query_type == HISTOGRAM_NAME):
			arguments = (luminosity,run,name)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=? AND RUN_NUMBER=? AND NAME=?', arguments):
				result += parseRow(row)
		elif(query_type == RUN_AND_LUMI):
			arguments = (luminosity,run)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=? AND RUN_NUMBER=?', arguments):
				result += parseRow(row)
		elif(query_type == LUMINOSITY):
			arguments = (luminosity,)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=?', arguments):
				result += parseRow(row)
		elif(query_type == RUN):
			arguments = (run,)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE RUN_NUMBER=?', arguments):
				result += parseRow(row)
			
		print result
		print len(result)
		connection.sendall(hex16(len(result)))
		connection.sendall(result)
		connection.close()
if __name__ == "__main__":
	main(sys.argv[1:])




import socket
import sys
import getopt
import struct
from ctypes import c_ushort 

RUN_AND_LUMI	= b"1"
RUN				= b"2"
LUMINOSITY		= b"3"
HISTOGRAM_NAME	= b"4"

def hex16(data):
	'''16bit int->hex converter'''
	return  '0x%004x' % (c_ushort(data).value)
#------------------------------------------------------------------------------      
def int16(data):
	'''16bit hex->int converter'''
	#return struct.unpack("H", data)[0]
	return c_ushort(int(data,16)).value

def main(argv):

	isLuminositySpecified = False
	isRunSpecified = False
	isHistogramNameSpecified = False
	
	try:
		opts, args = getopt.getopt(argv,"r:l:n:",["port="])
	except getopt.GetoptError:
		print 'query_database.py -r <run> -l <luminosity>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'query_database.py -r <run> -l <luminosity>'
			sys.exit()
		elif opt in ("-r"):
			run = arg
			isRunSpecified = True
		elif opt in ("-l"):
			luminosity = arg
			isLuminositySpecified = True
		elif opt in ("-n"):
			name = arg
			isHistogramNameSpecified = True			
		elif opt in ("--port"):
			if 9000 <= int(arg) <= 11000:
				port = int(arg)
			else:
				print "WARNING: Allowed ports are 9000-11000, using 10000 by default"

	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_address = ('localhost', port)
	#print 'INFO: Connecting to %s port %s' % server_address
	sock.connect(server_address)


	try:
		if(isHistogramNameSpecified and isLuminositySpecified and isRunSpecified):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", luminosity, "AND RUN_NUMBER =", run, "AND NAME =", name
			sock.sendall(bytearray(HISTOGRAM_NAME))
			sock.sendall(bytearray(bytes(luminosity).zfill(16)))			
			sock.sendall(bytearray(bytes(run).zfill(16)))
			sock.sendall(bytearray(bytes(str(len(name))).zfill(16)))
			sock.sendall(bytearray(name))

		elif(isLuminositySpecified and isRunSpecified):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", luminosity, "AND RUN_NUMBER =", run
			sock.sendall(bytearray(RUN_AND_LUMI))
			sock.sendall(bytearray(bytes(luminosity).zfill(16)))
			sock.sendall(bytearray(bytes(run).zfill(16)))
		elif(isLuminositySpecified and not isRunSpecified):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", luminosity
			sock.sendall(bytearray(LUMINOSITY))
			sock.sendall(bytearray(bytes(luminosity).zfill(16)))
		elif(not isLuminositySpecified and isRunSpecified):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE RUN_NUMBER =", run		
			sock.sendall(bytearray(RUN))
			sock.sendall(bytearray(bytes(run).zfill(16)))
			
		# Look for the response
		amount_received = 0
		query_output = ''
		sock.recv(2)
		length = int16(sock.recv(4))
		while amount_received < length:
			#test = input("press")
			data = sock.recv(16)
			amount_received += len(data)
			query_output += data
			#print >>sys.stderr, 'received "%s"' % data
	finally:
		#print >>sys.stderr, 'closing socket'
		sock.close()
	print "NAME, PATH, RUN_NUMBER, LUMISECTION, ENTRIES, X_MEAN, X_MEAN_ERROR, X_RMS, X_RMS_ERROR, X_UNDERFLOW,X_OVERFLOW, Y_MEAN, Y_MEAN_ERROR, Y_RMS, Y_RMS_ERROR,Y_UNDERFLOW, Y_OVERFLOW, Z_MEAN, Z_MEAN_ERROR, Z_RMS, Z_RMS_ERROR, Z_UNDERFLOW, Z_OVERFLOW"
	print query_output		
				
if __name__ == "__main__":
	main(sys.argv[1:])
	

import sys
import getopt
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons

RUN_AND_LUMI	= b"1"
RUN				= b"2"
LUMINOSITY		= b"3"
HISTOGRAM_NAME	= b"4"

def parseRow(row):
	row = str(row)
	row = row.replace("u\'", "\"")
	row = row.replace("\'", "\"")
	row_length = len(row)
	if(row_length>1):
		row = row[1:-1]
	return row+'\n\r'
#------------------------------------------------------------------------------      
def int16(data):
	return c_ushort(int(data,16)).value

def main(argv):
	database = ''
	default_database = 'db1.db'
	isDatabaseSpecified = False
	user = ''
	password = ''

	isLuminositySpecified = False
	isRunSpecified = False
	isHistogramNameSpecified = False
	
	try:
		opts, args = getopt.getopt(argv,"r:l:n:hu:d:p:",["port="])
	except getopt.GetoptError:
		print 'query_database.py -d <databasepath> -u <username> -p <password> -r <run> -l <luminosity> -n <histogram_name>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'query_database.py -d <databasepath> -u <username> -p <password> -r <run> -l <luminosity> -n <histogram_name>'
			sys.exit()
		elif opt in ("-d"):
			database = arg
			isDatabaseSpecified = True
		elif opt in ("-u"):
			user = arg
		elif opt in ("-p"):
			password = arg		
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
				port = 10000
				print "WARNING: Allowed ports are 9000-11000, using 10000 by default"

	if(not isHistogramNameSpecified and not isLuminositySpecified and not isRunSpecified):
		print 'query_database.py -r <run> -l <luminosity> -n <histogram_name>'
		sys.exit()
				
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

	try:
		c = conn.cursor()
		result = ''
		lumisections = []
		properties = []
		for i in range (0,19):
			properties.append([])
		property = []
		property_name = ['NAME', 'PATH', 'RUN_NUMBER', 'LUMISECTION', 'ENTRIES', 'X_MEAN', 'X_MEAN_ERROR', 'X_RMS', 'X_RMS_ERROR', 'X_UNDERFLOW', 'X_OVERFLOW', 'Y_MEAN', 'Y_MEAN_ERROR', 'Y_RMS', 'Y_RMS_ERROR', 'Y_UNDERFLOW', 'Y_OVERFLOW', 'Z_MEAN', 'Z_MEAN_ERROR', 'Z_RMS', 'Z_RMS_ERROR', 'Z_UNDERFLOW', 'Z_OVERFLOW']
		selection = 5
		if(isHistogramNameSpecified and isLuminositySpecified and isRunSpecified):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", luminosity, "AND RUN_NUMBER =", run, "AND NAME =", name
			arguments = (luminosity,run,name)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=? AND RUN_NUMBER=? AND NAME=?', arguments):
				result += parseRow(row)
		elif(isHistogramNameSpecified and not isLuminositySpecified and isRunSpecified):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE RUN_NUMBER =", run, "AND NAME =", name
			arguments = (run,name)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE RUN_NUMBER=? AND NAME=?', arguments):
				lumisections.append(row[3])
				for property in range (0,19):
					properties[property].append(row[property+4])
				result += parseRow(row)
				
		elif(isHistogramNameSpecified and isLuminositySpecified and not isRunSpecified):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", luminosity, "AND NAME =", name
			arguments = (luminosity,name)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=? AND NAME=?', arguments):
				result += parseRow(row)				
		elif(isLuminositySpecified and isRunSpecified):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", luminosity, "AND RUN_NUMBER =", run
			arguments = (luminosity,run)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=? AND RUN_NUMBER=?', arguments):
				result += parseRow(row)
		elif(isLuminositySpecified and not isRunSpecified):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", luminosity
			arguments = (luminosity,)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=?', arguments):
				result += parseRow(row)
		elif(not isLuminositySpecified and isRunSpecified):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE RUN_NUMBER =", run		
			arguments = (run,)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE RUN_NUMBER=?', arguments):
				result += parseRow(row)

		
		print "NAME, PATH, RUN_NUMBER, LUMISECTION, ENTRIES, X_MEAN, X_MEAN_ERROR, X_RMS, X_RMS_ERROR, X_UNDERFLOW, X_OVERFLOW, Y_MEAN, Y_MEAN_ERROR, Y_RMS, Y_RMS_ERROR, Y_UNDERFLOW, Y_OVERFLOW, Z_MEAN, Z_MEAN_ERROR, Z_RMS, Z_RMS_ERROR, Z_UNDERFLOW, Z_OVERFLOW"
		#print result
		
	except sqlite3.Error as e:
		print "ERROR: An error occurred:", e.args[0]
	
	if(isHistogramNameSpecified and not isLuminositySpecified and isRunSpecified):
		fig = plt.figure()
		fig.canvas.set_window_title('Histograms\' properties')
		ax = fig.add_subplot(111)
		plt.xticks(lumisections)
		x = lumisections
		y = properties[0]
		plot1, = ax.plot(x, y, 'ro')
		plt.axis([min(x)-2, max(x)+2, min(y)*0.98, max(y)*1.02])
		for xy in zip(x, y):                                                
			ax.annotate(' (%s)' % xy[1], xy=xy, textcoords='data')

		plt.xlabel('Luminosity')
		plt.title(name+'\n'+property_name[4])
		plt.grid()
		
		ax1 = subplot(110)
		subplots_adjust(bottom=0.25)	
		axcolor = 'lightgoldenrodyellow'
		axfreq = axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
		sfreq = Slider(axfreq, 'Property', 1, 19, valinit=1, valfmt='%0.0f')

		annotations = [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]

		def update(val):
			y = properties[int(val)-1]
			plot1.set_ydata(y)
			ax.set_title(name+'\n'+property_name[int(val)+3])
			ax.set_ylim([min(y)*0.98, max(y)*1.02])
			
			if (not annotations[int(val)-1]):
				for xy in zip(x, y):                                                # <--
					ax.annotate(' (%s)' % xy[1], xy=xy, textcoords='data')
				annotations[int(val)-1] = True
			draw()
		sfreq.on_changed(update)
		show()
		
		
if __name__ == "__main__":
	main(sys.argv[1:])
	

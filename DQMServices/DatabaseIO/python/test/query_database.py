import sys
#import getopt
import sqlite3
import argparse
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
	return row + '\n'
#------------------------------------------------------------------------------      
def int16(data):
	return c_ushort(int(data,16)).value

def main(argv):
	database = ''
	default_database = 'db1.db'
	isDatabaseSpecified = False
	user = ''
	password = ''

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--database", help="path to database")
	parser.add_argument("-u", "--user", help="database username")
	parser.add_argument("-p", "--password", help="database password")
	parser.add_argument("-r", "--run", help="specify run number for SQL query", type=int)
	parser.add_argument("-l", "--luminosity", help="specify luminosity number for SQL query", type=int)
	parser.add_argument("-n", "--histogram_name", help="specify histogram name for SQL query, if you also specify run (but no luminosity), program will display plots of histogram properties vs luminosities")
	parser.add_argument("-s", "--noplot", help="suspends plot display", action="store_true")
	parser.add_argument("-f", "--filename", help="writes results to file (terminal output suspended)")

	args = parser.parse_args()
	
	if(not args.histogram_name and not args.luminosity and not args.run):
		parser.print_help()
		sys.exit(2)
		
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
		if(args.histogram_name and args.luminosity and args.run):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", args.luminosity, "AND RUN_NUMBER =", args.run, "AND NAME =", args.histogram_name
			arguments = (args.luminosity,args.run,name)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=? AND RUN_NUMBER=? AND NAME=?', arguments):
				result += parseRow(row)
		elif(args.histogram_name and not args.luminosity and args.run):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE RUN_NUMBER =", args.run, "AND NAME =", args.histogram_name
			arguments = (args.run,args.histogram_name)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE RUN_NUMBER=? AND NAME=?', arguments):
				lumisections.append(row[3])
				for property in range (0,19):
					properties[property].append(row[property+4])
				result += parseRow(row)
		elif(args.histogram_name and args.luminosity and not args.run):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", args.luminosity, "AND NAME =", args.histogram_name
			arguments = (args.luminosity,args.histogram_name)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=? AND NAME=?', arguments):
				result += parseRow(row)
		elif(args.histogram_name and not args.luminosity and not args.run):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE NAME =", args.histogram_name
			arguments = (args.histogram_name,)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE NAME=?', arguments):
				result += parseRow(row)
		elif(args.luminosity and args.run):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", args.luminosity, "AND RUN_NUMBER =", args.run
			arguments = (args.luminosity,args.run)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=? AND RUN_NUMBER=?', arguments):
				result += parseRow(row)
		elif(args.luminosity and not args.run):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION =", args.luminosity
			arguments = (args.luminosity,)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE LUMISECTION=?', arguments):
				result += parseRow(row)
		elif(not args.luminosity and args.run):
			print "QUERY: SELECT * FROM HISTOGRAM_VALUES WHERE RUN_NUMBER =", args.run		
			arguments = (args.run,)
			for row in c.execute('SELECT * FROM HISTOGRAM_VALUES WHERE RUN_NUMBER=?', arguments):
				result += parseRow(row)

		
		if (result != ''):
			if(args.filename):
				f = open(args.filename, 'w')
				f.write("NAME, PATH, RUN_NUMBER, LUMISECTION, ENTRIES, X_MEAN, X_MEAN_ERROR, X_RMS, X_RMS_ERROR, X_UNDERFLOW, X_OVERFLOW, Y_MEAN, Y_MEAN_ERROR, Y_RMS, Y_RMS_ERROR, Y_UNDERFLOW, Y_OVERFLOW, Z_MEAN, Z_MEAN_ERROR, Z_RMS, Z_RMS_ERROR, Z_UNDERFLOW, Z_OVERFLOW\n")
				f.write(result)
				f.close()
			else:
				print "\nNAME, PATH, RUN_NUMBER, LUMISECTION, ENTRIES, X_MEAN, X_MEAN_ERROR, X_RMS, X_RMS_ERROR, X_UNDERFLOW, X_OVERFLOW, Y_MEAN, Y_MEAN_ERROR, Y_RMS, Y_RMS_ERROR, Y_UNDERFLOW, Y_OVERFLOW, Z_MEAN, Z_MEAN_ERROR, Z_RMS, Z_RMS_ERROR, Z_UNDERFLOW, Z_OVERFLOW\n"
				print result
		else:
			print "No results!"
			sys.exit(2)
		
	except sqlite3.Error as e:
		print "ERROR: An error occurred:", e.args[0]
	
	if(args.histogram_name and not args.luminosity and args.run and not args.noplot):
		print "\nPlots legend:\n ENTRIES\t1\n X_MEAN\t\t2\n X_MEAN_ERROR\t3\n X_RMS\t\t4\n X_RMS_ERROR\t5\n X_UNDERFLOW\t6\n X_OVERFLOW\t7\n Y_MEAN\t\t8\n Y_MEAN_ERROR\t9\n Y_RMS\t\t10\n Y_RMS_ERROR\t11\n Y_UNDERFLOW\t12\n Y_OVERFLOW\t13\n Z_MEAN\t\t14\n Z_MEAN_ERROR\t15\n Z_RMS\t\t16\n Z_RMS_ERROR\t17\n Z_UNDERFLOW\t18\n Z_OVERFLOW\t19\n"

		fig = plt.figure()
		fig.canvas.set_window_title('Histograms\' properties')
		ax = fig.add_subplot(111)
		plt.xticks(lumisections)
		x = lumisections
		y = properties[0]
		plot1, = ax.plot(x, y, 'ro')
		plt.axis([min(x)-2, max(x)+2, min(y)*0.98-0.0001, max(y)*1.02+0.0001]) #0 values generate bottom==top error for axis, thats why +-0.0001
		for xy in zip(x, y):                                                
			ax.annotate(' (%s)' % xy[1], xy=xy, textcoords='data')

		plt.xlabel('Luminosity')
		plt.title(args.histogram_name+'\n'+property_name[4])
		plt.grid()
		
		ax1 = subplot(111)
		subplots_adjust(bottom=0.25)	
		axcolor = 'lightgoldenrodyellow'
		axfreq = axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
		sfreq = Slider(axfreq, 'Property', 1, 19, valinit=1, valfmt='%0.0f')

		annotations = [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]

		def update(val):
			y = properties[int(val)-1]
			plot1.set_ydata(y)
			ax.set_title(args.histogram_name+'\n'+property_name[int(val)+3])
			ax.set_ylim([min(y)*0.98-0.0001, max(y)*1.02+0.0001])
			
			if (not annotations[int(val)-1]):
				for xy in zip(x, y):                                                # <--
					ax.annotate(' (%s)' % xy[1], xy=xy, textcoords='data')
				annotations[int(val)-1] = True
			draw()
		sfreq.on_changed(update)
		
		show()
		
		
if __name__ == "__main__":
	main(sys.argv[1:])
	

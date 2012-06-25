#!/usr/bin/env python

################################################
#
# printTopTriggers.py
#
#
# Usage: prints out the N max individual rates
#             and their cumulative rate
#
#
# Author : Michael Luk, TMD, 25 May 12
#
################################################


########################################
#
# banner
#
def banner():
        print '''
        +--------------------------------------------------------------
        |
        | printTopTriggers.py
        |
	| Usage:
        |      ./printTopTriggers.py -i rateEff.root (-m ntriggers default=10)
        |
        | author: Michael Luk, 25 May 2012
        |
        +--------------------------------------------------------------
            '''
banner()
_legend = "[printTopTriggers:] "        

#####################
#
# options
#
######################
from optparse import OptionParser
add_help_option = "./printTopTriggers.py -ACTION [other options]"

parser = OptionParser(add_help_option)
parser.add_option("-i", "--in-file", dest="infile", default=None, 
		  help="infile")
parser.add_option("-n", "--n-triggers", dest="ntriggers", default=10, 
		  help="number of triggers to print out")
parser.add_option("-m", "--histogram", dest="histogram", default="unique", 
		  help="which histogram to run over")
print _legend, 'parsing command line options...',
(options, args) = parser.parse_args()
print 'done'


####################
#
# Main Code
#
####################


import copy
from ROOT import TFile, TH1D, TIter
import ROOT


#used to order a dictionary
def itemgetter(*items):
	if len(items) == 1:
		item = items[0]
		def g(obj):
			return obj[item]
	else:
		def g(obj):
			return tuple(obj[item] for item in items)
	return g

#main printing function
def uniqueRate(_sFileName,_nTopTrigs):
	#opens rootfile
        f_s    = TFile(_sFileName,"read")

	#gets "unique" histogram - standard output of RateEff
	_shist = copy.deepcopy(f_s.Get(options.histogram))

	m_rate = {}
	for _ibins in range(0,_shist.GetNbinsX()+1):
		_nsig = _shist . GetBinContent(_ibins)
		_name = _shist . GetXaxis() . GetBinLabel(_ibins)
		m_rate[_name] = _nsig
		#print _nsig
	sorted_rates = sorted(m_rate.items(), key=itemgetter(1), reverse = True)

	counter     = 0
	rateCounter = 0.
	for _key in sorted_rates:
		counter += 1
		if counter == 1:
			print '----------------------------------------------------------------------------------------------------------------------------'
			print '%-80s | % .10s |'% ("Trigger Name "," Rate [Hz]")
			print '----------------------------------------------------------------------------------------------------------------------------'

		
		print str(counter)+")",
		print '%-80s | % .3f |'% (_key[0], _key[1])
		rateCounter += _key[1]
		if counter == _nTopTrigs:
			break

	print '----------------------------------------------------------------------------------------------------------------------------'
	print _legend,'total rate of top',counter,'= ','%.3f' % rateCounter ,'Hz'
	return
#run it!
if __name__ == '__main__':
	uniqueRate(options.infile, options.ntriggers)

### assumes QCD_Pt-.../res/ folders are in current directory ###

import glob, os, string

folderList = glob.glob('QCD*')

print '------------------------------------------------------------'
print 'directory'
print '    total,', 'passed,', 'efficiency'
print '------------------------------------------------------------'

for folder in folderList:
	eventsTotalList = []
	eventsPassedList = []
	fileList = glob.glob(folder + '/res/*.stdout')
	#if fileList != []:
	for f in fileList:
		fileObj = open(f,'r')
		for line in fileObj:
			if line.count('Events total') != 0:
				# events total line.split()[4]
				# events passed line.split()[7]
				# events failed line.split()[10]				
				eventsTotalList.append(int(line.split()[4]))
				eventsPassedList.append(int(line.split()[7]))
	sumTotal = sum(eventsTotalList)
	sumPassed = sum(eventsPassedList)
	if (type(sumTotal) is int) & (sumTotal > 0):
		print os.path.basename(folder)
		print '    {0}, {1}, {2}'.format(sumTotal, sumPassed, sumPassed/float(sumTotal))
	else:
		print os.path.basename(folder)
		print '    no logs found'

print '------------------------------------------------------------'

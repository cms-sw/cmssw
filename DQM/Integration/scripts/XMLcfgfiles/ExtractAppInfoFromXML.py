#!/usr/bin/env python3
"""Syntax:
	ExtracAppInfoFromXML [-sapc] file
Parameters:
	file	file from where to read a RCMS configuration
	-s  	list application servers found in the XML file
	-p  	list the ports used found in the XML file
	-a  	list the names of the applications configured in the XML file
	-c  	list the cfg (eg dqmfu09-1_cfg.py) files
Notes:	
	The default behavior is to present a table organized in the following way
		SERVER PORT CFG_FILE APP_NAME
	which is equivalent to using -sapc
	
	The options selected and their order will affect teeh fields shown and their
	respective sorting. eg.
		-sa will only show SERVER and APP_NAME and will sort first by SERVER and
		 then by APP_NAME 
		 
	OUTPUT is always unique in a per row bases
"""
from __future__ import print_function
################################################################################
from builtins import range
import sys, os.path
from xml.dom import minidom
################################################################################
# Some module's global variables.
xmldoc=""

def printXMLtree(head,l=0,bn=0):
	tabs=""
	for a in range(l):
		tabs+="\t"
	try:
		print("[%d-%d-%d]"%(l,bn,head.nodeType)+tabs+"+++++>"+head.tagName)
	except AttributeError as e:
		print("[%d-%d-%d]"%(l,bn,head.nodeType)+tabs+"+++++>"+str(e))
	print("[%d-%d-%d-v]"%(l,bn,head.nodeType)+tabs+"."+ (head.nodeValue or "None"))
	try:
		for katt,vatt in head.attributes.items():
			if katt!="environmentString":
				print(tabs+"%s=%s"%(katt,vatt))
			else:
				print(tabs+"%s= 'Some Stuff'"%(katt,))
	except:
		pass	
	i=0
	for node in head.childNodes:
		printXMLtree(node,l+1,i)
		i+=1
################################################################################
def compactNodeValue(head):
	firstborne=None
	for item in head.childNodes:
		if item.nodeType == 3:
			firstborne = item
			break
	if not firstborne:
		return
	for item in head.childNodes[1:]:
		if item.nodeType == 3:
			firstborne.nodeValue+=item.nodeValue
			item.nodeValue=None
			
################################################################################
def appendDataXML(head):
	"""Parses information that's XML format from value to the Docuemnt tree"""
	compactNodeValue(head)
	if head.firstChild.nodeValue:
		newNode=minidom.parseString(head.firstChild.nodeValue)
		for node in newNode.childNodes:
			head.appendChild(node.cloneNode(True))
		newNode.unlink()			
################################################################################
def getAppNameFromCfg(filename):
	"""it searches for the line containing the string consumerName, usually
	found as a property of the process, and returns the set value found.
	eg. 
	matches line:
		process.EventStreamHttpReader.consumerName = 'EcalEndcap DQM Consumer' 
	returns:
		EcalEndcap DQM Consumer	
	"""
	try:
		f = open(filename)
		consumer = f.readline()
		name=""
		while consumer :
			consumer=consumer.strip()
			if "consumerName" in consumer:
				name=consumer[consumer.index("'")+1:consumer.index("'",consumer.index("'")+1)] 
				break
			consumer = f.readline()
		f.close()
	except:
		sys.stderr.write("WARNING: Unable to open file: " + filename + " from <configFile> section of XML\n")
		name = "CONFIG FILE IS M.I.A"        
	return name
################################################################################
def getProcNameFromCfg(filename):
	"""it searches for the line containing the string consumerName, usually
	found as a property of the process, and returns the set value found.
	eg. 
	matches line:
		process = cms.Process ("ECALDQM") 
	returns:
		ECALDQM
	"""
	try:
		f = open(filename)
	except:
		sys.stderr.write("Unable to open file: " + filename + " from <configFile> section of XML\n")
		raise IOError
	consumer = f.readline()
	name=""
	while consumer :
		consumer=consumer.strip()
		if "cms.Process(" in consumer:
			name=consumer[consumer.index("(")+2:consumer.index(")")-1] 
			break
		consumer = f.readline()
	f.close()
	return name
################################################################################
def filterNodeList(branch1,nodeList):
	if len(branch1) > 0:
		branch=branch1[:len(branch1)]
		idx=0
		for item in range(len(nodeList)):
			vals=[v for (k,v) in nodeList[idx].attributes.items()]
			if branch[0] not in vals:
				del nodeList[idx]
			else:
				idx=idx+1
		del branch[0]
	elif len(branch1)==0:
		return nodeList	
	return filterNodeList(branch,nodeList)
		
################################################################################
def fillTable(order,branch=[]):
	global xmldoc
	table={} 
	if len(order)==0: 
		return table
	key=min(order.keys())	
	k=order[key]
	order.pop(key)
	if k=="s":
		lista=xmldoc.getElementsByTagName("XdaqExecutive")
		lista=filterNodeList(branch,lista)
		for item in lista:
			table[item.attributes["hostname"].value]=""
		for item in table.keys():
			table[item]=fillTable(order.copy(),branch + [item])
	elif k=="a":
		lista=xmldoc.getElementsByTagName("XdaqExecutive")
		lista=filterNodeList(branch,lista)
		for item in lista:
			pset=item.getElementsByTagName("parameterSet")
			if len(pset):
				arch=pset[0].firstChild.nodeValue[5:]
				appname=getAppNameFromCfg(arch) or getProcNameFromCfg(arch)
				table[appname]=""
			else:
				App=item.getElementsByTagName("xc:Application")
				table[App[0].attributes["class"].value]=""
		for item in table.keys():
			table[item]=fillTable(order.copy(),branch)
	elif k=="p":
		lista=xmldoc.getElementsByTagName("XdaqExecutive")
		lista=filterNodeList(branch,lista)
		for item in lista:
			table[item.attributes["port"].value]=""
		for item in table.keys():
			table[item]=fillTable(order.copy(),branch + [item])
	elif k=="c":
		lista=xmldoc.getElementsByTagName("XdaqExecutive")
		lista=filterNodeList(branch,lista)
		for item in lista:
			pset=item.getElementsByTagName("parameterSet")
			if not len(pset):
				table["No additional file"]=""	
			else:
				table[pset[0].firstChild.nodeValue]=""
		for item in table.keys():
			table[item]=fillTable(order.copy(),branch)
	else:
		pass
	return table
################################################################################
def SortAndGrid(table,order):
	"""table => {s:{p:{c:{a:{}}}}}"""
	grid=[]
	for (server,ports) in table.items():
		for (port,configfiles) in ports.items():
			for (configfile,appnames) in configfiles.items():
				for appname in appnames.keys():
					line=[]
					for col in order.values():
						if col=="s":
							line.append(server)
						if col=="p":
							line.append(port)
						if col=="c":
							line.append(configfile)
						if col=="a":
							line.append(appname)
					grid.append(line)
	grid.sort()
	return grid
################################################################################
def printGrid(grid):
	numcols=len(grid[0])
	PPGrid=grid[:]
	maxs=[]
	for col in range(numcols):
		maxs.append(0)
	for line in grid:
		for col in range(numcols):
			if len(line[col])>maxs[col]:
				maxs[col]=len(line[col])
	for line in PPGrid:
		pline=""
		for col in range(numcols):
			pline+=line[col].ljust(maxs[col]+2)
		print(pline)
			
################################################################################	
#getAppInfo                                                                    #
################################################################################
def getAppInfo(XMLf,s=0,a=2,p=1,c=3):
	"""	getAppInfo(XMLf,s=0,a=2,p=1,c=3) takes the file name of a valid RCMS 
		configuration and 4	variables that represent which fields are desired 
		and in which order. 
		
		It returns a touple containing a directory that contains all the 
		relevant information in the XMLf file and a list of rows each row 
		containing the fiels specified by the other four variables in the r
		espective order
		
		The fields are Servers (s) ports(p) Appnames a.k.a. consumer names(a) 
		and consumer config file. (Note: The consumerName is directly extracted 
		from the config file.) if one field is not desired it should be assigned
		a value of -1 eg s=-1. other wise their value is mapped from smallest to
		largest ==> left to right. Note the default values, they will take 
		precedence if not specifyed giving unexpected results
	"""
	global xmldoc
	try: 
		os.path.exists(XMLf)
	except:
		sys.stderr.write('File doesn\'t exist\n')
		sys.exit(2)
	try:
		xmldoc = minidom.parse(XMLf)
	except IOError:
		sys.stderr.write('Unable to locate file ' +XMLf +'\n')
		return ({},[])
	except:
		sys.stderr.write('Parser error\n')
		return ({},[])
		
	configFileNodes=xmldoc.getElementsByTagName("configFile")
	for node in configFileNodes:
		appendDataXML(node)
	## The table is always filled in a specific order, to properly get the data
	order={0:"s",1:"p",3:"a",2:"c"}
	#try:
	table=fillTable(order)
	#except:
	#	return ({},[])
	del order
	order={}
	if a != -1:
		order[a]="a"
	if c != -1:
		order[c]="c"
	if s != -1:
		order[s]="s"
	if p != -1:
		order[p]="p"
	grid=SortAndGrid(table,order)
	#printXMLtree(xmldoc)	
	#Clean Up
	xmldoc.unlink()
	return (table,grid)
	                                                          
################################################################################
if __name__ == "__main__":             
	XMLfile=""
	args=sys.argv
	args.remove(args[0])
	options=""
	for arg in args:
		if arg.startswith("-"):
			options+=arg.strip("-")
		else:
			XMLfile=arg
	if options.count("s")+options.count("a")+options.count("p")+options.count("c")!=len(options):
		sys.stderr.write(  "Sintax Error unrecognised option" )
		sys.stderr.write( __doc__ )
		sys.exit(2)
	if options.count("s")+options.count("a")+options.count("p")+options.count("c")==0:
		(apptable,appinfo)=getAppInfo(XMLfile)
	else:
		(apptable,appinfo)=getAppInfo(XMLfile,options.find("s"),options.find("a"),options.find("p"),options.find("c"))
	if appinfo != []:
		printGrid(appinfo)
	apptable

#!/usr/bin/env python

from xml.dom import minidom
from crabutil import colors
import numpy as numpy
import sys

color = colors.Paint()

class Parser():

    def __init__(self, filename, verbose):

        # parse an xml file by name
        self.mydoc = minidom.parse(filename)
        cNodes = self.mydoc.childNodes

	if verbose:
	        print "\nReading file..."
        	print color.BOLD + cNodes[0].toxml() + color.ENDC
	        print "\n"

    def GetFromXML(self, verbose):

        command = []
        samples_node = self.mydoc.getElementsByTagName('samples')
        for samples in samples_node:
            dataset_node = samples.getElementsByTagName('dataset')
            for dataset in dataset_node:

		try:
			samplename = dataset.getElementsByTagName("name")[0]
        	        typemode = dataset.getElementsByTagName("type")[0]
                	era = dataset.getElementsByTagName("era")[0]
	                year = dataset.getElementsByTagName("year")[0]
			xangle = dataset.getElementsByTagName("xangle")[0]
			mass = dataset.getElementsByTagName("mass")[0]
	                configfile = dataset.getElementsByTagName("config")[0]
        	        nevents = dataset.getElementsByTagName("eventsperjob")[0]
                	tag = dataset.getElementsByTagName("tagname")[0]
                	enable = dataset.getElementsByTagName("enable")[0]
			with_dataset = dataset.getElementsByTagName("with_dataset")[0]
			outLFNDirBase = dataset.getElementsByTagName("outLFNDirBase")[0]
			command.append([str(samplename.firstChild.data), str(typemode.firstChild.data), str(era.firstChild.data), str(year.firstChild.data), str(xangle.firstChild.data), str(mass.firstChild.data), str(configfile.firstChild.data), str(nevents.firstChild.data), str(tag.firstChild.data), str(enable.firstChild.data), str(with_dataset.firstChild.data), str(outLFNDirBase.firstChild.data)])
               		if verbose:
			   print color.OKBLUE + "\tSample: " + samplename.firstChild.data + color.ENDC
        	           print color.OKBLUE + "\tType: " + typemode.firstChild.data + color.ENDC
                	   print color.OKBLUE + "\tEra: " + era.firstChild.data + color.ENDC
	                   print color.OKBLUE + "\tYear: " + year.firstChild.data + color.ENDC
			   print color.OKBLUE + "\tX-Angle: " + xangle.firstChild.data + color.ENDC		   
			   print color.OKBLUE + "\tMass: " + mass.firstChild.data + color.ENDC
	                   print color.OKBLUE + "\tConfig File: " + configfile.firstChild.data + color.ENDC
        	           print color.OKBLUE + "\tN Events per job: " + nevents.firstChild.data + color.ENDC
                	   print color.OKBLUE + "\tTag Name: " + tag.firstChild.data + color.ENDC
                	   print color.OKBLUE + "\tEnable: " + enable.firstChild.data + color.ENDC
			   print color.OKBLUE + "\tWith Dataset: " + with_dataset.firstChild.data + color.ENDC
                	   print color.OKBLUE + "\toutLFNDirBase: " + outLFNDirBase.firstChild.data + color.ENDC
		
		except:
                        print color.FAIL+color.BOLD+'\tFailed to get all the parameters from XML file! Please, check your XML file, there is(are) some error(s)!'+color.ENDC+color.HEADER+color.ENDC
                        exit(0)

        return command


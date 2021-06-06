import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as opts
import csv
from io import open

options = opts.VarParsing ('standard')

options.register('MagField',
				 None,
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.float,
				 'Magnetic field value in Tesla')
options.register('Year',
				 None,
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.string,
				 'Current year for versioning')
options.register('Version',
				 None,
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.string,
				 'Template DB object version')
options.register('Append',
				 None,
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.string,
				 'Any additional string to add to the filename, i.e. "bugfix", etc.')
options.register('Map',
				 '../data/template2D_phase1_2017_IOV1/IOV1_phase1_map.csv',
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.string,
				 'Path to map file')
options.register('Delimiter',
				 ',',
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.string,
				 'Delimiter in csv file')
options.register('Quotechar',
				 '"',
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.string,
				 'Quotechar in csv file')
options.register('TemplateFilePath',
				 'CondTools/SiPixel/data/template2D_phase1_2017_IOV1',
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.string,
				 'Location of template files')
options.register('GlobalTag',
				 'auto:phase2_realistic',
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.string,
				 'Global tag for this run')
options.register('numerator',
				 False,
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.bool,
				 'Switch on to produce the DB object for the numerators of the reweighting factors')
options.register('denominator',
				 False,
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.bool,
				 'Switch on to produce the DB object for the denominators of the reweighting factors')
options.register('useVectorIndices',
				 False,
				 opts.VarParsing.multiplicity.singleton,
				 opts.VarParsing.varType.bool,
				 'Switch on in case Morris uses vector indices in csv file, eg. [0,(N-1)] instead of [1,N]')
options.parseArguments()

if options.numerator==False and options.denominator==False :
	print('ERROR: Neither numerator nor denominator option was selected. Please rerun with numerator/denominator=True options.')
	quit()
if options.numerator==True and options.denominator==True :
	print('ERROR: Both numerator and denominator options are true. Please rerun with only one of numerator/denominator=True.')
	quit()

MagFieldValue = 10.*options.MagField #code needs it in deciTesla
print('\nMagField = %f deciTesla \n'%(MagFieldValue))
version = options.Version
print('\nVersion = %s \n'%(version))
magfieldstrsplit = str(options.MagField).split('.')
MagFieldString = magfieldstrsplit[0]
if len(magfieldstrsplit)>1 :
	MagFieldString+=magfieldstrsplit[1]

#open the map file
mapfile = open(options.Map,'rU', newline='')
#read the csv file into a reader
mapfilereader = csv.reader(mapfile,delimiter=options.Delimiter,quotechar=options.Quotechar)
#separate into the different sections
barrel_rule_lines = []; endcap_rule_lines = []
barrel_exception_lines = []; endcap_exception_lines = []
sections = [barrel_rule_lines, endcap_rule_lines, barrel_exception_lines, endcap_exception_lines]
i=0; line = next(mapfilereader)
for i in range(len(sections)) :
	while line[0].find('TEMPLATE ID')==-1 : #skip to just before the section of info
		line=next(mapfilereader)
	try :
		line=next(mapfilereader)
	except StopIteration :
		print('Done reading input file')
		break
	while line[1]!='' : #add the lines that are the barrel rules
		sections[i].append(line) 
		try :
			line=next(mapfilereader)
		except StopIteration :
			print('Done reading input file')
			break
#print 'barrel rules = %s\nendcap rules = %s\nbarrel exceptions = %s\nendcap exceptions = %s'%(barrel_rule_lines,endcap_rule_lines,barrel_exception_lines,endcap_exception_lines) #DEBUG
#Make the lists of location strings and template IDs
barrel_locations = []
barrel_template_IDs = []
endcap_locations = []
endcap_template_IDs = []
template_filenames = []
prefix = options.TemplateFilePath+'/template_summary2D_zp'
suffix = '.out'
for s in range(len(sections)) :
	for line in sections[s] :
	#	print 'reading line: %s'%(line) #DEBUG
		template_ID_s = line[0]
		while len(template_ID_s)<4 :
			template_ID_s='0'+template_ID_s
		newtemplatefilename = prefix+template_ID_s+suffix
		template_ID = int(template_ID_s)
		if not newtemplatefilename in template_filenames :
			template_filenames.append(newtemplatefilename)
		if s%2==0 :
			lay, lad, mod = line[1], line[2], line[3]
	#		print '	lay = %s, lad = %s, mod = %s'%(lay, lad, mod) #DEBUG
			#barrel ID strings are "layer_ladder_module"
			laysplit = lay.split('-'); firstlay=int(laysplit[0]); lastlay= int(laysplit[1])+1 if len(laysplit)>1 else firstlay+1
			for i in range(firstlay,lastlay) :
				lay_string = str(i)+'_'
				ladsplit = lad.split('-'); firstlad=int(ladsplit[0]); lastlad= int(ladsplit[1])+1 if len(ladsplit)>1 else firstlad+1
				for j in range(firstlad,lastlad) :
					lad_string = lay_string+str(j)+'_'
					modsplit = mod.split('-'); firstmod=int(modsplit[0]); lastmod= int(modsplit[1])+1 if len(modsplit)>1 else firstmod+1
					for k in range(firstmod,lastmod) :
						location_string = lad_string+str(k)
						if s==0 :
	#						print '	Adding with location string "%s" and template ID %d'%(location_string,template_ID) #DEBUG
							barrel_locations.append(location_string)
							barrel_template_IDs.append(template_ID)
						else :
							location_index = barrel_locations.index(location_string)
							barrel_template_IDs[location_index]=template_ID
		else : 
			disk, blade, side, panel = line[1], line[2], line[3], line[4]
			#endcap ID strings are "disk_blade_side_panel"
			disksplit = disk.split('-'); firstdisk=int(disksplit[0]); lastdisk = int(disksplit[1])+1 if len(disksplit)>1 else firstdisk+1
			for i in range(firstdisk,lastdisk) :
				disk_string = str(i)+'_'
				bladesplit = blade.split('-'); firstblade=int(bladesplit[0]); lastblade = int(bladesplit[1])+1 if len(bladesplit)>1 else firstblade+1
				for j in range(firstblade,lastblade) :
					blade_string = disk_string+str(j)+'_'
					sidesplit = side.split('-'); firstside=int(sidesplit[0]); lastside = int(sidesplit[1])+1 if len(sidesplit)>1 else firstside+1
					for k in range(firstside,lastside) :
						side_string = blade_string+str(k)+'_'
						panelsplit = panel.split('-'); firstpanel=int(panelsplit[0]); lastpanel = int(panelsplit[1])+1 if len(panelsplit)>1 else firstpanel+1
						for m in range(firstpanel,lastpanel) :
							location_string = side_string+str(m)
							if s==1 :
								endcap_locations.append(location_string)
								endcap_template_IDs.append(template_ID)
							else :
								location_index = endcap_locations.index(location_string)
								endcap_template_IDs[location_index]=template_ID
#Debug print out assignments
#print 'BARREL ASSIGNMENTS:' #DEBUG
#for i in range(len(barrel_locations)) : #DEBUG
	#tempid = barrel_template_IDs[i] #DEBUG
	#lay, lad, mod = barrel_locations[i].split('_')[0], barrel_locations[i].split('_')[1], barrel_locations[i].split('_')[2] #DEBUG
	#print '	layer %s, ladder %s, module %s will have template ID %d assigned to it'%(lay,lad,mod,tempid) #DEBUG
#print 'ENDCAP ASSIGNMENTS:' #DEBUG
#for i in range(len(endcap_locations)) : #DEBUG
#	tempid = endcap_template_IDs[i] #DEBUG
#	disk, blade, side, panel = endcap_locations[i].split('_')[0], endcap_locations[i].split('_')[1], endcap_locations[i].split('_')[2], endcap_locations[i].split('_')[3] #DEBUG
#	print '	disk %s, blade %s, side %s, panel %s will have template ID %d assigned to it'%(disk,blade,side,panel,tempid) #DEBUG

from Configuration.StandardSequences.Eras import eras

process = cms.Process("SiPixel2DTemplateDBUpload",eras.Phase2)#C2)
process.load("CondCore.CondDB.CondDB_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.GlobalTag, '')

template_base = 'SiPixel2DTemplateDBObject_phase2_'+MagFieldString+'T'+'_v'+version
if options.numerator==True :
	template_base+='_num'
elif options.denominator==True :
	template_base+='_den'
if options.Append!=None :
	template_base+='_'+options.Append
#output SQLite filename
sqlitefilename = 'sqlite_file:'+template_base+'.db'

print('\nUploading %s with record SiPixel2DTemplateDBObjectRcd in file %s\n' % (template_base,sqlitefilename))

process.source = cms.Source("EmptyIOVSource",
		timetype = cms.string('runnumber'),
		firstValue = cms.uint64(1),
		lastValue = cms.uint64(1),
                interval = cms.uint64(1)
		)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
if options.numerator==True :
	process.PoolDBOutputService = cms.Service("PoolDBOutputService",
		DBParameters = cms.PSet(messageLevel = cms.untracked.int32(0),
                    authenticationPath = cms.untracked.string('.')
                    ),
		timetype = cms.untracked.string('runnumber'),
		connect = cms.string(sqlitefilename),
		toPut = cms.VPSet(cms.PSet(record = cms.string('SiPixel2DTemplateDBObjectRcd'),
		tag = cms.string(template_base)
		)
		)
		)
elif options.denominator==True :
	process.PoolDBOutputService = cms.Service("PoolDBOutputService",
	DBParameters = cms.PSet(messageLevel = cms.untracked.int32(0),
                         authenticationPath = cms.untracked.string('.')
                         ),
			timetype = cms.untracked.string('runnumber'),
			connect = cms.string(sqlitefilename),
			toPut = cms.VPSet(cms.PSet(record = cms.string('SiPixel2DTemplateDBObjectRcd'),
                            label=cms.string('unirradiated'),
                            tag = cms.string(template_base)
			)
			)
                    )
process.uploader = cms.EDAnalyzer("SiPixel2DTemplateDBObjectUploader",
	siPixelTemplateCalibrations = cms.vstring(template_filenames),
	theTemplateBaseString = cms.string(template_base),
	Version = cms.double(3.0),
	MagField = cms.double(MagFieldValue),
	detIds = cms.vuint32(1,2), #0 is for all, 1 is Barrel, 2 is EndCap
	barrelLocations = cms.vstring(barrel_locations),
	endcapLocations = cms.vstring(endcap_locations),
	barrelTemplateIds = cms.vuint32(barrel_template_IDs),
	endcapTemplateIds = cms.vuint32(endcap_template_IDs),
	useVectorIndices  = cms.untracked.bool(options.useVectorIndices),
	)
process.myprint = cms.OutputModule("AsciiOutputModule")
process.p = cms.Path(process.uploader)
process.CondDB.connect = sqlitefilename
process.CondDB.DBParameters.messageLevel = 0
process.CondDB.DBParameters.authenticationPath = './'

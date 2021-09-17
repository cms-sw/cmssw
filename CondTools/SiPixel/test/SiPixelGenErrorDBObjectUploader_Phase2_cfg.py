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
					 '../data/template1D_IOV0_phase1_MC/IOV0_phase1_MC_map.csv',
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
options.register('GenErrFilePath',
					 'CondTools/SiPixel/data/template1D_IOV0_phase1_MC',
					 opts.VarParsing.multiplicity.singleton,
					 opts.VarParsing.varType.string,
					 'Location of generr files')
options.register('GlobalTag',
					 'auto:phase2_realistic',
					 opts.VarParsing.multiplicity.singleton,
					 opts.VarParsing.varType.string,
					 'Global tag for this run')
options.register('useVectorIndices',
					 False,
					 opts.VarParsing.multiplicity.singleton,
					 opts.VarParsing.varType.bool,
					 'Switch on in case Morris uses vector indices in csv file, eg. [0,(N-1)] instead of [1,N]')
options.register('geometry',
                 'T5',
                 opts.VarParsing.multiplicity.singleton,
                 opts.VarParsing.varType.string,
                 'Tracker Geometry Default = T5')
options.parseArguments()

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
barrel_generr_IDs = []
endcap_locations = []
endcap_generr_IDs = []
template_filenames = []
prefix = options.GenErrFilePath+'/generror_summary_zp'
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
							barrel_generr_IDs.append(template_ID)
						else :
							location_index = barrel_locations.index(location_string)
							barrel_generr_IDs[location_index]=template_ID
		else : 
			disk, blade, side, panel = line[1], line[2], line[3], line[4]
			#endcap ID strings are "disk_blade_side_panel_plaquette"
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
								endcap_generr_IDs.append(template_ID)
							else :
								location_index = endcap_locations.index(location_string)
								endcap_generr_IDs[location_index]=template_ID
#Debug print out assignments
#print 'BARREL ASSIGNMENTS:' #DEBUG
#for i in range(len(barrel_locations)) : #DEBUG
#	tempid = barrel_generr_IDs[i] #DEBUG
#	lay, lad, mod = barrel_locations[i].split('_')[0], barrel_locations[i].split('_')[1], barrel_locations[i].split('_')[2] #DEBUG
#	print '	layer %s, ladder %s, module %s will have template ID %d assigned to it'%(lay,lad,mod,tempid) #DEBUG
#print 'ENDCAP ASSIGNMENTS:' #DEBUG
#for i in range(len(endcap_locations)) : #DEBUG
#	tempid = endcap_generr_IDs[i] #DEBUG
#	disk, blade, side = endcap_locations[i].split('_')[0], endcap_locations[i].split('_')[1], endcap_locations[i].split('_')[2], endcap_locations[i].split('_')[3] #DEBUG
#	print '	disk %s, blade %s, side %s, panel %s will have template ID %d assigned to it'%(disk,blade,side,panel,tempid) #DEBUG

from Configuration.StandardSequences.Eras import eras

process = cms.Process("SiPixelGenErrorDBUpload",eras.Phase2)#C2)
process.load("CondCore.CondDB.CondDB_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

geometry_cff = ''
recoGeometry_cff = ''
tGeometry = options.geometry
if tGeometry == 'T5':
    geometry_cff = 'GeometryExtended2026D17_cff'
    recoGeometry_cff = 'GeometryExtended2026D17Reco_cff'
    LA_value = 0.106
    tag = 'SiPixelLorentzAngle_Phase2_T5'
elif tGeometry == 'T6':
    geometry_cff = 'GeometryExtended2026D35_cff'
    recoGeometry_cff = 'GeometryExtended2026D35Reco_cff'
elif tGeometry == 'T14':
    geometry_cff = 'GeometryExtended2026D43_cff'
    recoGeometry_cff = 'GeometryExtended2026D43Reco_cff'
elif tGeometry == 'T15':
    geometry_cff = 'GeometryExtended2026D49_cff'
    recoGeometry_cff = 'GeometryExtended2026D49Reco_cff'
elif tGeometry == 'T16':
    geometry_cff = 'GeometryExtended2026D48_cff'
    recoGeometry_cff = 'GeometryExtended2026D48Reco_cff'
else:
    print("Unknown tracker geometry")
    print("What are you doing ?!?!?!?!")
    exit(1)
geometry_cff = 'Configuration.Geometry.' + geometry_cff
recoGeometry_cff = 'Configuration.Geometry.' + recoGeometry_cff
process.load(geometry_cff)
process.load(recoGeometry_cff)

global_tag_name = options.GlobalTag+'_'+tGeometry
#global_tag_name = options.GlobalTag+'_T15'

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, global_tag_name, '')

generror_base = 'SiPixelGenErrorDBObject_phase2_'+tGeometry+'_v'+version
if options.Append!=None :
	generror_base+='_'+options.Append
#output SQLite filename
sqlitefilename = 'sqlite_file:'+generror_base+'.db'

print('\nUploading %s with record SiPixelGenErrorDBObjectRcd in file %s\n' % (generror_base,sqlitefilename))

process.source = cms.Source("EmptyIOVSource",
							timetype = cms.string('runnumber'),
							firstValue = cms.uint64(1),
							lastValue = cms.uint64(1),
							interval = cms.uint64(1)
							)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
										  DBParameters = cms.PSet(messageLevel = cms.untracked.int32(0),
																  authenticationPath = cms.untracked.string('.')
																),
										  timetype = cms.untracked.string('runnumber'),
										  connect = cms.string(sqlitefilename),
										  toPut = cms.VPSet(cms.PSet(record = cms.string('SiPixelGenErrorDBObjectRcd'),
																	 tag = cms.string(generror_base)
																	)
															)
										)
process.uploader = cms.EDAnalyzer("SiPixelGenErrorDBObjectUploader",
								  siPixelGenErrorCalibrations = cms.vstring(template_filenames),
								  theGenErrorBaseString = cms.string(generror_base),
								  Version = cms.double(3.0),
								  MagField = cms.double(MagFieldValue),
								  detIds = cms.vuint32(1,2), #0 is for all, 1 is Barrel, 2 is EndCap
								  barrelLocations = cms.vstring(barrel_locations),
								  endcapLocations = cms.vstring(endcap_locations),
								  barrelGenErrIds = cms.vuint32(barrel_generr_IDs),
								  endcapGenErrIds = cms.vuint32(endcap_generr_IDs),
								  useVectorIndices  = cms.untracked.bool(options.useVectorIndices),
								)
process.myprint = cms.OutputModule("AsciiOutputModule")
process.p = cms.Path(process.uploader)
process.CondDB.connect = sqlitefilename
process.CondDB.DBParameters.messageLevel = 0
process.CondDB.DBParameters.authenticationPath = './'

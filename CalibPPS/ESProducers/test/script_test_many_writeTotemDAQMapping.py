from CondTools.CTPPS.mappings_PPSObjects_XML_cfi import *
import FWCore.ParameterSet.Config as cms
import os
import re
import subprocess
import sys

# Script which reads data from DB/XML and writes them to file
# Run with python3

# Data can be read from SQLite(Db) or from XML
fromDb = True
if(len(sys.argv)>1 and sys.argv[1].lower()=='false'):
  fromDb = False
  
  
# If True Mask and Mapping records will be read and wrote to file; If False only one of them will be processed
writeBothRecords = False
if(len(sys.argv)>2 and sys.argv[2].lower()=='true'):
  writeBothRecords = True
  
  
filesToRead = [totemTiming, timingDiamond, trackingStrip, totemT2] 
if(len(sys.argv)>3):
  filesToRead = [filesMap[mapName] for mapName in sys.argv[3:]]
  

# For each file change the variable values in the config so that they match the selected XML file and then run the config
test_script = os.path.join(os.path.dirname(os.path.realpath(__file__)),'test_writeTotemDAQMapping.py')
for fileContent in filesToRead:
    for fileInfo in fileContent["configuration"]:
        with open(test_script, 'r+') as f:
            content = f.read()
            # replace values specific for selected detector
            content = re.sub(r'subSystemName =.*', f'subSystemName = "{fileContent["subSystemName"]}"', content)
            content = re.sub(r'process.CondDB.connect =.*', f'process.CondDB.connect = "{fileContent["dbConnect"]}"', content)
            content = re.sub(r'process.totemDAQMappingESSourceXML.multipleChannelsPerPayload =.*', 
                             f'process.totemDAQMappingESSourceXML.multipleChannelsPerPayload = {fileContent["multipleChannelsPerPayload"]}', 
                             content)
            content = re.sub(r'process.totemDAQMappingESSourceXML.sampicSubDetId =.*', 
                             f'process.totemDAQMappingESSourceXML.sampicSubDetId = {fileInfo["sampicSubDetId"]}',
                             content)
            
            
            # replace name of output file and add adequate suffix to it ('_db' or '_xml')
            fileNameExt = "all_" + fileContent["subSystemName"] + "_db.txt" if fromDb else "all_" + fileContent["subSystemName"] +"_xml.txt"
            content = re.sub(r'fileName =.*', f'fileName = cms.untracked.string("{fileNameExt}"),' , content)
          
            # replace values specific for selected files
            content = re.sub(r'minIov =.*', f'minIov = {fileInfo["validityRange"].start()}', content)
            content = re.sub(r'maxIov =.*', f'maxIov = {fileInfo["validityRange"].end()}', content)
            content = re.sub(r'mappingFileNames =.*', f'mappingFileNames = {fileInfo["mappingFileNames"]},', content)
            content = re.sub(r'maskFileNames =.*', f'maskFileNames = {fileInfo["maskFileNames"]},', content)
               
            if fromDb:
              sourceClass = "PoolDBESSource"  
              obj = ""           
            else:    
              sourceClass = "TotemDAQMappingESSourceXML"     
              obj =  "totemDAQMappingESSourceXML"

            # replace records which needs to be get from DBSource (one or two of them)
            dbRecords = ""
            if writeBothRecords or fileContent == analysisMask:
              dbRecords += "cms.PSet(\nrecord = cms.string('TotemAnalysisMaskRcd'),\n tag = cms.string('AnalysisMask'),\n label = cms.untracked.string(subSystemName)),\n"
              replacement = f'process.es_prefer_totemTimingMapping = cms.ESPrefer("{sourceClass}", "{obj}", \
                {"TotemAnalysisMaskRcd" if fromDb else "TotemReadoutRcd"}=cms.vstring(f"TotemAnalysisMask/{fileContent["subSystemName"]}"))'
            
            if writeBothRecords or fileContent != analysisMask:
              dbRecords += "cms.PSet(\nrecord = cms.string('TotemReadoutRcd'),\n tag = cms.string('DiamondDAQMapping'),\n label = cms.untracked.string(subSystemName))\n"
              replacement = f'process.es_prefer_totemTimingMapping = cms.ESPrefer("{sourceClass}", "{obj}", \
                TotemReadoutRcd=cms.vstring(f"TotemDAQMapping/{fileContent["subSystemName"]}"))'

            
            enters = "\n\n\n\n" if not writeBothRecords else ""
            
            content = re.sub(r'readMap = cms.untracked.bool.*', f"readMap = cms.untracked.bool({writeBothRecords or fileContent != analysisMask}),", content)
            content = re.sub(r'readMask = cms.untracked.bool.*', f"readMask = cms.untracked.bool({writeBothRecords or fileContent == analysisMask}),", content)
            content = re.sub(r'toGet = cms.VPSet\(\n((?:.*\n){9})', f"toGet = cms.VPSet(\n"+dbRecords+"))\n"+enters, content)       
            
            # replace values in ESPrefer to read data from DB or from XML
            content = re.sub(r'process.es_prefer_totemTimingMapping =.*', replacement, content)  
            
            f.seek(0)
            f.write(content)
            f.truncate()
            
            
        subprocess.run(f'cmsRun {os.environ["CMSSW_BASE"]}/src/CalibPPS/ESProducers/test/test_writeTotemDAQMapping.py' , shell=True)

    

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
  
filesToRead = [totemTiming, timingDiamond, trackingStrip, totemT2] 
if(len(sys.argv)>2):
  filesToRead = [filesMap[mapName] for mapName in sys.argv[2:]]


# For each file change the variable values in the config so that they match the selected XML file and then run the config
for fileContent in filesToRead:
    for fileInfo in fileContent["configuration"]:
        with open(f'{os.environ["CMSSW_BASE"]}/src/CalibPPS/ESProducers/test/test_writeTotemDAQMapping.py', 'r+') as f:        
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
            content = re.sub(r'fileName =.*', f'fileName = cms.untracked.string("{fileNameExt}")' , content)
          
            # replace values specific for selected files
            content = re.sub(r'minIov =.*', f'minIov = {fileInfo["validityRange"].start()}', content)
            content = re.sub(r'maxIov =.*', f'maxIov = {fileInfo["validityRange"].end()}', content)
            content = re.sub(r'mappingFileNames =.*', f'mappingFileNames = {fileInfo["mappingFileNames"]},', content)
            content = re.sub(r'maskFileNames =.*', f'maskFileNames = {fileInfo["maskFileNames"]},', content)
            
            # replace values in ESPrefer to read data from DB or from XML
            if fromDb:
              replacement = f'process.es_prefer_totemTimingMapping = cms.ESPrefer("PoolDBESSource", "", \
                TotemReadoutRcd=cms.vstring(f"TotemDAQMapping/{fileContent["subSystemName"]}"))'
            else:          
              replacement = f'process.es_prefer_totemTimingMapping = cms.ESPrefer("TotemDAQMappingESSourceXML", \
                "totemDAQMappingESSourceXML", TotemReadoutRcd=cms.vstring("TotemDAQMapping/{fileContent["subSystemName"]}"))'
                
            content = re.sub(r'process.es_prefer_totemTimingMapping =.*', replacement, content)            
          
            f.seek(0)
            f.write(content)
            f.truncate()
            
            
        subprocess.run(f'cmsRun {os.environ["CMSSW_BASE"]}/src/CalibPPS/ESProducers/test/test_writeTotemDAQMapping.py' , shell=True)

    

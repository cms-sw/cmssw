from CondTools.CTPPS.mappings_PPSObjects_XML_cfi import *
import FWCore.ParameterSet.Config as cms
import os
import re
import subprocess
import sys
import shutil

# Script which reads data from XML and drops objects to SQLite
# Run with python3

filesToWrite = [totemTiming, timingDiamond, trackingStrip, totemT2, analysisMask] 
writeBothRecords = False

if(len(sys.argv)>1 and sys.argv[1].lower()=='true'):
  writeBothRecords = True

if(len(sys.argv)>2):
  filesToWrite = [filesMap[mapName] for mapName in sys.argv[2:]]


# For each file change the variable values in the config so that they match the selected XML file and then run the config
test_script = "write-ctpps-totem_daqmap_cfg.py"
orig_script = os.path.join(os.path.dirname(os.path.realpath(__file__)),test_script)
shutil.copyfile(orig_script, test_script)
for fileContent in filesToWrite:
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
          
          # replace values specific for selected files
          content = re.sub(r'minIov =.*', f'minIov = {fileInfo["validityRange"].start()}', content)
          content = re.sub(r'maxIov =.*', f'maxIov = {fileInfo["validityRange"].end()}', content)
          content = re.sub(r'mappingFileNames =.*', f'mappingFileNames = {fileInfo["mappingFileNames"]},', content)
          content = re.sub(r'maskFileNames =.*', f'maskFileNames = {fileInfo["maskFileNames"]},', content)
          
          
          mapRcd = ''
          maskRcd = ''
          if writeBothRecords or fileContent != analysisMask:
            mapRcd = 'TotemReadoutRcd'
          if writeBothRecords or fileContent == analysisMask:
            maskRcd = 'TotemAnalysisMaskRcd'
            
            
          content = re.sub(r'recordMap = cms.string.*', f"recordMap = cms.string('{mapRcd}'),", content)
          content = re.sub(r'recordMask = cms.string.*', f"recordMask = cms.string('{maskRcd}'),", content)
            
            
          f.seek(0)
          f.write(content)
          f.truncate()
            
            
      subprocess.run(f'cmsRun ./{test_script}' , shell=True)
    

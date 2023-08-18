from CondTools.CTPPS.mappings_PPSObjects_XML_cfi import *
import FWCore.ParameterSet.Config as cms
import os
import re
import subprocess
import sys
# Script which reads data from XML and drops objects to SQLite
# Run with python3

filesToWrite = [totemTiming, timingDiamond, trackingStrip, totemT2] 

if(len(sys.argv)>1):
  filesToWrite = [filesMap[mapName] for mapName in sys.argv[1:]]


# For each file change the variable values in the config so that they match the selected XML file and then run the config
for fileContent in filesToWrite:
    for fileInfo in fileContent["configuration"]:
      with open(f'{os.environ["CMSSW_BASE"]}/src/CondTools/CTPPS/test/write-ctpps-totem_daqmap_cfg.py', 'r+') as f:        
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
          
          f.seek(0)
          f.write(content)
          f.truncate()
            
            
      subprocess.run(f'cmsRun {os.environ["CMSSW_BASE"]}/src/CondTools/CTPPS/test/write-ctpps-totem_daqmap_cfg.py' , shell=True)
    

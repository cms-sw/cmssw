import re
import subprocess
import FWCore.ParameterSet.Config as cms

# Script which reads data from XML and drops objects to SQLite
# Run with python3

diamonds = {
  "dbConnect": "sqlite_file:CTPPSDiamondsScript_DAQMapping.db",
  "subSystemName" : "TimingDiamond",
  "sampicSubDetId": 6,
  "configuration": cms.VPSet(
    # 2016, before diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("1:min - 283819:max"),
      mappingFileNames = cms.vstring(),
      maskFileNames = cms.vstring()
    ),
    # 2016, after diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("283820:min - 292520:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2017
    cms.PSet(
      validityRange = cms.EventRange("292521:min - 310000:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2017.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2018
    cms.PSet(
      validityRange = cms.EventRange("310001:min - 339999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2018.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2022
    cms.PSet(
      validityRange = cms.EventRange("340000:min - 362919:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2022.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2023
    cms.PSet(
      validityRange = cms.EventRange("362920:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2023.xml"),
      maskFileNames = cms.vstring()
    )

  )
}

strips = {
    "dbConnect": "sqlite_file:CTPPSStrip_DAQMapping.db",
    "subSystemName": "TrackingStrip",
    "sampicSubDetId": 6,
    "configuration": cms.VPSet(
        # 2016, before TS2
        cms.PSet(
        validityRange = cms.EventRange("1:min - 280385:max"),
        mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2016_to_fill_5288.xml"),
        maskFileNames = cms.vstring()
        ),
        # 2016, during TS2
        cms.PSet(
        validityRange = cms.EventRange("280386:min - 281600:max"),
        mappingFileNames = cms.vstring(),
        maskFileNames = cms.vstring()
        ),
        # 2016, after TS2
        cms.PSet(
        validityRange = cms.EventRange("281601:min - 290872:max"),
        mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2016_from_fill_5330.xml"),
        maskFileNames = cms.vstring()
        ),
        # 2017
        cms.PSet(
        validityRange = cms.EventRange("290873:min - 311625:max"),
        mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2017.xml"),
        maskFileNames = cms.vstring()
        ),
        # 2018
        cms.PSet(
        validityRange = cms.EventRange("311626:min - 339999:max"),
        mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2018.xml"),
        maskFileNames = cms.vstring()
        ),
        # 2022
        cms.PSet(
        validityRange = cms.EventRange("340000:min - 999999999:max"),
        mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2022.xml"),
        maskFileNames = cms.vstring()
        )
    )
}

totemTiming = {
  "dbConnect": "sqlite_file:CTPPSTotemTiming_DAQMapping.db",
  "subSystemName": "TotemTiming",
  "sampicSubDetId" : 5,
  "configuration": cms.VPSet(
    # 2017, before detector inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("1:min - 310000:max"),
      mappingFileNames = cms.vstring(),
      maskFileNames = cms.vstring()
    ),
    # 2018
    cms.PSet(
      validityRange = cms.EventRange("310001:min - 339999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_timing_2018.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2022
    cms.PSet(
      validityRange = cms.EventRange("340000:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_timing_2022.xml"),
      maskFileNames = cms.vstring()
    )
  )
}

#-----------------------------------------------------------------

filesToWrite = [diamonds]  

# For each file change the variable values in the config so that they match the selected XML file and then run the config
for fileContent in filesToWrite:
    for fileInfo in fileContent["configuration"]:
      with open('CondTools/CTPPS/test/write-ctpps-totem_daqmap_cfg.py', 'r+') as f:        
          content = f.read()
          # replace values specific for selected detector
          content = re.sub(r'subSystemName =.*', f'subSystemName = "{fileContent["subSystemName"]}"', content)
          content = re.sub(r'process.CondDB.connect =.*', f'process.CondDB.connect = "{fileContent["dbConnect"]}"', content)
          content = re.sub(r'process.totemDAQMappingESSourceXML.sampicSubDetId =.*', 
                           f'process.totemDAQMappingESSourceXML.sampicSubDetId = cms.uint32({fileContent["sampicSubDetId"]})',
                           content)
          
          # replace values specific for selected files
          content = re.sub(r'minIov =.*', f'minIov = {fileInfo.validityRange.start()}', content)
          content = re.sub(r'maxIov =.*', f'maxIov = {fileInfo.validityRange.end()}', content)
          content = re.sub(r'mappingFileNames =.*', f'mappingFileNames = {fileInfo.mappingFileNames},', content)
          content = re.sub(r'maskFileNames =.*', f'maskFileNames = {fileInfo.maskFileNames},', content)
          
          f.seek(0)
          f.write(content)
          f.truncate()
            
            
      subprocess.run(f'cmsRun CondTools/CTPPS/test/write-ctpps-totem_daqmap_cfg.py' , shell=True)
    

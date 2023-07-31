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

totem_timing = {
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

files_to_write = [diamonds]  

for file_content in files_to_write:
    for file_info in file_content["configuration"]:
      with open('CondTools/CTPPS/test/write-ctpps-totem_daqmap_cfg.py', 'r+') as f:        
          content = f.read()
          content = re.sub(r'subSystemName =.*', f'subSystemName = "{file_content["subSystemName"]}"', content)
          content = re.sub(r'process.CondDB.connect =.*', f'process.CondDB.connect = "{file_content["dbConnect"]}"', content)
          content = re.sub(r'process.totemDAQMappingESSourceXML.sampicSubDetId =.*', f'process.totemDAQMappingESSourceXML.sampicSubDetId = cms.uint32({file_content["sampicSubDetId"]})', content)
          
          content = re.sub(r'min_iov =.*', f'min_iov = {file_info.validityRange.start()}', content)
          content = re.sub(r'max_iov =.*', f'max_iov = {file_info.validityRange.end()}', content)
          content = re.sub(r'mappingFileNames =.*', f'mappingFileNames = {file_info.mappingFileNames},', content)
          content = re.sub(r'maskFileNames =.*', f'maskFileNames = {file_info.maskFileNames},', content)
          
          f.seek(0)
          f.write(content)
          f.truncate()
            
            
      subprocess.run(f'cmsRun CondTools/CTPPS/test/write-ctpps-totem_daqmap_cfg.py' , shell=True)
    

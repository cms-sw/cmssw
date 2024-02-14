import FWCore.ParameterSet.Config as cms

timingDiamond = {
  "dbConnect": "sqlite_file:CTPPSTimingDiamond_DAQMapping.db",
  "subSystemName" : "TimingDiamond",
  "multipleChannelsPerPayload": False,
  "configuration": [
    # 2016, before diamonds inserted in DAQ
    {
      "sampicSubDetId": cms.uint32(6),
      "validityRange" : cms.EventRange("1:min - 283819:max"),
      "mappingFileNames": cms.vstring(),
      "maskFileNames" : cms.vstring()
    },
    # 2016, after diamonds inserted in DAQ
    {
      "sampicSubDetId": cms.uint32(6),
      "validityRange" : cms.EventRange("283820:min - 292520:max"),
      "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond.xml"),
      "maskFileNames" : cms.vstring()
    },
    # 2017
    {
      "sampicSubDetId": cms.uint32(6),
      "validityRange" : cms.EventRange("292521:min - 310000:max"),
      "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2017.xml"),
      "maskFileNames" : cms.vstring()
    },
    # 2018
    {
      "sampicSubDetId": cms.uint32(6),
      "validityRange" : cms.EventRange("310001:min - 339999:max"),
      "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2018.xml"),
      "maskFileNames" : cms.vstring()
    },
    # 2022
    {
      "sampicSubDetId": cms.uint32(6),
      "validityRange" : cms.EventRange("340000:min - 362919:max"),
      "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2022.xml"),
      "maskFileNames" : cms.vstring()
    },
    # 2023
    {
      "sampicSubDetId": cms.uint32(6),
      "validityRange" : cms.EventRange("362920:min - 999999999:max"),
      "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2023.xml"),
      "maskFileNames" : cms.vstring()
    }
  ]
}

trackingStrip = {
    "dbConnect": "sqlite_file:CTPPSTrackingStrip_DAQMapping.db",
    "subSystemName": "TrackingStrip",
    "multipleChannelsPerPayload": False,
    "configuration": [
        # 2016, before TS2
        {
          "sampicSubDetId": cms.uint32(6),
          "validityRange" : cms.EventRange("1:min - 280385:max"),
          "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2016_to_fill_5288.xml"),
          "maskFileNames" : cms.vstring()
        },
        # 2016, during TS2
        {
          "sampicSubDetId": cms.uint32(6),
          "validityRange" : cms.EventRange("280386:min - 281600:max"),
          "mappingFileNames": cms.vstring(),
          "maskFileNames" : cms.vstring()
        },
        # 2016, after TS2
        {
          "sampicSubDetId": cms.uint32(6),
          "validityRange" : cms.EventRange("281601:min - 290872:max"),
          "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2016_from_fill_5330.xml"),
          "maskFileNames" : cms.vstring()
        },
        # 2017
        {
          "sampicSubDetId": cms.uint32(6),
          "validityRange" : cms.EventRange("290873:min - 311625:max"),
          "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2017.xml"),
          "maskFileNames" : cms.vstring()
        },
        # 2018
        {
          "sampicSubDetId": cms.uint32(6),
          "validityRange" : cms.EventRange("311626:min - 339999:max"),
          "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2018.xml"),
          "maskFileNames" : cms.vstring()
        },
        # 2022
        {
          "sampicSubDetId": cms.uint32(6),
          "validityRange" : cms.EventRange("340000:min - 999999999:max"),
          "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2022.xml"),
          "maskFileNames" : cms.vstring()
        }
    ]
}

totemTiming = {
  "dbConnect": "sqlite_file:CTPPSTotemTiming_DAQMapping.db",
  "subSystemName": "TotemTiming",
  "multipleChannelsPerPayload": False,
  "configuration": [
    # 2017, before detector inserted in DAQ
    {
      "sampicSubDetId": cms.uint32(6),
      "validityRange" : cms.EventRange("1:min - 310000:max"),
      "mappingFileNames": cms.vstring(),
      "maskFileNames" : cms.vstring()
    },
    # 2018
    {
      "sampicSubDetId": cms.uint32(6),
      "validityRange" : cms.EventRange("310001:min - 339999:max"),
      "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_timing_2018.xml"),
      "maskFileNames" : cms.vstring()
    },
    # 2022
    {
      "sampicSubDetId": cms.uint32(5),
      "validityRange" : cms.EventRange("340000:min - 999999999:max"),
      "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_timing_2022.xml"),
      "maskFileNames" : cms.vstring()
    }
  ]
}

totemT2 = {
  "dbConnect": "sqlite_file:CTPPSTotemT2_DAQMapping.db",
  "subSystemName": "TotemT2",
  "multipleChannelsPerPayload": True,
  "configuration": [
    {
      "sampicSubDetId": cms.uint32(7),
      "validityRange" : cms.EventRange("1:min - 368022:max"),
      "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_nt2_2023.xml"),
      "maskFileNames" : cms.vstring()
    },
    {
      "sampicSubDetId": cms.uint32(7),
      "validityRange" : cms.EventRange("368023:min - 999999999:max"),
      "mappingFileNames": cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_nt2_2023_final.xml"),
      "maskFileNames" : cms.vstring()
    }
  ]
}

analysisMask = {
  "dbConnect": "sqlite_file:CTPPS_AnalysisMask.db",
  "subSystemName": "",
  "multipleChannelsPerPayload": False,
  "configuration": [
    {
      "sampicSubDetId": cms.uint32(7),
      "validityRange" : cms.EventRange("1:min - 999999999:max"),
      "mappingFileNames": cms.vstring(),
      "maskFileNames" : cms.vstring()
    }
  ]
}

filesMap = {
  "TotemTiming": totemTiming,
  "TimingDiamond":timingDiamond,
  "TrackingStrip": trackingStrip, 
  "TotemT2":totemT2,
  "AnalysisMask": analysisMask
}


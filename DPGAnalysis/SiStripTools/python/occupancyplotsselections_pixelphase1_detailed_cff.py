import FWCore.ParameterSet.Config as cms

# Pixel barrel : 4 layers x 8 modules
OccupancyPlotsPixelWantedSubDets = cms.VPSet ()
OccupancyPlotsBPIXWantedSubDets = cms.VPSet (
    cms.PSet(detSelection=cms.uint32(1001),detLabel=cms.string("BPIXL1m1"),selection=cms.untracked.vstring("0x1ef00ffc-0x12100004")), 
    cms.PSet(detSelection=cms.uint32(1002),detLabel=cms.string("BPIXL1m2"),selection=cms.untracked.vstring("0x1ef00ffc-0x12100008")), 
    cms.PSet(detSelection=cms.uint32(1003),detLabel=cms.string("BPIXL1m3"),selection=cms.untracked.vstring("0x1ef00ffc-0x1210000c")), 
    cms.PSet(detSelection=cms.uint32(1004),detLabel=cms.string("BPIXL1m4"),selection=cms.untracked.vstring("0x1ef00ffc-0x12100010")), 
    cms.PSet(detSelection=cms.uint32(1005),detLabel=cms.string("BPIXL1m5"),selection=cms.untracked.vstring("0x1ef00ffc-0x12100014")), 
    cms.PSet(detSelection=cms.uint32(1006),detLabel=cms.string("BPIXL1m6"),selection=cms.untracked.vstring("0x1ef00ffc-0x12100018")), 
    cms.PSet(detSelection=cms.uint32(1007),detLabel=cms.string("BPIXL1m7"),selection=cms.untracked.vstring("0x1ef00ffc-0x1210001c")), 
    cms.PSet(detSelection=cms.uint32(1008),detLabel=cms.string("BPIXL1m8"),selection=cms.untracked.vstring("0x1ef00ffc-0x12100020")), 
    cms.PSet(detSelection=cms.uint32(1011),detLabel=cms.string("BPIXL2m1"),selection=cms.untracked.vstring("0x1ef00ffc-0x12200004")), 
    cms.PSet(detSelection=cms.uint32(1012),detLabel=cms.string("BPIXL2m2"),selection=cms.untracked.vstring("0x1ef00ffc-0x12200008")), 
    cms.PSet(detSelection=cms.uint32(1013),detLabel=cms.string("BPIXL2m3"),selection=cms.untracked.vstring("0x1ef00ffc-0x1220000c")), 
    cms.PSet(detSelection=cms.uint32(1014),detLabel=cms.string("BPIXL2m4"),selection=cms.untracked.vstring("0x1ef00ffc-0x12200010")), 
    cms.PSet(detSelection=cms.uint32(1015),detLabel=cms.string("BPIXL2m5"),selection=cms.untracked.vstring("0x1ef00ffc-0x12200014")), 
    cms.PSet(detSelection=cms.uint32(1016),detLabel=cms.string("BPIXL2m6"),selection=cms.untracked.vstring("0x1ef00ffc-0x12200018")), 
    cms.PSet(detSelection=cms.uint32(1017),detLabel=cms.string("BPIXL2m7"),selection=cms.untracked.vstring("0x1ef00ffc-0x1220001c")), 
    cms.PSet(detSelection=cms.uint32(1018),detLabel=cms.string("BPIXL2m8"),selection=cms.untracked.vstring("0x1ef00ffc-0x12200020")), 
    cms.PSet(detSelection=cms.uint32(1021),detLabel=cms.string("BPIXL3m1"),selection=cms.untracked.vstring("0x1ef00ffc-0x12300004")), 
    cms.PSet(detSelection=cms.uint32(1022),detLabel=cms.string("BPIXL3m2"),selection=cms.untracked.vstring("0x1ef00ffc-0x12300008")), 
    cms.PSet(detSelection=cms.uint32(1023),detLabel=cms.string("BPIXL3m3"),selection=cms.untracked.vstring("0x1ef00ffc-0x1230000c")), 
    cms.PSet(detSelection=cms.uint32(1024),detLabel=cms.string("BPIXL3m4"),selection=cms.untracked.vstring("0x1ef00ffc-0x12300010")), 
    cms.PSet(detSelection=cms.uint32(1025),detLabel=cms.string("BPIXL3m5"),selection=cms.untracked.vstring("0x1ef00ffc-0x12300014")), 
    cms.PSet(detSelection=cms.uint32(1026),detLabel=cms.string("BPIXL3m6"),selection=cms.untracked.vstring("0x1ef00ffc-0x12300018")), 
    cms.PSet(detSelection=cms.uint32(1027),detLabel=cms.string("BPIXL3m7"),selection=cms.untracked.vstring("0x1ef00ffc-0x1230001c")), 
    cms.PSet(detSelection=cms.uint32(1028),detLabel=cms.string("BPIXL3m8"),selection=cms.untracked.vstring("0x1ef00ffc-0x12300020")), 
    cms.PSet(detSelection=cms.uint32(1031),detLabel=cms.string("BPIXL4m1"),selection=cms.untracked.vstring("0x1ef00ffc-0x12400004")), 
    cms.PSet(detSelection=cms.uint32(1032),detLabel=cms.string("BPIXL4m2"),selection=cms.untracked.vstring("0x1ef00ffc-0x12400008")), 
    cms.PSet(detSelection=cms.uint32(1033),detLabel=cms.string("BPIXL4m3"),selection=cms.untracked.vstring("0x1ef00ffc-0x1240000c")), 
    cms.PSet(detSelection=cms.uint32(1034),detLabel=cms.string("BPIXL4m4"),selection=cms.untracked.vstring("0x1ef00ffc-0x12400010")), 
    cms.PSet(detSelection=cms.uint32(1035),detLabel=cms.string("BPIXL4m5"),selection=cms.untracked.vstring("0x1ef00ffc-0x12400014")), 
    cms.PSet(detSelection=cms.uint32(1036),detLabel=cms.string("BPIXL4m6"),selection=cms.untracked.vstring("0x1ef00ffc-0x12400018")), 
    cms.PSet(detSelection=cms.uint32(1037),detLabel=cms.string("BPIXL4m7"),selection=cms.untracked.vstring("0x1ef00ffc-0x1240001c")), 
    cms.PSet(detSelection=cms.uint32(1038),detLabel=cms.string("BPIXL4m8"),selection=cms.untracked.vstring("0x1ef00ffc-0x12400020"))  
)
# R1 of disks 1-7 FPIX: 22 modules each
OccupancyPlotsFPIXR1WantedSubDets = cms.VPSet (
    cms.PSet(detSelection = cms.uint32(1041),detLabel = cms.string("FPIXmD1R1"),selection=cms.untracked.vstring("0x1fbff000-0x14841000",
                                                                                                               "0x1fbff000-0x14842000",
                                                                                                               "0x1fbff000-0x14843000",
                                                                                                               "0x1fbff000-0x14844000",
                                                                                                               "0x1fbff000-0x14845000",
                                                                                                               "0x1fbff000-0x14846000",
                                                                                                               "0x1fbff000-0x14847000",
                                                                                                               "0x1fbff000-0x14848000",
                                                                                                               "0x1fbff000-0x14849000",
                                                                                                               "0x1fbff000-0x1484a000",
                                                                                                               "0x1fbff000-0x1484b000",
                                                                                                               "0x1fbff000-0x1484c000",
                                                                                                               "0x1fbff000-0x1484d000",
                                                                                                               "0x1fbff000-0x1484e000",
                                                                                                               "0x1fbff000-0x1484f000",
                                                                                                               "0x1fbff000-0x14850000",
                                                                                                               "0x1fbff000-0x14851000",
                                                                                                               "0x1fbff000-0x14852000",
                                                                                                               "0x1fbff000-0x14853000",
                                                                                                               "0x1fbff000-0x14854000",
                                                                                                               "0x1fbff000-0x14855000",
                                                                                                               "0x1fbff000-0x14856000")),
    cms.PSet(detSelection = cms.uint32(1042),detLabel = cms.string("FPIXmD2R1"),selection=cms.untracked.vstring("0x1fbff000-0x14881000",
                                                                                                               "0x1fbff000-0x14882000",
                                                                                                               "0x1fbff000-0x14883000",
                                                                                                               "0x1fbff000-0x14884000",
                                                                                                               "0x1fbff000-0x14885000",
                                                                                                               "0x1fbff000-0x14886000",
                                                                                                               "0x1fbff000-0x14887000",
                                                                                                               "0x1fbff000-0x14888000",
                                                                                                               "0x1fbff000-0x14889000",
                                                                                                               "0x1fbff000-0x1488a000",
                                                                                                               "0x1fbff000-0x1488b000",
                                                                                                               "0x1fbff000-0x1488c000",
                                                                                                               "0x1fbff000-0x1488d000",
                                                                                                               "0x1fbff000-0x1488e000",
                                                                                                               "0x1fbff000-0x1488f000",
                                                                                                               "0x1fbff000-0x14890000",
                                                                                                               "0x1fbff000-0x14891000",
                                                                                                               "0x1fbff000-0x14892000",
                                                                                                               "0x1fbff000-0x14893000",
                                                                                                               "0x1fbff000-0x14894000",
                                                                                                               "0x1fbff000-0x14895000",
                                                                                                               "0x1fbff000-0x14896000")),
    cms.PSet(detSelection = cms.uint32(1043),detLabel = cms.string("FPIXmD3R1"),selection=cms.untracked.vstring("0x1fbff000-0x148c1000",
                                                                                                               "0x1fbff000-0x148c2000",
                                                                                                               "0x1fbff000-0x148c3000",
                                                                                                               "0x1fbff000-0x148c4000",
                                                                                                               "0x1fbff000-0x148c5000",
                                                                                                               "0x1fbff000-0x148c6000",
                                                                                                               "0x1fbff000-0x148c7000",
                                                                                                               "0x1fbff000-0x148c8000",
                                                                                                               "0x1fbff000-0x148c9000",
                                                                                                               "0x1fbff000-0x148ca000",
                                                                                                               "0x1fbff000-0x148cb000",
                                                                                                               "0x1fbff000-0x148cc000",
                                                                                                               "0x1fbff000-0x148cd000",
                                                                                                               "0x1fbff000-0x148ce000",
                                                                                                               "0x1fbff000-0x148cf000",
                                                                                                               "0x1fbff000-0x148d0000",
                                                                                                               "0x1fbff000-0x148d1000",
                                                                                                               "0x1fbff000-0x148d2000",
                                                                                                               "0x1fbff000-0x148d3000",
                                                                                                               "0x1fbff000-0x148d4000",
                                                                                                               "0x1fbff000-0x148d5000",
                                                                                                               "0x1fbff000-0x148d6000")),
   cms.PSet(detSelection = cms.uint32(1061),detLabel = cms.string("FPIXpD1R1"),selection=cms.untracked.vstring("0x1fbff000-0x15041000",
                                                                                                               "0x1fbff000-0x15042000",
                                                                                                               "0x1fbff000-0x15043000",
                                                                                                               "0x1fbff000-0x15044000",
                                                                                                               "0x1fbff000-0x15045000",
                                                                                                               "0x1fbff000-0x15046000",
                                                                                                               "0x1fbff000-0x15047000",
                                                                                                               "0x1fbff000-0x15048000",
                                                                                                               "0x1fbff000-0x15049000",
                                                                                                               "0x1fbff000-0x1504a000",
                                                                                                               "0x1fbff000-0x1504b000",
                                                                                                               "0x1fbff000-0x1504c000",
                                                                                                               "0x1fbff000-0x1504d000",
                                                                                                               "0x1fbff000-0x1504e000",
                                                                                                               "0x1fbff000-0x1504f000",
                                                                                                               "0x1fbff000-0x15050000",
                                                                                                               "0x1fbff000-0x15051000",
                                                                                                               "0x1fbff000-0x15052000",
                                                                                                               "0x1fbff000-0x15053000",
                                                                                                               "0x1fbff000-0x15054000",
                                                                                                               "0x1fbff000-0x15055000",
                                                                                                               "0x1fbff000-0x15056000")),
    cms.PSet(detSelection = cms.uint32(1062),detLabel = cms.string("FPIXpD2R1"),selection=cms.untracked.vstring("0x1fbff000-0x15081000",
                                                                                                               "0x1fbff000-0x15082000",
                                                                                                               "0x1fbff000-0x15083000",
                                                                                                               "0x1fbff000-0x15084000",
                                                                                                               "0x1fbff000-0x15085000",
                                                                                                               "0x1fbff000-0x15086000",
                                                                                                               "0x1fbff000-0x15087000",
                                                                                                               "0x1fbff000-0x15088000",
                                                                                                               "0x1fbff000-0x15089000",
                                                                                                               "0x1fbff000-0x1508a000",
                                                                                                               "0x1fbff000-0x1508b000",
                                                                                                               "0x1fbff000-0x1508c000",
                                                                                                               "0x1fbff000-0x1508d000",
                                                                                                               "0x1fbff000-0x1508e000",
                                                                                                               "0x1fbff000-0x1508f000",
                                                                                                               "0x1fbff000-0x15090000",
                                                                                                               "0x1fbff000-0x15091000",
                                                                                                               "0x1fbff000-0x15092000",
                                                                                                               "0x1fbff000-0x15093000",
                                                                                                               "0x1fbff000-0x15094000",
                                                                                                               "0x1fbff000-0x15095000",
                                                                                                               "0x1fbff000-0x15096000")),
    cms.PSet(detSelection = cms.uint32(1063),detLabel = cms.string("FPIXpD3R1"),selection=cms.untracked.vstring("0x1fbff000-0x150c1000",
                                                                                                               "0x1fbff000-0x150c2000",
                                                                                                               "0x1fbff000-0x150c3000",
                                                                                                               "0x1fbff000-0x150c4000",
                                                                                                               "0x1fbff000-0x150c5000",
                                                                                                               "0x1fbff000-0x150c6000",
                                                                                                               "0x1fbff000-0x150c7000",
                                                                                                               "0x1fbff000-0x150c8000",
                                                                                                               "0x1fbff000-0x150c9000",
                                                                                                               "0x1fbff000-0x150ca000",
                                                                                                               "0x1fbff000-0x150cb000",
                                                                                                               "0x1fbff000-0x150cc000",
                                                                                                               "0x1fbff000-0x150cd000",
                                                                                                               "0x1fbff000-0x150ce000",
                                                                                                               "0x1fbff000-0x150cf000",
                                                                                                               "0x1fbff000-0x150d0000",
                                                                                                               "0x1fbff000-0x150d1000",
                                                                                                               "0x1fbff000-0x150d2000",
                                                                                                               "0x1fbff000-0x150d3000",
                                                                                                               "0x1fbff000-0x150d4000",
                                                                                                               "0x1fbff000-0x150d5000",
                                                                                                               "0x1fbff000-0x150d6000"))
)
# R2 of disks 1-10 FPIX: 34 modules each: from 23 to 56 in disks 1-7 and from 1 to 34 in disks 8-10
OccupancyPlotsFPIXR2WantedSubDets = cms.VPSet (
    cms.PSet(detSelection = cms.uint32(1051),detLabel = cms.string("FPIXmD1R2"),selection=cms.untracked.vstring("0x1fbff000-0x14857000",
                                                                                                               "0x1fbff000-0x14858000",
                                                                                                               "0x1fbff000-0x14859000",
                                                                                                               "0x1fbff000-0x1485a000",
                                                                                                               "0x1fbff000-0x1485b000",
                                                                                                               "0x1fbff000-0x1485c000",
                                                                                                               "0x1fbff000-0x1485d000",
                                                                                                               "0x1fbff000-0x1485e000",
                                                                                                               "0x1fbff000-0x1485f000",
                                                                                                               "0x1fbff000-0x14860000",
                                                                                                               "0x1fbff000-0x14861000",
                                                                                                               "0x1fbff000-0x14862000",
                                                                                                               "0x1fbff000-0x14863000",
                                                                                                               "0x1fbff000-0x14864000",
                                                                                                               "0x1fbff000-0x14865000",
                                                                                                               "0x1fbff000-0x14866000",
                                                                                                               "0x1fbff000-0x14867000",
                                                                                                               "0x1fbff000-0x14868000",
                                                                                                               "0x1fbff000-0x14869000",
                                                                                                               "0x1fbff000-0x1486a000",
                                                                                                               "0x1fbff000-0x1486b000",
                                                                                                               "0x1fbff000-0x1486c000",
                                                                                                               "0x1fbff000-0x1486d000",
                                                                                                               "0x1fbff000-0x1486e000",
                                                                                                               "0x1fbff000-0x1486f000",
                                                                                                               "0x1fbff000-0x14870000",
                                                                                                               "0x1fbff000-0x14871000",
                                                                                                               "0x1fbff000-0x14872000",
                                                                                                               "0x1fbff000-0x14873000",
                                                                                                               "0x1fbff000-0x14874000",
                                                                                                               "0x1fbff000-0x14875000",
                                                                                                               "0x1fbff000-0x14876000",
                                                                                                               "0x1fbff000-0x14877000",
                                                                                                               "0x1fbff000-0x14878000")),
    cms.PSet(detSelection = cms.uint32(1052),detLabel = cms.string("FPIXmD2R2"),selection=cms.untracked.vstring("0x1fbff000-0x14897000",
                                                                                                               "0x1fbff000-0x14898000",
                                                                                                               "0x1fbff000-0x14899000",
                                                                                                               "0x1fbff000-0x1489a000",
                                                                                                               "0x1fbff000-0x1489b000",
                                                                                                               "0x1fbff000-0x1489c000",
                                                                                                               "0x1fbff000-0x1489d000",
                                                                                                               "0x1fbff000-0x1489e000",
                                                                                                               "0x1fbff000-0x1489f000",
                                                                                                               "0x1fbff000-0x148a0000",
                                                                                                               "0x1fbff000-0x148a1000",
                                                                                                               "0x1fbff000-0x148a2000",
                                                                                                               "0x1fbff000-0x148a3000",
                                                                                                               "0x1fbff000-0x148a4000",
                                                                                                               "0x1fbff000-0x148a5000",
                                                                                                               "0x1fbff000-0x148a6000",
                                                                                                               "0x1fbff000-0x148a7000",
                                                                                                               "0x1fbff000-0x148a8000",
                                                                                                               "0x1fbff000-0x148a9000",
                                                                                                               "0x1fbff000-0x148aa000",
                                                                                                               "0x1fbff000-0x148ab000",
                                                                                                               "0x1fbff000-0x148ac000",
                                                                                                               "0x1fbff000-0x148ad000",
                                                                                                               "0x1fbff000-0x148ae000",
                                                                                                               "0x1fbff000-0x148af000",
                                                                                                               "0x1fbff000-0x14870000",
                                                                                                               "0x1fbff000-0x148b1000",
                                                                                                               "0x1fbff000-0x148b2000",
                                                                                                               "0x1fbff000-0x148b3000",
                                                                                                               "0x1fbff000-0x148b4000",
                                                                                                               "0x1fbff000-0x148b5000",
                                                                                                               "0x1fbff000-0x148b6000",
                                                                                                               "0x1fbff000-0x148b7000",
                                                                                                               "0x1fbff000-0x148b8000")),
    cms.PSet(detSelection = cms.uint32(1053),detLabel = cms.string("FPIXmD3R2"),selection=cms.untracked.vstring("0x1fbff000-0x148d7000",
                                                                                                               "0x1fbff000-0x148d8000",
                                                                                                               "0x1fbff000-0x148d9000",
                                                                                                               "0x1fbff000-0x148da000",
                                                                                                               "0x1fbff000-0x148db000",
                                                                                                               "0x1fbff000-0x148dc000",
                                                                                                               "0x1fbff000-0x148dd000",
                                                                                                               "0x1fbff000-0x148de000",
                                                                                                               "0x1fbff000-0x148df000",
                                                                                                               "0x1fbff000-0x148e0000",
                                                                                                               "0x1fbff000-0x148e1000",
                                                                                                               "0x1fbff000-0x148e2000",
                                                                                                               "0x1fbff000-0x148e3000",
                                                                                                               "0x1fbff000-0x148e4000",
                                                                                                               "0x1fbff000-0x148e5000",
                                                                                                               "0x1fbff000-0x148e6000",
                                                                                                               "0x1fbff000-0x148e7000",
                                                                                                               "0x1fbff000-0x148e8000",
                                                                                                               "0x1fbff000-0x148e9000",
                                                                                                               "0x1fbff000-0x148ea000",
                                                                                                               "0x1fbff000-0x148eb000",
                                                                                                               "0x1fbff000-0x148ec000",
                                                                                                               "0x1fbff000-0x148ed000",
                                                                                                               "0x1fbff000-0x148ee000",
                                                                                                               "0x1fbff000-0x148ef000",
                                                                                                               "0x1fbff000-0x148f0000",
                                                                                                               "0x1fbff000-0x148f1000",
                                                                                                               "0x1fbff000-0x148f2000",
                                                                                                               "0x1fbff000-0x148f3000",
                                                                                                               "0x1fbff000-0x148f4000",
                                                                                                               "0x1fbff000-0x148f5000",
                                                                                                               "0x1fbff000-0x148f6000",
                                                                                                               "0x1fbff000-0x148f7000",
                                                                                                               "0x1fbff000-0x148f8000")),
    cms.PSet(detSelection = cms.uint32(1071),detLabel = cms.string("FPIXpD1R2"),selection=cms.untracked.vstring("0x1fbff000-0x15057000",
                                                                                                               "0x1fbff000-0x15058000",
                                                                                                               "0x1fbff000-0x15059000",
                                                                                                               "0x1fbff000-0x1505a000",
                                                                                                               "0x1fbff000-0x1505b000",
                                                                                                               "0x1fbff000-0x1505c000",
                                                                                                               "0x1fbff000-0x1505d000",
                                                                                                               "0x1fbff000-0x1505e000",
                                                                                                               "0x1fbff000-0x1505f000",
                                                                                                               "0x1fbff000-0x15060000",
                                                                                                               "0x1fbff000-0x15061000",
                                                                                                               "0x1fbff000-0x15062000",
                                                                                                               "0x1fbff000-0x15063000",
                                                                                                               "0x1fbff000-0x15064000",
                                                                                                               "0x1fbff000-0x15065000",
                                                                                                               "0x1fbff000-0x15066000",
                                                                                                               "0x1fbff000-0x15067000",
                                                                                                               "0x1fbff000-0x15068000",
                                                                                                               "0x1fbff000-0x15069000",
                                                                                                               "0x1fbff000-0x1506a000",
                                                                                                               "0x1fbff000-0x1506b000",
                                                                                                               "0x1fbff000-0x1506c000",
                                                                                                               "0x1fbff000-0x1506d000",
                                                                                                               "0x1fbff000-0x1506e000",
                                                                                                               "0x1fbff000-0x1506f000",
                                                                                                               "0x1fbff000-0x15070000",
                                                                                                               "0x1fbff000-0x15071000",
                                                                                                               "0x1fbff000-0x15072000",
                                                                                                               "0x1fbff000-0x15073000",
                                                                                                               "0x1fbff000-0x15074000",
                                                                                                               "0x1fbff000-0x15075000",
                                                                                                               "0x1fbff000-0x15076000",
                                                                                                               "0x1fbff000-0x15077000",
                                                                                                               "0x1fbff000-0x15078000")),
    cms.PSet(detSelection = cms.uint32(1072),detLabel = cms.string("FPIXpD2R2"),selection=cms.untracked.vstring("0x1fbff000-0x15097000",
                                                                                                               "0x1fbff000-0x15098000",
                                                                                                               "0x1fbff000-0x15099000",
                                                                                                               "0x1fbff000-0x1509a000",
                                                                                                               "0x1fbff000-0x1509b000",
                                                                                                               "0x1fbff000-0x1509c000",
                                                                                                               "0x1fbff000-0x1509d000",
                                                                                                               "0x1fbff000-0x1509e000",
                                                                                                               "0x1fbff000-0x1509f000",
                                                                                                               "0x1fbff000-0x150a0000",
                                                                                                               "0x1fbff000-0x150a1000",
                                                                                                               "0x1fbff000-0x150a2000",
                                                                                                               "0x1fbff000-0x150a3000",
                                                                                                               "0x1fbff000-0x150a4000",
                                                                                                               "0x1fbff000-0x150a5000",
                                                                                                               "0x1fbff000-0x150a6000",
                                                                                                               "0x1fbff000-0x150a7000",
                                                                                                               "0x1fbff000-0x150a8000",
                                                                                                               "0x1fbff000-0x150a9000",
                                                                                                               "0x1fbff000-0x150aa000",
                                                                                                               "0x1fbff000-0x150ab000",
                                                                                                               "0x1fbff000-0x150ac000",
                                                                                                               "0x1fbff000-0x150ad000",
                                                                                                               "0x1fbff000-0x150ae000",
                                                                                                               "0x1fbff000-0x150af000",
                                                                                                               "0x1fbff000-0x15070000",
                                                                                                               "0x1fbff000-0x150b1000",
                                                                                                               "0x1fbff000-0x150b2000",
                                                                                                               "0x1fbff000-0x150b3000",
                                                                                                               "0x1fbff000-0x150b4000",
                                                                                                               "0x1fbff000-0x150b5000",
                                                                                                               "0x1fbff000-0x150b6000",
                                                                                                               "0x1fbff000-0x150b7000",
                                                                                                               "0x1fbff000-0x150b8000")),
    cms.PSet(detSelection = cms.uint32(1073),detLabel = cms.string("FPIXpD3R2"),selection=cms.untracked.vstring("0x1fbff000-0x150d7000",
                                                                                                               "0x1fbff000-0x150d8000",
                                                                                                               "0x1fbff000-0x150d9000",
                                                                                                               "0x1fbff000-0x150da000",
                                                                                                               "0x1fbff000-0x150db000",
                                                                                                               "0x1fbff000-0x150dc000",
                                                                                                               "0x1fbff000-0x150dd000",
                                                                                                               "0x1fbff000-0x150de000",
                                                                                                               "0x1fbff000-0x150df000",
                                                                                                               "0x1fbff000-0x150e0000",
                                                                                                               "0x1fbff000-0x150e1000",
                                                                                                               "0x1fbff000-0x150e2000",
                                                                                                               "0x1fbff000-0x150e3000",
                                                                                                               "0x1fbff000-0x150e4000",
                                                                                                               "0x1fbff000-0x150e5000",
                                                                                                               "0x1fbff000-0x150e6000",
                                                                                                               "0x1fbff000-0x150e7000",
                                                                                                               "0x1fbff000-0x150e8000",
                                                                                                               "0x1fbff000-0x150e9000",
                                                                                                               "0x1fbff000-0x150ea000",
                                                                                                               "0x1fbff000-0x150eb000",
                                                                                                               "0x1fbff000-0x150ec000",
                                                                                                               "0x1fbff000-0x150ed000",
                                                                                                               "0x1fbff000-0x150ee000",
                                                                                                               "0x1fbff000-0x150ef000",
                                                                                                               "0x1fbff000-0x150f0000",
                                                                                                               "0x1fbff000-0x150f1000",
                                                                                                               "0x1fbff000-0x150f2000",
                                                                                                               "0x1fbff000-0x150f3000",
                                                                                                               "0x1fbff000-0x150f4000",
                                                                                                               "0x1fbff000-0x150f5000",
                                                                                                               "0x1fbff000-0x150f6000",
                                                                                                               "0x1fbff000-0x150f7000",
                                                                                                               "0x1fbff000-0x150f8000"))
)
#
OccupancyPlotsPixelWantedSubDets.extend(OccupancyPlotsBPIXWantedSubDets)
OccupancyPlotsPixelWantedSubDets.extend(OccupancyPlotsFPIXR1WantedSubDets)
OccupancyPlotsPixelWantedSubDets.extend(OccupancyPlotsFPIXR2WantedSubDets)
OccupancyPlotsPixelWantedSubDets.extend(cms.VPSet (
    cms.PSet(detSelection=cms.uint32(101),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1ef00000-0x12100000")),  
    cms.PSet(detSelection=cms.uint32(102),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1ef00000-0x12200000")),  
    cms.PSet(detSelection=cms.uint32(103),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1ef00000-0x12300000")),  
    cms.PSet(detSelection=cms.uint32(104),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1ef00000-0x12400000")),  
    cms.PSet(detSelection=cms.uint32(105),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1ef00000-0x12500000")),  
    cms.PSet(detSelection=cms.uint32(106),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1ef00000-0x12600000")),  
    cms.PSet(detSelection=cms.uint32(107),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1ef00000-0x12700000")),  
    cms.PSet(detSelection=cms.uint32(108),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1ef00000-0x12800000")),  
    cms.PSet(detSelection=cms.uint32(109),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1ef00000-0x12900000")),  
    cms.PSet(detSelection=cms.uint32(110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1ef00000-0x12a00000")),   
#
    cms.PSet(detSelection=cms.uint32(211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14840000")),  
    cms.PSet(detSelection=cms.uint32(212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14880000")),  
    cms.PSet(detSelection=cms.uint32(213),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x148c0000")),  
    cms.PSet(detSelection=cms.uint32(214),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14900000")),  
    cms.PSet(detSelection=cms.uint32(215),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14940000")),  
    cms.PSet(detSelection=cms.uint32(216),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14980000")),  
    cms.PSet(detSelection=cms.uint32(217),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x149c0000")),  
    cms.PSet(detSelection=cms.uint32(218),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14a00000")),  
    cms.PSet(detSelection=cms.uint32(219),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14a40000")),  
    cms.PSet(detSelection=cms.uint32(210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14a80000")),  
    cms.PSet(detSelection=cms.uint32(211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14ac0000")),  
    cms.PSet(detSelection=cms.uint32(212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14b00000")),  
    cms.PSet(detSelection=cms.uint32(213),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14b40000")),  
    cms.PSet(detSelection=cms.uint32(214),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14b80000")),  
    cms.PSet(detSelection=cms.uint32(215),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x14bc0000")),  
#   
    cms.PSet(detSelection=cms.uint32(251),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15040000")),  
    cms.PSet(detSelection=cms.uint32(252),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15080000")),  
    cms.PSet(detSelection=cms.uint32(253),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x150c0000")),  
    cms.PSet(detSelection=cms.uint32(254),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15100000")),  
    cms.PSet(detSelection=cms.uint32(255),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15140000")),  
    cms.PSet(detSelection=cms.uint32(256),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15180000")),  
    cms.PSet(detSelection=cms.uint32(257),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x151c0000")),  
    cms.PSet(detSelection=cms.uint32(258),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15200000")),  
    cms.PSet(detSelection=cms.uint32(259),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15240000")),  
    cms.PSet(detSelection=cms.uint32(260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15280000")),  
    cms.PSet(detSelection=cms.uint32(261),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x152c0000")),  
    cms.PSet(detSelection=cms.uint32(262),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15300000")),  
    cms.PSet(detSelection=cms.uint32(263),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15340000")),  
    cms.PSet(detSelection=cms.uint32(264),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x15380000")),  
    cms.PSet(detSelection=cms.uint32(265),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1fbc0000-0x153c0000"))
)
)
 
OccupancyPlotsFPIXmD1DetailedWantedSubDets = cms.VPSet(
    cms.PSet(detSelection = cms.uint32(1),detLabel = cms.string("FPIXmD1R1m1p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14841400")),
    cms.PSet(detSelection = cms.uint32(2),detLabel = cms.string("FPIXmD1R1m2p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14842400")),
    cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("FPIXmD1R1m3p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14843400")),
    cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("FPIXmD1R1m4p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14844400")),
    cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("FPIXmD1R1m5p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14845400")),
    cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("FPIXmD1R1m6p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14846400")),
    cms.PSet(detSelection = cms.uint32(7),detLabel = cms.string("FPIXmD1R1m7p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14847400")),
    cms.PSet(detSelection = cms.uint32(8),detLabel = cms.string("FPIXmD1R1m8p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14848400")),
    cms.PSet(detSelection = cms.uint32(9),detLabel = cms.string("FPIXmD1R1m9p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14849400")),
    cms.PSet(detSelection = cms.uint32(10),detLabel = cms.string("FPIXmD1R1m10p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1484a400")),
    cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("FPIXmD1R1m11p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1484b400")),
    cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("FPIXmD1R1m12p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1484c400")),
    cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("FPIXmD1R1m13p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1484d400")),
    cms.PSet(detSelection = cms.uint32(14),detLabel = cms.string("FPIXmD1R1m14p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1484e400")),
    cms.PSet(detSelection = cms.uint32(15),detLabel = cms.string("FPIXmD1R1m15p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1484f400")),
    cms.PSet(detSelection = cms.uint32(16),detLabel = cms.string("FPIXmD1R1m16p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14850400")),
    cms.PSet(detSelection = cms.uint32(17),detLabel = cms.string("FPIXmD1R1m17p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14851400")),
    cms.PSet(detSelection = cms.uint32(18),detLabel = cms.string("FPIXmD1R1m18p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14852400")),
    cms.PSet(detSelection = cms.uint32(19),detLabel = cms.string("FPIXmD1R1m19p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14853400")),
    cms.PSet(detSelection = cms.uint32(20),detLabel = cms.string("FPIXmD1R1m20p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14854400")),
    cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("FPIXmD1R1m21p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14855400")),
    cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("FPIXmD1R1m22p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14856400")),
    cms.PSet(detSelection = cms.uint32(23),detLabel = cms.string("FPIXmD1R2m23p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14857400")),
    cms.PSet(detSelection = cms.uint32(24),detLabel = cms.string("FPIXmD1R2m24p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14858400")),
    cms.PSet(detSelection = cms.uint32(25),detLabel = cms.string("FPIXmD1R2m25p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14859400")),
    cms.PSet(detSelection = cms.uint32(26),detLabel = cms.string("FPIXmD1R2m26p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1485a400")),
    cms.PSet(detSelection = cms.uint32(27),detLabel = cms.string("FPIXmD1R2m27p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1485b400")),
    cms.PSet(detSelection = cms.uint32(28),detLabel = cms.string("FPIXmD1R2m28p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1485c400")),
    cms.PSet(detSelection = cms.uint32(29),detLabel = cms.string("FPIXmD1R2m29p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1485d400")),
    cms.PSet(detSelection = cms.uint32(30),detLabel = cms.string("FPIXmD1R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1485e400")),
    cms.PSet(detSelection = cms.uint32(31),detLabel = cms.string("FPIXmD1R2m31p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1485f400")),
    cms.PSet(detSelection = cms.uint32(32),detLabel = cms.string("FPIXmD1R2m32p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14860400")),
    cms.PSet(detSelection = cms.uint32(33),detLabel = cms.string("FPIXmD1R2m33p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14861400")),
    cms.PSet(detSelection = cms.uint32(34),detLabel = cms.string("FPIXmD1R2m34p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14862400")),
    cms.PSet(detSelection = cms.uint32(35),detLabel = cms.string("FPIXmD1R2m35p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14863400")),
    cms.PSet(detSelection = cms.uint32(36),detLabel = cms.string("FPIXmD1R2m36p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14864400")),
    cms.PSet(detSelection = cms.uint32(37),detLabel = cms.string("FPIXmD1R2m37p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14865400")),
    cms.PSet(detSelection = cms.uint32(38),detLabel = cms.string("FPIXmD1R2m38p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14866400")),
    cms.PSet(detSelection = cms.uint32(39),detLabel = cms.string("FPIXmD1R2m39p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14867400")),
    cms.PSet(detSelection = cms.uint32(40),detLabel = cms.string("FPIXmD1R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14868400")),
    cms.PSet(detSelection = cms.uint32(41),detLabel = cms.string("FPIXmD1R2m41p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14869400")),
    cms.PSet(detSelection = cms.uint32(42),detLabel = cms.string("FPIXmD1R2m42p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1486a400")),
    cms.PSet(detSelection = cms.uint32(43),detLabel = cms.string("FPIXmD1R2m43p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1486b400")),
    cms.PSet(detSelection = cms.uint32(44),detLabel = cms.string("FPIXmD1R2m44p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1486c400")),
    cms.PSet(detSelection = cms.uint32(45),detLabel = cms.string("FPIXmD1R2m45p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1486d400")),
    cms.PSet(detSelection = cms.uint32(46),detLabel = cms.string("FPIXmD1R2m46p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1486e400")),
    cms.PSet(detSelection = cms.uint32(47),detLabel = cms.string("FPIXmD1R2m47p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1486f400")),
    cms.PSet(detSelection = cms.uint32(48),detLabel = cms.string("FPIXmD1R2m48p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14870400")),
    cms.PSet(detSelection = cms.uint32(49),detLabel = cms.string("FPIXmD1R2m49p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14871400")),
    cms.PSet(detSelection = cms.uint32(50),detLabel = cms.string("FPIXmD1R2m50p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14872400")),
    cms.PSet(detSelection = cms.uint32(51),detLabel = cms.string("FPIXmD1R2m51p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14873400")),
    cms.PSet(detSelection = cms.uint32(52),detLabel = cms.string("FPIXmD1R2m52p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14874400")),
    cms.PSet(detSelection = cms.uint32(53),detLabel = cms.string("FPIXmD1R2m53p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14875400")),
    cms.PSet(detSelection = cms.uint32(54),detLabel = cms.string("FPIXmD1R2m54p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14876400")),
    cms.PSet(detSelection = cms.uint32(55),detLabel = cms.string("FPIXmD1R2m55p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14877400")),
    cms.PSet(detSelection = cms.uint32(56),detLabel = cms.string("FPIXmD1R2m56p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14878400")),
#
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("FPIXmD1R1m1p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14841800")),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIXmD1R1m2p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14842800")),
    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("FPIXmD1R1m3p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14843800")),
    cms.PSet(detSelection = cms.uint32(104),detLabel = cms.string("FPIXmD1R1m4p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14844800")),
    cms.PSet(detSelection = cms.uint32(105),detLabel = cms.string("FPIXmD1R1m5p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14845800")),
    cms.PSet(detSelection = cms.uint32(106),detLabel = cms.string("FPIXmD1R1m6p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14846800")),
    cms.PSet(detSelection = cms.uint32(107),detLabel = cms.string("FPIXmD1R1m7p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14847800")),
    cms.PSet(detSelection = cms.uint32(108),detLabel = cms.string("FPIXmD1R1m8p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14848800")),
    cms.PSet(detSelection = cms.uint32(109),detLabel = cms.string("FPIXmD1R1m9p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14849800")),
    cms.PSet(detSelection = cms.uint32(110),detLabel = cms.string("FPIXmD1R1m10p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1484a800")),
    cms.PSet(detSelection = cms.uint32(111),detLabel = cms.string("FPIXmD1R1m11p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1484b800")),
    cms.PSet(detSelection = cms.uint32(112),detLabel = cms.string("FPIXmD1R1m12p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1484c800")),
    cms.PSet(detSelection = cms.uint32(113),detLabel = cms.string("FPIXmD1R1m13p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1484d800")),
    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("FPIXmD1R1m14p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1484e800")),
    cms.PSet(detSelection = cms.uint32(115),detLabel = cms.string("FPIXmD1R1m15p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1484f800")),
    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("FPIXmD1R1m16p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14850800")),
    cms.PSet(detSelection = cms.uint32(117),detLabel = cms.string("FPIXmD1R1m17p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14851800")),
    cms.PSet(detSelection = cms.uint32(118),detLabel = cms.string("FPIXmD1R1m18p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14852800")),
    cms.PSet(detSelection = cms.uint32(119),detLabel = cms.string("FPIXmD1R1m19p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14853800")),
    cms.PSet(detSelection = cms.uint32(120),detLabel = cms.string("FPIXmD1R1m20p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14854800")),
    cms.PSet(detSelection = cms.uint32(121),detLabel = cms.string("FPIXmD1R1m21p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14855800")),
    cms.PSet(detSelection = cms.uint32(122),detLabel = cms.string("FPIXmD1R1m22p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14856800")),
    cms.PSet(detSelection = cms.uint32(123),detLabel = cms.string("FPIXmD1R1m23p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14857800")),
    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("FPIXmD1R1m24p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14858800")),
    cms.PSet(detSelection = cms.uint32(125),detLabel = cms.string("FPIXmD1R1m25p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14859800")),
    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("FPIXmD1R1m26p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1485a800")),
    cms.PSet(detSelection = cms.uint32(127),detLabel = cms.string("FPIXmD1R1m27p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1485b800")),
    cms.PSet(detSelection = cms.uint32(128),detLabel = cms.string("FPIXmD1R1m28p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1485c800")),
    cms.PSet(detSelection = cms.uint32(129),detLabel = cms.string("FPIXmD1R1m29p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1485d800")),
    cms.PSet(detSelection = cms.uint32(130),detLabel = cms.string("FPIXmD1R1m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1485e800")),
    cms.PSet(detSelection = cms.uint32(131),detLabel = cms.string("FPIXmD1R1m31p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1485f800")),
    cms.PSet(detSelection = cms.uint32(132),detLabel = cms.string("FPIXmD1R1m32p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14860800")),
    cms.PSet(detSelection = cms.uint32(133),detLabel = cms.string("FPIXmD1R1m33p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14861800")),
    cms.PSet(detSelection = cms.uint32(134),detLabel = cms.string("FPIXmD1R1m34p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14862800")),
    cms.PSet(detSelection = cms.uint32(135),detLabel = cms.string("FPIXmD1R2m35p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14863800")),
    cms.PSet(detSelection = cms.uint32(136),detLabel = cms.string("FPIXmD1R2m36p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14864800")),
    cms.PSet(detSelection = cms.uint32(137),detLabel = cms.string("FPIXmD1R2m37p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14865800")),
    cms.PSet(detSelection = cms.uint32(138),detLabel = cms.string("FPIXmD1R2m38p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14866800")),
    cms.PSet(detSelection = cms.uint32(139),detLabel = cms.string("FPIXmD1R2m39p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14867800")),
    cms.PSet(detSelection = cms.uint32(140),detLabel = cms.string("FPIXmD1R2m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14868800")),
    cms.PSet(detSelection = cms.uint32(141),detLabel = cms.string("FPIXmD1R2m41p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14869800")),
    cms.PSet(detSelection = cms.uint32(142),detLabel = cms.string("FPIXmD1R2m42p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1486a800")),
    cms.PSet(detSelection = cms.uint32(143),detLabel = cms.string("FPIXmD1R2m43p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1486b800")),
    cms.PSet(detSelection = cms.uint32(144),detLabel = cms.string("FPIXmD1R2m44p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1486c800")),
    cms.PSet(detSelection = cms.uint32(145),detLabel = cms.string("FPIXmD1R2m45p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1486d800")),
    cms.PSet(detSelection = cms.uint32(146),detLabel = cms.string("FPIXmD1R2m46p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1486e800")),
    cms.PSet(detSelection = cms.uint32(147),detLabel = cms.string("FPIXmD1R2m47p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1486f800")),
    cms.PSet(detSelection = cms.uint32(148),detLabel = cms.string("FPIXmD1R2m48p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14870800")),
    cms.PSet(detSelection = cms.uint32(149),detLabel = cms.string("FPIXmD1R2m49p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14871800")),
    cms.PSet(detSelection = cms.uint32(150),detLabel = cms.string("FPIXmD1R2m50p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14872800")),
    cms.PSet(detSelection = cms.uint32(151),detLabel = cms.string("FPIXmD1R2m51p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14873800")),
    cms.PSet(detSelection = cms.uint32(152),detLabel = cms.string("FPIXmD1R2m52p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14874800")),
    cms.PSet(detSelection = cms.uint32(153),detLabel = cms.string("FPIXmD1R2m53p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14875800")),
    cms.PSet(detSelection = cms.uint32(154),detLabel = cms.string("FPIXmD1R2m54p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14876800")),
    cms.PSet(detSelection = cms.uint32(155),detLabel = cms.string("FPIXmD1R2m55p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14877800")),
    cms.PSet(detSelection = cms.uint32(156),detLabel = cms.string("FPIXmD1R2m56p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14878800")),
)
#
OccupancyPlotsFPIXmD2DetailedWantedSubDets = cms.VPSet(
    cms.PSet(detSelection = cms.uint32(201),detLabel = cms.string("FPIXmD2R1m1p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14881400")),
    cms.PSet(detSelection = cms.uint32(202),detLabel = cms.string("FPIXmD2R1m2p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14882400")),
    cms.PSet(detSelection = cms.uint32(203),detLabel = cms.string("FPIXmD2R1m3p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14883400")),
    cms.PSet(detSelection = cms.uint32(204),detLabel = cms.string("FPIXmD2R1m4p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14884400")),
    cms.PSet(detSelection = cms.uint32(205),detLabel = cms.string("FPIXmD2R1m5p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14885400")),
    cms.PSet(detSelection = cms.uint32(206),detLabel = cms.string("FPIXmD2R1m6p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14886400")),
    cms.PSet(detSelection = cms.uint32(207),detLabel = cms.string("FPIXmD2R1m7p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14887400")),
    cms.PSet(detSelection = cms.uint32(208),detLabel = cms.string("FPIXmD2R1m8p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14888400")),
    cms.PSet(detSelection = cms.uint32(209),detLabel = cms.string("FPIXmD2R1m9p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14889400")),
    cms.PSet(detSelection = cms.uint32(210),detLabel = cms.string("FPIXmD2R1m10p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1488a400")),
    cms.PSet(detSelection = cms.uint32(211),detLabel = cms.string("FPIXmD2R1m11p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1488b400")),
    cms.PSet(detSelection = cms.uint32(212),detLabel = cms.string("FPIXmD2R1m12p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1488c400")),
    cms.PSet(detSelection = cms.uint32(213),detLabel = cms.string("FPIXmD2R1m13p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1488d400")),
    cms.PSet(detSelection = cms.uint32(214),detLabel = cms.string("FPIXmD2R1m14p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1488e400")),
    cms.PSet(detSelection = cms.uint32(215),detLabel = cms.string("FPIXmD2R1m15p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1488f400")),
    cms.PSet(detSelection = cms.uint32(216),detLabel = cms.string("FPIXmD2R1m16p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14890400")),
    cms.PSet(detSelection = cms.uint32(217),detLabel = cms.string("FPIXmD2R1m17p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14891400")),
    cms.PSet(detSelection = cms.uint32(218),detLabel = cms.string("FPIXmD2R1m18p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14892400")),
    cms.PSet(detSelection = cms.uint32(219),detLabel = cms.string("FPIXmD2R1m19p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14893400")),
    cms.PSet(detSelection = cms.uint32(220),detLabel = cms.string("FPIXmD2R1m20p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14894400")),
    cms.PSet(detSelection = cms.uint32(221),detLabel = cms.string("FPIXmD2R1m21p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14895400")),
    cms.PSet(detSelection = cms.uint32(222),detLabel = cms.string("FPIXmD2R1m22p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14896400")),
    cms.PSet(detSelection = cms.uint32(223),detLabel = cms.string("FPIXmD2R2m23p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14897400")),
    cms.PSet(detSelection = cms.uint32(224),detLabel = cms.string("FPIXmD2R2m24p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14898400")),
    cms.PSet(detSelection = cms.uint32(225),detLabel = cms.string("FPIXmD2R2m25p1"),selection=cms.untracked.vstring("0x1fbffc00-0x14899400")),
    cms.PSet(detSelection = cms.uint32(226),detLabel = cms.string("FPIXmD2R2m26p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1489a400")),
    cms.PSet(detSelection = cms.uint32(227),detLabel = cms.string("FPIXmD2R2m27p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1489b400")),
    cms.PSet(detSelection = cms.uint32(228),detLabel = cms.string("FPIXmD2R2m28p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1489c400")),
    cms.PSet(detSelection = cms.uint32(229),detLabel = cms.string("FPIXmD2R2m29p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1489d400")),
    cms.PSet(detSelection = cms.uint32(230),detLabel = cms.string("FPIXmD2R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1489e400")),
    cms.PSet(detSelection = cms.uint32(231),detLabel = cms.string("FPIXmD2R2m31p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1489f400")),
    cms.PSet(detSelection = cms.uint32(232),detLabel = cms.string("FPIXmD2R2m32p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148a0400")),
    cms.PSet(detSelection = cms.uint32(233),detLabel = cms.string("FPIXmD2R2m33p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148a1400")),
    cms.PSet(detSelection = cms.uint32(234),detLabel = cms.string("FPIXmD2R2m34p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148a2400")),
    cms.PSet(detSelection = cms.uint32(235),detLabel = cms.string("FPIXmD2R2m35p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148a3400")),
    cms.PSet(detSelection = cms.uint32(236),detLabel = cms.string("FPIXmD2R2m36p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148a4400")),
    cms.PSet(detSelection = cms.uint32(237),detLabel = cms.string("FPIXmD2R2m37p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148a5400")),
    cms.PSet(detSelection = cms.uint32(238),detLabel = cms.string("FPIXmD2R2m38p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148a6400")),
    cms.PSet(detSelection = cms.uint32(239),detLabel = cms.string("FPIXmD2R2m39p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148a7400")),
    cms.PSet(detSelection = cms.uint32(240),detLabel = cms.string("FPIXmD2R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148a8400")),
    cms.PSet(detSelection = cms.uint32(241),detLabel = cms.string("FPIXmD2R2m41p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148a9400")),
    cms.PSet(detSelection = cms.uint32(242),detLabel = cms.string("FPIXmD2R2m42p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148aa400")),
    cms.PSet(detSelection = cms.uint32(243),detLabel = cms.string("FPIXmD2R2m43p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ab400")),
    cms.PSet(detSelection = cms.uint32(244),detLabel = cms.string("FPIXmD2R2m44p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ac400")),
    cms.PSet(detSelection = cms.uint32(245),detLabel = cms.string("FPIXmD2R2m45p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ad400")),
    cms.PSet(detSelection = cms.uint32(246),detLabel = cms.string("FPIXmD2R2m46p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ae400")),
    cms.PSet(detSelection = cms.uint32(247),detLabel = cms.string("FPIXmD2R2m47p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148af400")),
    cms.PSet(detSelection = cms.uint32(248),detLabel = cms.string("FPIXmD2R2m48p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148b0400")),
    cms.PSet(detSelection = cms.uint32(249),detLabel = cms.string("FPIXmD2R2m49p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148b1400")),
    cms.PSet(detSelection = cms.uint32(250),detLabel = cms.string("FPIXmD2R2m50p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148b2400")),
    cms.PSet(detSelection = cms.uint32(251),detLabel = cms.string("FPIXmD2R2m51p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148b3400")),
    cms.PSet(detSelection = cms.uint32(252),detLabel = cms.string("FPIXmD2R2m52p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148b4400")),
    cms.PSet(detSelection = cms.uint32(253),detLabel = cms.string("FPIXmD2R2m53p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148b5400")),
    cms.PSet(detSelection = cms.uint32(254),detLabel = cms.string("FPIXmD2R2m54p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148b6400")),
    cms.PSet(detSelection = cms.uint32(255),detLabel = cms.string("FPIXmD2R2m55p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148b7400")),
    cms.PSet(detSelection = cms.uint32(256),detLabel = cms.string("FPIXmD2R2m56p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148b8400")),
#
    cms.PSet(detSelection = cms.uint32(301),detLabel = cms.string("FPIXmD2R1m1p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14881800")),
    cms.PSet(detSelection = cms.uint32(302),detLabel = cms.string("FPIXmD2R1m2p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14882800")),
    cms.PSet(detSelection = cms.uint32(303),detLabel = cms.string("FPIXmD2R1m3p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14883800")),
    cms.PSet(detSelection = cms.uint32(304),detLabel = cms.string("FPIXmD2R1m4p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14884800")),
    cms.PSet(detSelection = cms.uint32(305),detLabel = cms.string("FPIXmD2R1m5p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14885800")),
    cms.PSet(detSelection = cms.uint32(306),detLabel = cms.string("FPIXmD2R1m6p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14886800")),
    cms.PSet(detSelection = cms.uint32(307),detLabel = cms.string("FPIXmD2R1m7p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14887800")),
    cms.PSet(detSelection = cms.uint32(308),detLabel = cms.string("FPIXmD2R1m8p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14888800")),
    cms.PSet(detSelection = cms.uint32(309),detLabel = cms.string("FPIXmD2R1m9p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14889800")),
    cms.PSet(detSelection = cms.uint32(310),detLabel = cms.string("FPIXmD2R1m10p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1488a800")),
    cms.PSet(detSelection = cms.uint32(311),detLabel = cms.string("FPIXmD2R1m11p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1488b800")),
    cms.PSet(detSelection = cms.uint32(312),detLabel = cms.string("FPIXmD2R1m12p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1488c800")),
    cms.PSet(detSelection = cms.uint32(313),detLabel = cms.string("FPIXmD2R1m13p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1488d800")),
    cms.PSet(detSelection = cms.uint32(314),detLabel = cms.string("FPIXmD2R1m14p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1488e800")),
    cms.PSet(detSelection = cms.uint32(315),detLabel = cms.string("FPIXmD2R1m15p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1488f800")),
    cms.PSet(detSelection = cms.uint32(316),detLabel = cms.string("FPIXmD2R1m16p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14890800")),
    cms.PSet(detSelection = cms.uint32(317),detLabel = cms.string("FPIXmD2R1m17p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14891800")),
    cms.PSet(detSelection = cms.uint32(318),detLabel = cms.string("FPIXmD2R1m18p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14892800")),
    cms.PSet(detSelection = cms.uint32(319),detLabel = cms.string("FPIXmD2R1m19p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14893800")),
    cms.PSet(detSelection = cms.uint32(320),detLabel = cms.string("FPIXmD2R1m20p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14894800")),
    cms.PSet(detSelection = cms.uint32(321),detLabel = cms.string("FPIXmD2R1m21p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14895800")),
    cms.PSet(detSelection = cms.uint32(322),detLabel = cms.string("FPIXmD2R1m22p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14896800")),
    cms.PSet(detSelection = cms.uint32(323),detLabel = cms.string("FPIXmD2R1m23p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14897800")),
    cms.PSet(detSelection = cms.uint32(324),detLabel = cms.string("FPIXmD2R1m24p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14898800")),
    cms.PSet(detSelection = cms.uint32(325),detLabel = cms.string("FPIXmD2R1m25p2"),selection=cms.untracked.vstring("0x1fbffc00-0x14899800")),
    cms.PSet(detSelection = cms.uint32(326),detLabel = cms.string("FPIXmD2R1m26p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1489a800")),
    cms.PSet(detSelection = cms.uint32(327),detLabel = cms.string("FPIXmD2R1m27p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1489b800")),
    cms.PSet(detSelection = cms.uint32(328),detLabel = cms.string("FPIXmD2R1m28p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1489c800")),
    cms.PSet(detSelection = cms.uint32(329),detLabel = cms.string("FPIXmD2R1m29p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1489d800")),
    cms.PSet(detSelection = cms.uint32(330),detLabel = cms.string("FPIXmD2R1m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1489e800")),
    cms.PSet(detSelection = cms.uint32(331),detLabel = cms.string("FPIXmD2R1m31p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1489f800")),
    cms.PSet(detSelection = cms.uint32(332),detLabel = cms.string("FPIXmD2R1m32p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148a0800")),
    cms.PSet(detSelection = cms.uint32(333),detLabel = cms.string("FPIXmD2R1m33p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148a1800")),
    cms.PSet(detSelection = cms.uint32(334),detLabel = cms.string("FPIXmD2R1m34p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148a2800")),
    cms.PSet(detSelection = cms.uint32(335),detLabel = cms.string("FPIXmD2R2m35p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148a3800")),
    cms.PSet(detSelection = cms.uint32(336),detLabel = cms.string("FPIXmD2R2m36p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148a4800")),
    cms.PSet(detSelection = cms.uint32(337),detLabel = cms.string("FPIXmD2R2m37p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148a5800")),
    cms.PSet(detSelection = cms.uint32(338),detLabel = cms.string("FPIXmD2R2m38p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148a6800")),
    cms.PSet(detSelection = cms.uint32(339),detLabel = cms.string("FPIXmD2R2m39p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148a7800")),
    cms.PSet(detSelection = cms.uint32(340),detLabel = cms.string("FPIXmD2R2m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148a8800")),
    cms.PSet(detSelection = cms.uint32(341),detLabel = cms.string("FPIXmD2R2m41p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148a9800")),
    cms.PSet(detSelection = cms.uint32(342),detLabel = cms.string("FPIXmD2R2m42p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148aa800")),
    cms.PSet(detSelection = cms.uint32(343),detLabel = cms.string("FPIXmD2R2m43p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ab800")),
    cms.PSet(detSelection = cms.uint32(344),detLabel = cms.string("FPIXmD2R2m44p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ac800")),
    cms.PSet(detSelection = cms.uint32(345),detLabel = cms.string("FPIXmD2R2m45p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ad800")),
    cms.PSet(detSelection = cms.uint32(346),detLabel = cms.string("FPIXmD2R2m46p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ae800")),
    cms.PSet(detSelection = cms.uint32(347),detLabel = cms.string("FPIXmD2R2m47p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148af800")),
    cms.PSet(detSelection = cms.uint32(348),detLabel = cms.string("FPIXmD2R2m48p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148b0800")),
    cms.PSet(detSelection = cms.uint32(349),detLabel = cms.string("FPIXmD2R2m49p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148b1800")),
    cms.PSet(detSelection = cms.uint32(350),detLabel = cms.string("FPIXmD2R2m50p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148b2800")),
    cms.PSet(detSelection = cms.uint32(351),detLabel = cms.string("FPIXmD2R2m51p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148b3800")),
    cms.PSet(detSelection = cms.uint32(352),detLabel = cms.string("FPIXmD2R2m52p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148b4800")),
    cms.PSet(detSelection = cms.uint32(353),detLabel = cms.string("FPIXmD2R2m53p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148b5800")),
    cms.PSet(detSelection = cms.uint32(354),detLabel = cms.string("FPIXmD2R2m54p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148b6800")),
    cms.PSet(detSelection = cms.uint32(355),detLabel = cms.string("FPIXmD2R2m55p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148b7800")),
    cms.PSet(detSelection = cms.uint32(356),detLabel = cms.string("FPIXmD2R2m56p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148b8800")),
)
#
OccupancyPlotsFPIXmD3DetailedWantedSubDets = cms.VPSet(
    cms.PSet(detSelection = cms.uint32(401),detLabel = cms.string("FPIXmD3R1m1p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148c1400")),
    cms.PSet(detSelection = cms.uint32(402),detLabel = cms.string("FPIXmD3R1m2p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148c2400")),
    cms.PSet(detSelection = cms.uint32(403),detLabel = cms.string("FPIXmD3R1m3p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148c3400")),
    cms.PSet(detSelection = cms.uint32(404),detLabel = cms.string("FPIXmD3R1m4p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148c4400")),
    cms.PSet(detSelection = cms.uint32(405),detLabel = cms.string("FPIXmD3R1m5p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148c5400")),
    cms.PSet(detSelection = cms.uint32(406),detLabel = cms.string("FPIXmD3R1m6p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148c6400")),
    cms.PSet(detSelection = cms.uint32(407),detLabel = cms.string("FPIXmD3R1m7p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148c7400")),
    cms.PSet(detSelection = cms.uint32(408),detLabel = cms.string("FPIXmD3R1m8p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148c8400")),
    cms.PSet(detSelection = cms.uint32(409),detLabel = cms.string("FPIXmD3R1m9p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148c9400")),
    cms.PSet(detSelection = cms.uint32(410),detLabel = cms.string("FPIXmD3R1m10p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ca400")),
    cms.PSet(detSelection = cms.uint32(411),detLabel = cms.string("FPIXmD3R1m11p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148cb400")),
    cms.PSet(detSelection = cms.uint32(412),detLabel = cms.string("FPIXmD3R1m12p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148cc400")),
    cms.PSet(detSelection = cms.uint32(413),detLabel = cms.string("FPIXmD3R1m13p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148cd400")),
    cms.PSet(detSelection = cms.uint32(414),detLabel = cms.string("FPIXmD3R1m14p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ce400")),
    cms.PSet(detSelection = cms.uint32(415),detLabel = cms.string("FPIXmD3R1m15p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148cf400")),
    cms.PSet(detSelection = cms.uint32(416),detLabel = cms.string("FPIXmD3R1m16p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148d0400")),
    cms.PSet(detSelection = cms.uint32(417),detLabel = cms.string("FPIXmD3R1m17p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148d1400")),
    cms.PSet(detSelection = cms.uint32(418),detLabel = cms.string("FPIXmD3R1m18p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148d2400")),
    cms.PSet(detSelection = cms.uint32(419),detLabel = cms.string("FPIXmD3R1m19p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148d3400")),
    cms.PSet(detSelection = cms.uint32(420),detLabel = cms.string("FPIXmD3R1m20p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148d4400")),
    cms.PSet(detSelection = cms.uint32(421),detLabel = cms.string("FPIXmD3R1m21p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148d5400")),
    cms.PSet(detSelection = cms.uint32(422),detLabel = cms.string("FPIXmD3R1m22p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148d6400")),
    cms.PSet(detSelection = cms.uint32(423),detLabel = cms.string("FPIXmD3R2m23p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148d7400")),
    cms.PSet(detSelection = cms.uint32(424),detLabel = cms.string("FPIXmD3R2m24p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148d8400")),
    cms.PSet(detSelection = cms.uint32(425),detLabel = cms.string("FPIXmD3R2m25p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148d9400")),
    cms.PSet(detSelection = cms.uint32(426),detLabel = cms.string("FPIXmD3R2m26p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148da400")),
    cms.PSet(detSelection = cms.uint32(427),detLabel = cms.string("FPIXmD3R2m27p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148db400")),
    cms.PSet(detSelection = cms.uint32(428),detLabel = cms.string("FPIXmD3R2m28p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148dc400")),
    cms.PSet(detSelection = cms.uint32(429),detLabel = cms.string("FPIXmD3R2m29p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148dd400")),
    cms.PSet(detSelection = cms.uint32(430),detLabel = cms.string("FPIXmD3R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148de400")),
    cms.PSet(detSelection = cms.uint32(431),detLabel = cms.string("FPIXmD3R2m31p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148df400")),
    cms.PSet(detSelection = cms.uint32(432),detLabel = cms.string("FPIXmD3R2m32p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148e0400")),
    cms.PSet(detSelection = cms.uint32(433),detLabel = cms.string("FPIXmD3R2m33p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148e1400")),
    cms.PSet(detSelection = cms.uint32(434),detLabel = cms.string("FPIXmD3R2m34p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148e2400")),
    cms.PSet(detSelection = cms.uint32(435),detLabel = cms.string("FPIXmD3R2m35p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148e3400")),
    cms.PSet(detSelection = cms.uint32(436),detLabel = cms.string("FPIXmD3R2m36p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148e4400")),
    cms.PSet(detSelection = cms.uint32(437),detLabel = cms.string("FPIXmD3R2m37p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148e5400")),
    cms.PSet(detSelection = cms.uint32(438),detLabel = cms.string("FPIXmD3R2m38p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148e6400")),
    cms.PSet(detSelection = cms.uint32(439),detLabel = cms.string("FPIXmD3R2m39p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148e7400")),
    cms.PSet(detSelection = cms.uint32(440),detLabel = cms.string("FPIXmD3R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148e8400")),
    cms.PSet(detSelection = cms.uint32(441),detLabel = cms.string("FPIXmD3R2m41p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148e9400")),
    cms.PSet(detSelection = cms.uint32(442),detLabel = cms.string("FPIXmD3R2m42p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ea400")),
    cms.PSet(detSelection = cms.uint32(443),detLabel = cms.string("FPIXmD3R2m43p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148eb400")),
    cms.PSet(detSelection = cms.uint32(444),detLabel = cms.string("FPIXmD3R2m44p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ec400")),
    cms.PSet(detSelection = cms.uint32(445),detLabel = cms.string("FPIXmD3R2m45p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ed400")),
    cms.PSet(detSelection = cms.uint32(446),detLabel = cms.string("FPIXmD3R2m46p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ee400")),
    cms.PSet(detSelection = cms.uint32(447),detLabel = cms.string("FPIXmD3R2m47p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148ef400")),
    cms.PSet(detSelection = cms.uint32(448),detLabel = cms.string("FPIXmD3R2m48p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148f0400")),
    cms.PSet(detSelection = cms.uint32(449),detLabel = cms.string("FPIXmD3R2m49p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148f1400")),
    cms.PSet(detSelection = cms.uint32(450),detLabel = cms.string("FPIXmD3R2m50p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148f2400")),
    cms.PSet(detSelection = cms.uint32(451),detLabel = cms.string("FPIXmD3R2m51p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148f3400")),
    cms.PSet(detSelection = cms.uint32(452),detLabel = cms.string("FPIXmD3R2m52p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148f4400")),
    cms.PSet(detSelection = cms.uint32(453),detLabel = cms.string("FPIXmD3R2m53p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148f5400")),
    cms.PSet(detSelection = cms.uint32(454),detLabel = cms.string("FPIXmD3R2m54p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148f6400")),
    cms.PSet(detSelection = cms.uint32(455),detLabel = cms.string("FPIXmD3R2m55p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148f7400")),
    cms.PSet(detSelection = cms.uint32(456),detLabel = cms.string("FPIXmD3R2m56p1"),selection=cms.untracked.vstring("0x1fbffc00-0x148f8400")),
#
    cms.PSet(detSelection = cms.uint32(501),detLabel = cms.string("FPIXmD3R1m1p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148c1800")),
    cms.PSet(detSelection = cms.uint32(502),detLabel = cms.string("FPIXmD3R1m2p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148c2800")),
    cms.PSet(detSelection = cms.uint32(503),detLabel = cms.string("FPIXmD3R1m3p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148c3800")),
    cms.PSet(detSelection = cms.uint32(504),detLabel = cms.string("FPIXmD3R1m4p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148c4800")),
    cms.PSet(detSelection = cms.uint32(505),detLabel = cms.string("FPIXmD3R1m5p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148c5800")),
    cms.PSet(detSelection = cms.uint32(506),detLabel = cms.string("FPIXmD3R1m6p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148c6800")),
    cms.PSet(detSelection = cms.uint32(507),detLabel = cms.string("FPIXmD3R1m7p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148c7800")),
    cms.PSet(detSelection = cms.uint32(508),detLabel = cms.string("FPIXmD3R1m8p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148c8800")),
    cms.PSet(detSelection = cms.uint32(509),detLabel = cms.string("FPIXmD3R1m9p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148c9800")),
    cms.PSet(detSelection = cms.uint32(510),detLabel = cms.string("FPIXmD3R1m10p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ca800")),
    cms.PSet(detSelection = cms.uint32(511),detLabel = cms.string("FPIXmD3R1m11p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148cb800")),
    cms.PSet(detSelection = cms.uint32(512),detLabel = cms.string("FPIXmD3R1m12p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148cc800")),
    cms.PSet(detSelection = cms.uint32(513),detLabel = cms.string("FPIXmD3R1m13p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148cd800")),
    cms.PSet(detSelection = cms.uint32(514),detLabel = cms.string("FPIXmD3R1m14p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ce800")),
    cms.PSet(detSelection = cms.uint32(515),detLabel = cms.string("FPIXmD3R1m15p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148cf800")),
    cms.PSet(detSelection = cms.uint32(516),detLabel = cms.string("FPIXmD3R1m16p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148d0800")),
    cms.PSet(detSelection = cms.uint32(517),detLabel = cms.string("FPIXmD3R1m17p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148d1800")),
    cms.PSet(detSelection = cms.uint32(518),detLabel = cms.string("FPIXmD3R1m18p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148d2800")),
    cms.PSet(detSelection = cms.uint32(519),detLabel = cms.string("FPIXmD3R1m19p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148d3800")),
    cms.PSet(detSelection = cms.uint32(520),detLabel = cms.string("FPIXmD3R1m20p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148d4800")),
    cms.PSet(detSelection = cms.uint32(521),detLabel = cms.string("FPIXmD3R1m21p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148d5800")),
    cms.PSet(detSelection = cms.uint32(522),detLabel = cms.string("FPIXmD3R1m22p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148d6800")),
    cms.PSet(detSelection = cms.uint32(523),detLabel = cms.string("FPIXmD3R1m23p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148d7800")),
    cms.PSet(detSelection = cms.uint32(524),detLabel = cms.string("FPIXmD3R1m24p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148d8800")),
    cms.PSet(detSelection = cms.uint32(525),detLabel = cms.string("FPIXmD3R1m25p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148d9800")),
    cms.PSet(detSelection = cms.uint32(526),detLabel = cms.string("FPIXmD3R1m26p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148da800")),
    cms.PSet(detSelection = cms.uint32(527),detLabel = cms.string("FPIXmD3R1m27p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148db800")),
    cms.PSet(detSelection = cms.uint32(528),detLabel = cms.string("FPIXmD3R1m28p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148dc800")),
    cms.PSet(detSelection = cms.uint32(529),detLabel = cms.string("FPIXmD3R1m29p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148dd800")),
    cms.PSet(detSelection = cms.uint32(530),detLabel = cms.string("FPIXmD3R1m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148de800")),
    cms.PSet(detSelection = cms.uint32(531),detLabel = cms.string("FPIXmD3R1m31p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148df800")),
    cms.PSet(detSelection = cms.uint32(532),detLabel = cms.string("FPIXmD3R1m32p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148e0800")),
    cms.PSet(detSelection = cms.uint32(533),detLabel = cms.string("FPIXmD3R1m33p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148e1800")),
    cms.PSet(detSelection = cms.uint32(534),detLabel = cms.string("FPIXmD3R1m34p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148e2800")),
    cms.PSet(detSelection = cms.uint32(535),detLabel = cms.string("FPIXmD3R2m35p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148e3800")),
    cms.PSet(detSelection = cms.uint32(536),detLabel = cms.string("FPIXmD3R2m36p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148e4800")),
    cms.PSet(detSelection = cms.uint32(537),detLabel = cms.string("FPIXmD3R2m37p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148e5800")),
    cms.PSet(detSelection = cms.uint32(538),detLabel = cms.string("FPIXmD3R2m38p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148e6800")),
    cms.PSet(detSelection = cms.uint32(539),detLabel = cms.string("FPIXmD3R2m39p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148e7800")),
    cms.PSet(detSelection = cms.uint32(540),detLabel = cms.string("FPIXmD3R2m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148e8800")),
    cms.PSet(detSelection = cms.uint32(541),detLabel = cms.string("FPIXmD3R2m41p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148e9800")),
    cms.PSet(detSelection = cms.uint32(542),detLabel = cms.string("FPIXmD3R2m42p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ea800")),
    cms.PSet(detSelection = cms.uint32(543),detLabel = cms.string("FPIXmD3R2m43p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148eb800")),
    cms.PSet(detSelection = cms.uint32(544),detLabel = cms.string("FPIXmD3R2m44p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ec800")),
    cms.PSet(detSelection = cms.uint32(545),detLabel = cms.string("FPIXmD3R2m45p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ed800")),
    cms.PSet(detSelection = cms.uint32(546),detLabel = cms.string("FPIXmD3R2m46p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ee800")),
    cms.PSet(detSelection = cms.uint32(547),detLabel = cms.string("FPIXmD3R2m47p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148ef800")),
    cms.PSet(detSelection = cms.uint32(548),detLabel = cms.string("FPIXmD3R2m48p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148f0800")),
    cms.PSet(detSelection = cms.uint32(549),detLabel = cms.string("FPIXmD3R2m49p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148f1800")),
    cms.PSet(detSelection = cms.uint32(550),detLabel = cms.string("FPIXmD3R2m50p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148f2800")),
    cms.PSet(detSelection = cms.uint32(551),detLabel = cms.string("FPIXmD3R2m51p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148f3800")),
    cms.PSet(detSelection = cms.uint32(552),detLabel = cms.string("FPIXmD3R2m52p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148f4800")),
    cms.PSet(detSelection = cms.uint32(553),detLabel = cms.string("FPIXmD3R2m53p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148f5800")),
    cms.PSet(detSelection = cms.uint32(554),detLabel = cms.string("FPIXmD3R2m54p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148f6800")),
    cms.PSet(detSelection = cms.uint32(555),detLabel = cms.string("FPIXmD3R2m55p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148f7800")),
    cms.PSet(detSelection = cms.uint32(556),detLabel = cms.string("FPIXmD3R2m56p2"),selection=cms.untracked.vstring("0x1fbffc00-0x148f8800")),
)

OccupancyPlotsFPIXmDetailedWantedSubDets = OccupancyPlotsFPIXmD1DetailedWantedSubDets
OccupancyPlotsFPIXmDetailedWantedSubDets.extend(OccupancyPlotsFPIXmD2DetailedWantedSubDets)
OccupancyPlotsFPIXmDetailedWantedSubDets.extend(OccupancyPlotsFPIXmD3DetailedWantedSubDets)

OccupancyPlotsFPIXpD1DetailedWantedSubDets = cms.VPSet(
    cms.PSet(detSelection = cms.uint32(2001),detLabel = cms.string("FPIXpD1R1m1p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15041400")),
    cms.PSet(detSelection = cms.uint32(2002),detLabel = cms.string("FPIXpD1R1m2p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15042400")),
    cms.PSet(detSelection = cms.uint32(2003),detLabel = cms.string("FPIXpD1R1m3p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15043400")),
    cms.PSet(detSelection = cms.uint32(2004),detLabel = cms.string("FPIXpD1R1m4p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15044400")),
    cms.PSet(detSelection = cms.uint32(2005),detLabel = cms.string("FPIXpD1R1m5p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15045400")),
    cms.PSet(detSelection = cms.uint32(2006),detLabel = cms.string("FPIXpD1R1m6p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15046400")),
    cms.PSet(detSelection = cms.uint32(2007),detLabel = cms.string("FPIXpD1R1m7p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15047400")),
    cms.PSet(detSelection = cms.uint32(2008),detLabel = cms.string("FPIXpD1R1m8p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15048400")),
    cms.PSet(detSelection = cms.uint32(2009),detLabel = cms.string("FPIXpD1R1m9p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15049400")),
    cms.PSet(detSelection = cms.uint32(2010),detLabel = cms.string("FPIXpD1R1m10p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1504a400")),
    cms.PSet(detSelection = cms.uint32(2011),detLabel = cms.string("FPIXpD1R1m11p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1504b400")),
    cms.PSet(detSelection = cms.uint32(2012),detLabel = cms.string("FPIXpD1R1m12p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1504c400")),
    cms.PSet(detSelection = cms.uint32(2013),detLabel = cms.string("FPIXpD1R1m13p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1504d400")),
    cms.PSet(detSelection = cms.uint32(2014),detLabel = cms.string("FPIXpD1R1m14p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1504e400")),
    cms.PSet(detSelection = cms.uint32(2015),detLabel = cms.string("FPIXpD1R1m15p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1504f400")),
    cms.PSet(detSelection = cms.uint32(2016),detLabel = cms.string("FPIXpD1R1m16p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15050400")),
    cms.PSet(detSelection = cms.uint32(2017),detLabel = cms.string("FPIXpD1R1m17p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15051400")),
    cms.PSet(detSelection = cms.uint32(2018),detLabel = cms.string("FPIXpD1R1m18p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15052400")),
    cms.PSet(detSelection = cms.uint32(2019),detLabel = cms.string("FPIXpD1R1m19p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15053400")),
    cms.PSet(detSelection = cms.uint32(2020),detLabel = cms.string("FPIXpD1R1m20p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15054400")),
    cms.PSet(detSelection = cms.uint32(2021),detLabel = cms.string("FPIXpD1R1m21p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15055400")),
    cms.PSet(detSelection = cms.uint32(2022),detLabel = cms.string("FPIXpD1R1m22p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15056400")),
    cms.PSet(detSelection = cms.uint32(2023),detLabel = cms.string("FPIXpD1R2m23p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15057400")),
    cms.PSet(detSelection = cms.uint32(2024),detLabel = cms.string("FPIXpD1R2m24p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15058400")),
    cms.PSet(detSelection = cms.uint32(2025),detLabel = cms.string("FPIXpD1R2m25p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15059400")),
    cms.PSet(detSelection = cms.uint32(2026),detLabel = cms.string("FPIXpD1R2m26p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1505a400")),
    cms.PSet(detSelection = cms.uint32(2027),detLabel = cms.string("FPIXpD1R2m27p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1505b400")),
    cms.PSet(detSelection = cms.uint32(2028),detLabel = cms.string("FPIXpD1R2m28p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1505c400")),
    cms.PSet(detSelection = cms.uint32(2029),detLabel = cms.string("FPIXpD1R2m29p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1505d400")),
    cms.PSet(detSelection = cms.uint32(2030),detLabel = cms.string("FPIXpD1R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1505e400")),
    cms.PSet(detSelection = cms.uint32(2031),detLabel = cms.string("FPIXpD1R2m31p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1505f400")),
    cms.PSet(detSelection = cms.uint32(2032),detLabel = cms.string("FPIXpD1R2m32p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15060400")),
    cms.PSet(detSelection = cms.uint32(2033),detLabel = cms.string("FPIXpD1R2m33p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15061400")),
    cms.PSet(detSelection = cms.uint32(2034),detLabel = cms.string("FPIXpD1R2m34p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15062400")),
    cms.PSet(detSelection = cms.uint32(2035),detLabel = cms.string("FPIXpD1R2m35p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15063400")),
    cms.PSet(detSelection = cms.uint32(2036),detLabel = cms.string("FPIXpD1R2m36p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15064400")),
    cms.PSet(detSelection = cms.uint32(2037),detLabel = cms.string("FPIXpD1R2m37p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15065400")),
    cms.PSet(detSelection = cms.uint32(2038),detLabel = cms.string("FPIXpD1R2m38p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15066400")),
    cms.PSet(detSelection = cms.uint32(2039),detLabel = cms.string("FPIXpD1R2m39p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15067400")),
    cms.PSet(detSelection = cms.uint32(2040),detLabel = cms.string("FPIXpD1R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15068400")),
    cms.PSet(detSelection = cms.uint32(2041),detLabel = cms.string("FPIXpD1R2m41p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15069400")),
    cms.PSet(detSelection = cms.uint32(2042),detLabel = cms.string("FPIXpD1R2m42p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1506a400")),
    cms.PSet(detSelection = cms.uint32(2043),detLabel = cms.string("FPIXpD1R2m43p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1506b400")),
    cms.PSet(detSelection = cms.uint32(2044),detLabel = cms.string("FPIXpD1R2m44p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1506c400")),
    cms.PSet(detSelection = cms.uint32(2045),detLabel = cms.string("FPIXpD1R2m45p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1506d400")),
    cms.PSet(detSelection = cms.uint32(2046),detLabel = cms.string("FPIXpD1R2m46p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1506e400")),
    cms.PSet(detSelection = cms.uint32(2047),detLabel = cms.string("FPIXpD1R2m47p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1506f400")),
    cms.PSet(detSelection = cms.uint32(2048),detLabel = cms.string("FPIXpD1R2m48p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15070400")),
    cms.PSet(detSelection = cms.uint32(2049),detLabel = cms.string("FPIXpD1R2m49p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15071400")),
    cms.PSet(detSelection = cms.uint32(2050),detLabel = cms.string("FPIXpD1R2m50p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15072400")),
    cms.PSet(detSelection = cms.uint32(2051),detLabel = cms.string("FPIXpD1R2m51p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15073400")),
    cms.PSet(detSelection = cms.uint32(2052),detLabel = cms.string("FPIXpD1R2m52p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15074400")),
    cms.PSet(detSelection = cms.uint32(2053),detLabel = cms.string("FPIXpD1R2m53p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15075400")),
    cms.PSet(detSelection = cms.uint32(2054),detLabel = cms.string("FPIXpD1R2m54p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15076400")),
    cms.PSet(detSelection = cms.uint32(2055),detLabel = cms.string("FPIXpD1R2m55p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15077400")),
    cms.PSet(detSelection = cms.uint32(2056),detLabel = cms.string("FPIXpD1R2m56p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15078400")),
#
    cms.PSet(detSelection = cms.uint32(2101),detLabel = cms.string("FPIXpD1R1m1p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15041800")),
    cms.PSet(detSelection = cms.uint32(2102),detLabel = cms.string("FPIXpD1R1m2p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15042800")),
    cms.PSet(detSelection = cms.uint32(2103),detLabel = cms.string("FPIXpD1R1m3p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15043800")),
    cms.PSet(detSelection = cms.uint32(2104),detLabel = cms.string("FPIXpD1R1m4p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15044800")),
    cms.PSet(detSelection = cms.uint32(2105),detLabel = cms.string("FPIXpD1R1m5p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15045800")),
    cms.PSet(detSelection = cms.uint32(2106),detLabel = cms.string("FPIXpD1R1m6p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15046800")),
    cms.PSet(detSelection = cms.uint32(2107),detLabel = cms.string("FPIXpD1R1m7p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15047800")),
    cms.PSet(detSelection = cms.uint32(2108),detLabel = cms.string("FPIXpD1R1m8p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15048800")),
    cms.PSet(detSelection = cms.uint32(2109),detLabel = cms.string("FPIXpD1R1m9p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15049800")),
    cms.PSet(detSelection = cms.uint32(2110),detLabel = cms.string("FPIXpD1R1m10p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1504a800")),
    cms.PSet(detSelection = cms.uint32(2111),detLabel = cms.string("FPIXpD1R1m11p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1504b800")),
    cms.PSet(detSelection = cms.uint32(2112),detLabel = cms.string("FPIXpD1R1m12p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1504c800")),
    cms.PSet(detSelection = cms.uint32(2113),detLabel = cms.string("FPIXpD1R1m13p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1504d800")),
    cms.PSet(detSelection = cms.uint32(2114),detLabel = cms.string("FPIXpD1R1m14p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1504e800")),
    cms.PSet(detSelection = cms.uint32(2115),detLabel = cms.string("FPIXpD1R1m15p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1504f800")),
    cms.PSet(detSelection = cms.uint32(2116),detLabel = cms.string("FPIXpD1R1m16p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15050800")),
    cms.PSet(detSelection = cms.uint32(2117),detLabel = cms.string("FPIXpD1R1m17p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15051800")),
    cms.PSet(detSelection = cms.uint32(2118),detLabel = cms.string("FPIXpD1R1m18p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15052800")),
    cms.PSet(detSelection = cms.uint32(2119),detLabel = cms.string("FPIXpD1R1m19p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15053800")),
    cms.PSet(detSelection = cms.uint32(2120),detLabel = cms.string("FPIXpD1R1m20p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15054800")),
    cms.PSet(detSelection = cms.uint32(2121),detLabel = cms.string("FPIXpD1R1m21p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15055800")),
    cms.PSet(detSelection = cms.uint32(2122),detLabel = cms.string("FPIXpD1R1m22p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15056800")),
    cms.PSet(detSelection = cms.uint32(2123),detLabel = cms.string("FPIXpD1R2m23p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15057800")),
    cms.PSet(detSelection = cms.uint32(2124),detLabel = cms.string("FPIXpD1R2m24p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15058800")),
    cms.PSet(detSelection = cms.uint32(2125),detLabel = cms.string("FPIXpD1R2m25p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15059800")),
    cms.PSet(detSelection = cms.uint32(2126),detLabel = cms.string("FPIXpD1R2m26p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1505a800")),
    cms.PSet(detSelection = cms.uint32(2127),detLabel = cms.string("FPIXpD1R2m27p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1505b800")),
    cms.PSet(detSelection = cms.uint32(2128),detLabel = cms.string("FPIXpD1R2m28p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1505c800")),
    cms.PSet(detSelection = cms.uint32(2129),detLabel = cms.string("FPIXpD1R2m29p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1505d800")),
    cms.PSet(detSelection = cms.uint32(2130),detLabel = cms.string("FPIXpD1R2m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1505e800")),
    cms.PSet(detSelection = cms.uint32(2131),detLabel = cms.string("FPIXpD1R2m31p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1505f800")),
    cms.PSet(detSelection = cms.uint32(2132),detLabel = cms.string("FPIXpD1R2m32p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15060800")),
    cms.PSet(detSelection = cms.uint32(2133),detLabel = cms.string("FPIXpD1R2m33p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15061800")),
    cms.PSet(detSelection = cms.uint32(2134),detLabel = cms.string("FPIXpD1R2m34p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15062800")),
    cms.PSet(detSelection = cms.uint32(2135),detLabel = cms.string("FPIXpD1R2m35p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15063800")),
    cms.PSet(detSelection = cms.uint32(2136),detLabel = cms.string("FPIXpD1R2m36p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15064800")),
    cms.PSet(detSelection = cms.uint32(2137),detLabel = cms.string("FPIXpD1R2m37p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15065800")),
    cms.PSet(detSelection = cms.uint32(2138),detLabel = cms.string("FPIXpD1R2m38p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15066800")),
    cms.PSet(detSelection = cms.uint32(2139),detLabel = cms.string("FPIXpD1R2m39p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15067800")),
    cms.PSet(detSelection = cms.uint32(2140),detLabel = cms.string("FPIXpD1R2m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15068800")),
    cms.PSet(detSelection = cms.uint32(2141),detLabel = cms.string("FPIXpD1R2m41p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15069800")),
    cms.PSet(detSelection = cms.uint32(2142),detLabel = cms.string("FPIXpD1R2m42p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1506a800")),
    cms.PSet(detSelection = cms.uint32(2143),detLabel = cms.string("FPIXpD1R2m43p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1506b800")),
    cms.PSet(detSelection = cms.uint32(2144),detLabel = cms.string("FPIXpD1R2m44p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1506c800")),
    cms.PSet(detSelection = cms.uint32(2145),detLabel = cms.string("FPIXpD1R2m45p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1506d800")),
    cms.PSet(detSelection = cms.uint32(2146),detLabel = cms.string("FPIXpD1R2m46p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1506e800")),
    cms.PSet(detSelection = cms.uint32(2147),detLabel = cms.string("FPIXpD1R2m47p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1506f800")),
    cms.PSet(detSelection = cms.uint32(2148),detLabel = cms.string("FPIXpD1R2m48p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15070800")),
    cms.PSet(detSelection = cms.uint32(2149),detLabel = cms.string("FPIXpD1R2m49p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15071800")),
    cms.PSet(detSelection = cms.uint32(2150),detLabel = cms.string("FPIXpD1R2m50p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15072800")),
    cms.PSet(detSelection = cms.uint32(2151),detLabel = cms.string("FPIXpD1R2m51p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15073800")),
    cms.PSet(detSelection = cms.uint32(2152),detLabel = cms.string("FPIXpD1R2m52p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15074800")),
    cms.PSet(detSelection = cms.uint32(2153),detLabel = cms.string("FPIXpD1R2m53p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15075800")),
    cms.PSet(detSelection = cms.uint32(2154),detLabel = cms.string("FPIXpD1R2m54p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15076800")),
    cms.PSet(detSelection = cms.uint32(2155),detLabel = cms.string("FPIXpD1R2m55p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15077800")),
    cms.PSet(detSelection = cms.uint32(2156),detLabel = cms.string("FPIXpD1R2m56p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15078800")),
)
OccupancyPlotsFPIXpD2DetailedWantedSubDets = cms.VPSet(
    cms.PSet(detSelection = cms.uint32(2201),detLabel = cms.string("FPIXpD2R1m1p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15081400")),
    cms.PSet(detSelection = cms.uint32(2202),detLabel = cms.string("FPIXpD2R1m2p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15082400")),
    cms.PSet(detSelection = cms.uint32(2203),detLabel = cms.string("FPIXpD2R1m3p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15083400")),
    cms.PSet(detSelection = cms.uint32(2204),detLabel = cms.string("FPIXpD2R1m4p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15084400")),
    cms.PSet(detSelection = cms.uint32(2205),detLabel = cms.string("FPIXpD2R1m5p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15085400")),
    cms.PSet(detSelection = cms.uint32(2206),detLabel = cms.string("FPIXpD2R1m6p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15086400")),
    cms.PSet(detSelection = cms.uint32(2207),detLabel = cms.string("FPIXpD2R1m7p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15087400")),
    cms.PSet(detSelection = cms.uint32(2208),detLabel = cms.string("FPIXpD2R1m8p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15088400")),
    cms.PSet(detSelection = cms.uint32(2209),detLabel = cms.string("FPIXpD2R1m9p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15089400")),
    cms.PSet(detSelection = cms.uint32(2210),detLabel = cms.string("FPIXpD2R1m10p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1508a400")),
    cms.PSet(detSelection = cms.uint32(2211),detLabel = cms.string("FPIXpD2R1m11p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1508b400")),
    cms.PSet(detSelection = cms.uint32(2212),detLabel = cms.string("FPIXpD2R1m12p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1508c400")),
    cms.PSet(detSelection = cms.uint32(2213),detLabel = cms.string("FPIXpD2R1m13p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1508d400")),
    cms.PSet(detSelection = cms.uint32(2214),detLabel = cms.string("FPIXpD2R1m14p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1508e400")),
    cms.PSet(detSelection = cms.uint32(2215),detLabel = cms.string("FPIXpD2R1m15p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1508f400")),
    cms.PSet(detSelection = cms.uint32(2216),detLabel = cms.string("FPIXpD2R1m16p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15090400")),
    cms.PSet(detSelection = cms.uint32(2217),detLabel = cms.string("FPIXpD2R1m17p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15091400")),
    cms.PSet(detSelection = cms.uint32(2218),detLabel = cms.string("FPIXpD2R1m18p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15092400")),
    cms.PSet(detSelection = cms.uint32(2219),detLabel = cms.string("FPIXpD2R1m19p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15093400")),
    cms.PSet(detSelection = cms.uint32(2220),detLabel = cms.string("FPIXpD2R1m20p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15094400")),
    cms.PSet(detSelection = cms.uint32(2221),detLabel = cms.string("FPIXpD2R1m21p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15095400")),
    cms.PSet(detSelection = cms.uint32(2222),detLabel = cms.string("FPIXpD2R1m22p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15096400")),
    cms.PSet(detSelection = cms.uint32(2223),detLabel = cms.string("FPIXpD2R2m23p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15097400")),
    cms.PSet(detSelection = cms.uint32(2224),detLabel = cms.string("FPIXpD2R2m24p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15098400")),
    cms.PSet(detSelection = cms.uint32(2225),detLabel = cms.string("FPIXpD2R2m25p1"),selection=cms.untracked.vstring("0x1fbffc00-0x15099400")),
    cms.PSet(detSelection = cms.uint32(2226),detLabel = cms.string("FPIXpD2R2m26p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1509a400")),
    cms.PSet(detSelection = cms.uint32(2227),detLabel = cms.string("FPIXpD2R2m27p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1509b400")),
    cms.PSet(detSelection = cms.uint32(2228),detLabel = cms.string("FPIXpD2R2m28p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1509c400")),
    cms.PSet(detSelection = cms.uint32(2229),detLabel = cms.string("FPIXpD2R2m29p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1509d400")),
    cms.PSet(detSelection = cms.uint32(2230),detLabel = cms.string("FPIXpD2R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1509e400")),
    cms.PSet(detSelection = cms.uint32(2231),detLabel = cms.string("FPIXpD2R2m31p1"),selection=cms.untracked.vstring("0x1fbffc00-0x1509f400")),
    cms.PSet(detSelection = cms.uint32(2232),detLabel = cms.string("FPIXpD2R2m32p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150a0400")),
    cms.PSet(detSelection = cms.uint32(2233),detLabel = cms.string("FPIXpD2R2m33p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150a1400")),
    cms.PSet(detSelection = cms.uint32(2234),detLabel = cms.string("FPIXpD2R2m34p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150a2400")),
    cms.PSet(detSelection = cms.uint32(2235),detLabel = cms.string("FPIXpD2R2m35p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150a3400")),
    cms.PSet(detSelection = cms.uint32(2236),detLabel = cms.string("FPIXpD2R2m36p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150a4400")),
    cms.PSet(detSelection = cms.uint32(2237),detLabel = cms.string("FPIXpD2R2m37p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150a5400")),
    cms.PSet(detSelection = cms.uint32(2238),detLabel = cms.string("FPIXpD2R2m38p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150a6400")),
    cms.PSet(detSelection = cms.uint32(2239),detLabel = cms.string("FPIXpD2R2m39p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150a7400")),
    cms.PSet(detSelection = cms.uint32(2240),detLabel = cms.string("FPIXpD2R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150a8400")),
    cms.PSet(detSelection = cms.uint32(2241),detLabel = cms.string("FPIXpD2R2m41p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150a9400")),
    cms.PSet(detSelection = cms.uint32(2242),detLabel = cms.string("FPIXpD2R2m42p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150aa400")),
    cms.PSet(detSelection = cms.uint32(2243),detLabel = cms.string("FPIXpD2R2m43p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ab400")),
    cms.PSet(detSelection = cms.uint32(2244),detLabel = cms.string("FPIXpD2R2m44p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ac400")),
    cms.PSet(detSelection = cms.uint32(2245),detLabel = cms.string("FPIXpD2R2m45p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ad400")),
    cms.PSet(detSelection = cms.uint32(2246),detLabel = cms.string("FPIXpD2R2m46p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ae400")),
    cms.PSet(detSelection = cms.uint32(2247),detLabel = cms.string("FPIXpD2R2m47p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150af400")),
    cms.PSet(detSelection = cms.uint32(2248),detLabel = cms.string("FPIXpD2R2m48p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150b0400")),
    cms.PSet(detSelection = cms.uint32(2249),detLabel = cms.string("FPIXpD2R2m49p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150b1400")),
    cms.PSet(detSelection = cms.uint32(2250),detLabel = cms.string("FPIXpD2R2m50p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150b2400")),
    cms.PSet(detSelection = cms.uint32(2251),detLabel = cms.string("FPIXpD2R2m51p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150b3400")),
    cms.PSet(detSelection = cms.uint32(2252),detLabel = cms.string("FPIXpD2R2m52p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150b4400")),
    cms.PSet(detSelection = cms.uint32(2253),detLabel = cms.string("FPIXpD2R2m53p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150b5400")),
    cms.PSet(detSelection = cms.uint32(2254),detLabel = cms.string("FPIXpD2R2m54p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150b6400")),
    cms.PSet(detSelection = cms.uint32(2255),detLabel = cms.string("FPIXpD2R2m55p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150b7400")),
    cms.PSet(detSelection = cms.uint32(2256),detLabel = cms.string("FPIXpD2R2m56p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150b8400")),
#
    cms.PSet(detSelection = cms.uint32(2301),detLabel = cms.string("FPIXpD2R1m1p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15081800")),
    cms.PSet(detSelection = cms.uint32(2302),detLabel = cms.string("FPIXpD2R1m2p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15082800")),
    cms.PSet(detSelection = cms.uint32(2303),detLabel = cms.string("FPIXpD2R1m3p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15083800")),
    cms.PSet(detSelection = cms.uint32(2304),detLabel = cms.string("FPIXpD2R1m4p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15084800")),
    cms.PSet(detSelection = cms.uint32(2305),detLabel = cms.string("FPIXpD2R1m5p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15085800")),
    cms.PSet(detSelection = cms.uint32(2306),detLabel = cms.string("FPIXpD2R1m6p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15086800")),
    cms.PSet(detSelection = cms.uint32(2307),detLabel = cms.string("FPIXpD2R1m7p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15087800")),
    cms.PSet(detSelection = cms.uint32(2308),detLabel = cms.string("FPIXpD2R1m8p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15088800")),
    cms.PSet(detSelection = cms.uint32(2309),detLabel = cms.string("FPIXpD2R1m9p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15089800")),
    cms.PSet(detSelection = cms.uint32(2310),detLabel = cms.string("FPIXpD2R1m10p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1508a800")),
    cms.PSet(detSelection = cms.uint32(2311),detLabel = cms.string("FPIXpD2R1m11p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1508b800")),
    cms.PSet(detSelection = cms.uint32(2312),detLabel = cms.string("FPIXpD2R1m12p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1508c800")),
    cms.PSet(detSelection = cms.uint32(2313),detLabel = cms.string("FPIXpD2R1m13p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1508d800")),
    cms.PSet(detSelection = cms.uint32(2314),detLabel = cms.string("FPIXpD2R1m14p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1508e800")),
    cms.PSet(detSelection = cms.uint32(2315),detLabel = cms.string("FPIXpD2R1m15p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1508f800")),
    cms.PSet(detSelection = cms.uint32(2316),detLabel = cms.string("FPIXpD2R1m16p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15090800")),
    cms.PSet(detSelection = cms.uint32(2317),detLabel = cms.string("FPIXpD2R1m17p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15091800")),
    cms.PSet(detSelection = cms.uint32(2318),detLabel = cms.string("FPIXpD2R1m18p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15092800")),
    cms.PSet(detSelection = cms.uint32(2319),detLabel = cms.string("FPIXpD2R1m19p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15093800")),
    cms.PSet(detSelection = cms.uint32(2320),detLabel = cms.string("FPIXpD2R1m20p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15094800")),
    cms.PSet(detSelection = cms.uint32(2321),detLabel = cms.string("FPIXpD2R1m21p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15095800")),
    cms.PSet(detSelection = cms.uint32(2322),detLabel = cms.string("FPIXpD2R1m22p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15096800")),
    cms.PSet(detSelection = cms.uint32(2323),detLabel = cms.string("FPIXpD2R1m23p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15097800")),
    cms.PSet(detSelection = cms.uint32(2324),detLabel = cms.string("FPIXpD2R1m24p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15098800")),
    cms.PSet(detSelection = cms.uint32(2325),detLabel = cms.string("FPIXpD2R1m25p2"),selection=cms.untracked.vstring("0x1fbffc00-0x15099800")),
    cms.PSet(detSelection = cms.uint32(2326),detLabel = cms.string("FPIXpD2R1m26p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1509a800")),
    cms.PSet(detSelection = cms.uint32(2327),detLabel = cms.string("FPIXpD2R1m27p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1509b800")),
    cms.PSet(detSelection = cms.uint32(2328),detLabel = cms.string("FPIXpD2R1m28p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1509c800")),
    cms.PSet(detSelection = cms.uint32(2329),detLabel = cms.string("FPIXpD2R1m29p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1509d800")),
    cms.PSet(detSelection = cms.uint32(2330),detLabel = cms.string("FPIXpD2R1m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1509e800")),
    cms.PSet(detSelection = cms.uint32(2331),detLabel = cms.string("FPIXpD2R1m31p2"),selection=cms.untracked.vstring("0x1fbffc00-0x1509f800")),
    cms.PSet(detSelection = cms.uint32(2332),detLabel = cms.string("FPIXpD2R1m32p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150a0800")),
    cms.PSet(detSelection = cms.uint32(2333),detLabel = cms.string("FPIXpD2R1m33p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150a1800")),
    cms.PSet(detSelection = cms.uint32(2334),detLabel = cms.string("FPIXpD2R1m34p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150a2800")),
    cms.PSet(detSelection = cms.uint32(2335),detLabel = cms.string("FPIXpD2R2m35p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150a3800")),
    cms.PSet(detSelection = cms.uint32(2336),detLabel = cms.string("FPIXpD2R2m36p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150a4800")),
    cms.PSet(detSelection = cms.uint32(2337),detLabel = cms.string("FPIXpD2R2m37p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150a5800")),
    cms.PSet(detSelection = cms.uint32(2338),detLabel = cms.string("FPIXpD2R2m38p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150a6800")),
    cms.PSet(detSelection = cms.uint32(2339),detLabel = cms.string("FPIXpD2R2m39p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150a7800")),
    cms.PSet(detSelection = cms.uint32(2340),detLabel = cms.string("FPIXpD2R2m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150a8800")),
    cms.PSet(detSelection = cms.uint32(2341),detLabel = cms.string("FPIXpD2R2m41p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150a9800")),
    cms.PSet(detSelection = cms.uint32(2342),detLabel = cms.string("FPIXpD2R2m42p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150aa800")),
    cms.PSet(detSelection = cms.uint32(2343),detLabel = cms.string("FPIXpD2R2m43p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ab800")),
    cms.PSet(detSelection = cms.uint32(2344),detLabel = cms.string("FPIXpD2R2m44p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ac800")),
    cms.PSet(detSelection = cms.uint32(2345),detLabel = cms.string("FPIXpD2R2m45p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ad800")),
    cms.PSet(detSelection = cms.uint32(2346),detLabel = cms.string("FPIXpD2R2m46p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ae800")),
    cms.PSet(detSelection = cms.uint32(2347),detLabel = cms.string("FPIXpD2R2m47p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150af800")),
    cms.PSet(detSelection = cms.uint32(2348),detLabel = cms.string("FPIXpD2R2m48p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150b0800")),
    cms.PSet(detSelection = cms.uint32(2349),detLabel = cms.string("FPIXpD2R2m49p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150b1800")),
    cms.PSet(detSelection = cms.uint32(2350),detLabel = cms.string("FPIXpD2R2m50p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150b2800")),
    cms.PSet(detSelection = cms.uint32(2351),detLabel = cms.string("FPIXpD2R2m51p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150b3800")),
    cms.PSet(detSelection = cms.uint32(2352),detLabel = cms.string("FPIXpD2R2m52p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150b4800")),
    cms.PSet(detSelection = cms.uint32(2353),detLabel = cms.string("FPIXpD2R2m53p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150b5800")),
    cms.PSet(detSelection = cms.uint32(2354),detLabel = cms.string("FPIXpD2R2m54p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150b6800")),
    cms.PSet(detSelection = cms.uint32(2355),detLabel = cms.string("FPIXpD2R2m55p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150b7800")),
    cms.PSet(detSelection = cms.uint32(2356),detLabel = cms.string("FPIXpD2R2m56p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150b8800")),
)
#
OccupancyPlotsFPIXpD3DetailedWantedSubDets = cms.VPSet(
    cms.PSet(detSelection = cms.uint32(2401),detLabel = cms.string("FPIXpD3R1m1p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150c1400")),
    cms.PSet(detSelection = cms.uint32(2402),detLabel = cms.string("FPIXpD3R1m2p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150c2400")),
    cms.PSet(detSelection = cms.uint32(2403),detLabel = cms.string("FPIXpD3R1m3p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150c3400")),
    cms.PSet(detSelection = cms.uint32(2404),detLabel = cms.string("FPIXpD3R1m4p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150c4400")),
    cms.PSet(detSelection = cms.uint32(2405),detLabel = cms.string("FPIXpD3R1m5p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150c5400")),
    cms.PSet(detSelection = cms.uint32(2406),detLabel = cms.string("FPIXpD3R1m6p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150c6400")),
    cms.PSet(detSelection = cms.uint32(2407),detLabel = cms.string("FPIXpD3R1m7p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150c7400")),
    cms.PSet(detSelection = cms.uint32(2408),detLabel = cms.string("FPIXpD3R1m8p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150c8400")),
    cms.PSet(detSelection = cms.uint32(2409),detLabel = cms.string("FPIXpD3R1m9p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150c9400")),
    cms.PSet(detSelection = cms.uint32(2410),detLabel = cms.string("FPIXpD3R1m10p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ca400")),
    cms.PSet(detSelection = cms.uint32(2411),detLabel = cms.string("FPIXpD3R1m11p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150cb400")),
    cms.PSet(detSelection = cms.uint32(2412),detLabel = cms.string("FPIXpD3R1m12p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150cc400")),
    cms.PSet(detSelection = cms.uint32(2413),detLabel = cms.string("FPIXpD3R1m13p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150cd400")),
    cms.PSet(detSelection = cms.uint32(2414),detLabel = cms.string("FPIXpD3R1m14p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ce400")),
    cms.PSet(detSelection = cms.uint32(2415),detLabel = cms.string("FPIXpD3R1m15p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150cf400")),
    cms.PSet(detSelection = cms.uint32(2416),detLabel = cms.string("FPIXpD3R1m16p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150d0400")),
    cms.PSet(detSelection = cms.uint32(2417),detLabel = cms.string("FPIXpD3R1m17p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150d1400")),
    cms.PSet(detSelection = cms.uint32(2418),detLabel = cms.string("FPIXpD3R1m18p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150d2400")),
    cms.PSet(detSelection = cms.uint32(2419),detLabel = cms.string("FPIXpD3R1m19p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150d3400")),
    cms.PSet(detSelection = cms.uint32(2420),detLabel = cms.string("FPIXpD3R1m20p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150d4400")),
    cms.PSet(detSelection = cms.uint32(2421),detLabel = cms.string("FPIXpD3R1m21p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150d5400")),
    cms.PSet(detSelection = cms.uint32(2422),detLabel = cms.string("FPIXpD3R1m22p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150d6400")),
    cms.PSet(detSelection = cms.uint32(2423),detLabel = cms.string("FPIXpD3R2m23p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150d7400")),
    cms.PSet(detSelection = cms.uint32(2424),detLabel = cms.string("FPIXpD3R2m24p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150d8400")),
    cms.PSet(detSelection = cms.uint32(2425),detLabel = cms.string("FPIXpD3R2m25p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150d9400")),
    cms.PSet(detSelection = cms.uint32(2426),detLabel = cms.string("FPIXpD3R2m26p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150da400")),
    cms.PSet(detSelection = cms.uint32(2427),detLabel = cms.string("FPIXpD3R2m27p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150db400")),
    cms.PSet(detSelection = cms.uint32(2428),detLabel = cms.string("FPIXpD3R2m28p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150dc400")),
    cms.PSet(detSelection = cms.uint32(2429),detLabel = cms.string("FPIXpD3R2m29p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150dd400")),
    cms.PSet(detSelection = cms.uint32(2430),detLabel = cms.string("FPIXpD3R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150de400")),
    cms.PSet(detSelection = cms.uint32(2431),detLabel = cms.string("FPIXpD3R2m31p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150df400")),
    cms.PSet(detSelection = cms.uint32(2432),detLabel = cms.string("FPIXpD3R2m32p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150e0400")),
    cms.PSet(detSelection = cms.uint32(2433),detLabel = cms.string("FPIXpD3R2m33p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150e1400")),
    cms.PSet(detSelection = cms.uint32(2434),detLabel = cms.string("FPIXpD3R2m34p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150e2400")),
    cms.PSet(detSelection = cms.uint32(2435),detLabel = cms.string("FPIXpD3R2m35p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150e3400")),
    cms.PSet(detSelection = cms.uint32(2436),detLabel = cms.string("FPIXpD3R2m36p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150e4400")),
    cms.PSet(detSelection = cms.uint32(2437),detLabel = cms.string("FPIXpD3R2m37p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150e5400")),
    cms.PSet(detSelection = cms.uint32(2438),detLabel = cms.string("FPIXpD3R2m38p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150e6400")),
    cms.PSet(detSelection = cms.uint32(2439),detLabel = cms.string("FPIXpD3R2m39p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150e7400")),
    cms.PSet(detSelection = cms.uint32(2440),detLabel = cms.string("FPIXpD3R2m30p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150e8400")),
    cms.PSet(detSelection = cms.uint32(2441),detLabel = cms.string("FPIXpD3R2m41p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150e9400")),
    cms.PSet(detSelection = cms.uint32(2442),detLabel = cms.string("FPIXpD3R2m42p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ea400")),
    cms.PSet(detSelection = cms.uint32(2443),detLabel = cms.string("FPIXpD3R2m43p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150eb400")),
    cms.PSet(detSelection = cms.uint32(2444),detLabel = cms.string("FPIXpD3R2m44p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ec400")),
    cms.PSet(detSelection = cms.uint32(2445),detLabel = cms.string("FPIXpD3R2m45p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ed400")),
    cms.PSet(detSelection = cms.uint32(2446),detLabel = cms.string("FPIXpD3R2m46p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ee400")),
    cms.PSet(detSelection = cms.uint32(2447),detLabel = cms.string("FPIXpD3R2m47p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150ef400")),
    cms.PSet(detSelection = cms.uint32(2448),detLabel = cms.string("FPIXpD3R2m48p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150f0400")),
    cms.PSet(detSelection = cms.uint32(2449),detLabel = cms.string("FPIXpD3R2m49p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150f1400")),
    cms.PSet(detSelection = cms.uint32(2450),detLabel = cms.string("FPIXpD3R2m50p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150f2400")),
    cms.PSet(detSelection = cms.uint32(2451),detLabel = cms.string("FPIXpD3R2m51p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150f3400")),
    cms.PSet(detSelection = cms.uint32(2452),detLabel = cms.string("FPIXpD3R2m52p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150f4400")),
    cms.PSet(detSelection = cms.uint32(2453),detLabel = cms.string("FPIXpD3R2m53p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150f5400")),
    cms.PSet(detSelection = cms.uint32(2454),detLabel = cms.string("FPIXpD3R2m54p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150f6400")),
    cms.PSet(detSelection = cms.uint32(2455),detLabel = cms.string("FPIXpD3R2m55p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150f7400")),
    cms.PSet(detSelection = cms.uint32(2456),detLabel = cms.string("FPIXpD3R2m56p1"),selection=cms.untracked.vstring("0x1fbffc00-0x150f8400")),
#
    cms.PSet(detSelection = cms.uint32(2501),detLabel = cms.string("FPIXpD3R1m1p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150c1800")),
    cms.PSet(detSelection = cms.uint32(2502),detLabel = cms.string("FPIXpD3R1m2p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150c2800")),
    cms.PSet(detSelection = cms.uint32(2503),detLabel = cms.string("FPIXpD3R1m3p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150c3800")),
    cms.PSet(detSelection = cms.uint32(2504),detLabel = cms.string("FPIXpD3R1m4p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150c4800")),
    cms.PSet(detSelection = cms.uint32(2505),detLabel = cms.string("FPIXpD3R1m5p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150c5800")),
    cms.PSet(detSelection = cms.uint32(2506),detLabel = cms.string("FPIXpD3R1m6p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150c6800")),
    cms.PSet(detSelection = cms.uint32(2507),detLabel = cms.string("FPIXpD3R1m7p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150c7800")),
    cms.PSet(detSelection = cms.uint32(2508),detLabel = cms.string("FPIXpD3R1m8p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150c8800")),
    cms.PSet(detSelection = cms.uint32(2509),detLabel = cms.string("FPIXpD3R1m9p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150c9800")),
    cms.PSet(detSelection = cms.uint32(2510),detLabel = cms.string("FPIXpD3R1m10p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ca800")),
    cms.PSet(detSelection = cms.uint32(2511),detLabel = cms.string("FPIXpD3R1m11p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150cb800")),
    cms.PSet(detSelection = cms.uint32(2512),detLabel = cms.string("FPIXpD3R1m12p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150cc800")),
    cms.PSet(detSelection = cms.uint32(2513),detLabel = cms.string("FPIXpD3R1m13p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150cd800")),
    cms.PSet(detSelection = cms.uint32(2514),detLabel = cms.string("FPIXpD3R1m14p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ce800")),
    cms.PSet(detSelection = cms.uint32(2515),detLabel = cms.string("FPIXpD3R1m15p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150cf800")),
    cms.PSet(detSelection = cms.uint32(2516),detLabel = cms.string("FPIXpD3R1m16p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150d0800")),
    cms.PSet(detSelection = cms.uint32(2517),detLabel = cms.string("FPIXpD3R1m17p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150d1800")),
    cms.PSet(detSelection = cms.uint32(2518),detLabel = cms.string("FPIXpD3R1m18p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150d2800")),
    cms.PSet(detSelection = cms.uint32(2519),detLabel = cms.string("FPIXpD3R1m19p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150d3800")),
    cms.PSet(detSelection = cms.uint32(2520),detLabel = cms.string("FPIXpD3R1m20p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150d4800")),
    cms.PSet(detSelection = cms.uint32(2521),detLabel = cms.string("FPIXpD3R1m21p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150d5800")),
    cms.PSet(detSelection = cms.uint32(2522),detLabel = cms.string("FPIXpD3R1m22p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150d6800")),
    cms.PSet(detSelection = cms.uint32(2523),detLabel = cms.string("FPIXpD3R1m23p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150d7800")),
    cms.PSet(detSelection = cms.uint32(2524),detLabel = cms.string("FPIXpD3R1m24p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150d8800")),
    cms.PSet(detSelection = cms.uint32(2525),detLabel = cms.string("FPIXpD3R1m25p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150d9800")),
    cms.PSet(detSelection = cms.uint32(2526),detLabel = cms.string("FPIXpD3R1m26p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150da800")),
    cms.PSet(detSelection = cms.uint32(2527),detLabel = cms.string("FPIXpD3R1m27p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150db800")),
    cms.PSet(detSelection = cms.uint32(2528),detLabel = cms.string("FPIXpD3R1m28p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150dc800")),
    cms.PSet(detSelection = cms.uint32(2529),detLabel = cms.string("FPIXpD3R1m29p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150dd800")),
    cms.PSet(detSelection = cms.uint32(2530),detLabel = cms.string("FPIXpD3R1m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150de800")),
    cms.PSet(detSelection = cms.uint32(2531),detLabel = cms.string("FPIXpD3R1m31p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150df800")),
    cms.PSet(detSelection = cms.uint32(2532),detLabel = cms.string("FPIXpD3R1m32p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150e0800")),
    cms.PSet(detSelection = cms.uint32(2533),detLabel = cms.string("FPIXpD3R1m33p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150e1800")),
    cms.PSet(detSelection = cms.uint32(2534),detLabel = cms.string("FPIXpD3R1m34p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150e2800")),
    cms.PSet(detSelection = cms.uint32(2535),detLabel = cms.string("FPIXpD3R2m35p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150e3800")),
    cms.PSet(detSelection = cms.uint32(2536),detLabel = cms.string("FPIXpD3R2m36p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150e4800")),
    cms.PSet(detSelection = cms.uint32(2537),detLabel = cms.string("FPIXpD3R2m37p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150e5800")),
    cms.PSet(detSelection = cms.uint32(2538),detLabel = cms.string("FPIXpD3R2m38p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150e6800")),
    cms.PSet(detSelection = cms.uint32(2539),detLabel = cms.string("FPIXpD3R2m39p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150e7800")),
    cms.PSet(detSelection = cms.uint32(2540),detLabel = cms.string("FPIXpD3R2m30p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150e8800")),
    cms.PSet(detSelection = cms.uint32(2541),detLabel = cms.string("FPIXpD3R2m41p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150e9800")),
    cms.PSet(detSelection = cms.uint32(2542),detLabel = cms.string("FPIXpD3R2m42p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ea800")),
    cms.PSet(detSelection = cms.uint32(2543),detLabel = cms.string("FPIXpD3R2m43p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150eb800")),
    cms.PSet(detSelection = cms.uint32(2544),detLabel = cms.string("FPIXpD3R2m44p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ec800")),
    cms.PSet(detSelection = cms.uint32(2545),detLabel = cms.string("FPIXpD3R2m45p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ed800")),
    cms.PSet(detSelection = cms.uint32(2546),detLabel = cms.string("FPIXpD3R2m46p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ee800")),
    cms.PSet(detSelection = cms.uint32(2547),detLabel = cms.string("FPIXpD3R2m47p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150ef800")),
    cms.PSet(detSelection = cms.uint32(2548),detLabel = cms.string("FPIXpD3R2m48p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150f0800")),
    cms.PSet(detSelection = cms.uint32(2549),detLabel = cms.string("FPIXpD3R2m49p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150f1800")),
    cms.PSet(detSelection = cms.uint32(2550),detLabel = cms.string("FPIXpD3R2m50p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150f2800")),
    cms.PSet(detSelection = cms.uint32(2551),detLabel = cms.string("FPIXpD3R2m51p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150f3800")),
    cms.PSet(detSelection = cms.uint32(2552),detLabel = cms.string("FPIXpD3R2m52p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150f4800")),
    cms.PSet(detSelection = cms.uint32(2553),detLabel = cms.string("FPIXpD3R2m53p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150f5800")),
    cms.PSet(detSelection = cms.uint32(2554),detLabel = cms.string("FPIXpD3R2m54p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150f6800")),
    cms.PSet(detSelection = cms.uint32(2555),detLabel = cms.string("FPIXpD3R2m55p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150f7800")),
    cms.PSet(detSelection = cms.uint32(2556),detLabel = cms.string("FPIXpD3R2m56p2"),selection=cms.untracked.vstring("0x1fbffc00-0x150f8800")),
)
 
OccupancyPlotsFPIXpDetailedWantedSubDets = OccupancyPlotsFPIXpD1DetailedWantedSubDets
OccupancyPlotsFPIXpDetailedWantedSubDets.extend(OccupancyPlotsFPIXpD2DetailedWantedSubDets)
OccupancyPlotsFPIXpDetailedWantedSubDets.extend(OccupancyPlotsFPIXpD3DetailedWantedSubDets)

OccupancyPlotsStripWantedSubDets = cms.VPSet (
     cms.PSet(detSelection=cms.uint32(1101),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600640c")),     # TIB+ L1 int m3
     cms.PSet(detSelection=cms.uint32(1102),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600680c")),     # TIB+ L1 ext m3
     cms.PSet(detSelection=cms.uint32(1103),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16006408")),     # TIB+ L1 int m2
     cms.PSet(detSelection=cms.uint32(1104),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16006808")),     # TIB+ L1 ext m2
     cms.PSet(detSelection=cms.uint32(1105),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16006404")),     # TIB+ L1 int m1
     cms.PSet(detSelection=cms.uint32(1106),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16006804")),     # TIB+ L1 ext m1
     cms.PSet(detSelection=cms.uint32(1107),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16005404")),     # TIB- L1 int m1
     cms.PSet(detSelection=cms.uint32(1108),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16005804")),     # TIB- L1 ext m1
     cms.PSet(detSelection=cms.uint32(1109),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16005408")),     # TIB- L1 int m2
     cms.PSet(detSelection=cms.uint32(1110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16005808")),     # TIB- L1 ext m2
     cms.PSet(detSelection=cms.uint32(1111),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600540c")),     # TIB- L1 int m3
     cms.PSet(detSelection=cms.uint32(1112),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600580c")),     # TIB- L1 ext m3
     cms.PSet(detSelection=cms.uint32(1201),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a80c")),     # TIB+ L2 ext m3
     cms.PSet(detSelection=cms.uint32(1202),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a40c")),     # TIB+ L2 int m3
     cms.PSet(detSelection=cms.uint32(1203),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a808")),     # TIB+ L2 ext m2
     cms.PSet(detSelection=cms.uint32(1204),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a408")),     # TIB+ L2 int m2
     cms.PSet(detSelection=cms.uint32(1205),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a804")),     # TIB+ L2 ext m1
     cms.PSet(detSelection=cms.uint32(1206),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a404")),     # TIB+ L2 int m1
     cms.PSet(detSelection=cms.uint32(1207),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16009804")),     # TIB- L2 ext m1
     cms.PSet(detSelection=cms.uint32(1208),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16009404")),     # TIB- L2 int m1
     cms.PSet(detSelection=cms.uint32(1209),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16009808")),     # TIB- L2 ext m2
     cms.PSet(detSelection=cms.uint32(1210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16009408")),     # TIB- L2 int m2
     cms.PSet(detSelection=cms.uint32(1211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600980c")),     # TIB- L2 ext m3
     cms.PSet(detSelection=cms.uint32(1212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600940c")),     # TIB- L2 int m3
     cms.PSet(detSelection=cms.uint32(1301),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e40c")),     # TIB+ L3 int m3
     cms.PSet(detSelection=cms.uint32(1302),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e80c")),     # TIB+ L3 ext m3
     cms.PSet(detSelection=cms.uint32(1303),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e408")),     # TIB+ L3 int m2
     cms.PSet(detSelection=cms.uint32(1304),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e808")),     # TIB+ L3 ext m2
     cms.PSet(detSelection=cms.uint32(1305),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e404")),     # TIB+ L3 int m1
     cms.PSet(detSelection=cms.uint32(1306),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e804")),     # TIB+ L3 ext m1
     cms.PSet(detSelection=cms.uint32(1307),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d404")),     # TIB- L3 int m1
     cms.PSet(detSelection=cms.uint32(1308),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d804")),     # TIB- L3 ext m1
     cms.PSet(detSelection=cms.uint32(1309),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d408")),     # TIB- L3 int m2
     cms.PSet(detSelection=cms.uint32(1310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d808")),     # TIB- L3 ext m2
     cms.PSet(detSelection=cms.uint32(1311),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d40c")),     # TIB- L3 int m3
     cms.PSet(detSelection=cms.uint32(1312),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d80c")),     # TIB- L3 ext m3
     cms.PSet(detSelection=cms.uint32(1401),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1601280c")),     # TIB+ L4 ext m3
     cms.PSet(detSelection=cms.uint32(1402),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1601240c")),     # TIB+ L4 int m3
     cms.PSet(detSelection=cms.uint32(1403),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16012808")),     # TIB+ L4 ext m2
     cms.PSet(detSelection=cms.uint32(1404),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16012408")),     # TIB+ L4 int m2
     cms.PSet(detSelection=cms.uint32(1405),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16012804")),     # TIB+ L4 ext m1
     cms.PSet(detSelection=cms.uint32(1406),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16012404")),     # TIB+ L4 int m1
     cms.PSet(detSelection=cms.uint32(1407),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16011804")),     # TIB- L4 ext m1
     cms.PSet(detSelection=cms.uint32(1408),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16011404")),     # TIB- L4 int m1
     cms.PSet(detSelection=cms.uint32(1409),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16011808")),     # TIB- L4 ext m2
     cms.PSet(detSelection=cms.uint32(1410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16011408")),     # TIB- L4 int m2
     cms.PSet(detSelection=cms.uint32(1411),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1601180c")),     # TIB- L4 ext m3
     cms.PSet(detSelection=cms.uint32(1412),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1601140c")),     # TIB- L4 int m3

     cms.PSet(detSelection=cms.uint32(2110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18002a00")),     # TID- D1 R1 Front
     cms.PSet(detSelection=cms.uint32(2120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003200")),     # TID- D2 R1 Front
     cms.PSet(detSelection=cms.uint32(2130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003a00")),     # TID- D3 R1 Front
     cms.PSet(detSelection=cms.uint32(2140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18004a00")),     # TID+ D1 R1 Front
     cms.PSet(detSelection=cms.uint32(2150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005200")),     # TID+ D2 R1 Front
     cms.PSet(detSelection=cms.uint32(2160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005a00")),     # TID+ D3 R1 Front

     cms.PSet(detSelection=cms.uint32(2210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18002c00")),     # TID- D1 R2 Front
     cms.PSet(detSelection=cms.uint32(2220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003400")),     # TID- D2 R2 Front
     cms.PSet(detSelection=cms.uint32(2230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003c00")),     # TID- D3 R2 Front
     cms.PSet(detSelection=cms.uint32(2240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18004c00")),     # TID+ D1 R2 Front
     cms.PSet(detSelection=cms.uint32(2250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005400")),     # TID+ D2 R2 Front
     cms.PSet(detSelection=cms.uint32(2260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005c00")),     # TID+ D3 R2 Front

     cms.PSet(detSelection=cms.uint32(2310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18002e00")),     # TID- D1 R3 Front
     cms.PSet(detSelection=cms.uint32(2320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003600")),     # TID- D2 R3 Front
     cms.PSet(detSelection=cms.uint32(2330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003e00")),     # TID- D3 R3 Front
     cms.PSet(detSelection=cms.uint32(2340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18004e00")),     # TID+ D1 R3 Front
     cms.PSet(detSelection=cms.uint32(2350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005600")),     # TID+ D2 R3 Front
     cms.PSet(detSelection=cms.uint32(2360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005e00")),     # TID+ D3 R3 Front

    cms.PSet(detSelection=cms.uint32(3101),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a006018")),     # TOB+ L1 m6
    cms.PSet(detSelection=cms.uint32(3102),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a006014")),     # TOB+ L1 m5
    cms.PSet(detSelection=cms.uint32(3103),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a006010")),     # TOB+ L1 m4
    cms.PSet(detSelection=cms.uint32(3104),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00600c")),     # TOB+ L1 m3
    cms.PSet(detSelection=cms.uint32(3105),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a006008")),     # TOB+ L1 m2
    cms.PSet(detSelection=cms.uint32(3106),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a006004")),     # TOB+ L1 m1
    cms.PSet(detSelection=cms.uint32(3107),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a005004")),     # TOB- L1 m1
    cms.PSet(detSelection=cms.uint32(3108),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a005008")),     # TOB- L1 m2
    cms.PSet(detSelection=cms.uint32(3109),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00500c")),     # TOB- L1 m3
    cms.PSet(detSelection=cms.uint32(3110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a005010")),     # TOB- L1 m4
    cms.PSet(detSelection=cms.uint32(3111),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a005014")),     # TOB- L1 m5
    cms.PSet(detSelection=cms.uint32(3112),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a005018")),     # TOB- L1 m6

    cms.PSet(detSelection=cms.uint32(3201),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a018")),     # TOB+ L2 m6
    cms.PSet(detSelection=cms.uint32(3202),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a014")),     # TOB+ L2 m5
    cms.PSet(detSelection=cms.uint32(3203),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a010")),     # TOB+ L2 m4
    cms.PSet(detSelection=cms.uint32(3204),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a00c")),     # TOB+ L2 m3
    cms.PSet(detSelection=cms.uint32(3205),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a008")),     # TOB+ L2 m2
    cms.PSet(detSelection=cms.uint32(3206),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a004")),     # TOB+ L2 m1
    cms.PSet(detSelection=cms.uint32(3207),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a009004")),     # TOB- L2 m1
    cms.PSet(detSelection=cms.uint32(3208),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a009008")),     # TOB- L2 m2
    cms.PSet(detSelection=cms.uint32(3209),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00900c")),     # TOB- L2 m3
    cms.PSet(detSelection=cms.uint32(3210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a009010")),     # TOB- L2 m4
    cms.PSet(detSelection=cms.uint32(3211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a009014")),     # TOB- L2 m5
    cms.PSet(detSelection=cms.uint32(3212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a009018")),     # TOB- L2 m6

    cms.PSet(detSelection=cms.uint32(3301),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e018")),     # TOB+ L3 m6
    cms.PSet(detSelection=cms.uint32(3302),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e014")),     # TOB+ L3 m5
    cms.PSet(detSelection=cms.uint32(3303),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e010")),     # TOB+ L3 m4
    cms.PSet(detSelection=cms.uint32(3304),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e00c")),     # TOB+ L3 m3
    cms.PSet(detSelection=cms.uint32(3305),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e008")),     # TOB+ L3 m2
    cms.PSet(detSelection=cms.uint32(3306),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e004")),     # TOB+ L3 m1
    cms.PSet(detSelection=cms.uint32(3307),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d004")),     # TOB- L3 m1
    cms.PSet(detSelection=cms.uint32(3308),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d008")),     # TOB- L3 m2
    cms.PSet(detSelection=cms.uint32(3309),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d00c")),     # TOB- L3 m3
    cms.PSet(detSelection=cms.uint32(3310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d010")),     # TOB- L3 m4
    cms.PSet(detSelection=cms.uint32(3311),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d014")),     # TOB- L3 m5
    cms.PSet(detSelection=cms.uint32(3312),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d018")),     # TOB- L3 m6

    cms.PSet(detSelection=cms.uint32(3401),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a012018")),     # TOB+ L4 m6
    cms.PSet(detSelection=cms.uint32(3402),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a012014")),     # TOB+ L4 m5
    cms.PSet(detSelection=cms.uint32(3403),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a012010")),     # TOB+ L4 m4
    cms.PSet(detSelection=cms.uint32(3404),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01200c")),     # TOB+ L4 m3
    cms.PSet(detSelection=cms.uint32(3405),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a012008")),     # TOB+ L4 m2
    cms.PSet(detSelection=cms.uint32(3406),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a012004")),     # TOB+ L4 m1
    cms.PSet(detSelection=cms.uint32(3407),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a011004")),     # TOB- L4 m1
    cms.PSet(detSelection=cms.uint32(3408),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a011008")),     # TOB- L4 m2
    cms.PSet(detSelection=cms.uint32(3409),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01100c")),     # TOB- L4 m3
    cms.PSet(detSelection=cms.uint32(3410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a011010")),     # TOB- L4 m4
    cms.PSet(detSelection=cms.uint32(3411),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a011014")),     # TOB- L4 m5
    cms.PSet(detSelection=cms.uint32(3412),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a011018")),     # TOB- L4 m6

    cms.PSet(detSelection=cms.uint32(3501),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a016018")),     # TOB+ L5 m6
    cms.PSet(detSelection=cms.uint32(3502),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a016014")),     # TOB+ L5 m5
    cms.PSet(detSelection=cms.uint32(3503),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a016010")),     # TOB+ L5 m4
    cms.PSet(detSelection=cms.uint32(3504),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01600c")),     # TOB+ L5 m3
    cms.PSet(detSelection=cms.uint32(3505),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a016008")),     # TOB+ L5 m2
    cms.PSet(detSelection=cms.uint32(3506),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a016004")),     # TOB+ L5 m1
    cms.PSet(detSelection=cms.uint32(3507),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a015004")),     # TOB- L5 m1
    cms.PSet(detSelection=cms.uint32(3508),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a015008")),     # TOB- L5 m2
    cms.PSet(detSelection=cms.uint32(3509),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01500c")),     # TOB- L5 m3
    cms.PSet(detSelection=cms.uint32(3510),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a015010")),     # TOB- L5 m4
    cms.PSet(detSelection=cms.uint32(3511),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a015014")),     # TOB- L5 m5
    cms.PSet(detSelection=cms.uint32(3512),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a015018")),     # TOB- L5 m6

    cms.PSet(detSelection=cms.uint32(3601),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a018")),     # TOB+ L6 m6
    cms.PSet(detSelection=cms.uint32(3602),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a014")),     # TOB+ L6 m5
    cms.PSet(detSelection=cms.uint32(3603),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a010")),     # TOB+ L6 m4
    cms.PSet(detSelection=cms.uint32(3604),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a00c")),     # TOB+ L6 m3
    cms.PSet(detSelection=cms.uint32(3605),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a008")),     # TOB+ L6 m2
    cms.PSet(detSelection=cms.uint32(3606),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a004")),     # TOB+ L6 m1
    cms.PSet(detSelection=cms.uint32(3607),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a019004")),     # TOB- L6 m1
    cms.PSet(detSelection=cms.uint32(3608),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a019008")),     # TOB- L6 m2
    cms.PSet(detSelection=cms.uint32(3609),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01900c")),     # TOB- L6 m3
    cms.PSet(detSelection=cms.uint32(3610),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a019010")),     # TOB- L6 m4
    cms.PSet(detSelection=cms.uint32(3611),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a019014")),     # TOB- L6 m5
    cms.PSet(detSelection=cms.uint32(3612),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a019018"))     # TOB- L6 m6
    )

OccupancyPlotsStripWantedSubDets.extend(
    cms.VPSet(
    cms.PSet(detSelection=cms.uint32(4110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044020")),    # TEC- D1 R1 back
    cms.PSet(detSelection=cms.uint32(4120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048020")),    # TEC- D2 R1 back
    cms.PSet(detSelection=cms.uint32(4130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c020")),    # TEC- D3 R1 back
#    cms.PSet(detSelection=cms.uint32(4140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050020")),    # TEC- D4 R1 back
#    cms.PSet(detSelection=cms.uint32(4150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054020")),    # TEC- D5 R1 back
#    cms.PSet(detSelection=cms.uint32(4160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058020")),    # TEC- D6 R1 back
#    cms.PSet(detSelection=cms.uint32(4170),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c020")),    # TEC- D7 R1 back
#    cms.PSet(detSelection=cms.uint32(4180),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060020")),    # TEC- D8 R1 back
#    cms.PSet(detSelection=cms.uint32(4190),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064020")),    # TEC- D9 R1 back

    cms.PSet(detSelection=cms.uint32(4210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044040")),    # TEC- D1 R2 back
    cms.PSet(detSelection=cms.uint32(4220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048040")),    # TEC- D2 R2 back
    cms.PSet(detSelection=cms.uint32(4230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c040")),    # TEC- D3 R2 back
    cms.PSet(detSelection=cms.uint32(4240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050040")),    # TEC- D4 R2 back
    cms.PSet(detSelection=cms.uint32(4250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054040")),    # TEC- D5 R2 back
    cms.PSet(detSelection=cms.uint32(4260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058040")),    # TEC- D6 R2 back
#    cms.PSet(detSelection=cms.uint32(4270),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c040")),    # TEC- D7 R2 back
#    cms.PSet(detSelection=cms.uint32(4280),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060040")),    # TEC- D8 R2 back
#    cms.PSet(detSelection=cms.uint32(4290),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064040")),    # TEC- D9 R2 back

    cms.PSet(detSelection=cms.uint32(4310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044060")),    # TEC- D1 R3 back
    cms.PSet(detSelection=cms.uint32(4320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048060")),    # TEC- D2 R3 back
    cms.PSet(detSelection=cms.uint32(4330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c060")),    # TEC- D3 R3 back
    cms.PSet(detSelection=cms.uint32(4340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050060")),    # TEC- D4 R3 back
    cms.PSet(detSelection=cms.uint32(4350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054060")),    # TEC- D5 R3 back
    cms.PSet(detSelection=cms.uint32(4360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058060")),    # TEC- D6 R3 back
    cms.PSet(detSelection=cms.uint32(4370),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c060")),    # TEC- D7 R3 back
    cms.PSet(detSelection=cms.uint32(4380),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060060")),    # TEC- D8 R3 back
#    cms.PSet(detSelection=cms.uint32(4390),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064060")),    # TEC- D9 R3 back

    cms.PSet(detSelection=cms.uint32(4410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044080")),    # TEC- D1 R4 back
    cms.PSet(detSelection=cms.uint32(4420),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048080")),    # TEC- D2 R4 back
    cms.PSet(detSelection=cms.uint32(4430),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c080")),    # TEC- D3 R4 back
    cms.PSet(detSelection=cms.uint32(4440),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050080")),    # TEC- D4 R4 back
    cms.PSet(detSelection=cms.uint32(4450),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054080")),    # TEC- D5 R4 back
    cms.PSet(detSelection=cms.uint32(4460),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058080")),    # TEC- D6 R4 back
    cms.PSet(detSelection=cms.uint32(4470),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c080")),    # TEC- D7 R4 back
    cms.PSet(detSelection=cms.uint32(4480),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060080")),    # TEC- D8 R4 back
    cms.PSet(detSelection=cms.uint32(4490),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064080")),    # TEC- D9 R4 back
    
    cms.PSet(detSelection=cms.uint32(4510),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440a0")),    # TEC- D1 R5 back
    cms.PSet(detSelection=cms.uint32(4520),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480a0")),    # TEC- D2 R5 back
    cms.PSet(detSelection=cms.uint32(4530),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0a0")),    # TEC- D3 R5 back
    cms.PSet(detSelection=cms.uint32(4540),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500a0")),    # TEC- D4 R5 back
    cms.PSet(detSelection=cms.uint32(4550),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540a0")),    # TEC- D5 R5 back
    cms.PSet(detSelection=cms.uint32(4560),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580a0")),    # TEC- D6 R5 back
    cms.PSet(detSelection=cms.uint32(4570),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0a0")),    # TEC- D7 R5 back
    cms.PSet(detSelection=cms.uint32(4580),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600a0")),    # TEC- D8 R5 back
    cms.PSet(detSelection=cms.uint32(4590),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640a0")),    # TEC- D9 R5 back

    cms.PSet(detSelection=cms.uint32(4610),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440c0")),    # TEC- D1 R6 back
    cms.PSet(detSelection=cms.uint32(4620),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480c0")),    # TEC- D2 R6 back
    cms.PSet(detSelection=cms.uint32(4630),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0c0")),    # TEC- D3 R6 back
    cms.PSet(detSelection=cms.uint32(4640),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500c0")),    # TEC- D4 R6 back
    cms.PSet(detSelection=cms.uint32(4650),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540c0")),    # TEC- D5 R6 back
    cms.PSet(detSelection=cms.uint32(4660),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580c0")),    # TEC- D6 R6 back
    cms.PSet(detSelection=cms.uint32(4670),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0c0")),    # TEC- D7 R6 back
    cms.PSet(detSelection=cms.uint32(4680),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600c0")),    # TEC- D8 R6 back
    cms.PSet(detSelection=cms.uint32(4690),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640c0")),    # TEC- D9 R6 back

    cms.PSet(detSelection=cms.uint32(4710),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440e0")),    # TEC- D1 R7 back
    cms.PSet(detSelection=cms.uint32(4720),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480e0")),    # TEC- D2 R7 back
    cms.PSet(detSelection=cms.uint32(4730),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0e0")),    # TEC- D3 R7 back
    cms.PSet(detSelection=cms.uint32(4740),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500e0")),    # TEC- D4 R7 back
    cms.PSet(detSelection=cms.uint32(4750),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540e0")),    # TEC- D5 R7 back
    cms.PSet(detSelection=cms.uint32(4760),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580e0")),    # TEC- D6 R7 back
    cms.PSet(detSelection=cms.uint32(4770),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0e0")),    # TEC- D7 R7 back
    cms.PSet(detSelection=cms.uint32(4780),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600e0")),    # TEC- D8 R7 back
    cms.PSet(detSelection=cms.uint32(4790),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640e0")),    # TEC- D9 R7 back



    cms.PSet(detSelection=cms.uint32(5110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084020")),    # TEC+ D1 R1 back
    cms.PSet(detSelection=cms.uint32(5120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088020")),    # TEC+ D2 R1 back
    cms.PSet(detSelection=cms.uint32(5130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c020")),    # TEC+ D3 R1 back
#    cms.PSet(detSelection=cms.uint32(5140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090020")),    # TEC+ D4 R1 back
#    cms.PSet(detSelection=cms.uint32(5150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094020")),    # TEC+ D5 R1 back
#    cms.PSet(detSelection=cms.uint32(5160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098020")),    # TEC+ D6 R1 back
#    cms.PSet(detSelection=cms.uint32(5170),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c020")),    # TEC+ D7 R1 back
#    cms.PSet(detSelection=cms.uint32(5180),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0020")),    # TEC+ D8 R1 back
#    cms.PSet(detSelection=cms.uint32(5190),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4020")),    # TEC+ D9 R1 back


    cms.PSet(detSelection=cms.uint32(5210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084040")),    # TEC+ D1 R2 back
    cms.PSet(detSelection=cms.uint32(5220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088040")),    # TEC+ D2 R2 back
    cms.PSet(detSelection=cms.uint32(5230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c040")),    # TEC+ D3 R2 back
    cms.PSet(detSelection=cms.uint32(5240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090040")),    # TEC+ D4 R2 back
    cms.PSet(detSelection=cms.uint32(5250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094040")),    # TEC+ D5 R2 back
    cms.PSet(detSelection=cms.uint32(5260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098040")),    # TEC+ D6 R2 back
#    cms.PSet(detSelection=cms.uint32(5270),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c040")),    # TEC+ D7 R2 back
#    cms.PSet(detSelection=cms.uint32(5280),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0040")),    # TEC+ D8 R2 back
#    cms.PSet(detSelection=cms.uint32(5290),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4040")),    # TEC+ D9 R2 back

    cms.PSet(detSelection=cms.uint32(5310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084060")),    # TEC+ D1 R3 back
    cms.PSet(detSelection=cms.uint32(5320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088060")),    # TEC+ D2 R3 back
    cms.PSet(detSelection=cms.uint32(5330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c060")),    # TEC+ D3 R3 back
    cms.PSet(detSelection=cms.uint32(5340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090060")),    # TEC+ D4 R3 back
    cms.PSet(detSelection=cms.uint32(5350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094060")),    # TEC+ D5 R3 back
    cms.PSet(detSelection=cms.uint32(5360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098060")),    # TEC+ D6 R3 back
    cms.PSet(detSelection=cms.uint32(5370),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c060")),    # TEC+ D7 R3 back
    cms.PSet(detSelection=cms.uint32(5380),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0060")),    # TEC+ D8 R3 back
#    cms.PSet(detSelection=cms.uint32(5390),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4060")),    # TEC+ D9 R3 back

    cms.PSet(detSelection=cms.uint32(5410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084080")),    # TEC+ D1 R4 back
    cms.PSet(detSelection=cms.uint32(5420),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088080")),    # TEC+ D2 R4 back
    cms.PSet(detSelection=cms.uint32(5430),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c080")),    # TEC+ D3 R4 back
    cms.PSet(detSelection=cms.uint32(5440),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090080")),    # TEC+ D4 R4 back
    cms.PSet(detSelection=cms.uint32(5450),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094080")),    # TEC+ D5 R4 back
    cms.PSet(detSelection=cms.uint32(5460),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098080")),    # TEC+ D6 R4 back
    cms.PSet(detSelection=cms.uint32(5470),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c080")),    # TEC+ D7 R4 back
    cms.PSet(detSelection=cms.uint32(5480),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0080")),    # TEC+ D8 R4 back
    cms.PSet(detSelection=cms.uint32(5490),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4080")),    # TEC+ D9 R4 back

    cms.PSet(detSelection=cms.uint32(5510),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840a0")),    # TEC+ D1 R5 back
    cms.PSet(detSelection=cms.uint32(5520),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880a0")),    # TEC+ D2 R5 back
    cms.PSet(detSelection=cms.uint32(5530),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0a0")),    # TEC+ D3 R5 back
    cms.PSet(detSelection=cms.uint32(5540),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900a0")),    # TEC+ D4 R5 back
    cms.PSet(detSelection=cms.uint32(5550),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940a0")),    # TEC+ D5 R5 back
    cms.PSet(detSelection=cms.uint32(5560),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980a0")),    # TEC+ D6 R5 back
    cms.PSet(detSelection=cms.uint32(5570),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0a0")),    # TEC+ D7 R5 back
    cms.PSet(detSelection=cms.uint32(5580),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00a0")),    # TEC+ D8 R5 back
    cms.PSet(detSelection=cms.uint32(5590),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40a0")),    # TEC+ D9 R5 back

    cms.PSet(detSelection=cms.uint32(5610),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840c0")),    # TEC+ D1 R6 back
    cms.PSet(detSelection=cms.uint32(5620),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880c0")),    # TEC+ D2 R6 back
    cms.PSet(detSelection=cms.uint32(5630),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0c0")),    # TEC+ D3 R6 back
    cms.PSet(detSelection=cms.uint32(5640),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900c0")),    # TEC+ D4 R6 back
    cms.PSet(detSelection=cms.uint32(5650),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940c0")),    # TEC+ D5 R6 back
    cms.PSet(detSelection=cms.uint32(5660),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980c0")),    # TEC+ D6 R6 back
    cms.PSet(detSelection=cms.uint32(5670),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0c0")),    # TEC+ D7 R6 back
    cms.PSet(detSelection=cms.uint32(5680),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00c0")),    # TEC+ D8 R6 back
    cms.PSet(detSelection=cms.uint32(5690),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40c0")),    # TEC+ D9 R6 back

    cms.PSet(detSelection=cms.uint32(5710),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840e0")),    # TEC+ D1 R7 back
    cms.PSet(detSelection=cms.uint32(5720),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880e0")),    # TEC+ D2 R7 back
    cms.PSet(detSelection=cms.uint32(5730),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0e0")),    # TEC+ D3 R7 back
    cms.PSet(detSelection=cms.uint32(5740),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900e0")),    # TEC+ D4 R7 back
    cms.PSet(detSelection=cms.uint32(5750),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940e0")),    # TEC+ D5 R7 back
    cms.PSet(detSelection=cms.uint32(5760),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980e0")),    # TEC+ D6 R7 back
    cms.PSet(detSelection=cms.uint32(5770),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0e0")),    # TEC+ D7 R7 back
    cms.PSet(detSelection=cms.uint32(5780),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00e0")),    # TEC+ D8 R7 back
    cms.PSet(detSelection=cms.uint32(5790),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40e0"))    # TEC+ D9 R7 back



    )
    )
